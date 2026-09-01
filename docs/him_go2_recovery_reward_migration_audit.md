# HIM Go2 Env/Config/Asset 与随机姿态起身奖励迁移审计

日期：2026-09-01

审计基线：

- Git HEAD：`8d325cb`
- Python 解释器：`/home/kk/legged_mjlab/.venv/bin/python`
- 工作树状态：本轮用户已经改过一部分源码；本文档是本次唯一写入对象。
- 结论边界：本文所有代码片段都是迁移建议，不代表已经写入源码；所有 `PASS/FAIL/NOT_RUN` 只适用于上述工作树和本地 `.venv`。

范围：

- 当前项目：`/home/kk/legged_mjlab`
- 参考项目：`/home/kk/github/uw-himloco-hop`
- 目标：判断当前 `him_go2` 的 env、config、asset 是否已经写好；梳理随机姿态 reset 下应如何迁移鼓励起身的 reward/curriculum；给出后续工作顺序。
- 约束：本次只写文档，不修改训练代码、配置代码、asset、测试或构建脚本；`legged_mjlab/wrappers/` 暂不审，等 env 部分完成后再看。

## 1. 总结论

当前不能认为 `him_go2` 已经写好到可训练状态。

比较准确的判断是：

| 部分 | 当前状态 | 判断 |
|---|---|---|
| HIM 配置骨架 | `HimGo2RoughCfg`、`HimGo2CfgPPO` 已有，声明了 45 单帧观测、6 帧历史、12 动作、HIM runner/policy/algorithm。见 `legged_mjlab/envs/him_go2/him_go2_config.py:6`、`:8`、`:216`。 | 基本写了，但字段仍有多处和 env 实现不一致。 |
| HIM wrapper / 外部旧版 `rsl_rl` 链路 | 本轮按你的要求暂不审 `wrappers/`。这里只把 45D 单帧 actor、12D action 当作 env 输出边界。 | 不作为当前 blocker；等 env 能 `make_env -> reset -> step` 后再回头冻结 wrapper/runner。 |
| asset 资源 | `go2.xml` 和 OBJ mesh 存在；XML 有 floating base、12 个关节、足端 collision/site、IMU sensor。见 `resources/robots/unitree_go2/xmls/go2.xml:41`、`:53`、`:155`。当前 `Go2Asset(HimGo2RoughCfg())` 已能基础构造，读到 XML path、12 个 joint、`base_link`。 | 资源和基础 Python 封装已过第一层 smoke；但 termination contact sensor 仍是不完整占位，action order 也还没冻结。 |
| env 实现 | `HimGo2Env` 已尝试按 mjlab manager 风格组装 scene、action、events、obs、reward、termination。见 `legged_mjlab/envs/him_go2/him_go2_env.py:41`。当前 AST 和 env import 已通过。 | 尚未闭环。本轮按你的要求不再跟踪 `__init__` 构造签名、Perlin terrain 旧写法和 `mdp` alias，剩余重点是 command、action、observation、curriculum、reward、reset。 |
| 随机姿态 reset | 配置有 roll/pitch 全范围随机和 z 随机。见 `him_go2_config.py:174`。 | 目前更像“空中随机姿态跌落”，不是贴地倒姿起身 reset。 |
| 单策略联合训练 | 你明确要让倒地恢复和 locomotion 一起训同一个 policy。当前 reward 仍以 locomotion 项为主，`orientation` 还无法区分正立和倒立。见 `him_go2_env.py:708-728`。 | 不应再写成 recovery-only 任务；应在同一个 env 里混合 fallen/upright reset，并用当前物理状态对 recovery 与 locomotion reward 做连续门控。 |

结论：config 和 asset 资源“已经有雏形”，但 env 还没有达到能训练 HIM 的程度。后面应该先把环境构造、地形、command、action、observation、reward 的 ABI 跑通，再把 recovery reset 和 locomotion reset 放进同一个任务分布里联合训练。

## 2. 当前项目结构和 Env 数据链

高信号文件：

| 文件 | 作用 |
|---|---|
| `legged_mjlab/envs/him_go2/him_go2_config.py` | `him_go2` 任务配置，包含 env/terrain/commands/init_state/control/asset/domain_rand/rewards/PPO。 |
| `legged_mjlab/envs/him_go2/him_go2_env.py` | mjlab `ManagerBasedRlEnv` 风格环境组装。 |
| `legged_mjlab/envs/him_go2/go2_asset.py` | Go2 MJCF、actuator、collision、sensor 配置封装。 |
| `resources/robots/unitree_go2/xmls/go2.xml` | 当前配置指向的 Go2 MJCF。 |
| `legged_mjlab/envs/him_go2/__init__.py` | 注册 `task_id="him_go2"`、`wrapper_name="him"`。 |
| `legged_mjlab/scripts/train.py` | 训练入口，默认 `--task him_go2`。 |
| `legged_mjlab/wrappers/` | 本轮暂不审；只作为后续 wrapper/runner ABI 冻结入口。 |

当前 env 侧数据链意图是：

```text
MjLab native obs:
  actor  [N,45]
  critic [N,235]
Action:
  policy action [N,12] -> JointPositionAction -> q_target
```

关键点：

- `him_go2_config.py:10-14` 声明单帧 45、历史 6、actor 270、privileged 235、动作 12。
- 本轮不判断 `wrappers/` 和外部旧版 `rsl_rl` 是否最终对齐，只要求 env native obs/action/reward 能先构造并 step。
- 因为 wrappers 暂不动，env 侧仍应先保持 actor 单帧 45 和 action 12，不要在当前阶段把目标高度或 mode id 加进 actor obs。

这意味着：单策略联合训练的第一版应该把 fallen/upright 差异主要放在 reset 分布里，command 分布保持一致，reward gate 由 `projected_gravity` 等当前物理状态推导，而不是先改 observation ABI。后面如果要显式给 policy 一个 recovery mode bit 或逐环境目标高度，再统一看 `wrappers/`、runner、部署和 checkpoint 兼容。

## 3. 环境和版本确认

`.venv` 中只读确认到：

| 包 | 版本 |
|---|---|
| `mjlab` | `1.6.0` |
| `mujoco` | `3.11.0` |
| `mujoco-warp` | `3.11.0` |
| `rsl_rl` | `1.0.2` |
| `rsl-rl-lib` | `5.4.2` |

根目录 `setup.py:14-17` pin 了 `mjlab==1.6.0`、`mujoco-warp==3.11.0`、`scipy==1.17.0`。`setup.py:7-24` 同时把当前仓库里的 `rsl_rl/rsl_rl` 打包成顶层 `rsl_rl`。

项目根目录当前未发现 `ARCHITECTURE_CONTEXT.md`。本任务约束为只写本分析文档，因此没有创建或更新该记忆文件。

风险：

- 环境中同时存在旧 `rsl_rl==1.0.2` 和新版 `rsl-rl-lib==5.4.2`，它们都可能占用顶层 `rsl_rl` 包名。
- `TaskRegistry.load_project_rsl()` 有保护逻辑：如果已导入的 `rsl_rl.runners` 没有 `HIMOnPolicyRunner`，会切到项目内源码。见 `legged_mjlab/utils/task_registry.py:135` 之后。
- 后续 smoke 必须打印/确认 `rsl_rl.__file__` 和 `HIMOnPolicyRunner` 来源，避免实际训练时导入新版 ABI。

## 4. Config 审计

### 4.1 已经比较明确的内容

`HimGo2RoughCfg` 已经覆盖了大部分 legged_gym 风格配置：

- 并行环境数：`4096`，见 `him_go2_config.py:9`。
- 单帧 actor obs：`45`，见 `him_go2_config.py:10`。
- HIM 历史长度：`6`，见 `him_go2_config.py:11`。
- privileged obs 意图：`45 + 3 + 187 = 235`，见 `him_go2_config.py:13`。
- 动作维度：`12`，见 `him_go2_config.py:14`。
- terrain generator、height scan 17 x 11 = 187，见 `him_go2_config.py:21-45`。
- Go2 默认关节角，见 `him_go2_config.py:59-76`。
- PD、力矩、action scale、decimation，见 `him_go2_config.py:78-88`。
- Go2 asset path 和 contact 名称，见 `him_go2_config.py:91-98`。
- domain randomization 项比较全，包含 payload、link mass、COM、joint friction/damping/armature、friction、restitution、push、PD、motor zero offset、motor strength、obs/action latency，见 `him_go2_config.py:100-177`。
- PPO 侧指定 `HIMOnPolicyRunner`、`HIMActorCritic`、`HIMPPO`，见 `him_go2_config.py:216-230`。

这些说明配置方向是对的：你是在搭 legged_gym 风格的 mjlab 训练框架，而不是直接采用 mjlab 官方新版 `rsl_rl` 流程。

### 4.2 Config 中当前仍需要修的错配

必须先修的配置/实现错配：

| 问题 | 证据 | 影响 |
|---|---|---|
| action clip 现在改成读取 `normalization.clip_actions`，但传给 `JointPositionActionCfg.clip` 的仍是 tuple。 | `him_go2_env.py:220-234` | mjlab 1.6.0 action term 需要按名称匹配的 dict；tuple 会在构建 action term 时进入错误类型路径。并且 `clip_actions=100` 不是 raw action 的 `[-1,1]` 安全门。 |
| `heading_command=False`，但 env 仍把 `ranges.heading` 传成非空 tuple。 | `him_go2_config.py:52-57`、`him_go2_env.py:477-482`；本地 mjlab 1.6.0 `UniformVelocityCommand.__init__()` 会拒绝该组合。 | command manager 构造会失败；关闭 heading command 时应传 `heading=None`。 |
| `randomize_restitution` 配了，但 env 没有 restitution event。 | `him_go2_config.py:131-133`、`him_go2_env.py:311-458` | 配置不会生效。 |
| reward 名称基本补齐了，但部分函数签名/返回值仍不能被 `RewardManager` 调用。 | `him_go2_config.py:181-202`、`him_go2_env.py:642-660`、`:730-910`；mjlab `RewardManager.compute()` 调 `term_cfg.func(env, **term_cfg.params)`。 | `_reward_dof_acc()` 返回 `None`；`_reward_joint_power/_reward_base_height/_reward_dof_pos_limits/_reward_torque_limits` 没有默认 `asset_cfg` 或 `soft_limit`，而注册时没传 `params`，后续 reward manager 会 TypeError 或 shape check 失败。 |
| `only_positive_rewards=True` 只是保存在 env 属性中，没有接到 mjlab RewardManager。 | `him_go2_env.py:53-54`、`.venv/.../reward_manager.py:116-130` | 不能假设总 reward 会裁剪到非负。 |

起身训练还需要注意：

- 当前 `commands` 是速度命令配置，`num_commands=4` 仍写着 heading，但 `_build_commands()` 实际只对策略返回 3 维 twist。见 `him_go2_config.py:50`、`him_go2_env.py:459-473`。
- 你现在要单策略联合训练，所以不能把速度命令长期清零；fallen/upright reset 应共享同一套 twist command 分布，reset bucket 只用于采样和统计，reward gate 应由当前物理状态推导。
- wrappers 暂不动时，先不要把目标站高 `h*` 或 mode id 加进 actor obs；首版可以用固定 `base_height_target=0.30` 和从当前状态可推断的 recovery gate。

## 5. Asset 审计

### 5.1 资源文件基本齐

当前配置指向：

```text
resources/robots/unitree_go2/xmls/go2.xml
```

XML 中能看到：

- base body 和 floating base：`go2.xml:41-43`。
- 12 个命名关节：如 `FL_hip_joint` 在 `go2.xml:55`，`FR_hip_joint` 在 `go2.xml:80`，`RL_hip_joint` 在 `go2.xml:105`，`RR_hip_joint` 在 `go2.xml:130`。
- 足端 collision geom 和 site：如 `FL_foot_collision` / `site FL` 在 `go2.xml:73-74`，其他腿在 `:98-99`、`:123-124`、`:148-149`。
- IMU gyro、velocimeter、accelerometer：`go2.xml:155-158`。
- `assets/` 下 OBJ mesh 存在。

所以 asset 资源层面不是主要问题。

### 5.2 Python asset 封装当前状态

这部分已经改过一轮，不能再按旧文档说“初始化阶段必失败”。

当前代码观察：

- `go2_asset.py:30-34` 已经先创建 `self.entity` 和 `self.sensor_mgr`，再调用 `_parse_cfg()`。
- `go2_asset.py:74` 已经使用 `self.cfg.init_state.default_joint_angles.keys()`，不再访问不存在的 `self.robot_cfg`。
- `go2_asset.py:88` 已经使用 `self.entity.get_spec()`，不再访问不存在的 `self.asset.entity`。
- `go2_asset.py:95`、`:101` 已经使用 `self.cfg.asset...`。
- 只读 smoke：`Go2Asset(HimGo2RoughCfg())` 已能返回 XML path，`len(joint_names)=12`，并确认 `base_link in body_names`。

剩余问题：

- `go2_asset.py:247-257` 的 `get_termination_contact_sensor()` 仍是不完整占位；当前 `terminate_after_contacts_on=[]` 时未必触发，但接口不能算写好。
- `joint_names` 当前直接沿用 `default_joint_angles` 的 dict 顺序，也就是 type-major 且腿顺序 `FL, RL, FR, RR`。这和 XML leg-major `FL, FR, RL, RR`、部署 YAML 的语义仍没有冻结成同一个 ABI。

### 5.3 动作和部署顺序风险

需要尽早冻结一个唯一的关节/action 顺序：

- `him_go2_config.py:61-75` 的 dict 顺序是先四个 hip，再四个 thigh，再四个 calf，而且腿顺序写成 `FL, RL, FR, RR`。
- XML 自然层级是 leg-major：`FL`、`FR`、`RL`、`RR`，每条腿 hip/thigh/calf。
- `deploy/deploy_mujoco/him_go2/policy/config.yaml:38-41` 注释为 `FL, RL, FR, RR` 的 leg-major；scale 已经能和训练侧对齐，但 default pose 顺序和 torque limit 仍要冻结。
- 部署 YAML 里 `action_scale=0.25`、`hip_scale=0.125`，当前训练 config 里 `action_scale=0.25`、`hip_reduction=0.5`，所以 hip 目标幅度已经能推到同一个 `0.125`；这一项不再作为 mismatch。
- 部署 YAML 小腿 torque limit 是 `35.55`，训练 config 小腿是 `45`。见 `config.yaml:31-34`、`him_go2_config.py:83`。

这是 sim2real 前的 P0 风险。训练、仿真部署、真机部署必须共享同一份 action order、default pose、scale、torque limit 契约，否则策略输出会控制错关节或以错误幅度控制。

## 6. Env 审计

### 6.1 当前导入已通过；本轮不再跟踪三项旧问题

这部分已经改过一轮：

- `him_go2_env.py:673` 和 `:687` 的 command assert 已经改成普通字符串，Python 3.11 AST 检查通过。
- `import legged_mjlab.envs.him_go2.him_go2_env` 已通过。
- `_build_scene()` 调 `_build_sensors()` 时已经传入 `entity_name`，见 `him_go2_env.py:111-115`。
- `_DEFAULT_ASSET_CFG` 已经改成 `SceneEntityCfg(HimGo2RoughCfg.asset.name)`，不再默认指向 `"robot"`，见 `him_go2_env.py:38`。
- `push_robot` 已经补 `interval_range_s` 和 `asset_cfg`，`frandomize_friction` 也已改成 `randomize_friction`，见 `him_go2_env.py:393-411`。
- terrain 实际 generator 分支已经改为 `replace(ROUGH_TERRAINS_CFG)`，见 `him_go2_env.py:153-157`。注释块里的旧 Perlin 方案按你的要求不再作为当前 blocker。

按你的要求，下面三项本轮不再写入当前 blocker：`HimGo2Env.__init__()` 构造签名、Perlin terrain 旧写法、两个模块同名 `mdp` alias。剩余仍需要处理的 env 侧问题是：

| 问题 | 证据 | 影响 |
|---|---|---|
| `heading_command=False` 但 `ranges.heading` 非空，且 `_build_commands()` 无条件 `tuple(ranges.heading)`。 | `him_go2_config.py:52-57`、`him_go2_env.py:477-482`。 | command manager 会拒绝该组合；如果把 config heading 改成 `None` 但不改 env 逻辑，又会在 `tuple(None)` 处 TypeError。 |
| `JointPositionActionCfg.clip` 仍是 tuple。 | `him_go2_env.py:220-234`；mjlab 1.6.0 签名为 `clip: dict[str, tuple] | None`。 | action manager 会把 tuple 当成错误类型；并且这个 clip 夹的是 processed target，不是 raw action。 |
| actor obs 的 `projected_gravity/joint_pos_rel/joint_vel_rel` 没传 `asset_cfg`。 | `him_go2_env.py:537-553`；mjlab 默认 `SceneEntityCfg("robot")`。 | observation manager 会找 `env.scene["robot"]`，而当前实体名是 `go2`。 |

已修好的旧项：`_build_observations()` 中的 `entity_name` 现在来自 `self.robot_cfg.asset.name`，不再在 `super().__init__()` 前访问 `self.cfg.asset.name`。

### 6.2 Reset 当前不是“贴地随机倒姿起身”

当前随机 reset 的配置和实现：

- 默认 root 高度：`init_state.pos = (0, 0, 0.42)`，见 `him_go2_config.py:59-60`。
- reset 随机 z 偏移：`base_pose_z_range = [0.35, 0.5]`，见 `him_go2_config.py:174-175`。
- reset root 使用 `mdp.reset_root_state_uniform`，pose range 中只随机 `z/roll/pitch`，`x/y/yaw` 固定，velocity 空。见 `him_go2_env.py:272-298`。
- mjlab 1.6.0 的 `reset_root_state_uniform` 是在 `default_root_state` 上加 pose sample，再加 env origin。见 `.venv/lib/python3.11/site-packages/mjlab/envs/mdp/events.py:170-190`。
- joint reset 是 `position_range=(-0.0,0.0)`、`velocity_range=(-0.0,0.0)`，见 `him_go2_env.py:300-308`。

因此实际初始 base z 约是：

```text
0.42 + [0.35, 0.50] = [0.77, 0.92] m
```

这不是“机器人已经倒在地上再起身”，而更像“从空中带随机 roll/pitch 掉下来”。这会引入两个问题：

- 策略可能学到空中翻身、蹬地弹跳等非目标行为。
- 高度奖励、姿态奖励和速度惩罚会和落地冲击耦合，训练信号不干净。

如果目标是恢复起身，reset 应更接近参考仓库：

- base z 使用低高度，例如 `0.10-0.20 m` 的绝对高度或相对 terrain origin 的高度。
- roll/pitch/yaw 一起随机，姿态采样最好按倾角分桶，而不是独立均匀 roll/pitch。
- joint position 在 default 附近小扰动，例如 `q_default + U(-0.3,0.3)`，并 clamp 到关节限位。
- root velocity 和 joint velocity 初期置零。
- 恢复成功不要触发 reset；episode 继续到 timeout，让“越早站起来，越久拿站稳奖励”成为主驱动力。

### 6.3 Termination 目前只有 timeout

当前 `_build_terminations()` 只注册了 `time_out`，见 `him_go2_env.py:591-600`。

对 recovery 任务来说，这是合理的方向：初始就可能 base/thigh/calf 接触地面，不能用这些 contact 作为失败终止。成功也不应直接终止。需要补的是：

- 成功 predicate 作为 reward/metric/curriculum 状态，不作为 termination。
- 可选的硬安全终止只应针对 NaN、仿真爆炸、极端高度、关节严重越界等，不应把“身体接触地面”当失败。

### 6.4 Reward 当前还不是“恢复 + 行走”单策略联合目标

当前 reward scales 在 `him_go2_config.py:180-202`，包括速度跟踪、base height、orientation、feet air time、stand still 等行走任务项。

主要问题：

- `orientation` 当前返回 `g_x^2 + g_y^2`，见 `him_go2_env.py:708-728`。这个量只表示“base 是否水平”，不能区分正立和完全倒立：正立与倒立的 `g_x^2+g_y^2` 都是 0。随机倒姿下，这会允许“倒立但水平”成为局部最优。
- `tracking_lin_vel`、`tracking_ang_vel` 对 locomotion 是主任务，不能在单策略联合训练里全局关闭；但机器人明显非正立时应该被 `w_loco` 暂时压低，否则策略会一边没站起来一边追速度命令。
- `feet_air_time`、`foot_clearance` 是步态项，不是起身项；联合训练里建议只在 `w_loco` 接近 1 时生效，倒地恢复阶段不生效。
- `collision` 如果针对 base/thigh/calf，恢复初始接触地面时会惩罚正常状态；联合训练里应按 reset grace 和正立 gate 过滤，不能全局惩罚倒地接触。
- `stand_still` 只适合低命令站立，它不是恢复成功项。联合训练里应继续按命令大小门控，不能替代 `joint_to_default/recovery_success`。
- `only_positive_rewards=True` 不会自动生效，因为 mjlab RewardManager 实际对 term 逐项 `value * weight * dt` 后求和，没有看到总 reward 裁剪。见 `.venv/.../reward_manager.py:116-130`。

## 7. 参考仓库 recovery 设计

参考仓库关键文件：

| 文件 | 关键内容 |
|---|---|
| `/home/kk/github/uw-himloco-hop/legged_gym/envs/dog/dog_recovery_config.py:20-176` | `DogRecoveryCfg`，恢复任务配置、奖励权重、分桶课程。 |
| `/home/kk/github/uw-himloco-hop/legged_gym/envs/base/legged_robot.py:210-259` | 成功 predicate、平稳门槛、成功不终止。 |
| `/home/kk/github/uw-himloco-hop/legged_gym/envs/base/legged_robot.py:745-763` | 速度命令清零、采样目标站高。 |
| `/home/kk/github/uw-himloco-hop/legged_gym/envs/base/legged_robot.py:801-879` | joint/root reset：低高度、roll/pitch/yaw 随机、速度清零。 |
| `/home/kk/github/uw-himloco-hop/legged_gym/envs/base/legged_robot.py:900-942` | 分桶成功率统计和低成功率桶过采样。 |
| `/home/kk/github/uw-himloco-hop/legged_gym/envs/base/legged_robot.py:1825-1918` | 起身 reward 函数。 |
| `/home/kk/github/uw-himloco-hop/legged_gym/scripts/play_recovery.py:106-132` | 评估时按 episode 内是否曾成功统计成功率。 |

### 7.1 Reset 语义

参考 recovery 每次 reset：

- 关节：`q = q_default + U(-0.3, 0.3)`，`qdot=0`。见 `legged_robot.py:801-806`。
- root：`x/y += U(-0.3,0.3)`，`z=U(0.10,0.20)`，`yaw=U(-0.5,0.5)`，root velocity 为 0。见 `legged_robot.py:869-879`。
- roll/pitch：按倾角桶采样，总倾角桶见 `dog_recovery_config.py:130-135`，采样逻辑见 `legged_robot.py:827-846`。
- 命令：速度维清零，目标站高 `h*` 在 `[0.20,0.35]` 采样。见 `dog_recovery_config.py:36-48`、`legged_robot.py:745-763`。

最后一条只适用于参考仓库的 recovery-only 任务；迁移到你当前单策略联合训练时，速度命令清零只能作为 debug smoke，不能作为正式训练分布。

### 7.2 成功语义

参考 recovery 的成功判定：

```text
upright:      projected_gravity_z < -0.9
height_ok:    abs(base_z - h*) < 0.05
joint_ok:     mean(abs(q - q_default)) < 0.2
smooth_ok:    recent_max_agitation < 5.0
recovered:    upright & height_ok & joint_ok & smooth_ok
```

证据：

- upright 不用 `abs()`，明确避免倒立骗成功。见 `legged_robot.py:232-234`。
- height tolerance 见 `legged_robot.py:235-237`。
- joint tolerance 见 `legged_robot.py:238-240`。
- smooth_success 门控见 `legged_robot.py:222-249`。
- 成功不再写入 `reset_buf`，只更新 `ever_success_buf`。见 `legged_robot.py:251-259`。

这个设计值得迁移：成功只作为统计和奖励，不作为 episode termination。这样策略会被激励尽早站起并保持到 timeout。迁移时建议把 `recent_max_agitation` 写成每控制步更新一次的状态量，例如 `A_t=max(decay*A_{t-1}, normalized_agitation_t)`；不要在 reward 函数里临时看当前一帧，否则高速翻滚后恰好一帧静止也可能被计为成功。

### 7.3 Reward 设计

参考 reward 的原始 scale 会在旧 legged_gym 里乘 `dt=0.02`。迁移到 mjlab 1.6.0 时，RewardManager 默认 `scale_rewards_by_dt=True`，见 `.venv/lib/python3.11/site-packages/mjlab/envs/manager_based_rl_env.py:157` 和 `.venv/.../reward_manager.py:116-128`，所以不要再手动乘 dt。

| Reward | 公式/语义 | 参考 scale | 迁移建议 |
|---|---|---:|---|
| `upright_linear` | `(1 - g_z) / 2`，正立约 1，倒立约 0。见 `legged_robot.py:1825-1829`。 | `+3.0` | 必迁移，替代当前 `orientation` 主项。 |
| `stand_height` | `exp(-(z-h*)^2 / sigma_h^2) * 1[g_z < -0.7]`，参考实现里 `sigma_h^2=0.05 m^2`，不是 `sigma_h=0.05 m`。见 `legged_robot.py:1831-1837`。 | `+1.5` | 可迁移；45D 首版只能用固定 `base_height_target`，46D 才能使用逐环境 `h*`。 |
| `joint_to_default` | `exp(-mean((q-q0)^2) / 0.25)`。见 `legged_robot.py:1839-1844`。 | `+1.5` | 可迁移，但建议正立附近再强化，避免倒地阶段硬拽腿。 |
| `recovery_success` | 首次成功给 `10.0`。见 `legged_robot.py:1855-1861`。 | `+1.0` | 思路可迁移，但参考实现置位顺序有 bug，不能原样搬。 |
| `upside_down_penalty` | `-clamp(g_z, min=0)`，压制完全倒立。见 `legged_robot.py:1866-1869`。 | `+2.0` | 可迁移；注意 sign convention。 |
| `action_rate` | `sum((a_t-a_{t-1})^2)`。见 `legged_robot.py:1871-1874`。 | `-0.2` | 初期不要太大，防止压制翻身发力。 |
| `lin_vel_xy` | `sum(v_xy^2)`。见 `legged_robot.py:1882-1884`。 | `-0.5` | 原地恢复可保留。 |
| `lin_vel_z` | `v_z^2`。见 `legged_robot.py:1886-1890`。 | `-0.5` | 防止腾空翻滚作弊。 |
| `ang_vel_xy` | `sum(omega_xy^2)`。见 `legged_robot.py:1892-1894`。 | `-0.1` | 保留低权重；过大时会压住起身动作。 |
| `torques` | `sum(tau^2)`。见 `legged_robot.py:1896-1898`。 | `-1e-5` | 可保留，但需在 mjlab 中确认实际 actuator force 接口。 |
| `dof_vel` | `sum(qdot^2)`。见 `legged_robot.py:1906-1908`。 | `-1e-4` | 可保留。 |
| `dof_acc` | `sum((last_qdot-qdot)^2)/dt`。见 `legged_robot.py:1910-1918`。 | `-1e-6` | 可保留，但当前项目缺 `_reward_dof_acc`。 |

不能按 recovery-only 语义原样全局迁移的点：

- `tracking_lin_vel`、`tracking_ang_vel` 在你的单策略任务里要保留，但必须乘 `w_loco`，不能在倒地阶段满权重发力。
- `feet_air_time`、`foot_clearance` 是 locomotion/步态项，保留时也应乘 `w_loco`，且 rough terrain 上要用局部地面高度。
- 原来的 `orientation` 不能作为 recovery 主项，因为它不能区分正立和倒立。
- 初始接触会命中的 base/thigh/calf collision 惩罚需要 reset grace 和正立 gate，不能全局惩罚。
- `stand_still` 只适合零命令且正立状态，不应替代 `joint_to_default` 或 recovery metric。

### 7.4 参考仓库不能原样照搬的点

参考实现也有问题，迁移时要修正：

1. `recovery_success` 的首次成功奖励可能恒为 0。`check_termination()` 在奖励计算前先 `already_succeeded_buf[recovered] = True`，见 `legged_robot.py:257-259`；而 reward 又用 `success_buf & ~already_succeeded_buf`，见 `legged_robot.py:1855-1861`。mjlab 迁移时要保证首次成功 reward 在 latch 更新前计算，或单独维护 `first_success_this_step`。
2. 训练成功判定包含 `smooth_success`，但 `play_recovery.py:119-125` 的统计只检查 upright/height/joint，没有检查 smooth gate，评估可能高估成功率。
3. 分桶 roll/pitch 用的是小角近似 `roll=theta*cos(phi)`、`pitch=theta*sin(phi)`，在 `theta` 接近 pi 时不再严格等于 SO(3) 上的均匀倾角采样。见 `legged_robot.py:840-846`。
4. 注释说 z 随倒地程度变化，但代码实际是独立 `U(0.10,0.20)`，见 `legged_robot.py:872-875`。

## 8. 单策略联合训练方案

你现在的目标不是拆出 `him_go2_recovery` 单独训练，而是让同一个 policy 同时学会倒地恢复和 locomotion。这里不能照抄参考仓库的 recovery-only 语义：参考仓库会把速度命令清零，并单独采样目标站高；你的正式联合训练版应让倒姿 reset 和正常 reset 都处在同一个 velocity-command 分布里。

### 8.1 推荐的任务分布

首版仍保持 env actor 单帧 45 和 action 12，不动 `wrappers/`：

```text
single policy:
  fallen reset episodes  -> 低高度随机倒姿 + 同一套 twist command 分布
  upright reset episodes -> 正常站姿/轻扰动 + 同一套 twist command 分布
  reward handoff         -> 由当前物理状态连续 gate，不由隐藏 mode 硬切
```

推荐 reset 采样语义：

```python
is_fall = torch.rand(n, device=env.device) < fall_reset_probability

# 两组都保留同一 velocity-command 分布。
# reset label 只用于采样、bucket 统计和 curriculum，不用于改变同一观测下的奖励目标。
reset_upright(env_ids[~is_fall])
reset_fallen(env_ids[is_fall], pose_bucket=bucket[is_fall])
```

可以在 debug smoke 里临时把 command 置零，单独看起身 reward 方向是否正确；但正式单策略训练不要长期把 fallen reset 的速度命令清零。否则策略站起来后还要再学一个分布切换，且恢复阶段没有见过“站起后立刻按命令走”的样本。

### 8.2 Gate 语义

联合 reward 的 gate 建议由当前物理状态推导，而不是由 `is_fall_reset`、`is_recovery_episode` 这类隐藏标签直接控制。原因是 actor observation 目前没有 mode id；如果 reward 用隐藏 mode 硬切，同一个 45D 观测可能对应不同目标，会给 PPO 制造 POMDP 噪声。更糟的是，正常 locomotion episode 中途摔倒时，如果 gate 只看 reset 标签，策略反而收不到 recovery 信号。

建议先用 `projected_gravity_b[:, 2]` 做连续 smoothstep：

```python
def _loco_recovery_weights(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    asset = env.scene[asset_cfg.name]
    grav_z = asset.data.projected_gravity_b[:, 2]

    # 正立时 projected gravity z 约为 -1；倒立时约为 +1。
    x = (((-grav_z) - 0.65) / 0.25).clamp(0.0, 1.0)
    w_loco = x * x * (3.0 - 2.0 * x)
    w_recovery = 1.0 - w_loco
    return w_loco, w_recovery
```

工程含义：

| 类别 | reward | 联合训练建议 |
|---|---|---|
| 全程正立/安全 | `upright_linear`、`upside_down_penalty`、`dof_pos_limits`、`torque_limits` | 不依赖隐藏 reset mode；`upright_linear` 让任何摔倒状态都有恢复方向。 |
| recovery 加权 | `stand_height`、`joint_to_default`、`hip_pos`、`lin_vel_xy` | 乘 `w_recovery` 或在非正立阶段更强，避免行走时被固定站姿奖励拖僵。 |
| locomotion 加权 | `tracking_lin_vel`、`tracking_ang_vel`、`feet_air_time`、`foot_clearance`、`feet_stumble` | 乘 `w_loco`，机器人没站起来时不要强追速度和步态。 |
| 平滑/能耗 | `action_rate`、`smoothness`、`joint_power`、`dof_vel`、`dof_acc` | 可以全程低权重；站稳后可更强，倒地初期不要压住必要的大动作。 |
| 特殊项 | `collision`、`stand_still`、`recovery_success` | `collision` 需要 reset grace 和正立 gate；`stand_still` 只用于零命令且正立；`recovery_success` 首版建议先做 metric，不作为主 reward。 |

### 8.3 单策略 reward 配置建议

一个合理的首版不是“recovery-only scales”，而是“locomotion 与 recovery 都保留、函数内部按物理状态 gate”的联合 scales：

```python
joint_training_scales = {
    # state-wide / recovery shaping
    "upright_linear": 3.0,
    "stand_height": 1.0,
    "joint_to_default": 0.8,
    "upside_down_penalty": 1.0,

    # locomotion driver
    "tracking_lin_vel": 1.0,
    "tracking_ang_vel": 0.5,
    "feet_air_time": 0.1,
    "foot_clearance": -0.01,

    # shared constraints
    "lin_vel_xy": -0.15,
    "lin_vel_z": -0.25,
    "ang_vel_xy": -0.05,
    "action_rate": -0.02,
    "smoothness": -0.01,
    "dof_pos_limits": -2.0,
    "torque_limits": -0.2,
    "joint_power": -2e-5,

    # replace old ambiguous orientation/base-height terms
    "orientation": 0.0,
    "base_height": 0.0,
}
```

权重只是起点，真正重要的是 gate。没有 gate 的联合 reward 会出现两个典型坏局部最优：

- 倒地时直接收到满权重速度 tracking，策略还没站起来就被要求追速度，容易抽搐或乱蹬。
- 行走时一直收到强 `joint_to_default/stand_height` 奖励，策略会倾向僵硬站姿，牺牲步态摆动。

### 8.4 Reset 和 curriculum

联合训练的 reset 不应只有“全倒姿”或“全行走”，而是一个混合采样器：

```text
with probability fall_reset_probability:
  sample recovery bucket
  reset low fallen pose
  sample the same twist command distribution
  mark reset_source = fallen
else:
  reset nominal upright pose
  sample the same twist command distribution
  mark reset_source = upright
```

`fall_reset_probability` 不建议固定不变。更稳的做法是：

- P0 smoke：`fall_reset_probability=0.0`，只验证 locomotion env 可以 reset/step。
- P1 recovery debug：`fall_reset_probability=1.0`，只验证倒姿 reset 和 recovery reward shape/方向，必要时临时零命令。
- P2 联合训练：从 `fall_reset_probability=0.2~0.4` 开始，保证 locomotion 样本仍足够多。
- P3 弱项过采样：对 fallen reset 的姿态 bucket 使用低成功率过采样，但对整体倒姿 reset 比例设上限，避免训练分布被倒姿完全吞掉。

姿态 bucket 仍可迁移参考仓库，但统计只应该看 fallen reset 的 episode：

```python
trials += torch.bincount(bucket, minlength=num_buckets)
wins += torch.bincount(
    bucket,
    weights=ever_recovered.float(),
    minlength=num_buckets,
)
success_rate = wins / trials.clamp_min(1)
raw_weights = weight_floor + 1.0 - success_rate
weights = raw_weights / raw_weights.sum().clamp_min(1.0e-6)
```

关节扰动也不要先采样再粗暴 clamp。更好的写法是在可行区间内直接采样，避免大量样本堆到限位边界：

```python
lo = torch.maximum(q_min + margin, q_default - delta)
hi = torch.minimum(q_max - margin, q_default + delta)
q = lo + torch.rand_like(q_default) * (hi - lo)
```

成功率统计应该按 episode 内是否曾成功，而不是 timeout 瞬间是否成功。第一次 reset 不应写入失败样本。上 rough terrain 后，高度成功判定和 `foot_clearance` 都要改成相对局部地面高度，不能只看世界 z。

### 8.5 当前阶段不要动 wrappers

因为你说 `wrappers/` 先不用管，所以本文档不再建议现在新增 46D height-command ABI、recovery wrapper 或 runner 改动。当前 env 阶段的边界是：

- actor 单帧继续保持 45。
- 固定 `base_height_target=0.30`，不要采样 actor 看不到的逐环境 `h*`。
- recovery/locomotion 的区别先通过 reset 姿态、同分布速度命令和物理状态 reward gate 表达。
- 等 env native `make_env -> reset -> step` 跑通后，再单独回来看 wrapper history、critic 维度、terminal obs 和旧版 `rsl_rl`。

## 9. 具体修改说明（文档级 Patch Guide）

这一节回答你强调的三个问题：**具体哪里错、为什么错、改完应该长什么样**。这里的代码都是“建议补丁片段”，本次没有写入源码；真正改代码时建议按 P0 到 P3 分批提交，每批只跑对应 smoke。

### 9.1 P0 当前仍应修：command、action、observation、reward contract

| 位置 | 错在哪 | 为什么错 | 推荐改法 |
|---|---|---|---|
| `him_go2_env.py:477-482` | `heading_command=False`，但仍传 `heading=tuple(ranges.heading)`。 | mjlab 1.6.0 的 `UniformVelocityCommand` 不接受“关闭 heading command 但 heading range 非空”的组合。 | `heading_range = tuple(ranges.heading) if heading_command else None`，然后传 `heading=heading_range`。 |
| `him_go2_env.py:220-234` | `JointPositionActionCfg.clip` 仍是 tuple。 | mjlab action term 需要 dict；而且这里 clip 的是 processed target，不是 policy raw action。 | 生成 `{joint_name: (q_min, q_max)}` 或 `{joint_name: (q0-scale, q0+scale)}` 这类 per-joint target clip dict。 |
| `him_go2_env.py:537-553` | actor 的 `projected_gravity/joint_pos_rel/joint_vel_rel` 没传 `asset_cfg`。 | mjlab 这些 observation helper 默认查 `SceneEntityCfg("robot")`，当前 scene 注册名是 `go2`。 | 显式传 `SceneEntityCfg(entity_name)` 或带 joint_names 的 `joint_cfg`。 |
| `him_go2_env.py:622-626` | terrain curriculum 只传 `command_name`，没传当前 asset cfg。 | `terrain_levels_vel` 很可能默认查 `"robot"`，和当前 `go2` 不一致。 | 如果该 mjlab 函数支持 `asset_cfg`，显式传 `SceneEntityCfg(entity_name)`；如果不支持，需要包装一个 go2 版本 curriculum。 |
| `him_go2_env.py:642-660` | reward builder 仍只用 `getattr(self, "_reward_" + name)`，不传 term params。 | 多个 reward 函数需要 `asset_cfg` / `soft_limit`；`RewardManager` 不会自动帮你补。 | 改成显式 `reward_map`，每个非零 scale 都绑定函数和 params。 |
| `him_go2_env.py:730-735` | `_reward_dof_acc()` 仍是 `pass`。 | RewardManager 要求每个 reward term 返回 `[num_envs]` tensor，`None` 会失败。 | 实现返回 tensor，或在实现前把 `dof_acc` scale 置零。 |

本轮按你的要求不再跟踪的旧项：

- `him_go2_env.py:39-60` 构造签名。
- `him_go2_env.py` 注释块中的 Perlin terrain 旧写法；实际 generator 分支已经走 `replace(ROUGH_TERRAINS_CFG)`。
- `him_go2_env.py:11-12` 两个模块同名 `mdp` alias。

已不再列为 P0 blocker 的旧项：

- `him_go2_env.py:673`、`:687` 的 f-string assert 已修好，AST 检查通过。
- `_build_scene()` 调 `_build_sensors()` 已传 `entity_name`，见当前 `him_go2_env.py:111-115`。
- `_DEFAULT_ASSET_CFG` 已经跟随 `HimGo2RoughCfg.asset.name`，不再默认指向 `"robot"`。
- `push_robot` 已补 `interval_range_s` 和 `asset_cfg`。
- 摩擦随机化字段名已改成 `randomize_friction`。

### 9.2 `Go2Asset` 当前只剩 ABI 和占位接口风险

已不再列为 blocker 的旧项：

- `Go2Asset.__init__()` 初始化顺序已经调整，`self.entity` / `self.sensor_mgr` 在 `_parse_cfg()` 前创建。
- `_parse_cfg()` 中的 `self.robot_cfg`、`self.asset.entity`、`self.asset.cfg` 旧引用已经改成 `self.cfg` / `self.entity`。
- `ContactSensorCfg` 上的 `exclude_parent_body=True` 已删除。
- `RayCastSensorCfg` 上的 `history_length` 已删除，并保留了 raycast 合理的 `exclude_parent_body=True`。
- `get_all_sensors()` 已接收并传递 `entity_name`。

当前仍需要处理的是：

| 位置 | 问题 | 为什么还要修 |
|---|---|---|
| `go2_asset.py:247-257` | `get_termination_contact_sensor()` 仍是不完整占位，`name=""`，`ContactMatch()` 空参数。 | 当前 `terminate_after_contacts_on=[]` 时可能不走到，但一旦启用 termination contact，这个接口会失败。 |
| `go2_asset.py:74` | `joint_names` 直接沿用 `default_joint_angles` dict 顺序。 | 这个顺序是 type-major / `FL, RL, FR, RR` 混排，和 XML leg-major、部署 YAML 之间仍未冻结 ABI。 |

建议先在 `Go2Asset` 里冻结策略关节顺序。首版新训练推荐用 XML leg-major 顺序 `FL, FR, RL, RR`，每条腿 `hip, thigh, calf`。如果你要兼容已有部署 YAML 或旧 checkpoint，需要在 ABI 文档里明确另一套顺序并做 name-based remap，不能靠 dict 顺序。

```python
GO2_POLICY_JOINT_ORDER = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)
```

### 9.3 Action、command、observation 的 ABI 修法

| 位置 | 错在哪 | 为什么错 | 推荐改法 |
|---|---|---|---|
| `him_go2_env.py:220-234` | 字段名已经改成 `hip_reduction`、`normalization.clip_actions` 和 `actuator_names`，但 `clip` 仍给 tuple，且容易被误解为 raw action 的 `[-1,1]` 限幅。 | mjlab 1.6.0 的 `JointPositionAction` 先算 `raw * scale + q0`，再对 processed target 做 `clip`；它不是 policy raw action 安全门。`JointPositionActionCfg.clip` 也会走 name resolver，建议使用按关节名匹配的 dict。 | raw action 需要在 wrapper/runner 边界做 finite + clamp；env action term 的 `clip` 应表达每关节目标位置安全范围。 |
| `him_go2_env.py:477-482` | `heading_command=False` 但仍传 `heading=tuple(ranges.heading)`。 | 本地 mjlab 1.6.0 `UniformVelocityCommand.__init__()` 会拒绝 `ranges.heading` 非空且 `heading_command=False` 的组合。 | `heading = None if not heading_command else tuple(ranges.heading)`。 |
| `him_go2_env.py:537`、`:544`、`:552` | `projected_gravity/joint_pos_rel/joint_vel_rel` 没传 `asset_cfg`。 | 这些 mjlab 默认 `SceneEntityCfg("robot")`，当前实体是 `go2`。 | 显式传 `SceneEntityCfg(entity_name, ...)`。 |

已修好但仍建议保留一次 smoke 核验的点：

- `hip_reduction` 字段名已经对齐 config。
- `clip_actions` 当前来自 `normalization.clip_actions`，不再访问不存在的 `control.action_clip`。
- `JointPositionActionCfg` 已使用 `actuator_names`，不再是 `actuator_name`。
- observation 构建阶段已经使用 `self.robot_cfg.asset.name`，不再在 `super().__init__()` 前访问 `self.cfg.asset.name`。

Action 修改后片段：

```python
def _build_actions(self):
    entity_name = self.robot_cfg.asset.name
    action_scale = float(self.robot_cfg.control.action_scale)
    hip_scale = action_scale * float(self.robot_cfg.control.hip_reduction)
    raw_action_limit = 1.0
    default_q = self.robot_cfg.init_state.default_joint_angles
    action_scales = {}
    target_clip = {}
    for joint_name in self.asset.joint_names:
        scale = hip_scale if "hip" in joint_name else action_scale
        action_scales[joint_name] = scale
        center = float(default_q[joint_name])
        target_clip[joint_name] = (
            center - raw_action_limit * scale,
            center + raw_action_limit * scale,
        )

    return {
        "joint_position": JointPositionActionCfg(
            entity_name=entity_name,
            actuator_names=self.asset.joint_names,
            scale=action_scales,
            use_default_offset=True,
            preserve_order=True,
            clip=target_clip,
        )
    }
```

但这仍然不是完整安全闭环。训练或部署入口还需要在 `env.step(actions)` 前做 raw action gate，否则 unbounded Normal policy 的异常输出仍会进入 action term；遇到 NaN/Inf 必须 fail-closed，不能只靠 `torch.clamp`。

```python
def sanitize_policy_action(actions: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(actions).all():
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
    return actions.clamp(-1.0, 1.0)
```

部署到真实硬件时还要再加 `q_safe_min/max`、`dq_safe_limit`、`delta_q_per_tick_limit` 和硬件侧 `tau_send` 限幅。本文这段只解决仿真训练 ABI，不构成真机安全边界。

这里也故意不用 `{".*hip_.*": hip_scale, ".*": action_scale}` 这种重叠 regex。mjlab 的 `resolve_matching_names_values()` 会在同一个 target 被多个 key 匹配时报错；每关节 exact key 更啰嗦，但不会产生覆盖歧义。

Command 修改后片段：

```python
def _build_commands(self, debug_vis):
    entity_name = self.robot_cfg.asset.name
    ranges = self.robot_cfg.commands.ranges
    period = self.robot_cfg.commands.resampling_time
    if isinstance(period, (int, float)):
        resampling_time_range = (float(period), float(period))
    else:
        resampling_time_range = tuple(period)

    heading_command = bool(self.robot_cfg.commands.heading_command)
    heading_range = tuple(ranges.heading) if heading_command else None

    return {
        "twist": UniformVelocityCommandCfg(
            entity_name=entity_name,
            resampling_time_range=resampling_time_range,
            rel_standing_envs=0.05,
            rel_forward_envs=0.25,
            debug_vis=debug_vis,
            heading_command=heading_command,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=tuple(ranges.lin_vel_x),
                lin_vel_y=tuple(ranges.lin_vel_y),
                ang_vel_z=tuple(ranges.ang_vel_yaw),
                heading=heading_range,
            ),
        )
    }
```

正式单策略联合训练不要在 fallen reset 上长期使用零速度命令。可以只在 debug smoke 中临时把 `lin_vel_x/y/ang_vel_z` 全部设为 `(0.0, 0.0)`，用来验证起身 reward 方向和 reset 稳定性；一旦进入联合训练，fallen/upright reset 都应继续使用同一套 twist command 采样分布。

Observation 修改后片段：

```python
def _build_observations(self, play: bool):
    entity_name = self.robot_cfg.asset.name
    joint_cfg = SceneEntityCfg(
        entity_name,
        joint_names=self.asset.joint_names,
        preserve_order=True,
    )
    asset_cfg = SceneEntityCfg(entity_name)

    clip_val = self.robot_cfg.normalization.clip_observations
    clip_obs = (-clip_val, clip_val)

    actor_terms = {
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
            clip=clip_obs,
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": f"{entity_name}/imu_ang_vel"},
            scale=self.robot_cfg.normalization.obs_scales.ang_vel,
            clip=clip_obs,
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            params={"asset_cfg": asset_cfg},
            clip=clip_obs,
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": joint_cfg},
            scale=self.robot_cfg.normalization.obs_scales.dof_pos,
            clip=clip_obs,
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": joint_cfg},
            scale=self.robot_cfg.normalization.obs_scales.dof_vel,
            clip=clip_obs,
        ),
        "last_action": ObservationTermCfg(
            func=mdp.last_action,
            params={"action_name": "joint_position"},
            clip=clip_obs,
        ),
    }

    critic_terms = copy.deepcopy(actor_terms)
    for term in critic_terms.values():
        term.delay_min_lag = 0
        term.delay_max_lag = 0
        term.noise = None

    critic_terms.update({
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": f"{entity_name}/imu_lin_vel"},
        ),
        "height_scan": ObservationTermCfg(
            func=mdp.height_scan,
            params={"sensor_name": "height_scan"},
            scale=self.robot_cfg.normalization.obs_scales.height_measurements,
        ),
    })
```

注意：上面片段省略了你现有 noise/delay 的完整接线，只展示必须修的 ABI 点。真正改时可以把原来的 `noise`、`delay_min_lag`、`delay_max_lag` 保留下来，但每个 actor term 的最终 shape 必须还是：

```text
command 3 + base_ang_vel 3 + projected_gravity 3 + joint_pos 12 + joint_vel 12 + last_action 12 = 45
```

### 9.4 Event 和 domain randomization 的修法

| 位置 | 错在哪 | 为什么错 | 推荐改法 |
|---|---|---|---|
| `him_go2_config.py:132-133` | 配了 `randomize_restitution`，env 没有实际接线。 | 本地 mjlab 1.6.0 的 `dr.geom` 没有直接 `geom_restitution` 函数。 | 先在文档/配置里标成未接线；要么补 mjlab DR 函数，要么通过 material/pair/XML 方案处理，不要写一个不存在的 `dr.geom_restitution`。 |
| `him_go2_env.py:272-298` | 用 `reset_root_state_uniform` 实现低姿态 recovery。 | 该函数是 `default_root_state + pose_sample + env_origin`，当前默认 z=0.42，再加 `[0.35,0.50]` 变成高空 reset。 | 行走任务可保留；recovery 任务要换自定义低高度 reset event。 |

已修好的旧项：`push_robot` 现在有 `interval_range_s` 和 `asset_cfg`，摩擦随机化也已经使用 `randomize_friction`。它们不再列为当前 blocker。

单策略联合训练下，DR 不必永久关闭，但建议分阶段打开。先用 `fall_reset_probability=0` 验证 locomotion env，再用 `fall_reset_probability=1` 验证 fallen reset/reward，最后把 fallen reset 按比例混进训练；push、friction、latency、motor strength 不要在第一轮全部拉满，否则无法判断失败来自 reset、reward、action ABI 还是随机化。

### 9.5 Reward 注册参数和单策略联合 reward

| 位置 | 错在哪 | 为什么错 | 推荐改法 |
|---|---|---|---|
| `him_go2_env.py:730-735` | `_reward_dof_acc()` 只有 `pass`。 | mjlab `RewardManager` 会检查 term 返回 shape；`None` 会直接失败。 | 返回 `[num_envs]` tensor，或者在实现前把 `dof_acc` scale 置零。 |
| `him_go2_env.py:738-757`、`:871-910` | `_reward_joint_power/_reward_base_height/_reward_dof_pos_limits/_reward_torque_limits` 需要 `asset_cfg`，`torque_limits` 还需要 `soft_limit`，但注册时没有传 `params`。 | `RewardManager.compute()` 调用方式是 `term_cfg.func(env, **term_cfg.params)`；当前 `RewardTermCfg` 只给了 `func/weight`。 | 用显式 `reward_map` 给每个 term 绑定 `params`，或者给函数参数提供安全默认值。 |
| `him_go2_env.py:708-728` | `_reward_orientation()` 惩罚 `g_x^2 + g_y^2`。 | 它只能判断 base 是否水平，不能区分正立和倒立；倒立水平时也接近 0。 | recovery 中不要用这个主项，改用 `upright_linear = 0.5 * (1 - g_z)`。 |
| `him_go2_config.py:180-202` | recovery 与 locomotion 的 reward 没有 gate。 | 单策略联合训练时，速度 tracking 和步态项对 locomotion 是对的，但在倒地阶段会压制起身；recovery 项如果全程开，又会让行走僵硬。 | 保留两类 reward，但按当前物理状态得到的 `w_recovery/w_loco` 门控，不要用 actor 看不到的隐藏 mode 硬切。 |
| `him_go2_config.py:204` | `only_positive_rewards=True`。 | mjlab RewardManager 没有 legged_gym 那种总 reward clamp。 | 联合训练首版设为 False；如果一定要 clamp，应在 env reward aggregation 边界显式实现并测试。 |

最小 reward builder 建议做显式 map，不是为了替代所有函数，而是为了把 `asset_cfg/site_cfg/actuator_cfg/soft_limit` 这类参数固定住：

```python
def _prepare_reward_function(self):
    scales = class_to_dict(self.robot_cfg.rewards.scales)
    entity_name = self.robot_cfg.asset.name
    asset_cfg = SceneEntityCfg(entity_name)
    joint_cfg = SceneEntityCfg(
        entity_name,
        joint_names=self.asset.joint_names,
        preserve_order=True,
    )
    hip_cfg = SceneEntityCfg(
        entity_name,
        joint_names=(
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        ),
        preserve_order=True,
    )
    foot_site_cfg = SceneEntityCfg(
        entity_name,
        site_names=("FL", "FR", "RL", "RR"),
        preserve_order=True,
    )
    actuator_cfg = SceneEntityCfg(
        entity_name,
        actuator_names=(".*",),
        preserve_order=True,
    )

    reward_map = {
        "tracking_lin_vel": (self._reward_tracking_lin_vel, {"asset_cfg": asset_cfg}),
        "tracking_ang_vel": (self._reward_tracking_ang_vel, {"asset_cfg": asset_cfg}),
        "upright_linear": (self._reward_upright_linear, {"asset_cfg": asset_cfg}),
        "stand_height": (self._reward_stand_height, {"asset_cfg": asset_cfg}),
        "joint_to_default": (self._reward_joint_to_default, {"asset_cfg": joint_cfg}),
        "upside_down_penalty": (
            self._reward_upside_down_penalty,
            {"asset_cfg": asset_cfg},
        ),
        "recovery_success": (
            self._reward_recovery_success,
            {"asset_cfg": asset_cfg},
        ),
        "lin_vel_xy": (self._reward_lin_vel_xy, {"asset_cfg": asset_cfg}),
        "lin_vel_z": (self._reward_lin_vel_z, {"asset_cfg": asset_cfg}),
        "ang_vel_xy": (self._reward_ang_vel_xy, {"asset_cfg": asset_cfg}),
        "orientation": (self._reward_orientation, {"asset_cfg": asset_cfg}),
        "base_height": (self._reward_base_height, {"asset_cfg": asset_cfg}),
        "joint_power": (self._reward_joint_power, {"asset_cfg": joint_cfg}),
        "dof_acc": (self._reward_dof_acc, {"asset_cfg": joint_cfg}),
        "dof_pos_limits": (self._reward_dof_pos_limits, {"asset_cfg": joint_cfg}),
        "torque_limits": (
            self._reward_torque_limits,
            {
                "asset_cfg": actuator_cfg,
                "soft_limit": self.robot_cfg.rewards.soft_torque_limit,
            },
        ),
        "action_rate": (self._reward_action_rate, {}),
        "smoothness": (self._reward_smoothness, {}),
        "feet_air_time": (self._reward_feet_air_time, {}),
        "foot_clearance": (self._reward_foot_clearance, {"asset_cfg": foot_site_cfg}),
        "collision": (self._reward_collision, {"asset_cfg": asset_cfg}),
        "stand_still": (self._reward_stand_still, {"asset_cfg": joint_cfg}),
        "hip_pos": (self._reward_hip_pos, {"asset_cfg": hip_cfg}),
    }

    reward_terms = {}
    for name, scale in scales.items():
        if scale == 0 or name == "termination":
            continue
        if name not in reward_map:
            raise KeyError(
                f"Reward '{name}' has non-zero scale but no registered function."
            )
        func, params = reward_map[name]
        reward_terms[name] = RewardTermCfg(
            weight=scale,
            func=func,
            params=params,
        )
    return reward_terms
```

上面片段里还隐含几个需要同步修的点：

- 如果给 `collision` 也传 `asset_cfg`，那么 `_reward_collision()` 的签名也要接这个参数，或者 reward map 里不要传。文档建议统一所有 asset 相关 reward 签名，减少后面维护时的分支。
- `foot_clearance` 当前读取 `asset_cfg.site_ids`，所以不能传 `joint_cfg`；应该传足端 site cfg。
- `torque_limits` 当前读取 `asset_cfg.actuator_ids`，不应假定 actuator name 等于 joint name；先用 `actuator_names=(".*",)` 覆盖所有 actuator，再由函数内部按 actuator ids 取 force/limit。

单策略联合训练的 scales 不应把 locomotion 项全局置零。更合理的是增加 recovery 项，并在函数内部 gate：

```python
joint_training_scales = {
    "upright_linear": 3.0,
    "stand_height": 1.0,
    "joint_to_default": 0.8,
    "recovery_success": 0.0,  # 首版先做 metric；确认不奖励普通站立 reset 后再打开
    "upside_down_penalty": 1.0,
    "lin_vel_xy": -0.15,
    "lin_vel_z": -0.25,
    "ang_vel_xy": -0.05,
    "tracking_lin_vel": 1.0,
    "tracking_ang_vel": 0.5,
    "feet_air_time": 0.1,
    "foot_clearance": -0.01,
    "action_rate": -0.02,
    "dof_pos_limits": -2.0,
    "torque_limits": -0.2,
    "collision": -0.2,
    "stand_still": -0.5,
    "orientation": 0.0,
    "base_height": 0.0,
}
```

对应 reward 函数建议：

```python
def _recovery_target_height_fixed(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.full(
        (env.num_envs,),
        float(self.robot_cfg.rewards.base_height_target),
        device=env.device,
    )


def _loco_recovery_weights(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> tuple[torch.Tensor, torch.Tensor]:
    asset = env.scene[asset_cfg.name]
    grav_z = asset.data.projected_gravity_b[:, 2]
    x = (((-grav_z) - 0.65) / 0.25).clamp(0.0, 1.0)
    w_loco = x * x * (3.0 - 2.0 * x)
    return w_loco, 1.0 - w_loco


def _update_recovery_success_state(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """每个控制步只调用一次；可以作为非终止 termination term，返回全 False。"""
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    target_height = self._recovery_target_height_fixed(env)
    grav_z = asset.data.projected_gravity_b[:, 2]

    w_loco, _ = self._loco_recovery_weights(env, asset_cfg)
    env.recovery_bonus_armed.logical_or_(w_loco < 0.20)

    upright = grav_z < -0.90
    height_ok = (height - target_height).abs() < env.recovery_height_success_tol_m
    joint_error = (asset.data.joint_pos - asset.data.default_joint_pos).abs().mean(dim=-1)
    joints_ok = joint_error < 0.20

    omega_ref_sq = float(env.recovery_omega_xy_ref) ** 2
    vz_ref_sq = float(env.recovery_vz_ref) ** 2
    agitation_now = (
        asset.data.root_link_ang_vel_b[:, :2].square().sum(dim=-1)
        / omega_ref_sq
        + asset.data.root_link_lin_vel_b[:, 2].square()
        / vz_ref_sq
    )
    env.recovery_recent_max_agitation = torch.maximum(
        env.recovery_recent_max_agitation * env.recovery_agitation_decay,
        agitation_now,
    )
    stable_now = env.recovery_recent_max_agitation < env.recovery_agitation_threshold
    recovered_now = upright & height_ok & joints_ok & stable_now

    env.recovery_stable_steps = torch.where(
        recovered_now,
        env.recovery_stable_steps + 1,
        torch.zeros_like(env.recovery_stable_steps),
    )
    recovered = recovered_now & (env.recovery_stable_steps >= env.recovery_hold_steps)

    first_success = recovered & env.recovery_bonus_armed & ~env.recovery_success_latched
    env.recovery_first_success.copy_(first_success)
    env.recovery_ever_success.logical_or_(recovered)
    env.recovery_success_latched.logical_or_(first_success)

    return torch.zeros_like(recovered, dtype=torch.bool)


def _reward_upright_linear(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    grav_z = env.scene[asset_cfg.name].data.projected_gravity_b[:, 2]
    return 0.5 * (1.0 - grav_z)


def _reward_tracking_lin_vel(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("twist")
    assert command is not None, "Command 'twist' not found."
    err = torch.sum((command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]).square(), dim=1)
    w_loco, _ = self._loco_recovery_weights(env, asset_cfg)
    return torch.exp(-err / self.robot_cfg.rewards.tracking_sigma) * w_loco


def _reward_stand_height(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    target_height = self._recovery_target_height_fixed(env)
    _, w_recovery = self._loco_recovery_weights(env, asset_cfg)
    upright_gate = (asset.data.projected_gravity_b[:, 2] < -0.70).float()
    sigma_sq = env.recovery_height_reward_sigma_sq_m2
    return torch.exp(-(height - target_height).square() / sigma_sq) * upright_gate * w_recovery


def _reward_joint_to_default(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    error = (asset.data.joint_pos - asset.data.default_joint_pos).square().mean(dim=-1)
    _, w_recovery = self._loco_recovery_weights(env, asset_cfg)
    return torch.exp(-error / 0.25) * w_recovery


def _reward_recovery_success(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    return 10.0 * env.recovery_first_success.float()


def _reward_upside_down_penalty(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    grav_z = env.scene[asset_cfg.name].data.projected_gravity_b[:, 2]
    return -torch.clamp(grav_z, min=0.0)


def _reward_lin_vel_xy(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    _, w_recovery = self._loco_recovery_weights(env, asset_cfg)
    return asset.data.root_link_lin_vel_b[:, :2].square().sum(dim=1) * w_recovery
```

注册时可以把状态更新放进 termination manager，但它不是失败终止项：

```python
terminations["recovery_state_update"] = TerminationTermCfg(
    func=self._update_recovery_success_state,
    time_out=False,
    params={"asset_cfg": SceneEntityCfg(self.robot_cfg.asset.name)},
)
```

这要求 `_update_recovery_success_state()` 永远返回全 False。它利用 mjlab “termination 先于 reward”的顺序生成 reward snapshot，但不改变 done 语义。`recovery_bonus_armed` 只能由物理状态推导，例如 episode 内曾经明显非正立；不要让普通站立 reset 第一帧直接拿 `recovery_success` bonus。

上面的 reward 片段是 45D 固定高度版本。如果切到 46D height-command ABI，才能把 `_recovery_target_height_fixed()` 换成读取 `env.recovery_height_cmd`；同时 actor observation、history、critic 前 49 维、wrapper 和 checkpoint metadata 都要一起改。按当前用户约束，这一步暂时不做。

这里有三个关键边界：

- 成功状态只能在每个控制步的一个位置更新一次；reward term 应只读取 `recovery_first_success` 这类 snapshot。如果测试或日志又手动调用 `reward_manager.compute()`，不应再次推进 latch。
- `recovery_success` 不应把成功写成 done。mjlab step 顺序是 termination 再 reward；如果把 state-update 放在 termination manager 里，它必须返回全 False，只负责生成 reward 前 snapshot。联合训练首版更保守的做法是先把 `recovery_success` scale 设为 `0.0`，只记录 metric，确认不会奖励普通站立 reset 后再打开。
- `stand_height` 用的是相对 `env.scene.env_origins[:, 2]` 的高度。上 rough terrain 后还要改成相对局部地面高度，否则坡面/台阶会把成功判定搞偏。`height_success_tolerance_m=0.05` 和 `height_reward_sigma_sq_m2=0.05` 是两个不同量，不能把后者误写成 `sigma_h=0.05 m`。
- 如果沿用参考里的 `smooth_success_decay=0.9` 且控制步 `dt=0.02`，半衰期是 `log(0.5)/log(0.9)=6.58` 步，也就是约 `0.132 s`，不是 `0.2 s`。更清楚的写法是 `decay = 2 ** (-step_dt / half_life_s)`。
- `dof_acc = ||qdot_t - qdot_{t-1}||^2 / dt` 在 mjlab RewardManager 又会乘一次 `dt`，最终量纲更接近 `||delta_qdot||^2`，不是严格的 acceleration-squared 时间积分；权重不能直接照搬 legged_gym 常数。

### 9.6 Fallen reset：把“空中随机姿态”改成“贴地倒姿起身”

当前错在 `him_go2_env.py:272-298` 使用内置加法 reset，并且 config 里默认 z=0.42、随机 z=[0.35,0.50]。改完不应该继续用这组参数表达 recovery。

推荐写一个自定义 fallen reset event，核心语义是：

```text
pose bucket -> roll/pitch/yaw -> quaternion
root z      -> low calibrated height above terrain origin
joint pos   -> directly sample inside feasible interval near q_default
velocities  -> root/joint velocities zero at reset
command     -> do not override to zero in formal joint training
episode state -> clear success latch, bucket id, fallen-reset metric state
```

建议代码片段：

```python
@torch.no_grad()
def reset_recovery_root_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    asset = env.scene[asset_cfg.name]
    n = len(env_ids)

    bucket = torch.multinomial(
        env.recovery_bucket_weights,
        num_samples=n,
        replacement=True,
    )
    euler = env.recovery_pose_euler[bucket].clone()
    euler += (2.0 * torch.rand_like(euler) - 1.0) * env.recovery_pose_jitter
    quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])

    pos = torch.empty((n, 3), device=env.device, dtype=quat.dtype)
    pos[:, :2] = env.scene.env_origins[env_ids, :2]
    pos[:, :2] += 0.05 * (
        2.0 * torch.rand((n, 2), device=env.device, dtype=quat.dtype) - 1.0
    )
    pos[:, 2] = (
        env.scene.env_origins[env_ids, 2]
        + env.recovery_root_height_m[bucket]
    )

    asset.write_root_link_pose_to_sim(
        torch.cat((pos, quat), dim=-1),
        env_ids=env_ids,
    )
    asset.write_root_link_velocity_to_sim(
        torch.zeros((n, 6), device=env.device, dtype=pos.dtype),
        env_ids=env_ids,
    )

    q0 = asset.data.default_joint_pos[env_ids]
    limits = asset.data.soft_joint_pos_limits[env_ids]
    q_min, q_max = limits[..., 0], limits[..., 1]
    margin = float(env.recovery_joint_limit_margin_rad)
    delta = float(env.recovery_joint_jitter_rad)
    lo = torch.maximum(q_min + margin, q0 - delta)
    hi = torch.minimum(q_max - margin, q0 + delta)
    q = lo + torch.rand_like(q0) * (hi - lo)
    asset.write_joint_state_to_sim(
        q,
        torch.zeros_like(q),
        env_ids=env_ids,
    )

    # 正式单策略联合训练不要在这里把 command manager 的 twist 命令写零。
    # fallen/upright reset 应共享同一套 velocity-command 采样分布。
    env.recovery_bucket[env_ids] = bucket
    env.recovery_ever_success[env_ids] = False
    env.recovery_success_latched[env_ids] = False
    env.recovery_first_success[env_ids] = False
    env.recovery_bonus_armed[env_ids] = False
    env.recovery_stable_steps[env_ids] = 0
    env.recovery_recent_max_agitation[env_ids] = 0.0
    env.fallen_reset_episode[env_ids] = True
```

这段代码还有两个需要你实际落地时确认的点：

- `quat_from_euler_xyz` 要从 `mjlab.utils.lab_api.math` 或项目已有 math helper 正确导入。
- `asset.data.soft_joint_pos_limits` 在本地 mjlab 1.6.0 的 `EntityData` 中存在，但仍要在 env 构造通过后确认 shape 是 `[num_envs, 12, 2]` 且 joint 顺序与 `self.asset.joint_names` 一致。
- 上面函数只负责 fallen reset。联合训练还需要一个外层 reset sampler，根据 `fall_reset_probability` 把 env_ids 分成 fallen/upright 两组；两组都不要改写成不同的 command 分布。

还要补一个重要安全边界：这个 reset 是仿真训练用的，不是实机动作。每个姿态桶的 `recovery_root_height_m` 必须通过碰撞/间隙校准得到，不能只看 `z` 是否在 `[0.10, 0.20]`。最低验收应包括：

- `bucket x joint jitter` 采样后 `sim.forward()` 不产生深穿透或明显悬空。
- 第一物理步接触冲量、关节目标、力矩目标没有异常尖峰。
- 无效姿态要拒绝采样并重采，而不是让 PPO 从爆炸 reset 里学习。

### 9.7 姿态桶 curriculum：按 episode 内是否曾成功更新

参考仓库的核心价值不是某个 reward 常数，而是“低成功率姿态桶过采样”。mjlab 迁移时建议放在 curriculum term 里，利用 `_reset_idx()` 中 curriculum 在 reset event 前执行的顺序，用上一轮 episode 的 `ever_success` 更新桶权重。

```python
@torch.no_grad()
def recovery_pose_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    valid = env.fallen_reset_episode[env_ids]
    ids = env_ids[valid]
    if ids.numel() == 0:
        success_rate = env.recovery_bucket_wins / env.recovery_bucket_trials.clamp_min(1)
        return {"fallen_bucket_mean_success": success_rate.mean()}

    bucket = env.recovery_bucket[ids]
    ever_recovered = env.recovery_ever_success[ids].float()
    num_buckets = env.recovery_bucket_weights.numel()

    trials = torch.bincount(bucket, minlength=num_buckets).float()
    wins = torch.bincount(
        bucket,
        weights=ever_recovered,
        minlength=num_buckets,
    )

    env.recovery_bucket_trials += trials
    env.recovery_bucket_wins += wins
    success_rate = env.recovery_bucket_wins / env.recovery_bucket_trials.clamp_min(1.0)

    covered = env.recovery_bucket_trials >= env.recovery_min_samples
    weakness = torch.where(
        covered,
        1.0 - success_rate,
        torch.ones_like(success_rate),
    )
    weights = (env.recovery_weight_floor + weakness).clamp_min(0.05)
    env.recovery_bucket_weights.copy_(weights / weights.sum())

    return {"fallen_bucket_mean_success": success_rate.mean()}
```

不要每个 reset 都把 tensor 拉到 CPU 做 Python list 统计；4096 env 下这样会引入同步点，训练吞吐会变差。

另外，terrain level 和 velocity command curriculum 不应把 fallen reset 的恢复时间当成 locomotion tracking 失败。联合训练里评估这些 curriculum 时建议只统计 `w_loco` 已经接近 1 的时间窗，或者只统计 upright reset / 已恢复后的片段。

### 9.8 仿真策略 ABI：不是实机 Sim2Real 放行契约

当前训练 config、XML、部署 YAML 的顺序和幅度不一致。这里应该先写一份**仿真策略 ABI**，保证训练、wrapper、runner、MuJoCo 部署回放能对齐。它不能被称作真机 Sim2Real 安全契约；真实 Go2 部署还必须另做低层通信、模式切换、CRC、watchdog、急停、传感器新鲜度、NaN/Inf fail-closed、逐关节方向和零位核验。

首版新训练推荐的仿真策略契约：

```yaml
abi_version: go2-him-sim-policy-v1
scope: simulation_policy_only
hardware_deployment: forbidden_until_hardware_safety_contract_exists
entity_name: go2
base_body_name: base_link

policy_joint_order:
  - FL_hip_joint
  - FL_thigh_joint
  - FL_calf_joint
  - FR_hip_joint
  - FR_thigh_joint
  - FR_calf_joint
  - RL_hip_joint
  - RL_thigh_joint
  - RL_calf_joint
  - RR_hip_joint
  - RR_thigh_joint
  - RR_calf_joint

mapping_rule: >
  policy 输出先按 policy_joint_order 转为 {joint_name: value}；
  simulator actuator 再按解析到的 actuator/joint 名称重排。
  禁止依赖 dict insertion、MJCF traversal、regex 返回顺序或数组下标隐式映射。

action_pipeline:
  raw_action: "a_raw from policy distribution, shape [N,12]"
  finite_gate: "NaN/Inf -> fail-closed or replace with neutral safe action"
  accepted_action: "a = clamp(a_raw, -1, 1)"
  q_des_policy: "q0 + a * action_scale_rad"
  q_des_sim_clip: "clip to documented simulator target range"
  note: >
    mjlab JointPositionActionCfg.clip 夹的是 processed target，不是 raw action；
    因此 raw action gate 必须在 wrapper/runner/env.step 入口显式完成。

action_scale_rad:
  hip: 0.125
  thigh: 0.25
  calf: 0.25

sim_effort_limit_nm_candidate:
  hip: 23.5
  thigh: 23.5
  calf: 35.55
  note: "来自当前配置/部署 YAML 的保守交集候选，只能用于仿真一致性，不是实机授权值。"

observation_abi:
  actor_frame_45:
    order:
      - command_3
      - base_ang_vel_3
      - projected_gravity_3
      - joint_pos_rel_12
      - joint_vel_rel_12
      - last_accepted_action_12
    last_action_definition: >
      进入 env.step 前已经通过 finite gate 和 [-1,1] clamp 的 accepted action。
      如果实际代码继续使用 mjlab raw action，则必须保证传入 action_manager 的就是 accepted action。
  actor_history_270:
    layout: frame_major
    order: [t, t-1, t-2, t-3, t-4, t-5]
  native_critic_235:
    order: [actor_frame_45, base_lin_vel_3, height_scan_187]
  him_runner_critic_48:
    source: native_critic_235[0:48]
    order: [actor_frame_45, base_lin_vel_3]
```

这里把 `hip` scale 写成 `0.125`，与当前训练 config 的 `action_scale=0.25, hip_reduction=0.5` 和部署 YAML 的 `hip_scale=0.125` 一致。这一项当前已经对齐，不再作为 blocker；后续真正要冻结的是关节顺序、target clip、raw action gate 和 torque limit 的一致性。

如果你已有 checkpoint 或部署代码明确依赖 `FL, RL, FR, RR` 顺序，那就不要直接采用上面的顺序；应把 ABI version 改成 `go2-him-legacy-deploy-order`，并写显式 remap。关键不是哪个顺序“天然正确”，而是全链路只能有一个真实顺序。

真机前还缺一份逐关节硬件映射表。没有这张表，不能把策略发到实机：

```yaml
hardware_mapping_required_before_real_robot:
  - policy_name: FL_hip_joint
    hardware_motor_id: TBD
    sign: TBD
    zero_offset_rad: TBD
    q_safe_min_rad: TBD
    q_safe_max_rad: TBD
    dq_safe_limit_rad_s: TBD
    delta_q_per_tick_limit_rad: TBD
    kp: TBD
    kd: TBD
    torque_limit_continuous_nm: TBD
    torque_limit_peak_nm: TBD
    tau_ff_policy: "disabled_or_bounded"
    control_mode: TBD
    verified_from_live_state: false
```

实机安全契约还必须定义：

- 起控前读取当前关节状态，并插值过渡到安全初始目标；不能从任意姿态直接跳到策略目标。
- 推理超时、通信断连、CRC/包错误、传感器陈旧、NaN/Inf 时的 fail-closed 动作。
- 硬件侧位置、速度、增量、力矩、电流、温度/电压降额和急停。
- shadow mode / HIL / 低力矩悬空验证通过后，才能讨论落地 recovery。

### 9.9 验证代码建议：严格只读和写入型验证要分开

当前文档-only 范围内，严格只读审计只能做不写 cache、不启动 env、不写日志的检查。不要用 `py_compile` 或 `compileall`，因为它们会写 `__pycache__`。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -c 'import ast,pathlib; ast.parse(pathlib.Path("legged_mjlab/envs/him_go2/him_go2_env.py").read_text())'
```

`rsl_rl` 后端来源要通过项目加载器确认，不能只看 `importlib.metadata.version("rsl_rl")` 或裸 `import rsl_rl`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -c 'import sys; from legged_mjlab.utils.task_registry import load_project_rsl; runners = load_project_rsl(); print(sys.modules["rsl_rl"].__file__); print(runners.__file__); print(runners.HIMOnPolicyRunner.__module__)'
```

pytest、`make_env/reset/step` 和训练 smoke 都属于写入型或有运行副作用的验证：`-B` 与 `-p no:cacheprovider` 只能抑制 Python bytecode 和 `.pytest_cache`，不能保证测试代码、导入库、仿真库或日志完全零写入。后续获准写入后，建议使用 `/tmp` 作为 smoke 日志目录，不要默认写仓库 `logs/`。

| 验证 | 当前状态 | 写入范围 | PASS 条件 |
|---|---|---|---|
| AST only | `PASS`，当前 `him_go2_env.py` 可被 Python 3.11 AST 解析 | 无项目写入 | Python 3.11 AST 通过。 |
| backend source gate | `NOT_RUN` | import 级副作用风险，禁 pycache | `rsl_rl.__file__` 和 `HIMOnPolicyRunner.__module__` 均指向项目内旧版实现。 |
| pytest contract | `NOT_RUN`，`.venv` 无 pytest，建议测试文件尚未创建 | 测试 cache/临时文件；需抑制 cache 或写 `/tmp` | `legged_mjlab/test/test_him_go2_contracts.py` 创建后通过。 |
| env smoke | `NOT_RERUN`，本轮不再跟踪构造签名/Perlin 旧问题 | MuJoCo/mjlab 进程状态，可能写临时资源 | `reset -> step` 形状和 finite 检查通过。 |
| training smoke | `REQUIRES_WRITE_SCOPE` | 必须显式 `--log-dir /tmp/legged_mjlab_smoke` | `num_envs=1, max_iterations=1` 只验证 runner ABI，不评价性能。 |

建议测试内容至少覆盖：

```python
def test_actor_critic_action_contract(env):
    obs, extras = env.reset()
    assert obs["actor"].shape[-1] == 45
    assert obs["critic"].shape[-1] in (48, 235)
    assert env.action_manager.action.shape[-1] == 12


def test_reward_terms_are_vectors(env):
    # stateful recovery env 中，同一控制步不要重复调用 compute。
    terms = env.reward_manager.compute(dt=env.step_dt)
    assert terms.shape == (env.num_envs,)
    assert torch.isfinite(terms).all()


def test_upright_reward_orders_states(fake_env):
    env_impl = object.__new__(HimGo2Env)
    fake_env.scene["go2"].data.projected_gravity_b[:, 2] = torch.tensor([-1.0, 0.0, 1.0])
    value = env_impl._reward_upright_linear(fake_env, SceneEntityCfg("go2"))
    assert value[0] > value[1] > value[2]


def test_recovery_reset_is_low_height(env):
    env.reset()
    z = env.scene["go2"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    assert z.max() < 0.45
    assert z.min() > 0.05


def test_recovery_reset_has_no_initial_spike(env):
    env.reset()
    env.sim.forward()
    # 下面阈值需要按 Go2 MJCF 和仿真接触模型标定；先写成必须检查的接口。
    assert torch.isfinite(env.scene["go2"].data.joint_pos).all()
    assert torch.isfinite(env.scene["go2"].data.joint_vel).all()
    assert env.reset_diagnostics["max_penetration_m"].max() < env.reset_diagnostics["penetration_limit_m"]
    assert env.reset_diagnostics["first_step_peak_tau_nm"].max() < env.reset_diagnostics["tau_limit_nm"]
```

这些测试不是为了证明能学会起身，也不是实机安全证明。它们只证明仿真 ABI、reset 分布、reward shape、正立方向和初始状态不过分异常。真机前还要额外覆盖关节映射、符号、零位、位置/速度/增量/力矩限幅、传感器失效、通信超时、watchdog 和急停。

## 10. 后续执行顺序

建议按下面顺序推进，不要先写起身 reward；env native 构造链没跑通前，reward 再好也进不到训练。

### P0：让剩余 env contract 能 `reset -> step`

目标：在你已处理或暂不跟踪构造签名、Perlin 旧写法、`mdp` alias 之后，把剩余 env contract 修到可以完成一次 reset/step。

必须修：

1. `heading_command=False` 时把 `ranges.heading` 传成 `None`，并避免 `tuple(None)`。
2. action `clip` 从 tuple 改成按关节名匹配的 processed target clip dict；raw action finite/clamp 先记录为 wrapper/runner 后续边界。
3. actor observation 的 `projected_gravity/joint_pos_rel/joint_vel_rel` 显式传 `asset_cfg=SceneEntityCfg("go2", ...)`。
4. curriculum 里的 `terrain_levels_vel` 如果默认 `SceneEntityCfg("robot")`，也要显式传当前实体；否则 obs 修好后还会在 curriculum 层找错 asset。
5. reward builder 改成显式 map，给 `asset_cfg/site_cfg/actuator_cfg/soft_limit` 绑定参数；`_reward_dof_acc()` 不能继续 `pass`。

已修好或本轮不再跟踪的旧项不再列为 P0：Python 3.11 f-string assert、`_build_sensors(entity_name=...)`、`_DEFAULT_ASSET_CFG` 默认 `"robot"`、`push_robot` interval/asset_cfg、`frandomize_friction` 字段名、两个 tracking reward 里的 command assert、构造签名、Perlin 注释块、`mdp` alias。

完成标准：

- `import legged_mjlab.envs.him_go2.him_go2_env` 继续通过。
- `Go2Asset(HimGo2RoughCfg())` 继续能解析 XML、12 个 joint、`base_link`。
- native reset 返回 actor `[N,45]`、critic `[N,235]`，step 一次 reward/done/obs 都 finite。

### P1：冻结 env native ABI

目标：只冻结 env 侧 ABI，不审 `wrappers/`。

必须明确：

- action order：12 维输出到底采用 XML leg-major，还是沿用当前 config dict 顺序；不能靠 dict insertion 或 regex 结果隐式决定。
- obs order：`[cmd(3), ang_vel(3), projected_gravity(3), q_rel(12), qd(12), last_action(12)]`。
- critic order：env native critic 是否固定为 `[actor_45, base_lin_vel_3, height_scan_187] = 235`。
- policy frequency：`sim.dt=0.005`、`decimation=4`，control dt = `0.02 s`。
- target clip、raw action gate、soft joint limit、torque limit 的语义分层。

完成标准：

- `num_envs=1` 和小 batch 下 reset/step 形状稳定。
- `env.action_manager.action` 和 `processed_actions` 的含义写清楚，不再把 mjlab target clip 当作 raw action clamp。
- 暂不要求 `HIMRslRlWrapper`、terminal obs、history layout 或旧版 runner 完成；这些等 env 完成后单独审。

### P2：实现单策略混合 reset

目标：同一个 env 中同时采样 upright locomotion reset 和 fallen recovery reset。

建议：

- 不新建 recovery-only 训练任务作为主线；可以保留 debug cfg，但正式训练仍是一个 `him_go2` 单策略任务。
- 外层 reset sampler 用 `fall_reset_probability` 分流，fallen/upright 两组都使用同一套 twist command 分布。
- fallen reset 改成低高度贴地姿态，roll/pitch/yaw bucket，root/joint velocity 置零。
- joint position 在可行区间内直接采样，不要先 `q0 + U(-delta,delta)` 再 clamp 到限位。
- reset label 只用于 bucket 统计、reset 诊断、curriculum，不用于隐藏式 reward phase 切换。
- 初始非足端接触要有 reset grace；不能把恢复初始的 base/thigh/calf 接触当失败。

完成标准：

- 随机 reset 后采样分布可打印/可视化：base z、`g_z`、joint deviation、bucket id、fallen/upright 比例。
- 站立、侧躺、仰倒、俯倒都能被采到。
- 每个姿态桶经过 `sim.forward()` 和第一步物理检查：无深穿透、无明显悬空、接触冲量和首步力矩没有异常尖峰。

### P3：迁移单策略联合 reward

目标：同一个 policy 同时收到恢复和行走两类训练信号，但由物理状态连续 gate。

首批建议：

- `upright_linear` 全程打开，替代当前不能区分正立/倒立的 `orientation` 主项。
- `stand_height/joint_to_default/hip_pos/lin_vel_xy` 乘 `w_recovery`。
- `tracking_lin_vel/tracking_ang_vel/feet_air_time/foot_clearance` 乘 `w_loco`，不是全局关闭。
- `collision` 加 reset grace 和正立 gate。
- `recovery_success` 先做 metric；确认不会奖励普通站立 reset 后再考虑打开为小权重 one-shot bonus。

完成标准：

- 每个 reward term shape 都是 `[num_envs]`，没有 NaN/Inf。
- 同一控制步内 success latch 只更新一次；重复调用 reward 日志不会推进状态。
- fallen reset 的成功率按 episode 内是否曾成功统计。
- terrain/velocity curriculum 只在 `w_loco` 有效的时间窗评价 locomotion，不把恢复过程当 tracking 失败。

### P4：放大训练与 wrapper 后审

目标：env native 通过后，再接旧版 HIM wrapper/runner 和更大规模训练。

建议：

- 先 plane + low DR 做最小训练 smoke，再逐步打开 rough terrain、push、latency、motor strength。
- 只有 env native reset/step/reward 稳定后，才回头审 `legged_mjlab/wrappers/`、外部旧版 `rsl_rl` 的 tuple ABI、terminal obs、history layout 和 timeout bootstrap。
- 不建议一开始就把全姿态起身、粗糙地形、速度跟踪、强 DR、强平滑惩罚全部拉满。

### 阶段门槛表

| 阶段 | 前置条件 | 写入范围 | 当前状态 |
|---|---|---|---|
| P0 env contract | command、action、obs、curriculum、reward 参数修复 | 源码修改后才可跑完整动态 smoke | `BLOCKED/NOT_RERUN`，AST/import/`Go2Asset` 已过；本轮不再把构造签名和 Perlin 旧写法作为当前文档 blocker。 |
| P1 env ABI | P0 通过；action/obs/critic/order 文档冻结 | env/config 文档和少量源码 | `NOT_RUN`，reset/step 未通过。 |
| P2 mixed reset | P1 通过；fallen/upright reset sampler | 需要运行 MuJoCo reset/forward/step | `REQUIRES_CODE_CHANGE`。 |
| P3 joint reward | P2 reset 通过；物理状态 gate 和状态机单点更新 | 需要 reward logs 和短训练日志 | `REQUIRES_CODE_CHANGE`。 |
| P4 wrapper/runner | env native 全链路通过 | wrappers/runner/deploy 文档与测试 | `DEFERRED`，按你的要求本轮不审 wrappers。 |

## 11. 验证命令建议

这些命令是后续改代码后的验证建议。本次没有修改训练代码，也没有跑训练。

当前只读/轻量动态验证分支确认到的状态：

| 检查 | 状态 | 说明 |
|---|---|---|
| Python AST | `PASS` | `him_go2_env.py` 可被 Python 3.11 AST 解析，旧 f-string assert 问题已不再成立。 |
| env import | `PASS` | `import legged_mjlab.envs.him_go2.him_go2_env` 返回 `import-ok`。 |
| 原始 XML asset | 通过轻量验证 | `resources/robots/unitree_go2/xmls/go2.xml` 可被 MuJoCo 解析；这只证明 XML/mesh 可用，不证明训练 ABI 已闭合。 |
| `Go2Asset` 构造 | `PASS` | `Go2Asset(HimGo2RoughCfg())` 已能返回 XML path，`len(joint_names)=12`，并确认 `base_link in body_names`。 |
| env smoke | `NOT_RERUN` | 本轮按用户要求不再跟踪构造签名和 Perlin 旧写法；剩余静态问题集中在 heading、action clip、actor obs asset_cfg、curriculum、reward params。 |
| command/action/obs contract | `NEEDS_FIX` | `heading_command=False` 仍传 heading；`JointPositionActionCfg.clip` 仍是 tuple；actor obs 仍缺 `asset_cfg`。 |
| HIM backend source | `NOT_RUN` | 必须调用 `load_project_rsl()` 后检查 `rsl_rl.__file__`、`runners.__file__`、`HIMOnPolicyRunner.__module__`；裸 `import rsl_rl` 不能证明训练时使用项目内旧版 runner。 |
| wrappers | `DEFERRED` | 按你的要求，`legged_mjlab/wrappers/` 本轮不审。 |
| 训练 smoke | `NOT_RUN` | `train.py` 会创建日志目录并启动训练；本次文档-only 范围内不执行。 |

严格只读优先级：

1. AST only：已通过；后续每次改 env 后仍建议复跑，不用 `py_compile/compileall`。
2. backend source gate：用 `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B` 调 `load_project_rsl()`，打印并断言项目内旧版 `rsl_rl` 来源。
3. API signature check：对 mjlab 1.6.0 的 action、command、terrain、obs/reward term 签名做只读确认，不靠记忆迁移 1.2.0 代码。

获准写代码并运行后再做：

1. reset shape：native reset 返回 `actor [1,45]`、`critic [1,235]`。
2. step smoke：零动作和小随机动作各 step 一次，确认 obs、reward、done、extras finite。
3. reward term smoke：逐项 reward 返回 `[N]`，stateful recovery term 不得在同一控制步重复推进 latch。
4. fallen reset safety smoke：每个姿态桶检查穿透、悬空、首步接触冲量、首步力矩尖峰。
5. 最小训练 smoke：`--num-envs 1 --max-iterations 1 --log-dir /tmp/legged_mjlab_smoke`，只验证 runner ABI，不看性能。
6. env 通过后再开 wrapper/runner smoke；本轮不把它作为当前 blocker。

## 12. 当前优先级清单

P0 env contract blocker：

- action `clip` 类型/语义还没有从 raw action gate 中分离。
- `heading_command=False` 时仍传非空 `ranges.heading`。
- actor observation term 缺显式 `asset_cfg`。
- curriculum 可能仍默认找 `"robot"` asset，需要检查并显式传 `go2`。
- reward builder 还没有给所有非零 term 绑定必要 params；`_reward_dof_acc()` 仍是 `pass`。

P1 单策略 reset/reward blocker：

- 当前 reset 是默认高度加 z 偏移，不是贴地倒姿。
- 关节 reset 没有可行区间内的随机扰动。
- 还没有 mixed upright/fallen reset sampler，也还没保证两组共享同一 twist command 分布。
- 当前 `orientation` 不能区分正立和倒立；联合 reward 还没有 `w_recovery/w_loco` 物理状态 gate。
- 成功 predicate、episode 内 success latch、`recent_max_agitation` 状态机、bucket curriculum 状态还不存在。
- `collision` 缺 reset grace，恢复初始接触可能被当成坏行为。
- terrain/velocity curriculum 尚未排除恢复阶段，可能把起身时间误记为 locomotion tracking 失败。

P2 env ABI / backend blocker：

- action order、default joint order、部署 YAML 顺序不统一。
- raw action finite/clamp、processed target clip、per-joint `q_safe` clamp 三层语义还没有分开实现。
- critic 235 与后续旧 HIM runner 具体使用维度仍未冻结；wrapper 暂不审，但 env 侧 native 235 要先稳定。
- `only_positive_rewards` 语义和 mjlab RewardManager 不一致。
- 45D 固定高度与 46D 动态高度 ABI 必须互斥；当前阶段不采样 actor 看不到的逐环境目标高度。

P3 sim2real blocker：

- 训练和部署的 action scale/hip scale 当前已对齐到 `0.25/0.125`，但关节顺序和 torque limit 仍不一致。
- deploy 目录目前更像 sim2sim 配置，没有完整策略调用、关节重排、安全限幅、watchdog、通信桥接。
- 本文第 9.8 只能作为仿真策略 ABI；缺少真实硬件 motor id、方向符号、零位、位置/速度/增量/力矩限幅、模式切换、CRC、传感器新鲜度、急停和 fail-closed 契约。

## 13. 推荐的最小下一步

最小可落地路线：

1. 依次修 `heading None -> action clip dict -> actor obs asset_cfg -> curriculum asset_cfg -> reward params/dof_acc`，每修一层就复跑小规模 smoke，确认下一个真实失败点。
2. env native `reset -> step` 通过后，冻结 actor 45、critic 235、action 12、关节顺序、target clip 和 raw action gate 语义。
3. 实现 mixed upright/fallen reset sampler；fallen reset 做低高度贴地姿态和可行区间关节扰动，但正式联合训练不把速度命令清零。
4. 把当前 `orientation` 从主 reward 里退掉，新增 `upright_linear` 和 `w_recovery/w_loco` 物理状态 gate；locomotion tracking 与步态项保留但只在 `w_loco` 有效时发力。
5. `recovery_success` 先做 metric，bucket curriculum 只统计 fallen reset episode 的 `ever_recovered`。
6. 用小规模 smoke 训练看 reward shape、成功率和曲线方向，日志写到显式目录，例如 `/tmp/legged_mjlab_smoke`。
7. env 部分稳定后，再回头审 `legged_mjlab/wrappers/`、旧版 `rsl_rl` backend source、terminal obs、history layout 和 timeout bootstrap。

一句话：先把 mjlab env ABI 跑通，再做同一个 `him_go2` 任务内的 mixed fallen/upright reset 和物理状态 gated reward；不要把正式方案写成 recovery-only，也不要在当前阶段动 wrappers。

## 14. 并发审查结果与已吸收修改

本轮按只读扫描结果收敛，审查本身没有修改代码：

| 分支 | 结果 | 已写回本文档的修改 |
|---|---|---|
| `local_parallel_scan` | `completed` | 按当前源码重新过滤文档：构造签名、Perlin 注释块、`mdp` alias 不再作为当前 blocker；已删除旧的 critic obs sensor blocker；保留 heading、action clip、actor obs asset_cfg、curriculum asset、reward params/dof_acc、reset/reward gate 等问题。 |
| `codebase_explorer` | `failed` | 子代理调用返回 403 insufficient balance，本轮未采用该分支作为证据来源。 |
| `math_verifier` | `completed / reject recovery-only route` | 沿用上一轮已吸收的数学审查：把文档从 recovery-only 迁移改成单策略联合训练；补物理状态 smoothstep gate、同 command 分布 mixed reset、fallen-only bucket 统计、可行区间 joint sample、terrain/velocity curriculum gate、`smooth_success_decay` 和 `dof_acc` 量纲说明。 |

当前已经确认并从 blocker 列表删除的旧问题：

- `him_go2_env.py` AST 已通过。
- `him_go2_env` import 已通过。
- `Go2Asset(HimGo2RoughCfg())` 基础构造已通过。
- `_build_sensors(entity_name=...)` 已修。
- `_DEFAULT_ASSET_CFG` 已不再默认 `"robot"`。
- `push_robot` 已补 `interval_range_s` 和 `asset_cfg`。
- 摩擦随机化字段名已改成 `randomize_friction`。
- 两个 tracking reward 里的 command assert 已改为普通字符串 assert。
- critic obs 的 `base_lin_vel` 和 `height_scan` sensor 参数已按当前 scene/sensor 名称对齐。
- Perlin 旧写法现在位于注释掉的旧 terrain 方案中，实际 generator 分支走 `replace(ROUGH_TERRAINS_CFG)`；本轮不再作为 blocker。
- 构造签名和 `mdp` alias 按你的要求暂不在本文档里继续跟踪。

仍然不能声明 `ALL_TESTS_PASSED`，原因是：

- 本次只修改文档，没有直接写入源码。
- env native `reset`、`step`、reward smoke、pytest 和训练均未跑通或未重跑。
- `wrappers/` 按你的要求本轮不审，旧版 `rsl_rl` runner ABI 也还没做最终验收。
- 真机相关内容仍缺硬件安全契约、逐关节实测映射和 watchdog/急停/断连策略。

因此本文档的定位是“当前源码状态下的迁移和修复指南”，不是已通过实现或发布说明。
