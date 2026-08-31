# HIM Go2 Env/Config/Asset 与随机姿态起身奖励迁移审计

日期：2026-08-31

审计基线：

- Git HEAD：`1cdefbb`
- Python 解释器：`/home/kk/legged_mjlab/.venv/bin/python`
- 工作树状态：当前工作树为 dirty，至少包含本文档和 `legged_mjlab/envs/him_go2/go2_asset.py` 的未提交修改；后者不是本次文档补丁写入对象。
- 结论边界：本文所有代码片段都是迁移建议，不代表已经写入源码；所有 `PASS/FAIL/NOT_RUN` 只适用于上述工作树和本地 `.venv`。

范围：

- 当前项目：`/home/kk/legged_mjlab`
- 参考项目：`/home/kk/github/uw-himloco-hop`
- 目标：判断当前 `him_go2` 的 env、config、asset 是否已经写好；梳理随机姿态 reset 下应如何迁移鼓励起身的 reward/curriculum；给出后续工作顺序。
- 约束：本次只写文档，不修改训练代码、配置代码、asset、测试或构建脚本。

## 1. 总结论

当前不能认为 `him_go2` 已经写好到可训练状态。

比较准确的判断是：

| 部分 | 当前状态 | 判断 |
|---|---|---|
| HIM 配置骨架 | `HimGo2RoughCfg`、`HimGo2CfgPPO` 已有，声明了 45 单帧观测、6 帧历史、12 动作、HIM runner/policy/algorithm。见 `legged_mjlab/envs/him_go2/him_go2_config.py:6`、`:8`、`:216`。 | 基本写了，但字段仍有多处和 env 实现不一致。 |
| HIM wrapper / 外部旧版 `rsl_rl` 链路 | `HIMRslRlWrapper` 固定 `[N,270] = 6 x 45` 历史，`HIMOnPolicyRunner` 在本地 `rsl_rl/` 中存在。见 `legged_mjlab/wrappers/him_wrapper.py:14`、`rsl_rl/rsl_rl/runners/him_on_policy_runner.py:165`。 | 结构基本明确，但 critic 235 与 runner 实际 48 的契约要重新写清。 |
| asset 资源 | `go2.xml` 和 OBJ mesh 存在；XML 有 floating base、12 个关节、足端 collision/site、IMU sensor。见 `resources/robots/unitree_go2/xmls/go2.xml:41`、`:53`、`:155`。 | 资源基本齐，但 `Go2Asset` Python 封装当前不能正常初始化。 |
| env 实现 | `HimGo2Env` 已尝试按 mjlab manager 风格组装 scene、action、events、obs、reward、termination。见 `legged_mjlab/envs/him_go2/him_go2_env.py:38`。 | 尚未闭环，有语法、构造、字段、reward 注册等硬阻塞。 |
| 随机姿态 reset | 配置有 roll/pitch 全范围随机和 z 随机。见 `him_go2_config.py:174`。 | 目前更像“空中随机姿态跌落”，不是贴地倒姿起身 reset。 |
| 起身奖励 | 当前 reward 仍是 locomotion 风格，`orientation` 还无法区分正立和倒立。见 `him_go2_env.py:697`。 | 需要单独迁移 recovery reward，不建议在现有行走 reward 上小修。 |

结论：config 和 asset 资源“已经有雏形”，但 env 还没有达到能训练 HIM 的程度。后面应该先把环境构造、观测/action/reward/termination 的 ABI 跑通，再加 recovery reset 和起身奖励。

## 2. 当前项目结构和 HIM 数据链

高信号文件：

| 文件 | 作用 |
|---|---|
| `legged_mjlab/envs/him_go2/him_go2_config.py` | `him_go2` 任务配置，包含 env/terrain/commands/init_state/control/asset/domain_rand/rewards/PPO。 |
| `legged_mjlab/envs/him_go2/him_go2_env.py` | mjlab `ManagerBasedRlEnv` 风格环境组装。 |
| `legged_mjlab/envs/him_go2/go2_asset.py` | Go2 MJCF、actuator、collision、sensor 配置封装。 |
| `resources/robots/unitree_go2/xmls/go2.xml` | 当前配置指向的 Go2 MJCF。 |
| `legged_mjlab/envs/him_go2/__init__.py` | 注册 `task_id="him_go2"`、`wrapper_name="him"`。 |
| `legged_mjlab/scripts/train.py` | 训练入口，默认 `--task him_go2`。 |
| `legged_mjlab/wrappers/him_wrapper.py` | 将 mjlab obs 转成旧 HIM runner 需要的 history obs 和 legacy step tuple。 |
| `rsl_rl/rsl_rl/runners/him_on_policy_runner.py` | 当前项目内的旧 HIM runner。 |

当前数据链意图是：

```text
MjLab native obs:
  actor  [N,45]
  critic [N,235]
      |
      v
HIMRslRlWrapper:
  actor history [N,270] = [t, t-1, t-2, t-3, t-4, t-5] x 45
  critic passthrough [N,235]
      |
      v
HIMOnPolicyRunner / HIMPPO:
  actor uses [N,270]
  critic contract is currently forced to 45 + 3 = 48
```

关键点：

- `him_go2_config.py:10-14` 声明单帧 45、历史 6、actor 270、privileged 235、动作 12。
- `him_wrapper.py:26-34` 强制 history length 必须是 6，one-step obs 必须是 45。
- `him_wrapper.py:171-180` reset 时只把当前帧放到第 0 帧，历史其余 5 帧置零。
- `him_on_policy_runner.py:165-168` 注释和实现都说明 HIM critic 只使用 `num_one_step_obs + 3`，也就是 48；多出来的 height scan 等 privileged 项不是当前 runner 的有效 contract。

这意味着：如果不改 wrapper/runner，就不要贸然把 recovery 的 actor 单帧改成 46 或更多维。可以先做 45 维兼容版起身训练：固定目标站高，用现有 projected gravity、base angular velocity、joint pos/vel、last action 学起身。之后再考虑专门的 recovery wrapper，把 height command 加进 actor。

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

### 4.2 Config 中需要修的错配

必须先修的配置/实现错配：

| 问题 | 证据 | 影响 |
|---|---|---|
| `control` 里是 `hip_reduction`，env 里访问 `hip_scale_reduction`。 | `him_go2_config.py:88` vs `him_go2_env.py:207` | 构建 action cfg 会 `AttributeError`。 |
| config 没有 `action_clip`，env 访问 `self.robot_cfg.control.action_clip`。 | `him_go2_env.py:211` | 构建 action cfg 会失败。 |
| `commands.resampling_time` 是 float，env 写成 `tuple(self.robot_cfg.commands.resampling_time)`。 | `him_go2_config.py:51`、`him_go2_env.py:462` | `tuple(10.0)` 会 `TypeError`；mjlab 需要 range tuple 时应显式表达。 |
| config 是 `randomize_friction`，env 写 `frandomize_friction`。 | `him_go2_config.py:128`、`him_go2_env.py:400` | 摩擦随机化分支访问不存在字段。 |
| `randomize_restitution` 配了，但 env 没有 restitution event。 | `him_go2_config.py:131-133`、`him_go2_env.py:302-448` | 配置不会生效。 |
| reward scales 里有 `dof_acc`、`foot_clearance`、`torque_limits`，env 中对应函数名不闭合。 | `him_go2_config.py:187`、`:190`、`:201`；`him_go2_env.py:631-649`、`:866` | reward builder 使用 `getattr(self, "_reward_" + name)`，会直接失败。 |
| `only_positive_rewards=True` 只是保存在 env 属性中，没有接到 mjlab RewardManager。 | `him_go2_env.py:53-54`、`.venv/.../reward_manager.py:116-130` | 不能假设总 reward 会裁剪到非负。 |

起身训练还需要注意：

- 当前 `commands` 是速度命令配置，`num_commands=4` 仍写着 heading，但 `_build_commands()` 实际只对策略返回 3 维 twist。见 `him_go2_config.py:50`、`him_go2_env.py:459-473`。
- 参考 recovery 任务把速度命令清零，并额外采样目标站高 `h*`。如果你保持 HIM 45 维 ABI，建议先不加 height command，使用固定 `base_height_target=0.30`。如果你要迁移成 46 维 recovery actor，就必须同步改 wrapper、runner 元数据、部署 YAML 和已有 checkpoint 兼容假设。

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

### 5.2 Python asset 封装未闭合

`Go2Asset` 当前会在初始化阶段失败：

- `go2_asset.py:27-33` 中 `__init__()` 先调用 `_parse_cfg()`，之后才创建 `self.entity` 和 `self.sensor_mgr`。
- `_parse_cfg()` 内 `go2_asset.py:74` 访问 `self.robot_cfg`，但这个属性从未定义，应是 `self.cfg`。
- `_parse_cfg()` 内 `go2_asset.py:88` 访问 `self.asset.entity.get_spec()`，但此时 `self.asset` 也不存在；`Go2Asset` 自己就是 asset 对象。
- `go2_asset.py:95`、`:101` 又访问 `self.asset.cfg.asset...`，同样会失败。

也就是说：MJCF 资源够，但 Python 适配层当前不是可用状态。环境第一步创建 `Go2Asset(self.robot_cfg)` 时就会卡住，见 `him_go2_env.py:46-47`。

### 5.3 动作和部署顺序风险

需要尽早冻结一个唯一的关节/action 顺序：

- `him_go2_config.py:61-75` 的 dict 顺序是先四个 hip，再四个 thigh，再四个 calf，而且腿顺序写成 `FL, RL, FR, RR`。
- XML 自然层级是 leg-major：`FL`、`FR`、`RL`、`RR`，每条腿 hip/thigh/calf。
- `deploy/deploy_mujoco/him_go2/policy/config.yaml:38-41` 注释为 `FL, RL, FR, RR` 的 leg-major，但数值和训练 config 也不完全一致。
- 部署 YAML 里 `hip_scale=0.125`，训练 config 里 `action_scale=0.25` 且 `hip_reduction=1.0`。见 `config.yaml:24-25`、`him_go2_config.py:85-88`。
- 部署 YAML 小腿 torque limit 是 `35.55`，训练 config 小腿是 `45`。见 `config.yaml:31-34`、`him_go2_config.py:83`。

这是 sim2real 前的 P0 风险。训练、仿真部署、真机部署必须共享同一份 action order、default pose、scale、torque limit 契约，否则策略输出会控制错关节或以错误幅度控制。

## 6. Env 审计

### 6.1 语法和构造硬阻塞

当前 `him_go2_env.py` 在 Python 3.11 下有语法级问题：

- `him_go2_env.py:663`
- `him_go2_env.py:676`

这两处 f-string 使用了 Python 3.12 才更宽松的写法；项目 `.venv` 是 Python 3.11，因此不能导入该文件。

即使语法修复后，还有连续构造问题：

| 问题 | 证据 | 影响 |
|---|---|---|
| `TaskRegistry.make_env()` 传 `device=`，但 `HimGo2Env.__init__()` 参数是 `sim_device, render_mode`。 | `task_registry.py:208`、`him_go2_env.py:39` | 训练入口 `train.py:53-58` 创建 env 会失败。 |
| `_build_scene()` 调 `_build_sensors(debug_vis=...)`，但 `_build_sensors()` 需要 `entity_name`。 | `him_go2_env.py:111-117` | scene 构建失败。 |
| `_build_observations()` 中 `entity_name = self.cfg.asset.name`，但此时还没 `super().__init__()`，`self.cfg` 不可靠；应使用 `self.robot_cfg` 语义。 | `him_go2_env.py:46-58`、`:476-480` | observation cfg 构建阶段失败或读错对象。 |
| `_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")`，但 scene entity 注册名是 `"go2"`。 | `him_go2_env.py:35`、`:103-110` | reward 默认查 `env.scene["robot"]`，实际没有。 |
| critic sensor name 使用 `"robot/imu_lin_vel"` 和 `"terrain_scan"`，但 asset entity 是 `go2`，height sensor name 是 `"height_scan"`。 | `him_go2_env.py:563-569`、`go2_asset.py:264-265` | critic obs 组装会找不到传感器。 |

### 6.2 Reset 当前不是“贴地随机倒姿起身”

当前随机 reset 的配置和实现：

- 默认 root 高度：`init_state.pos = (0, 0, 0.42)`，见 `him_go2_config.py:59-60`。
- reset 随机 z 偏移：`base_pose_z_range = [0.35, 0.5]`，见 `him_go2_config.py:174-175`。
- reset root 使用 `mdp.reset_root_state_uniform`，pose range 中只随机 `z/roll/pitch`，`x/y/yaw` 固定，velocity 空。见 `him_go2_env.py:263-288`。
- mjlab 1.6.0 的 `reset_root_state_uniform` 是在 `default_root_state` 上加 pose sample，再加 env origin。见 `.venv/lib/python3.11/site-packages/mjlab/envs/mdp/events.py:170-190`。
- joint reset 是 `position_range=(-0.0,0.0)`、`velocity_range=(-0.0,0.0)`，见 `him_go2_env.py:291-298`。

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

### 6.4 Reward 当前仍是行走任务，不适合随机倒姿起身

当前 reward scales 在 `him_go2_config.py:180-202`，包括速度跟踪、base height、orientation、feet air time、stand still 等行走任务项。

主要问题：

- `orientation` 当前返回 `g_x^2 + g_y^2`，见 `him_go2_env.py:697-717`。这个量只表示“base 是否水平”，不能区分正立和完全倒立：正立与倒立的 `g_x^2+g_y^2` 都是 0。随机倒姿下，这会允许“倒立但水平”成为局部最优。
- `tracking_lin_vel`、`tracking_ang_vel` 在恢复阶段会干扰起身。参考 recovery 中速度命令全部清零，不做 tracking。
- `feet_air_time`、`foot_clearance` 是步态项，不是起身项；恢复阶段应该关闭，或只在站稳后进入 locomotion fine-tune。
- `collision` 如果针对 base/thigh/calf，恢复初始接触地面时会惩罚正常状态；应关闭或改成 upright 后门控。
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

不建议迁移或应关闭：

- locomotion tracking：`tracking_lin_vel`、`tracking_ang_vel`。
- 步态项：`feet_air_time`、`foot_clearance`。
- 原来的 `orientation`。
- 初始接触会命中的 base/thigh/calf collision 惩罚。
- `stand_still` 这类“静止时回 default”的行走项，恢复任务要换成 recovery-specific joint reward。

### 7.4 参考仓库不能原样照搬的点

参考实现也有问题，迁移时要修正：

1. `recovery_success` 的首次成功奖励可能恒为 0。`check_termination()` 在奖励计算前先 `already_succeeded_buf[recovered] = True`，见 `legged_robot.py:257-259`；而 reward 又用 `success_buf & ~already_succeeded_buf`，见 `legged_robot.py:1855-1861`。mjlab 迁移时要保证首次成功 reward 在 latch 更新前计算，或单独维护 `first_success_this_step`。
2. 训练成功判定包含 `smooth_success`，但 `play_recovery.py:119-125` 的统计只检查 upright/height/joint，没有检查 smooth gate，评估可能高估成功率。
3. 分桶 roll/pitch 用的是小角近似 `roll=theta*cos(phi)`、`pitch=theta*sin(phi)`，在 `theta` 接近 pi 时不再严格等于 SO(3) 上的均匀倾角采样。见 `legged_robot.py:840-846`。
4. 注释说 z 随倒地程度变化，但代码实际是独立 `U(0.10,0.20)`，见 `legged_robot.py:872-875`。

## 8. 推荐迁移方案

### 8.1 先做两个任务变体

建议不要直接在 `him_go2` 行走任务里硬塞 recovery reward，而是拆成两个任务语义：

| 任务 | 目的 | 观测/命令 | reward |
|---|---|---|---|
| `him_go2` | 正常行走 / rough locomotion | 现有 45 维 actor + velocity command | locomotion reward |
| `him_go2_recovery` | 随机姿态起身 / 原地恢复 | 推荐先保持 45 维 actor，速度命令清零，固定目标高度；进阶版再加 height command | recovery reward |

这样能避免行走 reward 与起身 reward 相互干扰，也方便单独训练、评估成功率和后续做 distillation/fine-tune。

### 8.2 兼容旧 HIM 45 维的首版 recovery

如果目标是尽快跑通外部旧版 `rsl_rl`，首版建议保持：

```text
actor one-step = 45
history = 6 x 45 = 270
action = 12
critic exposed by env can remain 235, but runner actually uses first 48
target stand height = fixed 0.30 m
velocity command = zero or mostly zero
```

首版 recovery reward：

| 项 | 建议权重 | 门控 |
|---|---:|---|
| `upright_linear` | `+3.0` | 全程 |
| `stand_height_fixed` | `+1.0` 到 `+1.5` | `g_z < -0.7` 后启用，最好平滑门控 |
| `joint_to_default` | `+0.5` 到 `+1.5` | 初期弱一点；接近正立后强化 |
| `recovery_success` | `+1.0`，term 内首次成功给 `10` | 首次成功一次性 |
| `upside_down_penalty` | 若 term 返回负值则 `+1.0` 到 `+2.0`；若 term 返回正 cost 则负权重 | 全程 |
| `lin_vel_z` | `-0.3` 到 `-0.5` | 全程或站起后强化 |
| `ang_vel_xy` | `-0.05` 到 `-0.1` | 不要过大 |
| `lin_vel_xy` | `-0.3` 到 `-0.5` | 全程 |
| `action_rate` | `-0.05` 到 `-0.2` | 初期低，成功率上来后加 |
| `torques/dof_vel/dof_acc` | 小权重 | 先小，避免压制起身 |

需要关闭或置零：

- `tracking_lin_vel`
- `tracking_ang_vel`
- `feet_air_time`
- `foot_clearance`
- `orientation`
- `stand_still`
- recovery 初期的 base/thigh/calf collision 惩罚

### 8.3 进阶版 recovery：加入目标高度命令

如果你愿意打破现有 45 维 HIM ABI，可以迁移参考仓库的 height command：

```text
height-only recovery ABI:
  actor one-step = 46
  history = 6 x 46 = 276
  command/obs extra = h*

height + heading recovery ABI:
  actor one-step = 47
  history = 6 x 47 = 282
  command/obs extra = heading + h*
```

这两个 ABI 和 45D 首版互斥。45D 首版必须使用固定目标高度，不能每个 env 动态采样 `h*` 后又不把它给 actor；否则同一个 observation 对应不同最优动作，MDP 语义会被破坏。

但这会牵涉：

- `him_go2_config.py` 中 `num_one_step_observations`、`num_observations`。
- `HIMRslRlWrapper` 当前强制 45，需要新增 recovery wrapper 或放宽维度。
- `HIMOnPolicyRunner` 的 `num_one_step_obs`、critic 48 规则需要重新确认。
- 部署 YAML 里 `num_observations`、history、obs 顺序都要改。
- 旧 checkpoint 不能直接兼容。

因此推荐路线是：先做 45 维固定高度 recovery，把起身能力学出来；稳定后再做 46 维高度命令版本。

### 8.4 Reset 迁移要点

首版 recovery reset 建议：

| 状态 | 建议 |
|---|---|
| base position | `x/y` 小范围扰动；`z` 用低高度绝对值或 `env_origin_z + low_height`。 |
| base orientation | 不要独立均匀 roll/pitch；按倾角 bucket 采样，或者至少先分阶段从小角到全姿态。 |
| yaw | `U(-0.5,0.5)` 或更宽范围均可，起身任务不应依赖 yaw。 |
| root velocity | 初期置 0。后期再随机小速度。 |
| joint position | `q_default + U(-0.3,0.3)`，并 clamp 到 soft/hard joint limits。 |
| joint velocity | 初期置 0。 |
| domain randomization | 第一阶段关闭或保留很小范围；站起成功率稳定后再逐步打开。 |

mjlab 1.6.0 的内置 `reset_root_state_uniform` 是加法语义，因此如果想要“低高度绝对 reset”，可能需要自定义 reset event 或保证默认 root state 设成低姿态基准。否则当前 `0.42 + [0.35,0.5]` 会继续把机器人放到空中。

### 8.5 Curriculum 迁移要点

推荐迁移参考仓库的“姿态分桶 + 低成功率桶过采样”思想：

```text
bins:
  [0, 40 deg]
  [40, 90 deg]
  [90, 140 deg]
  [140, 180 deg]

bounded linear weight_i proportional to 0.20 + (1 - success_rate_i)
```

注意：

- 成功率统计应该按 episode 内是否曾成功，而不是 timeout 瞬间是否成功。
- 第一次 reset 不应写入失败样本。
- 分桶统计如果每次 `.cpu().tolist()` 会带来 GPU 同步，4096 env 下可以接受但要避免过于频繁或过细粒度。
- 如果后续上 rough terrain，高度成功判定要用相对局部地面高度，而不是单纯世界 z。

也可以采用倒数形式 `weight_i proportional to 1 / (success_rate_i + 0.1)`，但那是另一套采样温度。本文后续代码片段采用有界线性形式，因为它不会在某个桶短期成功率接近 0 时过度放大采样概率。

## 9. 具体修改说明（文档级 Patch Guide）

这一节回答你强调的三个问题：**具体哪里错、为什么错、改完应该长什么样**。这里的代码都是“建议补丁片段”，本次没有写入源码；真正改代码时建议按 P0 到 P3 分批提交，每批只跑对应 smoke。

### 9.1 P0 硬阻塞：先让 env 能 import 和构造

| 位置 | 错在哪 | 为什么错 | 推荐改法 |
|---|---|---|---|
| `him_go2_env.py:35` | `_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")` | 当前 scene 里实体名来自 `cfg.asset.name`，配置是 `"go2"`；reward/obs 默认会去找不存在的 `env.scene["robot"]`。 | 改成 `SceneEntityCfg("go2")`，或者所有 term 都显式传 `asset_cfg=SceneEntityCfg(entity_name, ...)`。 |
| `him_go2_env.py:39-60` | `__init__()` 参数是 `sim_device, render_mode`，但 registry 传 `device=` 和 `play=`。 | `TaskRegistry.make_env()` 会用关键字 `device`，当前签名不接这个名字，训练入口创建 env 会失败。 | 用 `device: str`，`render_mode` 给默认值，并把 `device` 传给 `super()`。 |
| `him_go2_env.py:8-9` | 两次把不同模块命名为 `mdp`。 | `mjlab.tasks.velocity.mdp` 在本地会 re-export `mjlab.envs.mdp`，短期不一定炸，但可读性和后续迁移很差，容易把 velocity command 与通用 mdp API 混在一起。 | 建议别用同名 alias：通用 MDP 用 `env_mdp`，velocity command 单独 import。 |
| `him_go2_env.py:111-117` | `_build_scene()` 调 `_build_sensors(debug_vis=...)`，漏传 `entity_name`。 | `_build_sensors(self, entity_name, debug_vis)` 需要实体名生成 contact/raycast sensor。 | `self._build_sensors(entity_name=entity_name, debug_vis=debug_vis)`。 |
| `him_go2_env.py:663`、`:676` | f-string 写成 `f"Command '{"twist"}' not found."`。 | Python 3.11 下这是语法错误；当前 `.venv` 是 Python 3.11。 | 改成普通字符串 `"Command 'twist' not found."`。 |

对应的修改后代码片段建议是：

```python
import torch
import copy
import math

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp import dr
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("go2")


class HimGo2Env(ManagerBasedRlEnv):
    def __init__(
        self,
        cfg: HimGo2RoughCfg,
        device: str,
        render_mode: str | None = None,
        play: bool = False,
        debug_vis: bool = False,
    ):
        self.robot_cfg = cfg
        self.asset = Go2Asset(self.robot_cfg)
        self.play = bool(play)

        self.managercfg = self._build_mjlab_managercfg(
            play=self.play,
            debug_vis=debug_vis,
        )

        # mjlab RewardManager 不会自动实现 legged_gym 的 only_positive_rewards。
        # recovery 首版建议先关闭，避免负项被文档假设裁剪但实际没有裁剪。
        self.only_positive_rewards = False

        super().__init__(
            cfg=self.managercfg,
            device=device,
            render_mode=render_mode,
        )
```

`_build_scene()` 对应改成：

```python
def _build_scene(self, asset, play, debug_vis: bool = False):
    entity_name = self.robot_cfg.asset.name
    return SceneCfg(
        num_envs=self.robot_cfg.env.num_envs,
        env_spacing=self.robot_cfg.env.env_spacing,
        terrain=self._build_terrain(),
        entities={
            entity_name: self.asset.entity.get_robot_cfg(),
        },
        sensors=tuple(
            self._build_sensors(
                entity_name=entity_name,
                debug_vis=debug_vis,
            )
        ),
        extent=self.robot_cfg.env.extent,
    )
```

两个 reward 里的 command assert 改成：

```python
command = env.command_manager.get_command("twist")
assert command is not None, "Command 'twist' not found."
```

注意：如果你采用 `env_mdp` alias，所有通用函数也要同步改名，例如 `env_mdp.reset_root_state_uniform`、`env_mdp.joint_pos_rel`、`env_mdp.height_scan`。不要只改 import，不改调用。

### 9.2 `Go2Asset` 初始化顺序和 mjlab 1.6 sensor 字段

| 位置 | 错在哪 | 为什么错 | 推荐改法 |
|---|---|---|---|
| `go2_asset.py:28-33` | 先 `_parse_cfg()`，再创建 `self.entity`。 | `_parse_cfg()` 内要解析 MJCF spec，当前又写成 `self.asset.entity.get_spec()`，对象还没创建。 | 先建 `self.entity` / `self.sensor_mgr`，再 parse。 |
| `go2_asset.py:74` | 用 `self.robot_cfg`。 | `Go2Asset` 只有 `self.cfg`，没有 `self.robot_cfg`。 | 改成 `self.cfg`，或显式冻结 `GO2_POLICY_JOINT_ORDER`。 |
| `go2_asset.py:88`、`:95`、`:101` | 用 `self.asset...`。 | 在 `Go2Asset` 自身方法里没有 `self.asset`。 | 改成 `self.entity.get_spec()` 和 `self.cfg.asset...`。 |
| `go2_asset.py:245` | `ContactSensorCfg(..., exclude_parent_body=True)`。 | 本地 mjlab 1.6.0 的 `ContactSensorCfg` 有 `history_length`，但没有 `exclude_parent_body`。 | 删除该参数；如果确实要排除父子自碰，需要换 contact match 设计或在 reward 侧过滤。 |
| `go2_asset.py:281` | `RayCastSensorCfg(..., history_length=...)`。 | 本地 mjlab 1.6.0 的 `RayCastSensorCfg` 有 `exclude_parent_body`，但没有 `history_length`。 | 删除 `history_length`；height scan 不按 contact sensor 的 history 方式配置。 |
| `go2_asset.py:284-290` | `get_all_sensors()` 调 sensor 函数时没传 `entity_name`。 | 这些函数签名都需要实体名。虽然当前 env 没走这个函数，但后续调用会失败。 | 给 `get_all_sensors(entity_name, debug_vis=False)`。 |

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


class Go2Asset:
    def __init__(self, cfg: HimGo2RoughCfg):
        self.cfg = cfg
        self.entity = self.entitycfg(self)
        self.sensor_mgr = self.sensor(self)
        self._parse_cfg()

    def _parse_cfg(self):
        raw_file = getattr(self.cfg.asset, "file", "")
        self.xml_path: Path = self._resolve_xml_path(raw_file)

        self.effort_limit = self.cfg.control.effort_limit
        self.stiffness = self.cfg.control.stiffness
        self.damping = self.cfg.control.damping
        self.armature = self.cfg.asset.armature

        self.pos = self.cfg.init_state.pos
        self.rot = self.cfg.init_state.rot
        self.default_joint_angles = self.cfg.init_state.default_joint_angles

        self.vel_limit = self.cfg.rewards.soft_dof_vel_limit
        self.pos_limit = self.cfg.rewards.soft_dof_pos_limit

        missing = set(GO2_POLICY_JOINT_ORDER) - set(self.default_joint_angles)
        if missing:
            raise ValueError(f"default_joint_angles 缺少关节: {sorted(missing)}")
        self.joint_names = GO2_POLICY_JOINT_ORDER

        if self.cfg.domain_rand.randomize_cmd_action_latency:
            self.action_delay = self.cfg.domain_rand.range_cmd_action_latency
            self.action_delay_min = self.action_delay[0]
            self.action_delay_max = self.action_delay[1]
            self.action_hold_prob = self.cfg.domain_rand.action_hold_prob
        else:
            self.action_delay = 0
            self.action_delay_min = 0
            self.action_delay_max = 0
            self.action_hold_prob = 0

        spec = self.entity.get_spec()
        self.body_names = tuple(
            body.name for body in spec.worldbody.find_all("body")
        )

        self.penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            self.penalized_contact_names.extend(
                body for body in self.body_names if name in body
            )

        self.termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            self.termination_contact_names.extend(
                body for body in self.body_names if name in body
            )

        self.penalized_contact_names = tuple(dict.fromkeys(self.penalized_contact_names))
        self.termination_contact_names = tuple(dict.fromkeys(self.termination_contact_names))
```

sensor 部分建议这样修：

```python
def get_illegal_contact_sensor(self, entity_name: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        name="nonfoot_ground_touch",
        primary=ContactMatch(
            mode="body",
            pattern=tuple(self.asset.penalized_contact_names),
            entity=entity_name,
        ),
        secondary=ContactMatch(
            mode="body",
            pattern="terrain",
        ),
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
        history_length=self.asset.cfg.control.decimation,
    )


def get_height_scan_sensor(
    self,
    entity_name: str,
    debug_vis: bool = False,
) -> RayCastSensorCfg:
    x_points = tuple(self.asset.cfg.terrain.measured_points_x)
    y_points = tuple(self.asset.cfg.terrain.measured_points_y)
    return RayCastSensorCfg(
        name="height_scan",
        frame=ObjRef(
            type="body",
            name="base_link",
            entity=entity_name,
        ),
        pattern=GridPatternCfg(
            size=(
                max(x_points) - min(x_points),
                max(y_points) - min(y_points),
            ),
            resolution=self.asset.cfg.terrain.horizontal_scale,
        ),
        ray_alignment="yaw",
        max_distance=5.0,
        exclude_parent_body=True,
        debug_vis=bool(debug_vis),
    )


def get_all_sensors(
    self,
    entity_name: str,
    debug_vis: bool = False,
) -> Dict[str, Any]:
    sensors = {
        "feet_ground_contact": self.get_foot_contact_sensor(entity_name),
        "illegal_contact": self.get_illegal_contact_sensor(entity_name),
    }
    if getattr(self.asset.cfg.terrain, "measure_heights", False):
        sensors["height_scan"] = self.get_height_scan_sensor(
            entity_name,
            debug_vis=debug_vis,
        )
    return sensors
```

### 9.3 Action、command、observation 的 ABI 修法

| 位置 | 错在哪 | 为什么错 | 推荐改法 |
|---|---|---|---|
| `him_go2_env.py:207` | env 读 `hip_scale_reduction`。 | config 里字段是 `hip_reduction`。 | 统一成 `hip_reduction`。 |
| `him_go2_env.py:211` | env 读 `control.action_clip`。 | config 没有这个字段；已有 `normalization.clip_actions`。 | 用 `normalization.clip_actions`，或者在 config 新增字段并全链路使用。 |
| `him_go2_env.py:220` | `JointPositionActionCfg` 参数写 `actuator_name`。 | mjlab 1.6.0 的字段是 `actuator_names`。 | 改成 `actuator_names=self.asset.joint_names`。 |
| `him_go2_env.py:224` | `clip` 给 tuple，且容易被误解为 raw action 的 `[-1,1]` 限幅。 | mjlab 1.6.0 的 `JointPositionAction` 先算 `raw * scale + q0`，再对 processed target 做 `clip`；它不是 policy raw action 安全门。 | raw action 需要在 wrapper/runner 边界做 finite + clamp；env action term 的 `clip` 应表达每关节目标位置安全范围。 |
| `him_go2_env.py:462` | `tuple(self.robot_cfg.commands.resampling_time)`。 | `resampling_time` 是 float，`tuple(10.0)` 会 TypeError。 | 标量转成 `(period, period)`。 |
| `him_go2_env.py:471` | `heading_command=False` 但仍传 `heading=tuple(ranges.heading)`。 | `UniformVelocityCommand` 初始化时会拒绝 `heading_command=False` 且 `ranges.heading` 非空。 | `heading = None if not heading_command else tuple(ranges.heading)`。 |
| `him_go2_env.py:480` | observation 构建阶段读 `self.cfg.asset.name`。 | `self.cfg` 是 `ManagerBasedRlEnv` 初始化后才更可靠；构建 cfg 阶段应读 `self.robot_cfg`。 | 改成 `self.robot_cfg.asset.name`。 |
| `him_go2_env.py:527`、`:534`、`:542` | `projected_gravity/joint_pos_rel/joint_vel_rel` 没传 `asset_cfg`。 | 这些 mjlab 默认 `SceneEntityCfg("robot")`，当前实体是 `go2`。 | 显式传 `SceneEntityCfg(entity_name, ...)`。 |
| `him_go2_env.py:565`、`:569` | critic sensor name 是 `"robot/imu_lin_vel"` 和 `"terrain_scan"`。 | 当前 builtin sensor 应跟实体名走，height scan 名是 `"height_scan"`。 | 改成 `f"{entity_name}/imu_lin_vel"` 和 `"height_scan"`。 |

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

如果是首版 recovery 任务，建议再单独把速度命令范围收成零，保持 45 维 actor：

```python
ranges=UniformVelocityCommandCfg.Ranges(
    lin_vel_x=(0.0, 0.0),
    lin_vel_y=(0.0, 0.0),
    ang_vel_z=(0.0, 0.0),
    heading=None,
)
```

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
            func=env_mdp.generated_commands,
            params={"command_name": "twist"},
            clip=clip_obs,
        ),
        "base_ang_vel": ObservationTermCfg(
            func=env_mdp.builtin_sensor,
            params={"sensor_name": f"{entity_name}/imu_ang_vel"},
            scale=self.robot_cfg.normalization.obs_scales.ang_vel,
            clip=clip_obs,
        ),
        "projected_gravity": ObservationTermCfg(
            func=env_mdp.projected_gravity,
            params={"asset_cfg": asset_cfg},
            clip=clip_obs,
        ),
        "joint_pos": ObservationTermCfg(
            func=env_mdp.joint_pos_rel,
            params={"asset_cfg": joint_cfg},
            scale=self.robot_cfg.normalization.obs_scales.dof_pos,
            clip=clip_obs,
        ),
        "joint_vel": ObservationTermCfg(
            func=env_mdp.joint_vel_rel,
            params={"asset_cfg": joint_cfg},
            scale=self.robot_cfg.normalization.obs_scales.dof_vel,
            clip=clip_obs,
        ),
        "last_action": ObservationTermCfg(
            func=env_mdp.last_action,
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
            func=env_mdp.builtin_sensor,
            params={"sensor_name": f"{entity_name}/imu_lin_vel"},
        ),
        "height_scan": ObservationTermCfg(
            func=env_mdp.height_scan,
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
| `him_go2_env.py:384-397` | `mode="interval"` 的 push 没有 `interval_range_s`，也没显式 `asset_cfg`。 | mjlab `EventTermCfg` 对 interval 模式要求 interval range；push 函数也接 `asset_cfg`。 | 给 `interval_range_s=tuple(push_interval_s)`，params 中加 `asset_cfg`。 |
| `him_go2_env.py:400` | `frandomize_friction` 拼写错误。 | config 里是 `randomize_friction`。 | 改字段名。 |
| `him_go2_config.py:132-133` | 配了 `randomize_restitution`，env 没有实际接线。 | 本地 mjlab 1.6.0 的 `dr.geom` 没有直接 `geom_restitution` 函数。 | 先在文档/配置里标成未接线；要么补 mjlab DR 函数，要么通过 material/pair/XML 方案处理，不要写一个不存在的 `dr.geom_restitution`。 |
| `him_go2_env.py:263-288` | 用 `reset_root_state_uniform` 实现低姿态 recovery。 | 该函数是 `default_root_state + pose_sample + env_origin`，当前默认 z=0.42，再加 `[0.35,0.50]` 变成高空 reset。 | 行走任务可保留；recovery 任务要换自定义低高度 reset event。 |

push 和 friction 的最低修法：

```python
if self.robot_cfg.domain_rand.push_robots:
    events["push_robot"] = EventTermCfg(
        mode="interval",
        interval_range_s=tuple(self.robot_cfg.domain_rand.push_interval_s),
        func=env_mdp.push_by_setting_velocity,
        params={
            "asset_cfg": SceneEntityCfg(entity_name),
            "velocity_range": {
                "x": tuple(self.robot_cfg.domain_rand.push_vel_xy),
                "y": tuple(self.robot_cfg.domain_rand.push_vel_xy),
                "z": tuple(self.robot_cfg.domain_rand.push_vel_z),
                "roll": tuple(self.robot_cfg.domain_rand.push_ang_rp),
                "pitch": tuple(self.robot_cfg.domain_rand.push_ang_rp),
                "yaw": tuple(self.robot_cfg.domain_rand.push_ang_y),
            },
        },
    )

if self.robot_cfg.domain_rand.randomize_friction:
    events["friction"] = EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": foot_cfg,
            "operation": "abs",
            "ranges": tuple(self.robot_cfg.domain_rand.friction_range),
            "shared_random": True,
        },
    )
```

首版 recovery 建议先关闭 push、friction、restitution、latency、强 motor strength DR，等固定 plane 起身能稳定后再逐项打开。原因不是这些随机化不重要，而是 recovery 早期的失败模式很多；一次打开太多变量会看不出 reward、reset 还是动作 ABI 在出错。

### 9.5 Reward 注册闭合和 recovery reward 替换

| 位置 | 错在哪 | 为什么错 | 推荐改法 |
|---|---|---|---|
| `him_go2_env.py:635-649` | 所有非零 reward scale 都用 `getattr(self, "_reward_" + name)` 动态找函数。 | config 里有 `dof_acc`、`foot_clearance`、`torque_limits`，当前函数名不闭合，会直接 AttributeError。 | P0 阶段要么把这些 scale 置零，要么显式 reward map 并补齐函数。 |
| `him_go2_env.py:697-717` | `_reward_orientation()` 惩罚 `g_x^2 + g_y^2`。 | 它只能判断 base 是否水平，不能区分正立和倒立；倒立水平时也接近 0。 | recovery 中不要用这个主项，改用 `upright_linear = 0.5 * (1 - g_z)`。 |
| `him_go2_config.py:180-202` | recovery 初期仍开 locomotion tracking、feet air time、foot clearance、collision、stand still。 | 这些项鼓励走路或惩罚正常倒地接触，会干扰起身。 | 新建 recovery scales，先关掉 locomotion 项。 |
| `him_go2_config.py:204` | `only_positive_rewards=True`。 | mjlab RewardManager 没有 legged_gym 那种总 reward clamp。 | recovery 首版设为 False；如果一定要 clamp，应在 env reward aggregation 边界显式实现并测试。 |

如果先让现有 locomotion env 可 import，最小 reward builder 可以先做显式 map，避免动态 `getattr` 静默引入错名：

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

    reward_map = {
        "tracking_lin_vel": (self._reward_tracking_lin_vel, {"asset_cfg": asset_cfg}),
        "tracking_ang_vel": (self._reward_tracking_ang_vel, {"asset_cfg": asset_cfg}),
        "lin_vel_z": (self._reward_lin_vel_z, {"asset_cfg": asset_cfg}),
        "ang_vel_xy": (self._reward_ang_vel_xy, {"asset_cfg": asset_cfg}),
        "orientation": (self._reward_orientation, {"asset_cfg": asset_cfg}),
        "base_height": (self._reward_base_height, {"asset_cfg": asset_cfg}),
        "torques": (self._reward_torques, {"asset_cfg": asset_cfg}),
        "dof_vel": (self._reward_dof_vel, {"asset_cfg": joint_cfg}),
        "dof_pos_limits": (self._reward_dof_pos_limits, {"asset_cfg": joint_cfg}),
        "action_rate": (self._reward_action_rate, {}),
        "collision": (self._reward_collision, {"asset_cfg": asset_cfg}),
        "stand_still": (self._reward_stand_still, {"asset_cfg": joint_cfg}),
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

recovery 任务建议用一套单独 scales：

```python
recovery_scales = {
    "upright_linear": 3.0,
    "stand_height": 1.5,
    "joint_to_default": 1.0,
    "recovery_success": 1.0,
    "upside_down": 1.0,
    "lin_vel_xy": -0.25,
    "lin_vel_z": -0.25,
    "ang_vel_xy": -0.05,
    "action_rate": -0.02,
    "dof_pos_limits": -2.0,
    "tracking_lin_vel": 0.0,
    "tracking_ang_vel": 0.0,
    "feet_air_time": 0.0,
    "foot_clearance": 0.0,
    "collision": 0.0,
    "stand_still": 0.0,
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

    first_success = recovered & ~env.recovery_success_latched
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


def _reward_stand_height(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    target_height = self._recovery_target_height_fixed(env)
    upright_gate = (asset.data.projected_gravity_b[:, 2] < -0.70).float()
    sigma_sq = env.recovery_height_reward_sigma_sq_m2
    return torch.exp(-(height - target_height).square() / sigma_sq) * upright_gate


def _reward_joint_to_default(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    error = (asset.data.joint_pos - asset.data.default_joint_pos).square().mean(dim=-1)
    return torch.exp(-error / 0.25)


def _reward_recovery_success(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    return 10.0 * env.recovery_first_success.float()


def _reward_upside_down(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    grav_z = env.scene[asset_cfg.name].data.projected_gravity_b[:, 2]
    return -torch.clamp(grav_z, min=0.0)
```

注册时可以把状态更新放进 termination manager，但它不是失败终止项：

```python
terminations["recovery_state_update"] = TerminationTermCfg(
    func=self._update_recovery_success_state,
    time_out=False,
    params={"asset_cfg": SceneEntityCfg(self.robot_cfg.asset.name)},
)
```

这要求 `_update_recovery_success_state()` 永远返回全 False。它利用 mjlab “termination 先于 reward”的顺序生成 reward snapshot，但不改变 done 语义。

上面的 reward 片段是 45D 固定高度版本。如果切到 46D height-command ABI，才能把 `_recovery_target_height_fixed()` 换成读取 `env.recovery_height_cmd`；同时 actor observation、history、critic 前 49 维、wrapper 和 checkpoint metadata 都要一起改。

这里有三个关键边界：

- 成功状态只能在每个控制步的一个位置更新一次；reward term 应只读取 `recovery_first_success` 这类 snapshot。如果测试或日志又手动调用 `reward_manager.compute()`，不应再次推进 latch。
- `recovery_success` 不应把成功写成 done。mjlab step 顺序是 termination 再 reward；如果把 state-update 放在 termination manager 里，它必须返回全 False，只负责生成 reward 前 snapshot。
- `stand_height` 用的是相对 `env.scene.env_origins[:, 2]` 的高度。上 rough terrain 后还要改成相对局部地面高度，否则坡面/台阶会把成功判定搞偏。`height_success_tolerance_m=0.05` 和 `height_reward_sigma_sq_m2=0.05` 是两个不同量，不能把后者误写成 `sigma_h=0.05 m`。

### 9.6 Recovery reset：把“空中随机姿态”改成“贴地倒姿起身”

当前错在 `him_go2_env.py:263-288` 使用内置加法 reset，并且 config 里默认 z=0.42、随机 z=[0.35,0.50]。改完不应该继续用这组参数表达 recovery。

推荐写一个自定义 reset event，核心语义是：

```text
pose bucket -> roll/pitch/yaw -> quaternion
root z      -> low calibrated height above terrain origin
joint pos   -> q_default + small jitter, clamped to joint limits
velocities  -> zero at first stage
episode state -> clear success latch and bucket id
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
    q = q0 + torch.empty_like(q0).uniform_(-0.3, 0.3)
    limits = asset.data.soft_joint_pos_limits[env_ids]
    q = q.clamp(limits[..., 0], limits[..., 1])
    asset.write_joint_state_to_sim(
        q,
        torch.zeros_like(q),
        env_ids=env_ids,
    )

    env.recovery_bucket[env_ids] = bucket
    env.recovery_ever_success[env_ids] = False
    env.recovery_success_latched[env_ids] = False
    env.recovery_first_success[env_ids] = False
    env.recovery_stable_steps[env_ids] = 0
    env.recovery_recent_max_agitation[env_ids] = 0.0
    env.recovery_episode_active[env_ids] = True
```

这段代码还有两个需要你实际落地时确认的点：

- `quat_from_euler_xyz` 要从 `mjlab.utils.lab_api.math` 或项目已有 math helper 正确导入。
- `asset.data.soft_joint_pos_limits` 在本地 mjlab 1.6.0 的 `EntityData` 中存在，但仍要在 `Go2Asset` 可构造后确认 shape 是 `[num_envs, 12, 2]` 且 joint 顺序与 `self.asset.joint_names` 一致。

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
    valid = env.recovery_episode_active[env_ids]
    ids = env_ids[valid]
    if ids.numel() == 0:
        return {"mean_success": env.recovery_success_ema.mean()}

    bucket = env.recovery_bucket[ids]
    success = env.recovery_ever_success[ids].float()
    num_buckets = env.recovery_bucket_weights.numel()

    trials = torch.bincount(bucket, minlength=num_buckets).float()
    wins = torch.bincount(bucket, weights=success, minlength=num_buckets)
    seen = trials > 0
    batch_rate = wins / trials.clamp_min(1.0)

    env.recovery_trials += trials
    env.recovery_success_ema[seen] = (
        0.9 * env.recovery_success_ema[seen]
        + 0.1 * batch_rate[seen]
    )

    covered = env.recovery_trials >= env.recovery_min_samples
    weakness = torch.where(
        covered,
        1.0 - env.recovery_success_ema,
        torch.ones_like(env.recovery_success_ema),
    )
    weights = (0.20 + weakness).clamp_min(0.05)
    env.recovery_bucket_weights.copy_(weights / weights.sum())

    return {"mean_success": env.recovery_success_ema.mean()}
```

不要每个 reset 都把 tensor 拉到 CPU 做 Python list 统计；4096 env 下这样会引入同步点，训练吞吐会变差。

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

这里故意把 `hip` scale 写成 `0.125`，不是沿用当前训练 config 的 `action_scale=0.25, hip_reduction=1.0`。原因是部署 YAML 当前写的是 `hip_scale=0.125`，而 Go2 髋关节横摆通常对幅度更敏感。你也可以选择训练继续用 hip 0.25，但那必须同步改部署配置和安全限幅，不能训练/部署两边不一致。

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
| AST only | `FAIL`，当前停在 `him_go2_env.py:663` | 无项目写入 | Python 3.11 AST 通过。 |
| backend source gate | `NOT_RUN` | import 级副作用风险，禁 pycache | `rsl_rl.__file__` 和 `HIMOnPolicyRunner.__module__` 均指向项目内旧版实现。 |
| pytest contract | `NOT_RUN`，`.venv` 无 pytest，建议测试文件尚未创建 | 测试 cache/临时文件；需抑制 cache 或写 `/tmp` | `legged_mjlab/test/test_him_go2_contracts.py` 创建后通过。 |
| env smoke | `REQUIRES_WRITE_SCOPE` | MuJoCo/mjlab 进程状态，可能写临时资源 | `make_env -> reset -> step` 形状和 finite 检查通过。 |
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

建议按下面顺序推进，不要先写起身 reward：

### P0：让 `him_go2` 能导入和构造

目标：最小 `num_envs=1` 可以 import、make_env、reset。

必须修：

1. Python 3.11 f-string 语法。
2. `TaskRegistry.make_env()` 与 `HimGo2Env.__init__()` 参数对齐。
3. `Go2Asset` 中 `self.robot_cfg` / `self.asset` 未定义的问题。
4. `_build_sensors()` 调用参数。
5. `self.cfg` 和 `self.robot_cfg` 的使用边界。
6. `hip_reduction`、`action_clip`、`randomize_friction` 等字段名。
7. reward 名称与函数闭合，所有非零 scale 都必须有对应 callable。

完成标准：

- 能 import `legged_mjlab.envs.him_go2.him_go2_env`。
- 能创建 `Go2Asset(HimGo2RoughCfg())` 并解析 MJCF。
- 能 `task_registry.make_env("him_go2", device="cpu", num_envs=1)`。

### P1：冻结 ABI

目标：把训练、wrapper、runner、部署约定统一。

必须明确：

- action order：12 维输出到底是 `FL hip/thigh/calf, FR..., RL..., RR...`，还是当前 config dict 顺序。
- obs order：`[cmd(3), ang_vel(3), projected_gravity(3), q_rel(12), qd(12), last_action(12)]`。
- history order：当前帧在前 `[t, t-1, ..., t-5]`。
- critic order：如果保留 env 输出 235，则说明 runner 当前只用前 48；如果要用 height scan，则 runner 也要改。
- policy frequency：`sim.dt=0.005`、`decimation=4`，control dt = `0.02 s`。
- timeout bootstrap：旧 HIM PPO 不完全等同新版 rsl_rl，需要确认 `infos` 中 timeout 信息传递和 bootstrap 语义。

完成标准：

- reset 后 actor shape `[N,45]`，wrapper shape `[N,270]`。
- critic shape 和 runner 实际使用维度一致。
- random action step 后返回 legacy HIM runner 需要的 tuple，terminal observation 不被 auto-reset 覆盖。
- 调用 `load_project_rsl()` 后断言 `sys.modules["rsl_rl"].__file__`、`runners.__file__`、`HIMOnPolicyRunner.__module__` 指向项目内旧版 `rsl_rl`，而不是 site-packages 新版 ABI。

### P2：单独做 recovery reset

目标：得到真实贴地随机姿态起身 reset。

建议：

- 新建 recovery cfg/env 变体，先用 plane terrain、零速度命令、低 DR。
- base z 改成低高度绝对 reset 或低高度基准 reset。
- joint reset 加 `U(-0.3,0.3)`。
- 添加 episode 级 `ever_success`、`first_success_this_step`、`recent_max_agitation`、`bucket_id` 等状态。
- 成功不终止，只统计和给 reward。

完成标准：

- 随机 reset 后采样分布可打印/可视化：base z、`g_z`、joint deviation。
- 站立姿态、侧躺、仰倒、俯倒都能被采到。
- 不出现刚 reset 就因为正常身体接触而失败终止。
- 每个姿态桶经过 `sim.forward()` 和第一步物理检查：无深穿透、无明显悬空、接触冲量和首步力矩没有异常尖峰。只检查 `z` 范围不够。

### P3：迁移 recovery reward

目标：先学会站起，再谈行走。

首批只开：

- `upright_linear`
- `stand_height_fixed`
- `joint_to_default`
- `recovery_success`
- `lin_vel_z`
- `ang_vel_xy`
- `lin_vel_xy`
- `action_rate`

暂不开：

- rough terrain curriculum
- 速度 tracking
- feet air time
- foot clearance
- 强 DR
- 强 torque/dof_acc 惩罚

完成标准：

- 每个 reward term shape 都是 `[num_envs]`。
- episode reward log 中主项数值范围合理，没有 NaN/Inf。
- 成功率按 episode 内是否曾成功统计。
- 先用小 env 数和短训练 smoke 验证曲线方向，再放大到 4096。
- success latch 只在每个控制步的一个位置更新；测试和日志不得在同一控制步重复调用会推进状态的 reward/update 函数。

### P4：恢复能力稳定后再接 locomotion

当前状态：`UNSELECTED`。下面只是路线选项，不代表已经可以执行或已经验证。

可选路径：

1. recovery policy 作为初始化，再 fine-tune locomotion。
2. recovery 和 locomotion 做 multi-stage curriculum。
3. recovery 成功后切换到 velocity command 训练。

不建议一开始就把全姿态起身、粗糙地形、速度跟踪、强 DR、强平滑惩罚全部打开。

### 阶段门槛表

| 阶段 | 前置条件 | 写入范围 | 当前状态 |
|---|---|---|---|
| P0 import/construct | Python 3.11 AST 修复、`Go2Asset` 构造修复 | 源码修改后才可跑动态 smoke | `FAIL`，当前 env 语法仍阻塞。 |
| P1 ABI | P0 通过；action/obs/critic/order 文档冻结 | 可能修改 config/wrapper/runner/deploy 文档 | `NOT_RUN`，backend source gate 尚未跑。 |
| P2 recovery reset | P1 通过；plane + low DR recovery cfg | 需要运行 MuJoCo reset/forward/step | `REQUIRES_WRITE_SCOPE`。 |
| P3 recovery reward | P2 reset 通过；状态机单点更新 | 需要 reward logs 和短训练日志 | `REQUIRES_WRITE_SCOPE`。 |
| P4 locomotion 接回 | P3 各姿态桶成功率稳定且无安全异常 | 训练日志/checkpoint | `UNSELECTED`。 |

## 11. 验证命令建议

这些命令是后续改代码后的验证建议。本次没有运行训练，也没有修改代码。

当前只读验证分支确认到的状态：

| 检查 | 状态 | 说明 |
|---|---|---|
| Python AST | 失败 | `.venv/bin/python -B -c 'import ast,pathlib; ast.parse(pathlib.Path("legged_mjlab/envs/him_go2/him_go2_env.py").read_text())'` 会在 `him_go2_env.py:663` 报 `f-string: expecting '}'`；`:676` 有同类写法。 |
| env import | 阻塞 | `import legged_mjlab.envs.him_go2.him_go2_env` 受同一语法错误阻塞，`import legged_mjlab.envs` 也会失败。 |
| 原始 XML asset | 通过轻量验证 | `mujoco.MjModel.from_xml_path("resources/robots/unitree_go2/xmls/go2.xml")` 可解析，模型约为 `nq=19, nv=18, nu=0, nbody=14`；这只证明 XML/mesh 可用，不证明 `Go2Asset` 封装可用。 |
| `Go2Asset` 构造 | 不可验证 | 先被 env 语法导入阻塞；修复语法后仍会遇到 `go2_asset.py:74` 的 `self.robot_cfg` 未定义等问题。 |
| HIM backend source | 未运行 | 必须调用 `load_project_rsl()` 后检查 `rsl_rl.__file__`、`runners.__file__`、`HIMOnPolicyRunner.__module__`；裸 `import rsl_rl` 不能证明训练时使用项目内旧版 runner。 |
| wrapper pytest | `NOT_RUN` | `.venv` 有 `torch==2.13.0` 但没有 `pytest`；系统 Python 有 pytest 但没有 torch。建议的新测试文件还没有创建；现有相关测试位于 `legged_mjlab/test/test_him_wrapper_terminal_privileged.py`，本次未运行。 |
| 训练 smoke | 未运行 | `train.py:62` 会创建日志目录，`:70` 会启动训练；本次文档-only 范围内不执行。 |

严格只读优先级：

1. AST only：对 `legged_mjlab/envs/him_go2/him_go2_env.py` 做 Python 3.11 AST 检查，不用 `py_compile/compileall`。
2. backend source gate：用 `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B` 调 `load_project_rsl()`，打印并断言项目内旧版 `rsl_rl` 来源。
3. 文件存在性检查：确认建议测试文件尚未创建时标 `NOT_RUN`，不要把模板命令写成当前可执行结果。

获准写入/运行后再做：

1. asset smoke：创建 `HimGo2RoughCfg()` 和 `Go2Asset(cfg)`，确认 MJCF、joint names、body names、sensor cfg 可解析。
2. env smoke：`task_registry.make_env("him_go2", device="cpu", num_envs=1)`。
3. reset shape：native reset 返回 `actor [1,45]`、`critic [1,235]` 或文档中冻结的新维度。
4. wrapper shape：`HIMRslRlWrapper.reset()` 返回 actor history `[1,270]`。
5. step smoke：随机零动作或小动作 step 一次，确认 reward、done、infos、terminal obs。
6. reward term smoke：逐项 reward 返回 `[N]`，没有 NaN/Inf；stateful recovery term 不得在同一控制步重复推进 latch。
7. recovery reset safety smoke：每个姿态桶检查穿透、悬空、首步接触冲量、首步力矩尖峰。
8. 最小训练 smoke：`--num-envs 1 --max-iterations 1 --log-dir /tmp/legged_mjlab_smoke`，只验证 runner ABI，不看性能。

## 12. 当前优先级清单

P0 blocker：

- `him_go2_env.py` Python 3.11 语法。
- `Go2Asset` 初始化引用未定义属性。
- `TaskRegistry.make_env()` 和 `HimGo2Env.__init__()` 参数不匹配。
- `_build_sensors()` 缺 `entity_name`。
- action cfg 字段名和 config 不匹配。
- command resampling time 类型不匹配。
- observation sensor name 不匹配。
- reward 名称和函数不闭合。

P1 blocker：

- action order、default joint order、部署 YAML 顺序不统一。
- raw action finite/clamp、processed target clip、per-joint `q_safe` clamp 三层语义还没有分开实现。
- critic 235 vs HIM runner 48 的实际契约未冻结。
- `only_positive_rewards` 语义和 mjlab RewardManager 不一致。
- 45D 固定高度与 46D 动态高度 ABI 还没有写成互斥任务；45D 下不能采样 actor 不可见的逐环境目标高度。

P2 recovery blocker：

- 当前 reset 是默认高度加 z 偏移，不是贴地倒姿。
- 关节 reset 没有随机扰动。
- 成功 predicate、episode 内 success latch、`recent_max_agitation` 状态机、bucket curriculum 状态还不存在。
- locomotion command/reward 没有与 recovery 分离。
- 每个姿态桶的 root height 还没有经过穿透、悬空、接触冲量和首步力矩校准。

P3 sim2real blocker：

- 训练和部署 action scale、hip scale、torque limit 不一致。
- deploy 目录目前更像 sim2sim 配置，没有完整策略调用、关节重排、安全限幅、watchdog、通信桥接。
- 本文第 9.8 只能作为仿真策略 ABI；缺少真实硬件 motor id、方向符号、零位、位置/速度/增量/力矩限幅、模式切换、CRC、传感器新鲜度、急停和 fail-closed 契约。

## 13. 推荐的最小下一步

最小可落地路线：

1. 先修到 `him_go2` 可以 `import -> make_env -> reset -> step`。
2. 用 `num_envs=1` 冻结 actor/critic/action shape、action order、raw action gate 和 target clip 语义。
3. 新建 recovery 配置语义，但先保持 45 维 HIM ABI，固定目标高度 `0.30 m`，速度命令清零。
4. 把当前 `orientation` 替换为 recovery 主项 `upright_linear`，关闭 locomotion tracking 和步态项。
5. 实现低高度随机倒姿 reset，并对每个姿态桶做穿透/悬空/首步力矩校准。
6. 加 `upright_linear + stand_height + joint_to_default + success + 轻约束` 的首版 reward；success 状态机只在每个控制步更新一次。
7. 用小规模 smoke 训练看成功率和 reward 曲线，日志写到显式目录，例如 `/tmp/legged_mjlab_smoke`。
8. 成功率稳定后再加入姿态 bucket curriculum、DR、粗糙地形和行走命令；如果要动态目标高度，再另立 46D ABI 并同步 wrapper/runner/deploy。

一句话：先把 mjlab env ABI 跑通，再做 45D 固定高度 recovery 任务分支；起身奖励的核心是 `upright_linear`、高度门控、默认姿态门控、单点更新的首次成功 latch 和低成功率姿态过采样，不是继续调当前 locomotion `orientation` 项。

## 14. 并发审查结果与已吸收修改

本轮按三个分支做了只读审查，审查本身没有修改代码：

| 分支 | 结果 | 已写回本文档的修改 |
|---|---|---|
| `safety_reviewer` | `REJECT` | 将第 9.8 从“Sim2Real ABI”改成“仿真策略 ABI”；补 raw action finite gate、accepted action、target clip、真机硬件映射表和 fail-closed blocker；强调该文档不能作为实机放行依据。 |
| `math_verifier` | `REJECT` | 修正 `stand_height` 指数尺度；把 45D 固定高度与 46D/47D 动态高度 ABI 写成互斥；把 success latch 从 reward 函数移到单点状态更新；补 `recent_max_agitation` 衰减峰值状态机；统一 bucket curriculum 为有界线性权重。 |
| `devops_build_engineer` | `BLOCKER/NOT_RUN` | 将验证拆成严格只读和获准写入后两类；补 `load_project_rsl()` backend source gate；标明 `.venv` 无 pytest、建议测试文件未创建；训练 smoke 必须显式写 `/tmp` 日志目录。 |

仍然不能声明 `ALL_TESTS_PASSED`，原因是：

- 没有修改源码，所以 `him_go2_env.py:663` 的 Python 3.11 语法错误仍在。
- 没有运行 `Go2Asset` 构造、`make_env`、reset、step、reward smoke、pytest 或训练。
- 真机相关内容仍缺硬件安全契约、逐关节实测映射和 watchdog/急停/断连策略。

因此本文档的定位是“迁移和修复指南”，不是已通过实现或发布说明。
