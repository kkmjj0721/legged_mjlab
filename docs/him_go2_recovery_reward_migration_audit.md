# HIM Go2 Env/Config/Asset 与随机姿态起身奖励迁移审计

日期：2026-08-31

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

这个设计值得迁移：成功只作为统计和奖励，不作为 episode termination。这样策略会被激励尽早站起并保持到 timeout。

### 7.3 Reward 设计

参考 reward 的原始 scale 会在旧 legged_gym 里乘 `dt=0.02`。迁移到 mjlab 1.6.0 时，RewardManager 默认 `scale_rewards_by_dt=True`，见 `.venv/lib/python3.11/site-packages/mjlab/envs/manager_based_rl_env.py:157` 和 `.venv/.../reward_manager.py:116-128`，所以不要再手动乘 dt。

| Reward | 公式/语义 | 参考 scale | 迁移建议 |
|---|---|---:|---|
| `upright_linear` | `(1 - g_z) / 2`，正立约 1，倒立约 0。见 `legged_robot.py:1825-1829`。 | `+3.0` | 必迁移，替代当前 `orientation` 主项。 |
| `stand_height` | `exp(-(z-h*)^2 / 0.05) * 1[g_z < -0.7]`。见 `legged_robot.py:1831-1837`。 | `+1.5` | 可迁移；如果不加 height command，则用固定 `base_height_target`。 |
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
actor one-step = 46
history = 6 x 46 = 276
command = [vx, vy, wz, h*] 或 [vx, vy, wz, heading, h*]
```

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

weight_i proportional to 1 / (success_rate_i + 0.1)
```

注意：

- 成功率统计应该按 episode 内是否曾成功，而不是 timeout 瞬间是否成功。
- 第一次 reset 不应写入失败样本。
- 分桶统计如果每次 `.cpu().tolist()` 会带来 GPU 同步，4096 env 下可以接受但要避免过于频繁或过细粒度。
- 如果后续上 rough terrain，高度成功判定要用相对局部地面高度，而不是单纯世界 z。

## 9. 后续执行顺序

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

### P4：恢复能力稳定后再接 locomotion

可选路径：

1. recovery policy 作为初始化，再 fine-tune locomotion。
2. recovery 和 locomotion 做 multi-stage curriculum。
3. recovery 成功后切换到 velocity command 训练。

不建议一开始就把全姿态起身、粗糙地形、速度跟踪、强 DR、强平滑惩罚全部打开。

## 10. 验证命令建议

这些命令是后续改代码后的验证建议。本次没有运行训练，也没有修改代码。

当前只读验证分支确认到的状态：

| 检查 | 状态 | 说明 |
|---|---|---|
| Python AST | 失败 | `.venv/bin/python -B -c 'import ast,pathlib; ast.parse(pathlib.Path("legged_mjlab/envs/him_go2/him_go2_env.py").read_text())'` 会在 `him_go2_env.py:663` 报 `f-string: expecting '}'`；`:676` 有同类写法。 |
| env import | 阻塞 | `import legged_mjlab.envs.him_go2.him_go2_env` 受同一语法错误阻塞，`import legged_mjlab.envs` 也会失败。 |
| 原始 XML asset | 通过轻量验证 | `mujoco.MjModel.from_xml_path("resources/robots/unitree_go2/xmls/go2.xml")` 可解析，模型约为 `nq=19, nv=18, nu=0, nbody=14`；这只证明 XML/mesh 可用，不证明 `Go2Asset` 封装可用。 |
| `Go2Asset` 构造 | 不可验证 | 先被 env 语法导入阻塞；修复语法后仍会遇到 `go2_asset.py:74` 的 `self.robot_cfg` 未定义等问题。 |
| wrapper pytest | 环境阻塞 | `.venv` 有 `torch==2.13.0` 但没有 `pytest`；系统 Python 有 pytest 但没有 torch。需要先统一测试解释器环境。 |
| 训练 smoke | 未运行 | `train.py:62` 会创建日志目录，`:70` 会启动训练；本次文档-only 范围内不执行。 |

只读/短 smoke 优先级：

1. 语法检查：对 `legged_mjlab/envs/him_go2/him_go2_env.py` 做 Python 3.11 AST/compile 检查。
2. 版本来源检查：打印 `mjlab.__version__`、`mujoco.__version__`、`rsl_rl.__file__`、`HIMOnPolicyRunner` 来源。
3. asset smoke：创建 `HimGo2RoughCfg()` 和 `Go2Asset(cfg)`，确认 MJCF、joint names、body names、sensor cfg 可解析。
4. env smoke：`task_registry.make_env("him_go2", device="cpu", num_envs=1)`。
5. reset shape：native reset 返回 `actor [1,45]`、`critic [1,235]` 或文档中冻结的新维度。
6. wrapper shape：`HIMRslRlWrapper.reset()` 返回 actor history `[1,270]`。
7. step smoke：随机零动作或小动作 step 一次，确认 reward、done、infos、terminal obs。
8. reward term smoke：逐项 reward 返回 `[N]`，没有 NaN/Inf。
9. 最小训练 smoke：`--num-envs 1 --max-iterations 1`，只验证 runner ABI，不看性能。

## 11. 当前优先级清单

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
- critic 235 vs HIM runner 48 的实际契约未冻结。
- `only_positive_rewards` 语义和 mjlab RewardManager 不一致。

P2 recovery blocker：

- 当前 reset 是默认高度加 z 偏移，不是贴地倒姿。
- 关节 reset 没有随机扰动。
- 成功 predicate、episode 内 success latch、bucket curriculum 状态还不存在。
- locomotion command/reward 没有与 recovery 分离。

P3 sim2real blocker：

- 训练和部署 action scale、hip scale、torque limit 不一致。
- deploy 目录目前更像 sim2sim 配置，没有完整策略调用、关节重排、安全限幅、watchdog、通信桥接。

## 12. 推荐的最小下一步

最小可落地路线：

1. 先修到 `him_go2` 可以 `import -> make_env -> reset -> step`。
2. 用 `num_envs=1` 冻结 actor/critic/action shape 和顺序。
3. 新建 recovery 配置语义，但先保持 45 维 HIM ABI，固定目标高度 `0.30 m`，速度命令清零。
4. 把当前 `orientation` 替换为 recovery 主项 `upright_linear`，关闭 locomotion tracking 和步态项。
5. 实现低高度随机倒姿 reset，并记录 episode 内成功。
6. 加 `upright_linear + stand_height + joint_to_default + success + 轻约束` 的首版 reward。
7. 用小规模 smoke 训练看成功率和 reward 曲线。
8. 成功率稳定后再加入姿态 bucket curriculum、DR、粗糙地形和行走命令。

一句话：先把 mjlab env ABI 跑通，再做 recovery 任务分支；起身奖励的核心是 `upright_linear`、高度门控、默认姿态门控、首次成功 latch 和低成功率姿态过采样，不是继续调当前 locomotion `orientation` 项。
