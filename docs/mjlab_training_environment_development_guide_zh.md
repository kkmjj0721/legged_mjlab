# 面向 legged_gym / HIMLoco 的 mjlab 训练环境开发手册

> 文档状态：设计与代码阅读手册（只读取证版）  
> 适用工作区：`/home/kk/github/legged_mjlab`  
> 参考项目：`/home/kk/github/unitree_rl_mjlab`、`/home/kk/github/HIMLoco`  
> 相关对照：`/home/kk/github/uw-himloco`  
> 本轮边界：只读代码、整理文档；不安装依赖、不运行训练/仿真、不修改业务代码。

## 0. 先给结论

你想要的目标，最稳妥的实现路线不是把 HIMLoco 的 `LeggedRobot` 基类原样搬到 mjlab，而是把两套项目各自最有价值的部分拆开组合：

| 借鉴对象 | 应该借鉴的内容 | 不应直接搬运的内容 |
| --- | --- | --- |
| HIMLoco / legged_gym | 清晰的任务入口、机器人任务目录、配置分层、`task_registry` 思维、`train/play` 工作流、HIM 的观测历史与 privileged critic 概念 | `isaacgym.gymapi`、PhysX tensor、继承式大基类中的物理循环、`_reward_xxx` 反射式奖励注册 |
| unitree_rl_mjlab | mjlab 的 manager-based env cfg、`SceneCfg`、`EntityCfg`、传感器、Observation/Action/Command/Event/Reward/Termination/Curriculum term、每个机器人独立的任务注册 | 把所有内容重新塞回一个大 `legged_robot.py`；无契约地复制 Isaac Gym 的 buffer 和 simulator API |
| 当前 `legged_mjlab` | 已有的 Go2 MJCF/mesh 资源和一小段配置骨架 | 目前为空的 base task、空的 legged robot、空的脚本目录、若干 0 字节占位文件，以及仍引用不存在的 `src` 包路径 |

推荐的最终形态是：

```text
目标任务包
  ├─ 机器人资产适配：MJCF + mesh + EntityCfg + actuator + 初始状态
  ├─ 任务基础配置：场景、传感器、观测、动作、命令、事件、奖励、终止、课程
  ├─ 机器人差异配置：只填名称、尺度、碰撞、执行器和机器人专属奖励参数
  ├─ 任务注册：task_id -> env_cfg / play_env_cfg / rl_cfg / runner
  ├─ 训练入口：加载注册任务 -> 创建 ManagerBasedRlEnv -> wrapper -> RSL runner
  └─ 可选 HIM 适配：在标准 mjlab env 与 RSL runner 之间定义 history/estimator 合同
```

最重要的工程顺序是：

1. 先让一个标准 PPO 的平地 Go2 任务形成可追踪的 mjlab 契约。
2. 再加入 rough terrain、raycast/contact sensor、domain randomization 和 curriculum。
3. 再加入 HIM history/estimator；不要一开始同时迁移 HIM、复杂地形和自定义控制器。
4. 最后才做 sim2sim、ONNX 或真机部署。

这样可以把“环境不对”“观测维度不对”“HIM estimator 不对”“部署动作映射不对”四类问题分开定位。

---

## 1. 证据规则、当前状态和范围

### 1.1 文档中的三种标记

- `[事实]`：能在当前工作树的文件中直接定位，后面给出相对路径和行号。
- `[设计建议]`：面向目标项目的推荐方案，不代表当前代码已经具备该能力。
- `[待验证]`：需要未来安装依赖、导入包、创建环境、运行仿真或做实验才能确认；本轮不声称已验证。

参考代码会持续变化，所以行号是本次静态扫描的定位锚点；若后续文件被修改，应以符号和邻近内容复核。

### 1.2 当前工作树快照

#### `unitree_rl_mjlab`

- `[事实]` `setup.py:6-9` 声明 `mjlab==1.2.0` 和 `mujoco-warp==3.5.0`。
- `[事实]` 任务代码在 `src/tasks/velocity/` 和 `src/tasks/tracking/`，机器人常量在 `src/assets/robots/`，入口脚本在 `scripts/`。
- `[事实]` `src/tasks/__init__.py:1-5` 通过 `import_packages` 导入任务包，黑名单为 `utils` 和 `.mdp`；具体任务在每个机器人配置包的 `__init__.py` 调用 `register_mjlab_task`。
- `[事实]` README 已提供训练、play、动作文件转换、部署流程，但安装文档仍同时包含训练和 C++/真机所需内容，不能把所有依赖都当成“训练环境最小依赖”。

#### `HIMLoco`

- `[事实]` 当前目录有 `legged_gym/`、`rsl_rl/`、`deploy/`、`sim2sim/`、`docs/` 和 `ARCHITECTURE_CONTEXT.md`。
- `[事实]` 当前 `HIMLoco/.git` 是空目录而不是有效 Git 元数据；根目录没有 README 和 setup.py。不能从这个副本推断分支、提交或上游版本。
- `[事实]` 可直接引用的安装元数据是 `legged_gym/setup.py:4-15`，包名为 `legged_gym`，依赖 `isaacgym`、`rsl-rl`、`matplotlib`。
- `[事实]` `HIMLoco/ARCHITECTURE_CONTEXT.md:3-20` 的根路径仍写成 `/home/kk/HIMLoco`，并包含 GO2W sim2sim/HIM 导出记忆；这些内容是历史记忆，不等于当前副本的运行验证。
- `[事实]` `uw-himloco/README.md` 是一个更完整的相关变体说明，但它不是当前 `HIMLoco` 目录本身，不能混用相对路径。

#### `legged_mjlab`

- `[事实]` 根目录有很短的 `README.md` 和一个当前为 0 字节的 `setup.py`；没有有效的 packaging 元数据，也没有 `pyproject.toml`。
- `[事实]` `legged_mjlab/envs/base/legged_robot_config.py:1-15` 只有一小段旧式配置骨架。
- `[事实]` `legged_mjlab/envs/base/base_task.py` 和 `legged_mjlab/envs/base/legged_robot.py` 当前为空文件。
- `[事实]` `legged_mjlab/envs/him_go1/`、`legged_mjlab/scripts/` 目前没有实现文件；`legged_mjlab/test/test_env.py`、`legged_mjlab/utils/helpers.py`、`legged_mjlab/utils/logger.py`、`docs/setup.md` 和 `docs/设计.md` 当前是 0 字节占位文件。`docs/前言.md` 已有一份关于 Manager-Based 与单体 Tensor 取舍的设计前言，但还不是安装、环境或训练实现文档；本手册是 `docs/` 下唯一完整的训练环境开发手册。
- `[事实]` `resources/robots/unitree_go2/` 已有 MJCF 和 mesh，`go2_constants.py:7` 却仍然 `from src import SRC_PATH`，而目标项目没有 `src` 包；`go2_constants.py:18-21` 还把 XML 路径拼接到了 `src/assets/...`，与目标当前资源位置不一致。
- `[事实]` 目标项目同时存在空 base class、部分复制来的旧式配置、空的占位 setup/test/utils 文件和部分 mjlab 风格 Go2 常量。这不是“已能运行的 mjlab 训练环境”，而是迁移中的骨架。

### 1.3 本轮没有做的事情

以下事情全部没有执行：

- 没有创建 Conda/venv，没有 `pip install`、`apt install` 或下载依赖。
- 没有导入 `mjlab`、`isaacgym` 或 `torch` 来做运行时探测。
- 没有启动训练、MuJoCo viewer、Isaac Gym viewer、sim2sim、ONNX 或真机控制。
- 没有运行测试、编译 C++ 或修改任何 Python/C++/XML/YAML 业务文件。

因此本文是一份基于代码结构的开发手册，不是“环境已经跑通”的报告。

---

## 2. 两个参考项目到底是怎样组织的

## 2.1 unitree_rl_mjlab：manager-based 的任务配置系统

### 2.1.1 目录职责

```text
unitree_rl_mjlab/
├── src/
│   ├── assets/robots/                 # 机器人常量、MJCF 规格、执行器和实体配置
│   └── tasks/
│       ├── velocity/                  # 速度跟踪任务
│       │   ├── velocity_env_cfg.py    # 共享基础 env cfg 工厂
│       │   ├── mdp/                   # 本项目自定义观测/奖励/终止/命令/课程项
│       │   ├── config/<robot>/        # 机器人具体 env cfg、rl cfg、注册
│       │   └── rl/runner.py           # 可选任务专用 runner
│       └── tracking/                  # 动作模仿任务
├── scripts/
│   ├── train.py
│   ├── play.py
│   ├── list_envs.py
│   └── csv_to_npz.py
├── doc/setup_zh.md
└── setup.py
```

这是当前目录事实的概括，依据为 `rg --files src scripts`、`src/tasks/__init__.py` 和 `setup.py`。

### 2.1.2 从命令到环境的调用链

训练入口的静态调用链是：

```text
scripts/train.py:195-218
  -> import src.tasks                         # 触发注册
  -> list_tasks()
  -> load_env_cfg(task_id)
  -> load_rl_cfg(task_id)
  -> ManagerBasedRlEnv(cfg=...)
  -> RslRlVecEnvWrapper(...)
  -> load_runner_cls(task_id) 或 MjlabOnPolicyRunner
  -> runner.learn(...)
```

更完整的细节如下：

1. `scripts/train.py:23-39` 定义 `TrainConfig`，把 `env` 和 `agent` 作为两个顶层配置，并预留 motion file、video、NaN guard、GPU 列表。
2. `scripts/train.py:36-39` 通过任务注册表取得 env cfg 和 RL cfg。
3. `scripts/train.py:42-140` 选择设备/seed，处理 tracking motion file，构造 `ManagerBasedRlEnv`，套 `RslRlVecEnvWrapper`，创建 runner，写入 env/agent YAML，最后调用 `learn`。
4. `scripts/train.py:145-192` 负责日志目录、`CUDA_VISIBLE_DEVICES`、`MUJOCO_GL=egl` 和可选多 GPU `torchrunx`。
5. `scripts/train.py:195-218` 先导入任务，再由 tyro 把任务 ID 和其余 CLI override 解析成配置。
6. `scripts/play.py:42-177` 使用 `load_env_cfg(task_id, play=True)`，支持 zero/random/trained 三种 policy 模式、checkpoint、motion file、视频和 native/viser viewer。
7. `scripts/list_envs.py:6-45` 只负责导入任务并列出注册的 task ID。

### 2.1.3 任务注册：按机器人拆分，而不是靠一个巨型类

以 Go2 为例：

- `src/tasks/velocity/config/go2/__init__.py:1-8` 导入注册函数、runner、env cfg 工厂和 RL cfg 工厂。
- `src/tasks/velocity/config/go2/__init__.py:10-24` 分别注册 `Unitree-Go2-Rough` 和 `Unitree-Go2-Flat`。
- 每一个注册项明确给出训练 env cfg、play env cfg、rl cfg 和 runner class。
- `src/tasks/velocity/config/go2/env_cfgs.py:22-136` 先由 `make_velocity_env_cfg()` 生成共享配置，再填机器人实体、脚名、接触传感器、地形、viewer、奖励参数和 illegal-contact termination。
- `src/tasks/velocity/config/go2/env_cfgs.py:139-170` 从 rough cfg 派生 flat cfg，移除地形扫描和 terrain curriculum，并修改平地 play 的命令范围。
- `src/tasks/velocity/config/go2/rl_cfg.py:10-45` 定义 actor、critic、PPO 超参数、实验名、保存间隔、rollout steps 和最大迭代次数。

这个结构的关键不是文件数量，而是每个文件只回答一个问题：

| 文件 | 它回答的问题 |
| --- | --- |
| `velocity_env_cfg.py` | 速度任务有哪些通用传感器和 manager term？ |
| `config/go2/env_cfgs.py` | Go2 的 body/geom/site/joint 名称和 rough/flat 差异是什么？ |
| `config/go2/rl_cfg.py` | Go2 任务采用什么 actor/critic/PPO 配置？ |
| `config/go2/__init__.py` | 外部通过什么 task ID 找到这些配置？ |
| `mdp/*.py` | 每个 term 的计算函数与统计信息是什么？ |
| `assets/robots/unitree_go2/*.py` | MJCF、执行器、碰撞和初始状态如何编译成 EntityCfg？ |

### 2.1.4 基础 velocity env cfg 的组成

`src/tasks/velocity/velocity_env_cfg.py:36-431` 是理解 mjlab 任务的最好入口。它按如下顺序组装配置：

1. **传感器**：`RayCastSensorCfg`，用于 terrain scan；`GridPatternCfg` 定义扫描范围和分辨率（`43-51`）。
2. **actor observation**：角速度、投影重力、速度命令、步态 phase、相对关节位置、相对关节速度、上一动作、height scan（`58-91`）。
3. **critic observation**：复制 actor terms，再加 base linear velocity、脚高、脚空中时间、接触状态和接触力（`93-121`）。
4. **observation group**：actor 开启 corruption，critic 关闭 corruption；当前 velocity 基础配置的 `history_length` 是 1（`123-135`）。
5. **metrics**：例如平均动作加速度（`142-146`）。
6. **action**：`JointPositionActionCfg`，实体名为 robot，默认对所有 actuator 匹配（`152-159`）。
7. **command**：`UniformVelocityCommandCfg`，包含采样时间、站立比例、heading 控制和速度范围（`165-180`）。
8. **events**：reset、关节 reset、周期 push、脚摩擦随机化、encoder bias、base COM 偏移（`186-255`）。
9. **rewards**：速度跟踪、姿态、关节加速度、动作变化、足端 gait/clearance/slip/landing 等（`261-354`）。
10. **terminations**：timeout 和 bad orientation（`360-366`）。
11. **curriculum**：terrain level 与 velocity command stages（`372-387`）。
12. **scene/sim/viewer**：terrain generator、sensor、env 数、MuJoCo timestep、iterations、decimation、episode length（`393-431`）。

注意：这套配置把“数据和调度”交给 manager；任务代码不需要再维护 Isaac Gym 风格的 `obs_buf`、`rew_buf`、`contact_forces` 全局 tensor 生命周期。

### 2.1.5 机器人资产适配

以 `src/assets/robots/unitree_go2/go2_constants.py` 为参照：

- `18-33` 定位 XML、收集 mesh bytes、从 `mujoco.MjSpec` 创建规格。
- `40-66` 定义 hip/thigh/calf 三类 position actuator 的正则匹配、刚度、阻尼、effort limit 和 armature。
- `73-82` 定义初始位置、默认关节角和速度。
- `88-112` 定义脚部/全碰撞规则。
- `118-125` 组合 articulation 信息。
- `128-139` 用 `get_go2_robot_cfg()` 返回新 `EntityCfg`，避免共享配置实例被多个任务互相修改。

这个文件的接口可以抽象为：

```text
get_robot_cfg() -> EntityCfg
  ├─ spec_fn() -> MjSpec
  ├─ assets(meshdir) -> dict[str, bytes]
  ├─ init_state
  ├─ collisions
  └─ articulation(actuators, joint limit policy)
```

上面是接口结构说明，不是要复制到目标项目的实现代码。

### 2.1.6 MDP term 的组织方式

`src/tasks/velocity/mdp/` 将功能分成 observations、rewards、terminations、velocity_command、curriculums；这些函数由 cfg 中的 `ObservationTermCfg`、`RewardTermCfg` 等引用。

- observation term 从环境、scene entity、sensor 或 command manager 读取数据。
- reward term 返回每个环境一个标量，并可通过 `env.extras["log"]` 写统计指标。
- termination term 返回每个环境的布尔值。
- command term 管理命令采样和 debug visualization。
- curriculum term 只负责课程更新，不应该偷偷改变与其无关的物理参数。

例如 `src/tasks/velocity/mdp/terminations.py:13-25` 的 `illegal_contact` 只接收 env、sensor name 和阈值，从 contact sensor data 判断非法接触；任务 cfg 再决定用哪个 sensor 和阈值。

### 2.1.7 tracking 任务的特点

`src/tasks/tracking/tracking_env_cfg.py:42-315` 给出了动作模仿与 velocity task 的差异：

- command 变成 `MotionCommandCfg`，配置 motion file、anchor body、body names 和参考动作扰动。
- actor observation 包含 motion anchor、IMU、joint state、上一动作；critic 还包含 body position/orientation（`49-122`）。
- reward 以 root/body 的位置、姿态、线速度、角速度误差为主（`211-253`）。
- termination 以参考动作的 anchor/末端偏差为主（`259-281`）。
- G1 配置在 `src/tasks/tracking/config/g1/env_cfgs.py:16-101` 填入 body names、foot friction、self collision sensor、play override，并可移除 state-estimation 相关 actor observation。

### 2.1.8 unitree_rl_mjlab 的实际优点与实际风险

优点：

- manager term 让观测、奖励、终止、事件和课程可独立阅读。
- 共享 env cfg 工厂 + 每机器人 override 避免重复整份配置。
- 训练和 play 配置显式分离。
- task ID、runner、checkpoint、motion file 的边界清楚。

风险：

- `src/tasks/velocity/velocity_env_cfg.py` 在 `26-33` 同时导入 `mjlab.tasks.velocity.mdp as mdp` 和 `src.tasks.velocity.mdp as mdp`，后者覆盖前者；阅读或迁移时必须确认每个 term 实际来自哪一个模块。
- README 中的 task 列表、默认日志路径和脚本实现需要逐项核对，不能只复制 README。
- `setup.py` 的 `packages=["src"]` 很窄；未来目标工程应明确打包子包和资源的策略，而不是照抄这一行。

## 2.2 HIMLoco：Isaac Gym 的继承式环境 + 全局注册表 + HIM runner

### 2.2.1 目录和职责

```text
HIMLoco/
├── legged_gym/
│   ├── legged_gym/envs/base/
│   │   ├── base_config.py
│   │   ├── base_task.py
│   │   ├── legged_robot.py
│   │   └── legged_robot_config.py
│   ├── legged_gym/envs/<robot>/*_config.py
│   ├── legged_gym/utils/
│   │   ├── task_registry.py
│   │   ├── terrain.py
│   │   ├── helpers.py
│   │   └── logger.py
│   ├── legged_gym/scripts/train.py
│   ├── legged_gym/scripts/play.py
│   └── legged_gym/tests/
├── rsl_rl/
│   ├── modules/him_actor_critic.py
│   ├── modules/him_estimator.py
│   ├── algorithms/him_ppo.py
│   ├── runners/him_on_policy_runner.py
│   └── storage/him_rollout_storage.py
├── resources/
├── sim2sim/
└── deploy/
```

`HIMLoco` 本地副本的 `legged_gym` 实际嵌套层级以 `find` 为准；相关 `uw-himloco` 变体的 README 使用的是另一种根目录布局，因此命令和路径不可直接混用。

### 2.2.2 配置继承和注册

- `legged_gym/legged_gym/envs/base/base_config.py:33-54` 通过反射递归实例化嵌套配置 class。
- `legged_gym/legged_gym/envs/base/legged_robot_config.py:3-221` 定义 env、terrain、commands、init state、control、asset、domain randomization、rewards、normalization、noise、viewer、sim。
- 其中 env 默认是 2048 个环境、45 维单步观测、6 步 history、12 动作，并将 privileged observation 交给 critic；证据为 `legged_robot_config.py:3-13`。
- `legged_gym/legged_gym/envs/base/legged_robot_config.py:223-...` 继续定义 PPO、policy、algorithm 和 runner 配置；`runner_class_name` 为 `HIMOnPolicyRunner`，配置中指定 `HIMActorCritic`/`HIMPPO`。
- 任务注册在 `legged_gym/legged_gym/envs/__init__.py:11-15` 和 `HIMLoco/legged_gym/legged_gym/envs/__init__.py` 相关行：先 import 配置，再调用 `task_registry.register(name, task_class, env_cfg, train_cfg)`。
- `legged_gym/legged_gym/utils/task_registry.py:14-97` 保存三张表：task class、env cfg、train cfg；`make_env` 负责解析参数、seed、仿真参数和构造环境；`make_alg_runner` 负责日志目录、runner 和 resume。

机器人配置通过继承覆盖参数。例如 `legged_gym/legged_gym/envs/my_robot/my_robot_config.py:4-202` 覆盖 terrain、commands、initial pose、control、asset、domain randomization、reward scales、noise、normalization 和 PPO runner；`go2w_config.py:3-179` 进一步把单步 observation 变成 57、action 变成 16，并加入 wheel 相关控制/奖励字段。

### 2.2.3 Isaac Gym 的 step 时序

`legged_gym/legged_gym/envs/base/legged_robot.py` 的核心时序如下：

```text
policy action
  -> clip action
  -> 可选 action delay
  -> decimation 次 _compute_torques + gym.simulate
  -> 刷新 root/body/contact/dof tensor
  -> 采样 command / terrain height / disturbance / push
  -> check_termination
  -> compute_reward
  -> reset_idx
  -> compute_observations
  -> 更新 last_actions / last_dof_vel
  -> 返回 obs, privileged_obs, reward, done, extras, termination obs
```

精确锚点：

- `legged_robot.py:40-65` 是 action clip、delay、decimation 和返回值。
- `legged_robot.py:67-104` 是 physics 后刷新、终止、奖励、reset、观测和历史状态更新。
- `legged_robot.py:106-109` 是 contact termination 与 timeout。
- `legged_robot.py:163-176` 是按 reward scale 累加、正奖励裁剪和 termination reward。
- `legged_robot.py:380-398` 是 PD/velocity/torque 三种控制类型、motor strength 和 torque clipping。
- `legged_robot.py:400-433` 是 reset、push 和外部扰动。
- `legged_robot.py:435-460` 是 terrain/command curriculum。
- `legged_robot.py:577-597` 用 `_reward_` + 名称反射生成 reward function 列表。

这个时序很清晰，但它是 Isaac Gym/PhysX tensor 的实现方式；在 mjlab 中应把其中的“语义”映射到 manager term 和 MuJoCo sensor，而不是复制调用序列。

### 2.2.4 HIMLoco 的观测和 history 合同

`legged_robot.py:178-198` 将单步观测拼接为：

1. 速度 command（三维）；
2. base angular velocity（三维）；
3. projected gravity（三维）；
4. 相对 joint position；
5. joint velocity；
6. 当前 action；
7. privileged-only 的 base linear velocity；
8. disturbance；
9. height measurements（rough terrain 时）。

然后按 `num_one_step_obs` 把最新单步观测放在 history 前端，旧历史向后移动（`197-198`）。

基础四足 HIM 的重要维度是：

| 项目 | 维度 | 证据 |
| --- | ---: | --- |
| 单步 actor obs | 45 | `legged_robot_config.py:3-9` |
| history length | 6 | `num_observations = 45 * 6`；`legged_robot.py:30-32` |
| actor history obs | 270 | 同上 |
| privileged 单步 obs | 45 + 3 + 3 + 187 = 238 | `legged_robot_config.py:8-9` |
| action | 12 | `legged_robot_config.py:10` |
| HIM actor extra | velocity 3 + latent 16 | `rsl_rl/modules/him_actor_critic.py:91-100` |
| HIM actor input | 45 + 3 + 16 = 64 | `him_actor_critic.py:96-100` |

GO2W 的配置在 `go2w_config.py:3-10` 把单步 actor obs 改为 57、history 改为 342、privileged 单步 obs 改为 262、动作改为 16。不要把 45、46、57、342、270、276、64、76 等数字写死到新项目；它们必须由同一个 observation contract 产生。

### 2.2.5 HIM 的训练链路

HIM 不是一个普通的 MLP policy；它至少包含以下边界：

```text
history obs
  -> HIMEstimator.encoder
      -> predicted velocity (3)
      -> normalized latent z (16)
  -> actor(current one-step obs + velocity + z)
  -> action distribution
critic privileged obs
  -> critic value
next critic obs + history
  -> estimator supervised / prototype update
```

证据：

- `rsl_rl/modules/him_actor_critic.py:74-100` 定义 history size、单步维度、动作维度、estimator 和 actor/critic 输入。
- `him_actor_critic.py:166-183` 在 actor 中对 estimator 输出做 `no_grad`，拼接当前单步 observation、velocity 和 latent。
- `rsl_rl/modules/him_estimator.py:11-58` 用 history 展平输入 encoder，用 one-step next obs 做 target，并创建 prototype embedding。
- `him_estimator.py:64-116` 输出 velocity/latent，并用 velocity estimation loss + swap loss 更新 estimator。
- `rsl_rl/algorithms/him_ppo.py:90-113` 记录 action、value、obs、critic obs、next critic obs；`102-113` 处理 timeout bootstrapping、storage 和 actor reset。
- `him_ppo.py:125-190` 在 PPO mini-batch 更新中同时执行 estimator update、surrogate loss、value loss 和 optimizer step。
- `rsl_rl/runners/him_on_policy_runner.py:61-75` 要求 env 具有 `num_one_step_obs`，并用 HIM actor/algorithm 初始化 storage。
- `rsl_rl/storage/him_rollout_storage.py:53-99` 额外保存 `next_privileged_observations`、action distribution 参数和 transition。

对 mjlab 的意义是：如果只使用标准 `MjlabOnPolicyRunner`，不需要立刻迁移 HIM；如果要迁移 HIM，必须先写清楚一个 adapter 的输入/输出，而不能只把 `HIMActorCritic` import 进来。

### 2.2.6 HIMLoco 版本和代码卫生风险

- `HIMLoco` 当前副本没有有效 Git history，无法确认与 `uw-himloco` 的分支对应关系。
- `play.py:29` 使用绝对路径 `/home/kk/HIMLoco/.../model_1200.pt`；这是本机历史路径，不能作为通用使用方式。
- `go2w_config.py` 和一些脚本包含固定动作维度、固定路径或 wheel 特化逻辑；迁移时应改写成显式的 asset/observation/action contract。
- `HIMActorCritic.reset()` 当前是空实现（`him_actor_critic.py:148-152`）；现有 HIM 是无 recurrent hidden state 的 estimator，但未来若换 GRU，reset/mask 必须成为 runner 合同的一部分。
- 现有架构记忆中关于 GO2W 导出修复的结论只覆盖既有 wrapper，不证明新 mjlab 环境的 actor/critic 维度正确。

---


## 2.3 mjlab 与 MuJoCo 文档的关键结论

本轮同时对照了本地 `unitree_rl_mjlab` 和官方文档。官方文档的版本会随 mjlab 发布变化；当前本地 Unitree 项目仍以 `setup.py` 中的 `mjlab==1.2.0`、`mujoco-warp==3.5.0` 为自己的参考组合，不能把 main 分支文档中的新字段直接当成该组合已经支持。

建议固定阅读这些入口：

- [mjlab Architecture Overview](https://mujocolab.github.io/mjlab/main/source/architecture_overview.html)：simulation layer、manager layer、生命周期和 step 顺序。
- [mjlab Entity](https://mujocolab.github.io/mjlab/main/source/entity/index.html)：`EntityCfg`、`spec_fn`、初始状态、执行器和运行时 `EntityData`。
- [mjlab Actuators](https://mujocolab.github.io/mjlab/main/source/actuators.html)：`EntityCfg.articulation`、position/velocity/effort actuator。
- [mjlab Scene](https://mujocolab.github.io/mjlab/main/source/scene.html)：`SceneCfg`、实体前缀、sensor 注册、MJCF composition 和 environment origins。
- [mjlab Observations](https://mujocolab.github.io/mjlab/main/source/observations.html)：term、group、noise、delay 和 history。
- [mjlab Actions](https://mujocolab.github.io/mjlab/main/source/actions.html)：action term、scale/offset/clip、actuator 匹配和 decimation 子步。
- [mjlab Sensors](https://mujocolab.github.io/mjlab/main/source/sensors/index.html)：builtin/contact/raycast sensor 数据和 shape。
- [mjlab Events](https://mujocolab.github.io/mjlab/main/source/events.html)：startup、reset、interval、step 生命周期。
- [mjlab Terminations](https://mujocolab.github.io/mjlab/main/source/terminations.html)：terminal failure 与 timeout/truncation 的区别。
- [mjlab Training with RSL-RL](https://mujocolab.github.io/mjlab/main/source/training/rsl_rl.html)：task registry、wrapper、runner 和训练配置。
- [MuJoCo Python binding](https://mujoco.readthedocs.io/en/latest/python.html)：`MjSpec`、`MjModel`、`MjData`、`mj_step` 和资源打包。
- [MuJoCo Simulation](https://mujoco.readthedocs.io/en/latest/programming/simulation.html)：forward dynamics、sensor data 和模拟循环。
- [MuJoCo XML Reference](https://mujoco.readthedocs.io/en/latest/XMLreference.html)：joint、geom、actuator、sensor 和 asset 的 MJCF 语义。

### 2.3.1 mjlab 的两层边界

官方 mjlab 文档把系统分成 simulation layer 和 manager layer：

```text
simulation layer
  MjSpec -> MjModel/MjData -> MuJoCo Warp parallel worlds
  Entity / Actuator / Sensor / Scene / Terrain

manager layer
  Observation / Action / Command / Event / Reward
  Termination / Curriculum / Metrics / Recorder
  ManagerBasedRlEnvCfg -> ManagerBasedRlEnv -> RSL-RL wrapper/runner
```

这解释了新项目的目录决策：

- `robots/unitree_go2/entity_cfg.py` 负责 simulation layer 的机器人实体；它是 Python wrapper，不是原始资源目录。
- `envs/go2/go2_config.py` 负责把实体、scene、sensor 和 manager terms 组合成一个 RL task。
- `envs/base/base_task.py` 只负责外部 wrapper 或兼容接口，不能复制 `MjData`、reset buffer 和物理循环。
- `scripts/train.py` 负责 task ID、CLI、日志和 runner；它不应该成为第三个 manager layer。

`SceneCfg.entities` 中的 key 会成为实体命名空间。例如实体 key 为 `robot` 时，MJCF 中的 `base_link` 在组合场景里应按 `robot/base_link` 访问；XML 内置 IMU 也会按 `robot/<sensor_name>` 暴露。传感器本身注册在 `SceneCfg.sensors`，不是塞进 `EntityCfg` 的任意字段。

### 2.3.2 mjlab 的控制频率和生命周期

`ManagerBasedRlEnvCfg` 的环境步长是：

```text
physics_dt = cfg.sim.mujoco.timestep
policy_dt  = physics_dt * cfg.decimation
episode_steps = ceil(cfg.episode_length_s / policy_dt)
```

例如 `timestep=0.005`、`decimation=4` 时，policy step 为 `0.020 s`，即 50 Hz。

概念上的顺序是：

```text
build: entity spec_fn -> compose MjSpec -> compile MjModel -> parallel data
initialize: managers -> resolve SceneEntityCfg -> history/delay -> startup events
reset: reset scene -> reset events -> commands -> first observation
step(action): process action -> decimation 次 physics -> termination/reward/metrics
            -> step/interval events -> partial reset -> forward/commands/sense
            -> next observation
```

因此，一个 policy step 不是调用一次 `mujoco.mj_step`。`decimation` 属于环境配置合同；动作的缩放、offset 和 actuator 目标属于 ActionManager/actuator 合同。

奖励还要记录时间缩放语义：当前官方 mjlab 架构说明 RewardManager 会按 step duration 处理加权奖励，但本地任务锁定的 `mjlab==1.2.0` 仍应以实际 API/实现复核。迁移 HIMLoco 时不要一边保留旧 `dt` 缩放、一边再让 manager 自动缩放，避免奖励整体被重复乘以时间步长。

动作处理至少要记录以下顺序：

```text
policy action
  -> 按 action term 切片
  -> scale
  -> offset / default joint pose
  -> clip
  -> 每个 decimation physics substep 写入 actuator target
```

观测的处理顺序则是 `compute -> noise -> clip -> scale -> delay -> history`。`history_length=N` 默认展平为 `[num_envs, N*D]`，并按 term-major 顺序拼接；reset 会清空 history，首个有效 observation 再回填所有历史槽位。这个语义与 HIMLoco 的旧 `obs_buf`/零填充历史不同，必须写进 contract。

### 2.3.3 MuJoCo 原生对象和 mjlab 对象

| MuJoCo 原生对象 | mjlab 访问方式 | 文档应记录 |
| --- | --- | --- |
| `MjSpec` | `EntityCfg.spec_fn()` 返回 | XML、mesh bytes、spec editor 和 asset 名 |
| `MjModel` | `env.sim.model` 或编译结果 | joint/body/geom/actuator ID |
| `MjData` | `env.sim.data`、`Entity.data`、sensor data | `[num_envs, ...]` shape、frame 和单位 |
| XML actuator | `EntityCfg.articulation.actuators` | joint regex、stiffness/damping、effort/armature |
| XML sensor | builtin/contact/raycast sensor | sensor 名、字段、更新频率和 shape |

MuJoCo 原生理解模板：

```python
import mujoco

spec = mujoco.MjSpec.from_file("robot.xml")
model = spec.compile()
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
mujoco.mj_step(model, data, nstep=4)
```

mjlab 通常不在任务类中直接创建这些对象；`spec_fn`、`SceneCfg` 和 `ManagerBasedRlEnv` 负责组合与批量化。

原生 MuJoCo 的 `qpos`/`qvel` 是未批量化的 `[nq]`/`[nv]`；浮动基 free joint 通常贡献 7 个 `qpos` 和 6 个 `qvel`，所以 `nq`/`nv` 既不等于 actuated joint 数，也不等于 policy action 维度。mjlab/MuJoCo Warp 在外层增加并行 world 维后，才通常表现为 `[N, nq]`、`[N, nv]`；迁移时必须分别记录 native shape、batched shape 和 action target shape。

资源注入需要单独锁版本。当前 Unitree 参考代码使用 `spec.assets = dict[str, bytes]`，但 MuJoCo Python 文档已经把这种字典注入方式标为 deprecated，并推荐 `MjVfs`；而 mjlab 的 `EntityCfg.spec_fn` 只返回 `MjSpec`，VFS 的生命周期还涉及后续 scene compile。因此不能只把一行 `spec.assets` 机械替换成局部 `with mujoco.MjVfs()`；应先确认目标 mjlab 版本如何把 VFS 传到解析和编译阶段，再决定资源 wrapper 的实现。

### 2.3.4 shape、history、reset

官方 mjlab 约定实体数据第一维是并行环境数：

```text
EntityData root/joint/sensor: [N, ...]
observation term:              [N, width]
reward term:                   [N]
termination term:              [N] bool
```

`SceneEntityCfg` 的名称正则在 manager 初始化阶段解析为 ID；term 运行时应使用已解析的 `joint_ids/body_ids/site_ids/geom_ids`，不要每步用字符串查找。

mjlab 的 `history_length` 与 HIMLoco 旧 `obs_buf` 不是同一个契约：mjlab 的 ObservationManager 会处理 history reset/backfill；HIMLoco 的历史、partial reset、terminal critic snapshot 和 estimator target 仍需单独定义。仅把 `history_length=6` 写进 cfg，不能自动得到可用 HIM。

### 2.3.5 版本锁定原则

文档中的 API 片段分为 `[事实]`（当前 Unitree 代码）和 `[官方模板]`（当前官方文档）。每次升级 mjlab/MuJoCo 后，至少重新核对 `ManagerBasedRlEnvCfg` 字段、`register_mjlab_task` 签名、`EntityCfg`/actuator 字段、`EntityData` 属性、history/reset 语义、RSL wrapper 构造参数，以及 `MjSpec` asset 注入行为。

## 3. 两种框架的对应关系：迁移时怎么想

| legged_gym / HIMLoco 概念 | mjlab / unitree_rl_mjlab 对应物 | 迁移规则 |
| --- | --- | --- |
| `LeggedRobotCfg.env` | `ManagerBasedRlEnvCfg` 的 `scene`、`decimation`、`episode_length_s` 与 observation groups | 维度和时间参数分散到明确字段，不再由一个 class 保存所有 buffer 数量 |
| `asset.file` URDF | `EntityCfg.spec_fn` + MJCF/XML + `get_assets` | 先将 URDF/MJCF 资源编译成 EntityCfg；不要在环境类里手工加载资源 |
| `control.stiffness/damping` | `BuiltinPositionActuatorCfg` 或 mjlab actuator cfg | actuator 名称正则、effort、armature、默认偏移都必须记录 |
| `default_joint_angles` | `EntityCfg.InitialStateCfg.joint_pos` | 使用 joint name regex，保留单位和零位定义 |
| `commands` class | `CommandTermCfg` / `UniformVelocityCommandCfg` / 自定义 command | 命令生成、重采样和 debug visualization 由 command manager 持有 |
| `compute_observations()` | `ObservationTermCfg` + `ObservationGroupCfg` | 一个大拼接函数拆成命名 term；actor/critic 共享与差异显式记录 |
| `obs_buf` history | `ObservationGroupCfg.history_length`；HIM 另加 history adapter | 标准 PPO 可用 manager history；HIM 需要确认每一帧的顺序、噪声和 reset 行为 |
| `_compute_torques()` | action term + actuator | policy action 与物理 actuator 之间的缩放、offset、clip、decimation 要独立成合同 |
| `_reward_xxx()` | `RewardTermCfg(func=...)` | reward 函数不依赖隐藏的全局 reward scale；权重和 term 分离 |
| `check_termination()` | `TerminationTermCfg` | 每个终止条件命名、返回布尔值；timeout 单独标记 |
| `domain_rand` flags | `EventTermCfg`、`dr.*`、reset/startup/interval mode | 把随机化发生的时机和对象写进事件配置 |
| `terrain.py` | `TerrainEntityCfg` + terrain generator | terrain 生成器、sensor 和 curriculum 分开；不要在 reward 函数里生成地形 |
| `task_registry.register` | `register_mjlab_task` | 保留 task ID 入口，但注册对象变成 env/play/rl/runner 四元组 |
| `OnPolicyRunner` | `MjlabOnPolicyRunner` | 先用标准 runner；只有 HIM adapter 合同稳定后才引入自定义 runner |
| `HIMOnPolicyRunner` | 自定义 `HimOnPolicyRunner` 或 wrapper + runner adapter | 需要 `num_one_step_obs`、history、privileged obs、next critic obs 和 estimator update 合同 |
| `export_policy_as_jit/onnx` | mjlab/RSL 导出或自定义 exporter | 必须由 obs/action contract 推导输入输出，禁止硬编码 45/57/64/76 |

### 3.1 哪些 Isaac Gym API 不能直接搬

下面这些名称如果出现在新的 mjlab 任务代码中，应当视为迁移警报：

- `gymapi.acquire_gym()`、`gymtorch.unwrap_tensor()`、`gym.simulate()`、`gym.refresh_*_tensor()`。
- `self.root_states`、`self.contact_forces`、`self.dof_pos` 等由 PhysX pipeline 管理的全局 buffer。
- `create_actor`、`set_actor_dof_properties`、`set_actor_root_state_tensor_indexed`。
- 通过 `_reward_` + `getattr` 反射建立奖励函数。
- 用固定切片直接解释 actor/critic/HIM obs，而没有由 cfg 记录字段来源。

在 mjlab 中，应优先查当前 `unitree_rl_mjlab` 使用的 `SceneEntityCfg`、`ContactSensorCfg`、`RayCastSensorCfg`、`ObservationTermCfg`、`RewardTermCfg`、`EventTermCfg` 和 `TerminationTermCfg`。

---

## 4. 目标 `legged_mjlab` 的目录：采用 legged_gym 的 env-centric 组织

你指出得对：上一版目录把 mjlab 的 manager 概念直接映射成 `tasks/<task>/mdp`，结果目录看起来像一个小型 Isaac Lab，而不像 HIMLoco/legged_gym。目标项目应当先回答“机器人环境在哪里、注册入口在哪里、训练脚本在哪里”，然后把 mjlab manager 配置作为环境内部实现细节。

推荐的主结构如下。它保留 HIMLoco 的 `envs/base`、`envs/<robot>`、`utils/task_registry.py`、`scripts/train.py/play.py` 组织方式；同时让 `ManagerBasedRlEnvCfg`、`EntityCfg`、MuJoCo `MjSpec`、sensor 和 actuator 成为每个环境的底层实现。

```text
legged_mjlab/
├── pyproject.toml                         # 推荐；唯一根项目安装入口
├── README.md
├── docs/
│   ├── mjlab_training_environment_development_guide_zh.md
│   ├── architecture.md                    # 稳定后从本手册拆出的架构说明
│   └── contracts/                         # obs/action/asset/checkpoint 表
├── legged_mjlab/                          # Python 包根，类似 HIMLoco/legged_gym/legged_gym
│   ├── __init__.py                         # ROOT_DIR、版本和最小公共导出
│   ├── robots/
│   │   └── unitree_go2/
│   │       ├── __init__.py
│   │       └── entity_cfg.py               # EntityCfg/spec/actuator/碰撞
│   ├── envs/
│   │   ├── __init__.py                     # 导入 env 配置和 task registry 注册点
│   │   ├── base/
│   │   │   ├── base_config.py              # 轻量配置基类/复制工具；不放物理循环
│   │   │   ├── base_task.py                # ManagerBasedRlEnv 的薄适配层
│   │   │   ├── legged_robot.py             # 只保存腿式环境共同的访问/适配语义
│   │   │   └── legged_robot_config.py      # 通用 episode/control/obs/reward 参数
│   │   ├── velocity/
│   │   │   ├── __init__.py
│   │   │   ├── velocity_env_cfg.py          # 共享 manager cfg 工厂
│   │   │   └── mdp/                         # 共享 velocity term 与 mjlab MDP 转发
│   │   │       ├── __init__.py
│   │   │       ├── observations.py
│   │   │       ├── rewards.py
│   │   │       ├── terminations.py
│   │   │       └── curriculums.py
│   │   ├── go2/
│   │   │   ├── go2_config.py               # rough/flat 组合 Go2 资产和 manager cfg
│   │   │   ├── rl_cfg.py                   # 标准 RSL-RL PPO 配置
│   │   │   └── go2_terms.py                # 只有 Go2 专属 term 变多时才拆出
│   │   │   └── go2_env.py                  # 仅在需要自定义 env 行为时存在
│   │   ├── him_go2/
│   │   │   ├── him_go2_config.py           # HIM Go2 配置和维度合同
│   │   │   ├── him_go2.py                  # history/estimator 薄适配，不复制物理引擎
│   │   │   └── him_runner.py               # 只有标准 runner 不足时才添加
│   │   └── <future_robot>/
│   │       └── <robot>_config.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── task_registry.py                # mjlab registry façade；不保存第二份表
│   │   ├── helpers.py                      # CLI、seed、日志路径、checkpoint 元数据
│   │   ├── terrain.py                       # 仅保留项目自定义 terrain 辅助
│   │   ├── math.py                          # 项目确有需要时的张量数学
│   │   └── logger.py                        # 日志和评估指标适配
│   ├── scripts/
│   │   ├── train.py                        # 解析参数、建 env、建 runner、learn
│   │   ├── play.py                         # 加载 checkpoint、渲染、评估
│   │   └── list_envs.py                    # 列举 task ID
│   └── resources/                           # 发布 wheel 时随包安装的原始资源
│       └── robots/unitree_go2/
│           └── xmls/
├── resources/robots/unitree_go2/           # 迁移期间保留的原始资源镜像
└── rsl_rl/                                 # 当前副本/参考；不在 baseline 中复制修改
```

这里有一个重要的“目录和 API 分离”原则：

- 目录像 `legged_gym`：机器人环境放在 `envs/<robot>`，公共基类放在 `envs/base`，注册和入口放在 `utils`/`scripts`。
- 内部像 mjlab：`go2_config.py` 不写 `gymapi` 的创建/刷新循环，而是构造 `EntityCfg`、`SceneCfg`、`ObservationGroupCfg`、`ActionTermCfg`、`RewardTermCfg`、`TerminationTermCfg`、`EventTermCfg` 和 `ManagerBasedRlEnvCfg`。
- `base_task.py` 不再复制 HIMLoco 的 `BaseTask.step()`；它只能包装 `ManagerBasedRlEnv` 的生命周期，避免两套 reset、decimation、sensor 更新顺序互相打架。
- `go2_env.py` 不是必须文件。只要任务可以用 cfg + manager term 表达，就不要为了模仿 `LeggedRobot` 而创建空的环境子类。
- `him_go2.py` 只处理历史帧、estimator、privileged critic 和 runner 契约；MuJoCo 的物理状态仍由 mjlab scene/entity 管理。

### 4.1 为什么这个结构比上一版更合适

| 设计问题 | 上一版的问题 | 本版处理 |
| --- | --- | --- |
| 任务入口 | `tasks/velocity/config/go2` 层级过深，阅读者先遇到 manager 再看到机器人 | 先进入 `envs/go2/go2_config.py`，一份配置能追踪资产、场景、观测和训练入口 |
| 基类职责 | 容易重新实现一套 `LeggedRobot` 物理循环 | `base_task.py` 只做 mjlab env 的薄适配，step/reset 由 mjlab 负责 |
| MDP term | 独立 `tasks/<task>/mdp` 让小项目目录碎片化 | Go2 baseline 先集中在 `go2_config.py`；term 很多时再拆成 `envs/go2/go2_terms.py` 或按 observation/reward/event 分文件，不强制预建目录 |
| 机器人差异 | 资产、任务、RL cfg 分散在多个 package | `envs/go2/go2_config.py` 先保持单入口；超过可读范围再拆 `go2_asset.py`/`go2_rewards.py` |
| HIM | 与 velocity task 并列为另一套 manager 树 | 放在 `envs/him_go2/`，作为 baseline env 的算法适配层 |
| 资源 | `resources/` 与 Python 包根没有明确发布策略 | 迁移期保留外部原始资源；稳定后把 XML/mesh 放入 `legged_mjlab/resources`，把 Python wrapper 放入 `legged_mjlab/robots`，并用 `importlib.resources` 或包路径访问 |

### 4.2 目录规则

1. `envs/base/` 只保留公共接口和默认参数，不保留 Go2 的 joint/site/geom 名称。
2. `envs/go2/go2_config.py` 是第一阶段唯一需要频繁阅读的任务文件；它可以包含 manager term 字典，但不包含 MuJoCo 物理循环。
3. 当 term 数量确实超过一个文件的可读范围时，才拆成 `envs/go2/observations.py`、`rewards.py`、`events.py`；拆分按“环境”而不是按“所有任务共享的抽象层”进行。
4. `envs/__init__.py` 或 `utils/task_registry.py` 只能有一个注册路径；每个 task ID 只注册一次。
5. `scripts/` 只负责 CLI、设备、日志、checkpoint、runner 和关闭环境；不能在脚本中实现 reward/observation。
6. HIM 不是 `base` 的默认能力；先完成普通 PPO 的 Go2 flat，再添加 `him_go2`。
7. `rsl_rl/` 如果保留在仓库中，必须在文档中声明它是独立包/参考副本；baseline 不得同时隐式导入两个不同来源的 RSL-RL。

### 4.3 当前目标项目需要先解决的结构阻塞

这是阅读结果，不是本轮实施：

| 阻塞 | 证据 | 后续文档动作 |
| --- | --- | --- |
| 没有有效安装元数据 | 根目录 `setup.py` 当前为 0 字节，且无 `pyproject.toml` | 未来增加唯一 packaging 入口，并声明 mjlab/torch/mujoco-warp 等版本；本轮不写 |
| base task 为空 | `legged_mjlab/envs/base/base_task.py` | 不再沿用旧类，先决定完全采用 mjlab env 还是保留兼容 adapter |
| legged robot 为空 | `legged_mjlab/envs/base/legged_robot.py` | 不复制 HIMLoco 大基类；以 manager cfg 组合任务 |
| `him_go1/` 为空 | 目录没有文件 | 先以 Go2 baseline 验证架构，再为 Go1 设计资产/观测合同 |
| `scripts/` 为空 | `legged_mjlab/scripts/` 没有实现文件 | 从 unitree 的 train/play/list 读取入口语义，未来重新实现而非复制路径 |
| 资源路径与包名冲突 | `resources/robots/unitree_go2/go2_constants.py:7,18-21` | 统一 `SRC_PATH`/包根/资源根，只保留一种路径策略 |
| 旧 RSL-RL 是未跟踪副本 | `rsl_rl/setup.py` 与 `rsl_rl/rsl_rl/*` | 先标注来源和版本；标准 mjlab runner 可用时不要维护两套 PPO |


### 4.4 可直接复制的 packaging 模板（未来文件）

下面的代码块是文档模板，不是本轮写入工程的代码。第一版建议只选一个 `pyproject.toml`；不要同时维护根 `setup.py`、嵌套 `rsl_rl/setup.py` 和另一个隐式 package root。

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "legged-mjlab"
version = "0.1.0"
description = "Legged-gym-style environments built on mjlab"
requires-python = ">=3.11,<3.12"
dependencies = [
  # 这是与当前 unitree_rl_mjlab 对照的示例组合，必须先做版本矩阵。
  "mjlab==1.2.0",
  "mujoco-warp==3.5.0",
  "torch",
  "tyro",
  "prettytable",
]

[project.scripts]
legged-mjlab-list-envs = "legged_mjlab.scripts.list_envs:main"
legged-mjlab-train = "legged_mjlab.scripts.train:main"
legged-mjlab-play = "legged_mjlab.scripts.play:main"

[project.optional-dependencies]
viewer = ["viser"]

[tool.setuptools.packages.find]
where = ["."]
include = ["legged_mjlab*"]

[tool.setuptools.package-data]
legged_mjlab = ["resources/robots/**/*"]
```

这段模板只解决“根项目如何声明包”的形状，不证明上述版本在当前机器可安装。发布前还要生成并提交一份与 Python/CUDA 平台对应的 constraints 文件，至少锁定 `torch`、`tyro`、`mujoco`、`mjlab`、`mujoco-warp` 和构建工具；不能用未锁定的 `torch` 让不同机器自动解析出不同 ABI。`rsl_rl` 若仍是独立 fork，应单独记录来源、版本和导入路径；不要让根项目通过同名包把两个 RSL-RL 混在一起。

### 4.5 可直接复制的 Go2 资源 wrapper

目标：原始 XML/mesh 放在 `legged_mjlab/resources/robots/unitree_go2/`，Python 适配器放在 `legged_mjlab/robots/unitree_go2/entity_cfg.py`。开发期可以暂时从仓库外部 `resources/` 读取，但最终只保留一种路径策略。

```python
# legged_mjlab/robots/unitree_go2/entity_cfg.py
from importlib.resources import as_file, files

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

_RESOURCE_ROOT = files("legged_mjlab").joinpath(
  "resources", "robots", "unitree_go2"
)
_FOOT_REGEX = "^[FR][LR]_foot_collision$"


def get_spec() -> mujoco.MjSpec:
  # 当前 unitree_rl_mjlab 的可复制写法：把 mesh 文件注入 spec.assets。
  # 如果锁定版本要求 MjVfs，应按对应版本 API 改写这一段。
  # Materialize the whole robot resource directory, not only go2.xml: the
  # XML refers to the sibling assets/ directory.
  with as_file(_RESOURCE_ROOT) as resource_root:
    xml_path = resource_root / "xmls" / "go2.xml"
    spec = mujoco.MjSpec.from_file(str(xml_path))
    asset_bytes: dict[str, bytes] = {}
    update_assets(asset_bytes, xml_path.parent / "assets", spec.meshdir)
    spec.assets = asset_bytes
    return spec


GO2_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.32),
  joint_pos={
    ".*thigh_joint": 0.9,
    ".*calf_joint": -1.8,
    ".*R_hip_joint": 0.1,
    ".*L_hip_joint": -0.1,
  },
  joint_vel={".*": 0.0},
)

GO2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*hip_.*",),
      stiffness=20.0, damping=1.0, effort_limit=23.5, armature=0.01,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*thigh_.*",),
      stiffness=20.0, damping=1.0, effort_limit=23.5, armature=0.01,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*calf_.*",),
      stiffness=40.0, damping=2.0, effort_limit=45.0, armature=0.02,
    ),
  ),
  soft_joint_pos_limit_factor=0.9,
)

GO2_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=0,
  condim={_FOOT_REGEX: 3, ".*_collision": 1},
  priority={_FOOT_REGEX: 1},
  friction={_FOOT_REGEX: (0.6,)},
  solimp={_FOOT_REGEX: (0.9, 0.95, 0.023)},
)


def get_go2_robot_cfg() -> EntityCfg:
  # 每次返回新 EntityCfg；不要让 flat/play/HIM 共享会被修改的实例。
  return EntityCfg(
    spec_fn=get_spec,
    init_state=GO2_INIT_STATE,
    articulation=GO2_ARTICULATION,
    collisions=(GO2_COLLISION,),
  )
```

需要逐项替换/确认的内容：XML 相对路径、mesh 名、foot geom 名、joint regex、actuator 顺序、零位、effort limit、collision policy。`".*"` 只表示匹配，不代表硬件关节顺序已经确定。

### 4.6 可直接复制的 Go2 flat manager 配置

先把 flat 任务压缩到一份 `envs/go2/go2_config.py`，确保读者能从一个文件追踪实体、scene、observation、action、command、reward 和 termination。以下示例使用当前 mjlab/Unitree 代码中出现的公开配置类；自定义 reward 函数见下一段。

```python
# legged_mjlab/envs/go2/go2_config.py
import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg

from legged_mjlab.robots.unitree_go2.entity_cfg import get_go2_robot_cfg
from .go2_terms import body_orientation_l2, track_linear_velocity


def make_go2_flat_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  num_envs = 1 if play else 4096
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "command": ObservationTermCfg(
      func=envs_mdp.generated_commands, params={"command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }
  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
  }
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_go2_robot_cfg()},
      num_envs=num_envs,
    ),
    observations={
      "actor": ObservationGroupCfg(
        terms=actor_terms,
        concatenate_terms=True,
        enable_corruption=not play,
        history_length=1,
      ),
      "critic": ObservationGroupCfg(
        terms=critic_terms,
        concatenate_terms=True,
        enable_corruption=False,
        history_length=1,
      ),
    },
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",),
        scale=0.25, use_default_offset=True,
      ),
    },
    commands={
      "twist": UniformVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(3.0, 8.0),
        heading_command=True,
        ranges=UniformVelocityCommandCfg.Ranges(
          lin_vel_x=(-1.0, 2.0),
          lin_vel_y=(-1.0, 1.0),
          ang_vel_z=(-1.0, 1.0),
          heading=(-math.pi, math.pi),
        ),
      ),
    },
    rewards={
      "track_lin_vel": RewardTermCfg(
        func=track_linear_velocity,
        weight=1.0,
        params={"command_name": "twist", "std": 0.5},
      ),
      "upright": RewardTermCfg(
        func=body_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
      ),
      "termination": RewardTermCfg(func=envs_mdp.is_terminated, weight=-200.0),
    },
    terminations={
      "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
      "fallen": TerminationTermCfg(
        func=envs_mdp.bad_orientation, params={"limit_angle": 1.0},
      ),
    },
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.005)),
    decimation=4,
    episode_length_s=20.0,
  )
```


#### 4.6.1 rough 只增加 terrain/sensor/event/curriculum

下面只是解释“如何增量添加 rough 专属项”的短结构模板，不是最终函数继承方向。它保留了上一轮 flat-first 的示意命名；本地 Unitree 的可执行设计是 `rough -> flat`，请以 4.12.7 的 `unitree_go2_rough_env_cfg()` / `unitree_go2_flat_env_cfg()` 为唯一复制版本。rough 不应该重新复制整份 Go2 环境；从共享 velocity factory 得到一份配置，再显式增加 rough 专属内容。名称必须和实际 XML 一致。

```python
from dataclasses import replace

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG


def make_go2_rough_cfg(play: bool = False):
  cfg = make_go2_flat_cfg(play=play)
  cfg.events = dict(cfg.events or {})
  cfg.curriculum = dict(cfg.curriculum or {})
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=replace(ROUGH_TERRAINS_CFG),
    max_init_terrain_level=5,
  )

  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base_link", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
  )
  feet_contact = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=("FR_foot_collision", "FL_foot_collision",
               "RR_foot_collision", "RL_foot_collision"),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (terrain_scan, feet_contact)
  cfg.events["push_robot"] = EventTermCfg(
    func=envs_mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(5.0, 6.0),
    params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
  )
  if play:
    cfg.curriculum = {}
    cfg.observations["actor"].enable_corruption = False
  return cfg
```

这个模板省略了 rough reward 和 height-scan term 的具体权重；它们应在 observation/reward contract 中逐项加入。注意 `ContactSensorCfg` 的 `force` shape 依赖 primary 数量和 `num_slots`，不能把它误当成固定的 `[N, 4, 3]`。

这段 flat cfg 有意不放 terrain scan、contact history、friction randomization 和 curriculum；这些功能只能在 rough 配置中显式加入。`play=True` 也只能做可视化 override，不能改变训练 checkpoint 所需的 actor input shape。

### 4.7 可直接复制的 MDP term

mjlab term 的最小规则是：输入从 `env.scene`、`command_manager` 或明确的 `SceneEntityCfg` 获得；输出保持 batch 第一维；不在 reward 函数中修改共享仿真状态。

```python
# legged_mjlab/envs/go2/go2_terms.py
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg


def track_linear_velocity(env, std: float, command_name: str, asset_cfg=SceneEntityCfg("robot")):
  robot = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found"
  actual = robot.data.root_link_lin_vel_b
  xy_error = torch.sum((command[:, :2] - actual[:, :2]) ** 2, dim=1)
  z_error = actual[:, 2] ** 2
  return torch.exp(-(xy_error + 2.0 * z_error) / std**2)


def body_orientation_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
  robot = env.scene[asset_cfg.name]
  return torch.sum(robot.data.projected_gravity_b[:, :2] ** 2, dim=1)
```

可复制前必须核对当前版本的 `EntityData` 属性名。当前 Unitree 代码使用 `root_link_lin_vel_b`、`projected_gravity_b`；不要把 Isaac Gym 的 `base_lin_vel`、`projected_gravity` 属性名直接套过来。

### 4.8 可直接复制的 task 注册与列举入口

HIMLoco 的注册思想保留，但实际注册委托 mjlab registry，不再创建第二套全局表。

本节仍是 API 边界的短模板，使用 `make_go2_flat_cfg` 等简化名称；完整的两个 task、PPO cfg 和 rough/flat 真实继承关系以 4.12.7--4.12.8 为准。

```python
# legged_mjlab/envs/go2/__init__.py
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .go2_config import make_go2_flat_cfg, make_go2_ppo_cfg

register_mjlab_task(
  task_id="Legged-Mjlab-Go2-Flat",
  env_cfg=make_go2_flat_cfg(),
  play_env_cfg=make_go2_flat_cfg(play=True),
  rl_cfg=make_go2_ppo_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)
```

如果希望保留 HIMLoco/legged_gym 的 `utils.task_registry` 使用习惯，它只能做转发门面，不能另建一个会与 mjlab registry 漂移的全局字典：

```python
# legged_mjlab/utils/task_registry.py
from mjlab.tasks.registry import (
  list_tasks as _list_tasks,
  load_env_cfg as _load_env_cfg,
  load_rl_cfg as _load_rl_cfg,
  load_runner_cls as _load_runner_cls,
)


def list_tasks():
  return _list_tasks()


def load_env_cfg(task_id: str, play: bool = False):
  return _load_env_cfg(task_id, play=play)


def load_rl_cfg(task_id: str):
  return _load_rl_cfg(task_id)


def load_runner_cls(task_id: str):
  return _load_runner_cls(task_id)
```

```python
# legged_mjlab/envs/go2/go2_config.py
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def make_go2_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu"),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu"),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      gamma=0.99,
      lam=0.95,
    ),
    experiment_name="go2_velocity",
    num_steps_per_env=24,
    max_iterations=10001,
    save_interval=100,
  )
```

这组字段来自当前 `unitree_rl_mjlab/src/tasks/velocity/config/go2/rl_cfg.py` 的结构；具体字段仍必须按锁定的 mjlab 版本核对。

```python
# legged_mjlab/envs/__init__.py
from . import go2  # noqa: F401  # 导入即完成唯一注册
```

```python
# legged_mjlab/scripts/list_envs.py
import legged_mjlab.envs  # noqa: F401
from legged_mjlab.utils.task_registry import list_tasks

if __name__ == "__main__":
  for task_id in list_tasks():
    print(task_id)
```

注册失败必须区分：模块没有导入、task ID 重复、配置工厂抛错、runner cfg 缺失。不要用 `try/except Exception: pass` 把注册失败隐藏成“任务列表为空”。

### 4.9 可直接复制的标准 train/play 薄入口

第一版入口只做下面这条链：`task_id -> load cfg -> ManagerBasedRlEnv -> RslRlVecEnvWrapper -> runner`。这是当前 Unitree 脚本的真实边界；不要把 manager term 写进脚本。

```python
# legged_mjlab/scripts/train.py（最小结构模板）
from dataclasses import asdict
from pathlib import Path

import legged_mjlab.envs  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from legged_mjlab.utils.task_registry import (
  load_env_cfg, load_rl_cfg, load_runner_cls,
)


def train(task_id: str, log_dir: str = "logs") -> None:
  env_cfg = load_env_cfg(task_id)
  agent_cfg = load_rl_cfg(task_id)
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), str(Path(log_dir) / task_id), "cuda:0")
  runner.learn(
    num_learning_iterations=agent_cfg.max_iterations,
    init_at_random_ep_len=True,
  )
  env.close()
```

```python
# legged_mjlab/scripts/play.py（最小结构模板）
from dataclasses import asdict

import legged_mjlab.envs  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from legged_mjlab.utils.task_registry import (
  load_env_cfg, load_rl_cfg, load_runner_cls,
)


def play(task_id: str, checkpoint: str):
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device="cuda:0")
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True)
  policy = runner.get_inference_policy(device="cuda:0")
  # 后续交给 viewer；不要在这里重写 env.step 生命周期。
  return policy
```

真实项目还应加入 viewer、video、device、checkpoint metadata 和 `env.close()`。示例的重点是边界，不是宣称当前骨架可以运行。

### 4.10 可直接复制的 HIM history adapter

mjlab 的普通 observation history 不自动满足 HIMLoco 的 `HIMPPO` 接口。建议先写一个独立 adapter，定义 `D/H/C/A`，再决定是否需要自定义 runner。

```python
# legged_mjlab/envs/him_go2/him_go2.py
import torch


class HistoryAdapter:
  def __init__(self, num_envs: int, one_step_dim: int, history_length: int, device: str):
    self.D = one_step_dim
    self.H = history_length
    self.buffer = torch.zeros(num_envs, self.H, self.D, device=device)

  def reset(self, obs: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
    assert obs.ndim == 2 and obs.shape[-1] == self.D
    if env_ids is None:
      self.buffer[:] = obs.unsqueeze(1)
    else:
      self.buffer[env_ids] = obs[env_ids].unsqueeze(1)

  def append(self, obs: torch.Tensor) -> torch.Tensor:
    assert obs.shape[-1] == self.D
    self.buffer[:, 1:] = self.buffer[:, :-1].clone()
    self.buffer[:, 0] = obs
    return self.buffer.flatten(start_dim=1)

  @property
  def actor_obs_dim(self) -> int:
    return self.H * self.D
```

这里选择“reset 后用当前帧回填”的语义只是设计示例；若要复现 HIMLoco 旧行为，必须改成零填充并将差异写进 contract。无论选哪一种，都要用 `env_ids` 做 partial reset，不能每次 reset 整个 batch。

HIM 的 target slice 不能隐式猜测。若 `next_critic_obs` 按当前参考的布局保存，至少显式检查：

```python
def validate_him_shapes(obs_history, next_critic_obs, D, H, C):
  assert obs_history.ndim == 2
  assert obs_history.shape[1] == H * D
  assert next_critic_obs.ndim == 2
  assert next_critic_obs.shape[1] == C
  assert C >= D + 3
  target_velocity = next_critic_obs[:, D:D + 3]
  target_input = next_critic_obs[:, 3:D + 3]
  return target_velocity, target_input
```

当前 HIMLoco GO2W 参考值是 `D=57, H=6, C=262, A=16, L=16`，对应 history `[N,342]`、actor input `[N,76]`、critic input `[N,262]`；这些数字只能作为参考实验合同，不能直接写入新 Go2 任务。

若要对照当前仓库内 vendored `rsl_rl` 的真实构造方式，可以先用下面的最小片段核对参数形状。它是 HIMLoco 兼容性参考，不是推荐第一天接入的 target runner：

```python
# 仅用于核对 legged_mjlab/rsl_rl 的旧 HIM 接口
from rsl_rl.algorithms import HIMPPO
from rsl_rl.modules import HIMActorCritic

D = 57
H = 6
C = 262
A = 16

actor_critic = HIMActorCritic(
  num_actor_obs=D * H,
  num_critic_obs=C,
  num_one_step_obs=D,
  num_actions=A,
  actor_hidden_dims=[512, 256, 128],
  critic_hidden_dims=[512, 256, 128],
  activation="elu",
  init_noise_std=1.0,
)

algorithm = HIMPPO(
  actor_critic=actor_critic,
  num_learning_epochs=5,
  num_mini_batches=4,
  learning_rate=1.0e-3,
  gamma=0.99,
  lam=0.95,
  device="cuda:0",
)

algorithm.init_storage(
  num_envs=4096,
  num_transitions_per_env=24,
  actor_obs_shape=[D * H],
  critic_obs_shape=[C],
  action_shape=[A],
)
```

当前 `HIMActorCritic` 的 actor 输入层仍固定为 `D + 3 + 16`；因此 `latent_dim=16` 不是上述构造函数的显式参数。新项目若要改变 latent 维度，必须同时改 actor、estimator、checkpoint metadata 和 contract，不能只改环境中的一个常量。

### 4.11 复制这些片段前的逐项检查

1. 先把 `mjlab`、`mujoco-warp`、Torch、Python 版本写入矩阵，再选择对应版本的 API 文档。
2. 先复制 `entity_cfg.py`，确认 XML/mesh/MjSpec/actuator 能构造；不要先复制 HIM runner。
3. 再复制 flat `ManagerBasedRlEnvCfg`，逐项记录 actor/critic term 的名称、顺序、shape、scale、noise 和 frame。
4. 再复制 `register_mjlab_task` 和标准 train/play 薄入口，保持 task ID 唯一。
5. flat 通过 reset/one-step/checkpoint smoke 后，才加入 rough sensor、terrain、events 和 curriculum。
6. rough 通过后，才加入 history adapter；最后才接 HIM estimator、next critic obs 和 HIMPPO storage。
7. 每个模板都必须标注 `[待验证]`，直到实际 import、MJCF compile、reset、step、checkpoint 和评估矩阵完成。

### 4.12 完整文件级复制示例：Go2 velocity baseline

前面的 4.4--4.10 是用于解释边界的短模板；如果只复制其中一个片段，确实不能得到一个完整的环境。本节给出一套按文件落盘的 baseline。它采用 `legged_gym/HIMLoco` 的 env-centric 入口，但把物理、scene、sensor、action、reward、termination 和 event 全部交给 mjlab manager。

这套示例以当前本地参考工程的版本形状为基线：`mjlab==1.2.0`、`mujoco-warp==3.5.0`。本地工作树尚未安装、导入或编译这套未来文件，因此每段代码仍带 `[待验证]` 标记；“可直接复制”表示文件边界、调用顺序和数据合同完整，不表示可以跳过版本/API smoke test。官方 `main` 文档如果与本地锁定版本冲突，以锁定版本的源码和 `inspect.signature()` 为准。

#### 4.12.1 最终文件关系

```text
legged_mjlab/
└── legged_mjlab/
    ├── __init__.py
    ├── robots/unitree_go2/{__init__.py,entity_cfg.py}
    ├── resources/robots/unitree_go2/xmls/
    │   ├── go2.xml
    │   └── assets/*
    ├── envs/
    │   ├── __init__.py
    │   ├── velocity/
    │   │   ├── __init__.py
    │   │   ├── velocity_env_cfg.py
    │   │   └── mdp/
    │   │       ├── __init__.py
    │   │       ├── observations.py
    │   │       ├── rewards.py
    │   │       ├── terminations.py
    │   │       └── curriculums.py
    │   ├── go2/
    │   │   ├── __init__.py
    │   │   ├── go2_config.py
    │   │   └── rl_cfg.py
    │   └── him_go2/
    │       └── history_adapter.py
    ├── utils/{task_registry.py,checkpoint_contract.py}
    └── scripts/{__init__.py,train.py,play.py,list_envs.py}
```

依赖方向只有一条：

```text
robots/unitree_go2
        ↓
envs/velocity/mdp  →  envs/velocity/velocity_env_cfg.py
                                      ↓
                         envs/go2/go2_config.py
                                      ↓
                          task registry / scripts
                                      ↓
                         standard PPO → optional HIM
```

`velocity_env_cfg.py` 是共享 manager cfg 工厂，不能引用 Go2 的 `base_link`、foot geom 或 joint 名称；这些名称只在 `go2_config.py` 注入。这样第二个机器人可以复用 velocity 环境，而不会复制一份 `LeggedRobot` 大基类。

资源落盘是这套文件关系的一部分：在执行 `import legged_mjlab` 前，必须把当前仓库根 `resources/robots/unitree_go2/` 的 XML 和 `xmls/assets/` 原样复制到包内 `legged_mjlab/resources/robots/unitree_go2/`。只复制 Python wrapper 而不复制包内资源，`files("legged_mjlab")` 会找不到 `go2.xml`；开发期的仓库根资源只能作为镜像，不能作为 wheel 的隐式运行时依赖。

#### 4.12.2 `mdp/__init__.py`：只做转发和显式覆盖

```python
# legged_mjlab/envs/velocity/mdp/__init__.py
"""Velocity MDP terms used by the legged_mjlab environments."""

# These two imports expose the version-pinned mjlab terms.  If the locked
# mjlab release does not export the second module, remove it and import the
# exact equivalent from that release's public task API.
from mjlab.envs.mdp import *  # noqa: F401,F403
from mjlab.tasks.velocity.mdp import *  # noqa: F401,F403

from .observations import foot_air_time, foot_contact, foot_contact_forces, foot_height, phase
from .curriculums import terrain_levels_vel
from .rewards import body_orientation_l2, track_angular_velocity, track_linear_velocity
from .terminations import illegal_contact

__all__ = [
  "body_orientation_l2",
  "foot_air_time",
  "foot_contact",
  "foot_contact_forces",
  "foot_height",
  "illegal_contact",
  "phase",
  "terrain_levels_vel",
  "track_angular_velocity",
  "track_linear_velocity",
]
```

这里的 `*` 不是为了隐藏依赖，而是复用 mjlab 对应版本已经提供的通用 term；Go2 自己实现的函数在后面的显式 import 中覆盖或补充。不要在此文件中再维护一套与 `mjlab.envs.mdp` 同名的全局注册表。

#### 4.12.3 `observations.py`：明确每个 term 的 batch shape

```python
# legged_mjlab/envs/velocity/mdp/observations.py
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def foot_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """World-frame z position of selected foot sites; returns [B, num_sites]."""
  asset = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Current air time per primary contact; returns [B, num_feet]."""
  sensor: ContactSensor = env.scene[sensor_name]
  return sensor.data.current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Binary contact flag; returns [B, num_feet]."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found > 0).to(dtype=torch.float32)


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Stable force feature; returns [B, num_feet * 3]."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  force = sensor.data.force.flatten(start_dim=1)
  return torch.sign(force) * torch.log1p(torch.abs(force))


def phase(
  env: ManagerBasedRlEnv,
  period: float,
  command_name: str,
) -> torch.Tensor:
  """Two-channel gait phase; zeroes the phase while standing."""
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command {command_name!r} not found"
  time = env.episode_length_buf.to(torch.float32) * env.step_dt
  angle = 2.0 * torch.pi * time / period
  result = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
  moving = torch.linalg.vector_norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  return result * (moving > 0.1).unsqueeze(-1)
```

关键点是不要从 `asset.data.site_pos_w` 推断固定四脚数量，也不要把 MuJoCo native `site_pos` 当成带环境 batch 的数组。mjlab entity/sensor data 的第一维是环境 batch；`asset_cfg.site_ids` 由名称解析后得到整数索引。

#### 4.12.4 `rewards.py`：奖励只读状态，dt 缩放交给唯一一层

```python
# legged_mjlab/envs/velocity/mdp/rewards.py
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command {command_name!r} not found"
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  return torch.exp(-(xy_error + 2.0 * z_error) / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command {command_name!r} not found"
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  return torch.exp(-(z_error + 0.05 * xy_error) / std**2)


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Squared projected gravity in the base xy plane; returns [B]."""
  asset = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
```

本例把 `RewardTermCfg.weight` 作为唯一的 manager 权重，把 reward 原始函数写成每个环境一个标量 `[B]`。当前 mjlab 文档默认会按 policy step duration 缩放 reward；如果为了复现 HIMLoco 而在环境里先乘过 `dt`，必须关闭/调整其中一层，不能两次缩放。

#### 4.12.5 `terminations.py`：contact history 的 shape 不是常量

```python
# legged_mjlab/envs/velocity/mdp/terminations.py
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Return [B]; history mode checks every stored substep."""
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # [B, num_primary, history_length, 3]
    force_mag = torch.linalg.vector_norm(data.force_history, dim=-1)
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)
  assert data.found is not None
  return torch.any(data.found, dim=-1)
```

`reduce="netforce"`、`num_slots=1` 和 `history_length=4` 都会影响 shape。只有完成 entity compile 后，才可以从 sensor data 打印实际的 `force.shape`；不要把 `[B,4,3]` 写死在网络输入合同中。

#### 4.12.5.1 `curriculums.py`：把 terrain level 作为独立 term

```python
# legged_mjlab/envs/velocity/mdp/curriculums.py
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Move environments between terrain rows from travelled distance."""
  asset: Entity = env.scene[asset_cfg.name]
  terrain = env.scene.terrain
  assert terrain is not None and terrain.cfg.terrain_generator is not None
  command = env.command_manager.get_command(command_name)
  assert command is not None

  distance = torch.linalg.vector_norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )
  move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
  move_down = distance < torch.linalg.vector_norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  move_down &= ~move_up
  terrain.update_env_origins(env_ids, move_up, move_down)
  return torch.mean(terrain.terrain_levels.to(torch.float32))
```

不要把这个 curriculum 函数写进 `go2_config.py`；它只依赖 terrain、command 和 robot entity，可以复用于其他 velocity robot。`flat` 配置必须显式删除 `terrain_levels`，因为它把 `terrain_generator` 设成了 `None`。

#### 4.12.6 `velocity_env_cfg.py`：共享 manager cfg 工厂

```python
# legged_mjlab/envs/velocity/velocity_env_cfg.py
from __future__ import annotations

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig

from . import mdp


def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
  """Build a robot-neutral rough-terrain velocity cfg.

  Robot-specific entity, body, site and geom names are intentionally empty
  here.  Go2 fills them in after this factory returns.
  """
  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    debug_vis=True,
  )

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
    ),
    "command": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "phase": ObservationTermCfg(
      func=mdp.phase,
      params={"period": 0.6, "command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1),
      scale=1.0 / terrain_scan.max_distance,
    ),
  }
  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
    ),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      scale=1.0 / terrain_scan.max_distance,
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=())},
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(ROUGH_TERRAINS_CFG),
        max_init_terrain_level=5,
      ),
      sensors=(terrain_scan,),
      num_envs=1,
      extent=2.0,
    ),
    observations={
      "actor": ObservationGroupCfg(
        terms=actor_terms,
        concatenate_terms=True,
        enable_corruption=True,
        history_length=1,
      ),
      "critic": ObservationGroupCfg(
        terms=critic_terms,
        concatenate_terms=True,
        enable_corruption=False,
        history_length=1,
      ),
    },
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=0.25,
        use_default_offset=True,
      ),
    },
    commands={
      "twist": UniformVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.05,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
          lin_vel_x=(-1.0, 2.0),
          lin_vel_y=(-1.0, 1.0),
          ang_vel_z=(-1.0, 1.0),
          heading=(-math.pi, math.pi),
        ),
      ),
    },
    events={
      "reset_base": EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
          "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0), "yaw": (-3.14, 3.14)},
          "velocity_range": {},
        },
      ),
      "reset_robot_joints": EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
          "position_range": (0.0, 0.0),
          "velocity_range": (0.0, 0.0),
          "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        },
      ),
      "push_robot": EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 6.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.4, 0.4), "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)}},
      ),
      "foot_friction": EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
          "asset_cfg": SceneEntityCfg("robot", geom_names=()),
          "operation": "abs",
          "ranges": (0.3, 1.6),
          "shared_random": True,
        },
      ),
      "base_com": EventTermCfg(
        mode="startup",
        func=dr.body_com_offset,
        params={
          "asset_cfg": SceneEntityCfg("robot", body_names=()),
          "operation": "add",
          "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
        },
      ),
    },
    rewards={
      "track_linear_velocity": RewardTermCfg(
        func=mdp.track_linear_velocity,
        weight=1.0,
        params={"command_name": "twist", "std": math.sqrt(0.25)},
      ),
      "track_angular_velocity": RewardTermCfg(
        func=mdp.track_angular_velocity,
        weight=1.0,
        params={"command_name": "twist", "std": math.sqrt(0.5)},
      ),
      "body_orientation_l2": RewardTermCfg(
        func=mdp.body_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=())},
      ),
      "is_terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-200.0),
      "joint_acc_l2": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-2.5e-7),
      "joint_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-10.0),
      "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.05),
    },
    terminations={
      "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
      "fell_over": TerminationTermCfg(func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}),
    },
    curriculum={
      "terrain_levels": CurriculumTermCfg(func=mdp.terrain_levels_vel, params={"command_name": "twist"}),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
```

`JointPositionActionCfg` 和 `CurriculumTermCfg` 在这里都使用显式 import；不要依赖 `mjlab.envs.mdp` 的星号导出。这正是必须按锁定版本验证、不能凭 `main` 文档猜测的 API 边界。

#### 4.12.7 `go2_config.py`：真实的 rough -> flat 继承方向

本地 `unitree_rl_mjlab` 的事实是 `unitree_go2_rough_env_cfg()` 先调用共享 velocity factory，`unitree_go2_flat_env_cfg()` 再从 rough 配置切换到 plane 并移除 terrain scan。下面保持这个方向；不要使用“flat 是父类、rough 再加 terrain”的反向版本。

```python
# legged_mjlab/envs/go2/go2_config.py
from __future__ import annotations

from mjlab.managers import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from legged_mjlab.robots.unitree_go2.entity_cfg import get_go2_robot_cfg
from legged_mjlab.envs.velocity import mdp
from legged_mjlab.envs.velocity.velocity_env_cfg import make_velocity_env_cfg


def unitree_go2_rough_env_cfg(play: bool = False):
  cfg = make_velocity_env_cfg()
  cfg.scene.num_envs = 1 if play else 4096
  cfg.scene.entities = {"robot": get_go2_robot_cfg()}
  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0
  # Rough terrain has more contacts and raycast/contact matches than flat.
  # These values are a starting contract and still require overflow tests.
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  # The generic ray sensor has no body name.  Resolve it only after the Go2
  # entity is selected.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  foot_names = ("FR", "FL", "RR", "RL")
  foot_sites = foot_names
  foot_geoms = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=foot_geoms, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_collision\d*$",
      exclude=foot_geoms,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground, nonfoot_ground)

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_generator is not None
  cfg.scene.terrain.terrain_generator.curriculum = True

  # Resolve all per-robot selectors in one place.
  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = foot_sites
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geoms
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground.name, "force_threshold": 10.0},
  )

  # The rough configuration is also the common source for play mode.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events.pop("foot_friction", None)
    cfg.events.pop("base_com", None)
    cfg.curriculum = {}
    cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.scene.terrain.terrain_generator.num_cols = 5
    cfg.scene.terrain.terrain_generator.num_rows = 5
    cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go2_flat_env_cfg(play: bool = False):
  """Flat is a specialization of rough so sensor/reward contracts stay aligned."""
  cfg = unitree_go2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )
  cfg.observations["actor"].terms.pop("height_scan")
  cfg.observations["critic"].terms.pop("height_scan")
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.ranges.lin_vel_x = (-0.5, 1.0)
    twist.ranges.lin_vel_y = (-0.5, 0.5)
    twist.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
```

这段代码的职责边界很窄：`rough` 只负责 Go2 名称、terrain/contact sensor、terrain curriculum 和 play override；`flat` 只负责继承 rough 后删除 terrain scan、切换 plane 和删除 terrain curriculum。两者共享 actor 的核心项，避免 checkpoint 因“flat/rough 各写一份 actor term”而悄悄产生维度漂移；但 flat 删除了 `height_scan`，所以最终 `D_actor_flat` 与 `D_actor_rough` 可能不同，必须分别记录并拒绝跨变体 checkpoint 加载。

#### 4.12.8 `rl_cfg.py` 与注册：一个 task ID 一条入口

```python
# legged_mjlab/envs/go2/rl_cfg.py
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def unitree_go2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="go2_velocity",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )
```

```python
# legged_mjlab/envs/go2/__init__.py
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .go2_config import unitree_go2_flat_env_cfg, unitree_go2_rough_env_cfg
from .rl_cfg import unitree_go2_ppo_runner_cfg


register_mjlab_task(
  task_id="Unitree-Go2-Rough",
  env_cfg=unitree_go2_rough_env_cfg(),
  play_env_cfg=unitree_go2_rough_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)
register_mjlab_task(
  task_id="Unitree-Go2-Flat",
  env_cfg=unitree_go2_flat_env_cfg(),
  play_env_cfg=unitree_go2_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)
```

```python
# legged_mjlab/envs/__init__.py
from . import go2  # noqa: F401  # import triggers the two registrations
```

如果项目确实需要 HIMLoco 风格的 `utils.task_registry`，只做 mjlab registry 的 façade：

```python
# legged_mjlab/utils/task_registry.py
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

__all__ = ["list_tasks", "load_env_cfg", "load_rl_cfg", "load_runner_cls"]
```

不同 mjlab 版本的 registry 查询函数名称可能不同；注册本身应以 `register_mjlab_task` 的版本源码为准。不能因为 façade 的 import 失败，就在脚本中偷偷创建一个第二张 task 表。

#### 4.12.9 三个入口文件：脚本只做 orchestration

```python
# legged_mjlab/scripts/list_envs.py
import legged_mjlab.envs  # noqa: F401

from legged_mjlab.utils.task_registry import list_tasks


def main() -> None:
  for task_id in list_tasks():
    print(task_id)


if __name__ == "__main__":
  main()
```

```python
# legged_mjlab/scripts/train.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import tyro

import legged_mjlab.envs  # noqa: F401
from legged_mjlab.utils.task_registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper


@dataclass
class TrainArgs:
  task_id: str = "Unitree-Go2-Flat"
  device: str = "cuda:0"
  log_root: str = "logs"
  num_envs: int | None = None


def main(args: TrainArgs) -> None:
  env_cfg = load_env_cfg(args.task_id, play=False)
  if args.num_envs is not None:
    env_cfg.scene.num_envs = args.num_envs
  agent_cfg = load_rl_cfg(args.task_id)
  runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  vec_env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  run_dir = Path(args.log_root) / args.task_id / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  run_dir.mkdir(parents=True, exist_ok=False)
  # Persist env/agent/contract metadata here before starting learn.  The
  # exact serializer is version-specific; never overwrite an existing run.
  runner = runner_cls(vec_env, asdict(agent_cfg), str(run_dir), args.device)
  try:
    runner.learn(
      num_learning_iterations=agent_cfg.max_iterations,
      init_at_random_ep_len=True,
    )
  finally:
    vec_env.close()


if __name__ == "__main__":
  main(tyro.cli(TrainArgs))
```

```python
# legged_mjlab/scripts/play.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import tyro

import legged_mjlab.envs  # noqa: F401
from legged_mjlab.utils.task_registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass
class PlayArgs:
  task_id: str = "Unitree-Go2-Flat"
  checkpoint: str = ""
  device: str = "cuda:0"
  viewer: Literal["native", "viser"] = "native"


def main(args: PlayArgs) -> None:
  if not args.checkpoint:
    raise ValueError("--checkpoint is required")
  env_cfg = load_env_cfg(args.task_id, play=True)
  agent_cfg = load_rl_cfg(args.task_id)
  runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  vec_env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  runner = runner_cls(vec_env, asdict(agent_cfg), device=args.device)
  try:
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True)
    policy = runner.get_inference_policy(device=args.device)
    if args.viewer == "native":
      NativeMujocoViewer(vec_env, policy).run()
    else:
      ViserPlayViewer(vec_env, policy).run()
  finally:
    vec_env.close()


if __name__ == "__main__":
  main(tyro.cli(PlayArgs))
```

这里的 `play.py` 在 viewer 返回后才执行 `vec_env.close()`；不要把 `close()` 放在 viewer 调用之前。`NativeMujocoViewer`/`ViserPlayViewer` 的构造签名和 reset/step 返回值仍需按锁定版本做 smoke test；viewer 只负责可视化，不应被塞进 reward/term，也不能替代动作限幅、NaN 防护和硬件急停合同。

#### 4.12.10 HIM 适配文件：先固定合同，再接旧算法

普通 mjlab observation history 解决的是“manager 如何保存历史”，而旧 HIMLoco `HIMActorCritic` 还要求 `num_one_step_obs`、固定的 latent 维度、`next_critic_obs` 和旧 runner 的额外返回值。因此 HIM 只能是 `envs/him_go2/history_adapter.py` 的第二阶段适配，不应在 `go2_config.py` 中直接硬编码。

```python
# legged_mjlab/envs/him_go2/history_adapter.py
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HimContract:
  one_step_dim: int                 # D
  history_length: int               # H
  critic_dim: int                   # C
  action_dim: int                   # A
  latent_dim: int = 16              # legacy HIMActorCritic fixed value

  @property
  def actor_history_dim(self) -> int:
    return self.one_step_dim * self.history_length

  def validate(self) -> None:
    if min(self.one_step_dim, self.history_length, self.critic_dim, self.action_dim) <= 0:
      raise ValueError(f"Invalid HIM contract: {self}")
    if self.latent_dim != 16:
      raise ValueError("The vendored HIMActorCritic currently hard-codes latent_dim=16")


class HistoryAdapter:
  """Keep [B, H, D] history and expose term-major [B, H*D] tensors.

  Frame 0 is the newest frame, matching the legacy HIM estimator.  This
  implementation zero-fills reset environments before writing the first
  current frame; preserving old frames is a different, explicit contract.
  """

  def __init__(self, num_envs: int, contract: HimContract, device: str):
    contract.validate()
    self.contract = contract
    self.buffer = torch.zeros(
      num_envs,
      contract.history_length,
      contract.one_step_dim,
      device=device,
    )

  def reset(self, obs: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
    # `obs` is always the full [B, D] manager observation.  Making this
    # explicit prevents a partial batch from being silently indexed twice.
    self._check_one_step(obs)
    if obs.shape[0] != self.buffer.shape[0]:
      raise ValueError("reset() expects the full [B, D] observation batch")
    if env_ids is None:
      self.buffer.zero_()
      self.buffer[:, 0] = obs
    else:
      self.buffer[env_ids] = 0.0
      self.buffer[env_ids, 0] = obs[env_ids]

  def append(self, obs: torch.Tensor) -> torch.Tensor:
    self._check_one_step(obs)
    if self.contract.history_length > 1:
      self.buffer[:, 1:] = self.buffer[:, :-1].clone()
    self.buffer[:, 0] = obs
    return self.history

  @property
  def history(self) -> torch.Tensor:
    return self.buffer.flatten(start_dim=1)

  def snapshot_terminal_critic(
    self,
    next_critic_obs: torch.Tensor,
    env_ids: torch.Tensor,
  ) -> torch.Tensor:
    """Clone terminal next-critic rows before manager reset mutates them."""
    if next_critic_obs.ndim != 2:
      raise ValueError("next_critic_obs must be a [B, C] tensor")
    if next_critic_obs.shape[0] != self.buffer.shape[0]:
      raise ValueError("next_critic_obs and history must have the same batch size")
    return next_critic_obs[env_ids].clone()

  def _check_one_step(self, obs: torch.Tensor) -> None:
    if obs.ndim != 2 or obs.shape[-1] != self.contract.one_step_dim:
      raise ValueError(
        f"Expected [B, {self.contract.one_step_dim}], got {tuple(obs.shape)}"
      )


def him_target_slices(
  next_critic_obs: torch.Tensor,
  contract: HimContract,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return target velocity and target one-step input for legacy HIMEstimator.

  Required critic layout:
      [ actor_one_step(D), base_lin_vel(3), privileged_tail ]
  """
  contract.validate()
  if next_critic_obs.ndim != 2 or next_critic_obs.shape[-1] != contract.critic_dim:
    raise ValueError(f"Expected critic shape [B, {contract.critic_dim}]")
  if contract.critic_dim < contract.one_step_dim + 3:
    raise ValueError("critic_dim must contain D + 3 target fields")
  target_velocity = next_critic_obs[:, contract.one_step_dim:contract.one_step_dim + 3]
  target_input = next_critic_obs[:, 3:contract.one_step_dim + 3]
  return target_velocity, target_input


def validate_him_batch(
  obs_history: torch.Tensor,
  critic_obs: torch.Tensor,
  actions: torch.Tensor,
  contract: HimContract,
) -> None:
  contract.validate()
  expected = {
    "obs_history": (contract.actor_history_dim, obs_history.shape[-1]),
    "critic_obs": (contract.critic_dim, critic_obs.shape[-1]),
    "actions": (contract.action_dim, actions.shape[-1]),
  }
  for name, (want, got) in expected.items():
    if want != got:
      raise ValueError(f"{name}: expected last dim {want}, got {got}")


def add_timeout_bootstrap(
  rewards: torch.Tensor,
  infos: dict,
  next_critic_obs: torch.Tensor,
  actor_critic,
  gamma: float,
) -> torch.Tensor:
  """Correct timeout target: bootstrap V(next_critic_obs), not V(s_t)."""
  timeouts = infos.get("time_outs")
  if timeouts is None:
    return rewards
  next_value = actor_critic.evaluate(next_critic_obs).detach().reshape(-1)
  return rewards + gamma * timeouts.to(rewards.device).reshape(-1) * next_value
```

旧 `HIMPPO.process_env_step()` 使用的是 transition 中已经保存的当前 `V(s_t)`；对于 timeout，这不是正确的 bootstrap 状态。若保留旧算法，至少把这一段替换为上面的 `add_timeout_bootstrap()`，并把 terminal transition 的 `next_critic_obs` 在 reset 前保存下来。`dones` 仍可表示 episode 结束，但 `time_outs` 必须单独保留，不能把自然终止和时间截断混成一个布尔量。

构造旧 HIM 网络时，合同必须先被计算和检查：

```python
# 只在 contract smoke 通过后使用；不是新的算法实现
from rsl_rl.algorithms import HIMPPO
from rsl_rl.modules import HIMActorCritic

contract = HimContract(one_step_dim=D, history_length=H, critic_dim=C, action_dim=A)
contract.validate()

actor_critic = HIMActorCritic(
  num_actor_obs=contract.actor_history_dim,
  num_critic_obs=contract.critic_dim,
  num_one_step_obs=contract.one_step_dim,
  num_actions=contract.action_dim,
  actor_hidden_dims=[512, 256, 128],
  critic_hidden_dims=[512, 256, 128],
  activation="elu",
  init_noise_std=1.0,
)

algorithm = HIMPPO(
  actor_critic=actor_critic,
  num_learning_epochs=5,
  num_mini_batches=4,
  learning_rate=1.0e-3,
  gamma=0.99,
  lam=0.95,
  device="cuda:0",
)
```

这里的 `D/H/C/A` 必须从已编译环境的 observation/action contract 得到，不能默认沿用旧 GO2W 的 `57/6/262/16`。旧估计器的 target slice 还要求 `next_critic_obs = [actor_one_step(D), base_lin_vel(3), privileged_tail]`；只要 critic term 顺序变化，就必须同步修改这个合同和 checkpoint metadata。

最后还要处理旧 runner 的接口差异。HIMLoco 旧 runner 读取的 step 结果是七项：

```python
obs, privileged_obs, rewards, dones, infos, termination_ids, termination_privileged_obs = env.step(actions)
```

而 mjlab/RSL-RL wrapper 的返回签名和 reset 语义应以锁定版本为准。不能只增加一个 history buffer 就把标准 `RslRlVecEnvWrapper` 当成旧 HIM runner；需要一个明确的 adapter，把 `critic_obs`、terminal `next_critic_obs`、`time_outs` 和额外 termination 信息逐项映射。

#### 4.12.11 每个文件到底调用了哪一层 API

| 文件 | 直接调用 | 不应承担的职责 | 复制后第一条检查 |
| --- | --- | --- | --- |
| `robots/unitree_go2/entity_cfg.py` | `EntityCfg`、`MjSpec`、actuator、collision | 不创建 env、不注册 task | `get_spec()` 能读取 XML，mesh 资产完整 |
| `envs/velocity/mdp/*.py` | `env.scene`、`ContactSensor.data`、`command_manager` | 不修改 scene、不控制 reset | 输入 batch 维度和输出 `[B,...]` |
| `velocity_env_cfg.py` | `SceneCfg`、observation/action/reward/event/termination cfg | 不写 Go2 名称 | 无实体名也能构造通用配置 |
| `go2_config.py` | `get_go2_robot_cfg()`、geom/site/body selector、rough/flat override | 不实现 physics loop | 12 joints/12 actuators 和传感器 selector 可解析 |
| `rl_cfg.py` | `RslRlOnPolicyRunnerCfg` | 不创建 MuJoCo env | actor/critic 输入维度与 env contract 相同 |
| `__init__.py` | `register_mjlab_task` | 不吞异常 | 导入一次得到两个唯一 task ID |
| `train.py` | env wrapper、runner、日志和关闭 | 不定义 reward/obs | `num_envs=1` 可 reset/zero-step |
| `history_adapter.py` | `[B,D] -> [B,H*D]`、partial reset、terminal snapshot | 不改变 MuJoCo 状态 | 最新帧、部分 reset、target slice 全部可断言 |

mjlab 的运行时顺序应保持为：`process_action -> decimation 次 sim.step/apply_action -> scene.update -> termination -> reward -> metrics/events -> reset -> command -> sensor -> observation`。脚本、HIM adapter 和自定义 term 都不应重新实现这一顺序；否则就会出现“动作已写入但 sensor 尚未更新”或“reset 后 history 仍是旧环境”的时序错误。

#### 4.12.12 复制顺序和最小验收矩阵

按以下顺序落盘，任何一步失败都不要继续接下一层：

1. 复制 `entity_cfg.py` 和资源，执行 `MjSpec`/mesh/asset 检查。
2. 复制 `mdp` term，使用随机 tensor 做 `[B]`、`[B,4]`、`[B,4,3]` shape smoke。
3. 复制 `velocity_env_cfg.py`，只创建通用 cfg，不导入 Go2 注册。
4. 复制 `go2_config.py`，检查 rough/flat 两个 cfg 的 entity、sensor、selector 和 observation term。
5. 以 `num_envs=1` 编译、reset、零动作单步；确认每个 observation finite。
6. 逐项记录 actor `D_actor`、critic `C_critic`、action `A_action`，再设置 4096 环境训练。
7. 先注册并运行标准 PPO；checkpoint 能 strict load 后再启用 terrain curriculum。
8. 最后复制 `history_adapter.py` 和自定义 HIM runner；验证 timeout、partial reset、target slice 和七项旧接口映射。

| 检查 | 预期 | 失败时停止原因 |
| --- | --- | --- |
| package/import | `import legged_mjlab.envs` 成功 | 任务注册尚未稳定 |
| registry | Flat/Rough 各出现一次 | 重复注册或模块未导入 |
| registry API | 对锁定版本运行 `inspect.signature(register_mjlab_task/load_env_cfg/load_runner_cls)` 并做 fresh-process round-trip | 不能混用 main 文档和本地 registry 签名 |
| asset compile | `MjSpec`、mesh、12 joints、12 actuators 存在 | 不能进入 manager 配置 |
| reset/step | 零动作后 obs/reward/done 全部 finite，batch 为 `N` | scene 或 term shape 错 |
| four variants | Flat/Rough × train/play 各自 reset、zero-action step、finite obs；play 不改 actor contract | 不能把一个变体的 smoke 结果外推给另一个 |
| action | `A` 等于匹配 actuator 数量 | checkpoint/action map 不可信 |
| flat/rough contract | 分别记录 `D_actor`、`C_critic`、term 顺序和 asset hash；禁止跨变体加载 | 共享 term 不代表共享 checkpoint |
| timeout | `timestep=0.005`, `decimation=4`, `20s` 对应 1000 个 policy step | bootstrap 语义未锁定 |
| rough sensor | contact/raycast 实际 shape 来自 compiled scene | 不能硬编码 sensor 维度 |
| checkpoint | task/asset/D/C/A/H metadata strict match | 禁止 silent load |
| HIM | `next_critic_obs`、`time_outs`、partial reset 通过断言 | 不得启动 HIM 训练 |

本轮不运行这些命令、不创建 Python 文件，也不宣称验收通过；本节是后续实现者可以逐文件复制的文档合同。

checkpoint metadata 不能只停留在表格里。一个最小的可复制 helper 可以放在 `legged_mjlab/utils/checkpoint_contract.py`：

```python
# legged_mjlab/utils/checkpoint_contract.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ContractMetadata:
  task_id: str
  asset_hash: str
  actor_dim: int
  critic_dim: int
  action_dim: int
  history_length: int
  joint_names: tuple[str, ...]
  action_scale: tuple[float, ...]
  timestep: float
  decimation: int
  solver_signature: str


def write_contract(path: str | Path, metadata: ContractMetadata) -> None:
  Path(path).write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n")


def assert_contract(path: str | Path, expected: ContractMetadata) -> None:
  actual = json.loads(Path(path).read_text())
  if actual != asdict(expected):
    raise ValueError(
      "Checkpoint contract mismatch; refuse to load.\n"
      f"expected={asdict(expected)!r}\nactual={actual!r}"
    )
```

训练入口应在 runner 创建前写入 metadata，play 入口应在 `runner.load()` 前调用 `assert_contract()`。`strict=True` 只约束 runner state 的 tensor/key 加载，不能替代这份 task/asset/joint/solver 合同；实际项目还应把 XML/mesh hash 和 observation term 顺序 hash 接入 `asset_hash` 或独立字段。

#### 4.12.13 运行安全与确定性门槛

`EntityCfg` 中的 `effort_limit`、`soft_joint_pos_limit_factor` 和 reward penalty 不能替代 MuJoCo XML/actuator 的硬安全边界。标准 PPO 或 play 可以在仿真中运行，不代表已经具备真机放行条件。实现时必须补齐下面的合同，并把值写入 checkpoint metadata：

| 安全面 | 必须固定/验证 | 未完成时的处理 |
| --- | --- | --- |
| 动作输入 | `isfinite(action)`、policy clip、joint order/sign、target range、slew-rate | NaN/Inf 或越界立即拒绝，进入 hold/zero 的仿真 fail-safe；不得继续送入 actuator |
| 执行器 | XML `ctrlrange`/`forcerange`、gear/transmission、joint hard limit、stiffness/damping/effort/armature | 未逐 actuator 对表不得做 sim2real |
| 接触求解 | timestep、CCD、solver iterations、`nconmax`/`njmax`、contact sensor match 上限 | 必须做穿透、漏接触、接触截断、非法接触和 self-collision 回放 |
| 选择器 | foot geom/site、terrain body、IMU sensor 的零匹配/多匹配行为 | zero-match 或 ambiguous-match 应 fail closed，不允许静默得到空 observation |
| play | 固定 seed；关闭 corruption、push、friction/COM 随机化和 curriculum；记录 command range | play 只能用于确定性回归，不能一边随机化一边比较 checkpoint |
| checkpoint | task、asset hash、XML/mesh 版本、D/C/A/H、joint map、scale/sign、timestep/decimation、solver 配置 | metadata 不匹配时 strict reject，不能只检查网络 tensor shape |
| 传感器 | finite、范围、时间新鲜度、恒值/断连检测、失败后的 hold/zero/termination | 没有健康检查前不允许连接真实 IMU/contact/command 通道 |
| 真机线程 | watchdog、通信超时、急停、RTOS 优先级和锁顺序 | 这些合同缺失时只能做仿真文档和 headless smoke |

这张表是 release gate，而不是建议清单。尤其不能把“训练 reward 里惩罚了关节限位”解释成“硬限位已经存在”，也不能把 `strict=True` 解释成“joint order、asset hash 和动作缩放已经匹配”。

## 5. 安装与使用手册（命令只供未来执行）

本节把“现有项目的安装事实”和“目标项目未来的安装方案”分开。所有命令均未在本轮执行。

## 5.1 unitree_rl_mjlab 的现有安装信息

`unitree_rl_mjlab/doc/setup_zh.md:3-43` 推荐 Ubuntu 22.04、NVIDIA GPU、驱动 550+、Python 3.11 的 Conda 环境，并给出 MiniConda、系统包和 editable install 命令。`setup.py:6-9` 是 Python 包依赖的更直接证据。

后续使用者可按该项目自己的文档执行，逻辑顺序是：

```text
准备 NVIDIA/CUDA/Conda
  -> 创建 Python 3.11 环境
  -> 获取 unitree_rl_mjlab
  -> 按 setup.py 安装 mjlab==1.2.0 与 mujoco-warp==3.5.0
  -> 按需要安装 C++ 部署/仿真依赖
  -> 进入仓库根目录运行 scripts/list_envs.py
  -> 先训练 flat，再训练 rough，再做 play
```

现有文档中的命令示例：

```bash
conda create -n unitree_rl_mjlab python=3.11
conda activate unitree_rl_mjlab
cd /path/to/unitree_rl_mjlab
pip install -e .
python scripts/list_envs.py
python scripts/train.py Unitree-Go2-Flat --env.scene.num-envs=4096
python scripts/play.py Unitree-Go2-Flat --checkpoint_file=/path/to/model.pt
```

注意：

- 这些命令是 README/setup 中的使用示例或按其入口推导的运行顺序，不代表本轮已执行。
- `doc/setup_zh.md` 中的 `apt install` 主要为后续 C++/部署链路准备；训练环境要先区分 Python、GPU 和部署依赖。
- README 中有 `Unitree-G1-Flat`、`Unitree-Go2-Flat` 等任务 ID；实际 task ID 应以 `list_envs.py` 输出为准。
- tracking 任务还需要 motion file；`scripts/train.py:64-82` 会检查路径并将其写入 `MotionCommandCfg`。
- `scripts/train.py:166-192` 在多 GPU 分支动态导入 `torchrunx`，但当前 `unitree_rl_mjlab/setup.py` 没有声明它；多 GPU 训练只能记为“代码路径存在、依赖和运行时未验证”。
- 当前 `scripts/train.py:148-154` 的日志根按 `logs/rsl_rl/<experiment>/<timestamp>[_run_name]` 组织，并由 runner 写入 `params/env.yaml`、`params/agent.yaml` 等快照；目标项目可以借鉴这种可追踪布局，但仍应额外加入本手册定义的 contract metadata。
- `scripts/play.py:86-108` 包含不直接给本地 checkpoint、而通过 `wandb_run_path` 获取/缓存 checkpoint 的代码分支；但当前可见的 `PlayConfig` 定义在 `scripts/play.py:22-40` 未声明 `registry_name` 和 `wandb_run_path` 字段，因此该路径属于需要版本/运行时复核的代码路径，不能当成已经可用的通用接口。本文的本地 `--checkpoint_file` 示例不覆盖该分支。

## 5.2 HIMLoco 的现有安装信息

当前 `HIMLoco` 根目录没有完整安装 README；可用元数据为 `HIMLoco/legged_gym/setup.py:4-15`，它声明：

- package name：`legged_gym`；
- dependencies：`isaacgym`、`rsl-rl`、`matplotlib`；
- 使用 `find_packages()`。

`HIMLoco/rsl_rl/setup.py` 是另一个独立的 RSL-RL 包入口；不能把 `legged_gym/setup.py` 的声明理解为已经包含当前仓库内所有 `rsl_rl` 代码。

另外，HIMLoco 的 helper 参数中出现 `--horovod` 选项，但当前静态阅读没有证明对应的多 GPU/分布式执行链路已经实现；它只能作为待验证的命令行兼容项，不能据此承诺并行训练可用。

因此后续安装流程只能写成“待实际环境确认”的流程：

```text
准备与 Isaac Gym 版本匹配的 Python/CUDA/Torch
  -> 准备 rsl_rl 包
  -> 从 HIMLoco/legged_gym 安装 legged_gym
  -> 导入 legged_gym.envs 触发 task_registry 注册
  -> 运行对应的 train.py/play.py
```

不要把 `uw-himloco/README.md` 的路径直接复制到当前 `HIMLoco`；两个目录层级和分支内容不同。Isaac Gym 的下载和许可方式也必须遵循实际版本的官方要求，[待验证] 本文不替用户安装。

## 5.3 目标 `legged_mjlab` 的安装设计

当前目标项目尚未形成可支持安装的 packaging 元数据：根目录 `setup.py` 当前为空且没有 `pyproject.toml`。本轮没有执行 `pip install -e .`，因此不报告具体 pip 失败信息；未来安装方案应满足：

1. 一个面向根项目的唯一 packaging 入口；这不等于必须删除或合并未来独立维护的 `rsl_rl/setup.py`。
2. 明确 Python 版本、PyTorch、mjlab、mujoco-warp 和 RSL-RL 版本组合。
3. 包含 `legged_mjlab`、任务子包和机器人配置子包。
4. 资源通过包内路径或稳定的项目根路径找到，不依赖 `/home/kk`、`/home/zhy` 等个人绝对路径。
5. 训练依赖与 sim2sim/deploy 依赖分组，避免训练用户安装 C++ 运行时。
6. 安装完成后，先提供只读的任务列举命令，再提供单环境/小批量 play，最后才是大规模训练。

[设计建议] 目标项目第一版可以使用与 `unitree_rl_mjlab` 相同的 `mjlab==1.2.0`/`mujoco-warp==3.5.0` 组合作为基线，但必须在实际安装时确认 API 和 CUDA/Torch 兼容，不能仅凭 setup.py 宣称已兼容。

目标项目的命令应与上面的参考项目命令分开：

```bash
# 在 pyproject.toml、包内 resources 和 constraints 验证完成后执行
python -m pip install -e ".[viewer]"
python -m legged_mjlab.scripts.list_envs
python -m legged_mjlab.scripts.train --task-id Unitree-Go2-Flat --num-envs 1
python -m legged_mjlab.scripts.play --task-id Unitree-Go2-Flat --checkpoint /path/to/model.pt --viewer viser
```

如果启用 `[project.scripts]`，等价入口是 `legged-mjlab-list-envs`、`legged-mjlab-train` 和 `legged-mjlab-play`。当前 `legged_mjlab` 尚未拥有这些入口；命令只是未来合同，不能在本轮执行。

---

## 6. 从零搭建目标训练环境的分阶段手册

以下每一步都说明“要阅读什么、未来要写什么、接口必须是什么、完成的证据是什么”。本轮只写手册，不执行这些步骤。

## 阶段 A：冻结范围和版本

### 目标

先做一个 `Unitree-Go2-Flat` 风格的标准速度跟踪任务，动作是关节位置目标，critic 可选 privileged observation；不在第一阶段加入 HIM、轮足、复杂 terrain 或部署。

### 要阅读

- `unitree_rl_mjlab/setup.py:1-17`
- `unitree_rl_mjlab/src/tasks/velocity/velocity_env_cfg.py:36-431`
- `unitree_rl_mjlab/src/tasks/velocity/config/go2/`
- `unitree_rl_mjlab/scripts/train.py:23-140`
- `unitree_rl_mjlab/scripts/play.py:22-177`

### 未来要写的文档/配置内容

- 一张版本矩阵：Python、Torch、mjlab、mujoco-warp、RSL-RL、GPU driver。
- 一张初始 task contract：task ID、动作维度、单步 actor obs、critic obs、控制周期、episode 时长。
- 一张 non-goal 清单：HIM、sim2sim、真机、ONNX 暂不进入 baseline。

### 通过证据

- 能说清楚每个依赖来自哪个 manifest。
- 不存在两个互相冲突的包根和资源根。
- 任务 ID 和动作/观测维度已经写入文档表格。

## 阶段 B：建立唯一包根和资源根

### 要阅读

- `unitree_rl_mjlab/src/__init__.py:1-2`
- `unitree_rl_mjlab/src/assets/robots/unitree_go2/go2_constants.py:18-33`
- `legged_mjlab/resources/robots/unitree_go2/go2_constants.py:7-21`

### 未来要写的内容

本项目固定采用一个包根，不再保留 `src` 作为第二个候选包根：

- Python wrapper 和环境代码统一位于 `legged_mjlab/`；
- Go2 资产 wrapper 位于 `legged_mjlab/robots/unitree_go2/entity_cfg.py`；
- 稳定发布时的 XML/mesh 位于 `legged_mjlab/resources/robots/unitree_go2/`；
- 迁移阶段仓库根 `resources/robots/unitree_go2/` 只是 raw asset 镜像，不承担 Python 包职责。

不要让 `resources/robots/.../go2_constants.py` 继续引用不存在的 `src.SRC_PATH`；原来的常量 wrapper 应迁移为 `entity_cfg.py`，并将资源访问改成包资源或显式的仓库资源根。

### 必须固定

```text
PROJECT_ROOT
  -> legged_mjlab/resources/robots/unitree_go2/xmls/go2.xml
  -> legged_mjlab/resources/robots/unitree_go2/xmls/assets/*

迁移期 raw mirror
  -> resources/robots/unitree_go2/xmls/go2.xml
  -> resources/robots/unitree_go2/xmls/assets/*
```

路径解析不能依赖当前 shell 工作目录，也不能依赖某个开发者的 home 目录。

### 通过证据

- 任何机器人配置都能从包根得到 XML 和 mesh 目录。
- 资产 wrapper 的 import 路径不再出现 `from src import SRC_PATH` 的悬空依赖。
- raw asset、Python wrapper、task cfg 三者职责分开。

## 阶段 C：完成 Go2 EntityCfg/执行器/碰撞合同

### 要阅读

- `unitree_rl_mjlab/src/assets/robots/unitree_go2/go2_constants.py:24-139`
- `legged_mjlab/resources/robots/unitree_go2/xmls/go2.xml`
- `legged_mjlab/resources/robots/unitree_go2/xmls/scene_go2.xml`

### 未来要写的内容

只写机器人资产 wrapper，不写环境类。它必须描述：

1. XML/MJCF 主文件和 mesh bytes 的来源。
2. robot entity 名称。
3. base body、foot site、foot collision geom、joint 和 actuator 的名称/正则。
4. 初始 root pose、默认 joint positions、joint velocities。
5. 每类 actuator 的 stiffness、damping、effort limit、armature。
6. collision filtering、脚接触规则和 self-collision 策略。
7. 每次调用是否返回新配置实例，避免 config mutation。

### 资产接口契约

| 项目 | 输入 | 输出 | 失败表现 |
| --- | --- | --- | --- |
| `spec_fn` | 无或资源根 | 可编译 MJCF spec | XML 不存在、mesh 缺失、名字不匹配时立即报错 |
| `get_assets` | mjlab meshdir | `dict[str, bytes]` | 资源目录缺失或 asset 名称不一致 |
| `get_robot_cfg` | 无 | 新的 `EntityCfg` | 不允许返回跨任务共享且会被修改的单例 |
| actuator 匹配 | joint name regex | actuator 数量/顺序 | 有 joint 未匹配或多重匹配时记录并阻断 |

### 通过证据

- MJCF 可被 mjlab 的实体构造流程接受。[待验证] 本轮不编译 spec。
- 每个 actuator 匹配到预期 joint，动作顺序和 effort limit 有表格记录。
- feet contact sensor 使用的 geom/site/body 名称都能在 XML 中找到。

## 阶段 D：建立共享 velocity env cfg

### 要阅读

完整阅读 `unitree_rl_mjlab/src/tasks/velocity/velocity_env_cfg.py`，重点是：

- sensor：`43-51`；
- actor/critic observations：`58-136`；
- action：`152-159`；
- commands：`165-180`；
- events：`186-255`；
- rewards：`261-354`；
- terminations/curriculum：`360-387`；
- scene/sim：`393-431`。

### 未来要写的内容

建立一个共享配置工厂，至少包含：

1. terrain scanner 和接触 sensor 的占位定义。
2. actor observation group：只放 policy 运行时可获得的量。
3. critic observation group：允许 privileged state，但明确它不进入部署 policy。
4. joint position action term。
5. velocity command term。
6. reset、push、friction、COM 和 encoder bias 等事件。
7. 速度跟踪、姿态、动作平滑、关节限制、足端接触等 reward term。
8. timeout、fallen/bad orientation、illegal contact 等 termination term。
9. terrain 和 command curriculum。
10. MuJoCo timestep、iterations、decimation、episode length。

### 关键原则

- shared cfg 只提供通用语义和默认值；机器人具体名称在 robot cfg 里填。
- 每个 term 的 `func`、参数、权重和命名都能单独追踪。
- actor/critic terms 的顺序和拼接方式必须写入 observation contract；不能只依赖字典迭代顺序。
- play cfg 要关闭 noise/corruption、push、curriculum 或把它们改成适合可视化的模式。

## 阶段 E：把 HIMLoco 的每个功能拆成 mjlab term

### E.1 观测

迁移表：

| HIMLoco 观测 | mjlab 目标 term | 数据来源 |
| --- | --- | --- |
| command | `generated_commands` | command manager |
| base angular velocity | sensor term | IMU/builtin sensor |
| projected gravity | `projected_gravity` | robot orientation |
| relative joint position | `joint_pos_rel` | robot entity |
| joint velocity | `joint_vel_rel` | robot entity |
| last action | `last_action` | action manager |
| base linear velocity | critic-only sensor term | privileged sensor/state |
| height scan | `height_scan` | raycast sensor |
| contact/air time/force | contact sensor terms | contact sensor |

不要在一个函数中拼接全部状态；每项观测必须有名称、维度、缩放、噪声、是否 actor/critic、是否可部署的记录。

### E.2 动作和控制周期

HIMLoco 是 `decimation=4`、`dt=0.005` 的典型控制结构；目标 mjlab 配置也可以采用“policy action period = physics timestep × decimation”的合同，但动作的实际执行应由 mjlab action/actuator 配置负责。

必须记录：

- policy action 的范围和 clip；
- action 到 joint target 的 scale/offset；
- actuator 的 stiffness/damping/effort limit；
- physics timestep、decimation、policy frequency；
- reset 时 action/last action 的初值；
- sim2real 时 joint order、sign、zero offset 的映射。

### E.3 奖励

将 HIMLoco 的 `_reward_tracking_lin_vel`、`_reward_tracking_ang_vel`、orientation、dof acceleration、torque、collision、feet air time、feet slip 等逐项迁移为独立 reward term。奖励权重不应该藏在函数内部。

每个 reward term 的文档模板：

```text
名称：
物理含义：
输入 entity/sensor/command：
输出 shape：[num_envs]
正负号约定：
权重与 dt 缩放：
是否改变共享状态：必须为否，除非明确记录
统计指标：
训练/评估是否都启用：
```

HIMLoco 中 `compute_reward()` 在正奖励裁剪后再加入 termination reward（`legged_robot.py:163-176`）；mjlab 中应把这一语义明确写进 reward manager 配置或 runner 合同，不要默默假设两者完全相同。

### E.4 终止

至少区分：

- time out：用于 bootstrap 的截断，不一定等价于摔倒；
- illegal contact：来自 contact sensor；
- bad orientation/fallen：来自 body orientation；
- 任务特定终止：例如 motion anchor 或末端偏差。

所有 termination term 返回 `[num_envs]` 布尔值；训练 wrapper 需要把 timeout 信息放进 `infos` 或等价字段，供 PPO 正确 bootstrap。

### E.5 随机化和课程

将 HIMLoco 的 `domain_rand` 分成三类：

1. startup：初始 friction、COM、encoder bias；
2. reset：root pose、joint offset、质量或动力学参数；
3. interval：push、disturbance、command resampling。

terrain curriculum 与 command curriculum 作为独立 curriculum term；不要让 reward 函数根据隐式条件改变 terrain level。

## 阶段 F：机器人专属 cfg、任务注册和 play cfg

### 要写的文件职责

```text
robots/unitree_go2/entity_cfg.py
  -> 加载包内 MJCF/mesh
  -> 填 EntityCfg、actuator、collision、initial state
  -> 只维护机器人实体名称与资源合同

envs/go2/go2_config.py
  -> 调用 flat/rough 配置工厂
  -> 填 base/feet/contact/joint/site/geom 名称
  -> 配置 rough/flat 差异和 play overrides
  -> 组合 actor/critic/action/command/reward/termination

envs/go2/__init__.py
  -> 注册唯一 task ID
  -> 绑定 train env cfg
  -> 绑定 play env cfg
  -> 绑定 RL cfg
  -> 绑定默认 runner
```

建议至少先注册两个 ID：

- `Legged-Mjlab-Go2-Flat`：无 terrain generator、无 height scan 的 baseline。
- `Legged-Mjlab-Go2-Rough`：启用 terrain generator、height scan、contact sensor 和 curriculum。

命名只是设计建议；最终命名要与项目的 namespace 约定一致。

### 注册验收

- 导入任务包后 task registry 能列出 task ID。[待验证]
- 同一个 task ID 不在多个 `__init__.py` 注册。
- `load_env_cfg(task_id, play=False/True)` 的 actor/critic/action 维度一致。
- play cfg 不改变训练 checkpoint 所需的 actor input 维度。
- rough -> flat 的派生逻辑只删除 terrain-specific terms，不误删 contact/reward/command。

## 阶段 G：先接标准 RSL-RL，再接 HIM

### G.1 标准 PPO baseline

先复制“接口语义”而不是 HIMLoco 的 runner 实现：

```text
train.py
  -> import tasks
  -> list/load task cfg
  -> ManagerBasedRlEnv
  -> RslRlVecEnvWrapper
  -> MjlabOnPolicyRunner
  -> learn
```

`unitree_rl_mjlab/scripts/train.py:92-140` 是当前最直接的参照。第一阶段只需要 actor observation、critic observation、action、reward、done、timeout 和 checkpoint 合同。

### G.2 HIM 适配的最小边界

如果要在 mjlab 中复用 HIMLoco 的思想，建议新增一个“history/estimator adapter”，而不是修改 mjlab 的核心 env：

| adapter 输入 | adapter 内部 | adapter 输出 |
| --- | --- | --- |
| manager actor obs `[B, D]` | reset 时清空 history；每个 policy step 前后按固定顺序滚动历史 | HIM actor history `[B, H*D]` |
| manager critic obs | 生成/保存 next critic obs | critic obs、next critic obs |
| action manager 输出 | 记录 policy action；若存在 action delay，明确记录实际执行 proxy | action history/训练记录 |
| done + timeout | reset 对应环境的 history；保留 timeout bootstrap 信息 | RSL-compatible `dones/infos` |

HIM adapter 必须明确：

1. 最新帧是在 history 前还是后；HIMLoco 的 `legged_robot.py:197-198` 是最新帧在前。
2. 每帧是否包含 noise/corruption。
3. actor history 的单步维度 `D` 从哪里得到，不允许通过 `int(total_obs / D)` 猜测而不检查整除。
4. privileged critic 是否与 actor 使用相同 history length。
5. `next_critic_obs` 中 estimator target 的语义和切片来源。
6. episode reset、timeout、partial env reset 如何处理 history。
7. action delay 时 history 记录 sampled action 还是 executed action proxy。

### G.3 HIM actor/estimator 合同

以 HIMLoco 当前实现为参考，必须在文档和 cfg 中记录：

```text
num_one_step_obs = D
history_length = H
actor_obs = H * D
estimator input = H * D
estimator velocity output = 3
estimator latent output = L（当前参考为 16）
actor input = D + 3 + L
critic input = privileged observation dimension
action dimension = robot actuator/action dimension
```

当前 HIMLoco 参考值是 D=45、H=6、L=16、actor input=64；GO2W 参考值是 D=57、H=6、actor input=76。目标 Go2 不能直接沿用这些数字，必须由实际 observation terms 计算后在 contract 中确认。

### G.4 为什么不建议第一天迁移 HIM

HIM runner 比标准 PPO 多了 `num_one_step_obs`、next privileged obs、estimator update、history reset 和额外 storage 字段。若 baseline 的 sensor 名称、action order 或 termination semantics 尚未稳定，HIM loss 下降也无法说明环境正确。

## 阶段 H：训练、日志、checkpoint 和 play

### train 入口必须负责的事情

参照 `unitree_rl_mjlab/scripts/train.py:42-192`，目标脚本只应负责：

- 解析 task ID 和配置 override；
- 选择 CPU/GPU、seed 和可选多 GPU；
- 加载 task cfg；
- 绑定 motion file（仅 tracking 任务）；
- 创建 env 和 vec wrapper；
- 创建默认/任务专属 runner；
- 创建唯一日志目录；
- 保存 env/agent 配置快照；
- resume checkpoint；
- 调用 learn 并关闭 env。

### play 入口必须负责的事情

参照 `unitree_rl_mjlab/scripts/play.py:42-177`，目标脚本应支持：

- zero/random/trained policy 或明确只支持 trained；
- checkpoint path；
- num envs；
- 禁用 termination 的演示模式（若确有必要）；
- video；
- viewer backend；
- play cfg 的 noise/curriculum/push override。

### checkpoint 合同

每个 run 目录至少应保存：

```text
logs/rsl_rl/<experiment>/<timestamp>/
├── model_<iteration>.pt
├── params/env.yaml
├── params/agent.yaml
├── videos/                         # 可选
└── metadata/                       # baseline 必须：git commit、task id、asset hash、obs/action contract hash
```

模型文件单独存在并不证明可恢复；恢复前必须检查：

- task ID；
- actor/critic observation dimension；
- HIM 的 D/H/L（如果启用）；
- action dimension、joint order、scale；
- runner/algorithm class；
- simulator timestep/decimation；
- asset XML 和 mesh 版本。

## 阶段 I：评估和 sim2sim 边界

训练环境的最低评估矩阵：

| 场景 | policy | 必看指标 |
| --- | --- | --- |
| flat / zero command | trained | 是否稳定站立、action 是否饱和 |
| flat / forward velocity | trained | velocity tracking、姿态、能耗 |
| rough / slow command | trained | terrain scan、接触、跌倒率 |
| rough / push | trained | recovery、termination 原因 |
| reset | zero/random | history/action/command 是否清零或按契约初始化 |
| checkpoint resume | trained | 配置和维度是否完全一致 |

sim2sim 之前必须单独建立：

- MuJoCo XML 与训练 asset 是否同源；
- joint order、sign、zero offset；
- action scaling 和 PD 参数；
- command scaling；
- observation scaling、历史顺序、噪声关闭策略；
- ONNX/JIT 输入输出 shape；
- 接触和 termination 语义是否能在目标仿真器重现。

本轮不执行这些检查，也不把现有 HIMLoco 的 sim2sim 记忆当成目标环境已通过的证据。

---

## 7. 代码阅读路线：每次改动前应该看什么

### 7.1 阅读 unitree_rl_mjlab 的推荐顺序

1. `setup.py`：确认版本和包入口。
2. `src/tasks/__init__.py`：理解任务导入/注册。
3. `scripts/list_envs.py`：理解外部 task ID。
4. `scripts/train.py`：理解训练边界和日志。
5. `src/tasks/velocity/velocity_env_cfg.py`：理解 manager 配置全貌。
6. `src/assets/robots/unitree_go2/go2_constants.py`：理解实体/执行器/资源。
7. `src/tasks/velocity/config/go2/env_cfgs.py`：理解机器人差异配置。
8. `src/tasks/velocity/config/go2/rl_cfg.py`：理解 actor/critic/PPO。
9. `src/tasks/velocity/mdp/*.py`：逐项读 term。
10. `scripts/play.py`：确认 play 时哪些配置被改变。
11. `src/tasks/tracking/`：最后再读 motion imitation。

### 7.2 阅读 HIMLoco 的推荐顺序

1. `legged_gym/setup.py`：确认 Isaac Gym 包依赖。
2. `legged_gym/legged_gym/envs/__init__.py`：确认注册项。
3. `legged_gym/legged_gym/utils/task_registry.py`：理解 env/runner 创建。
4. `legged_gym/legged_gym/envs/base/legged_robot_config.py`：列出所有配置字段。
5. `legged_gym/legged_gym/envs/base/base_task.py`：理解 VecEnv 对外接口。
6. `legged_gym/legged_gym/envs/base/legged_robot.py:40-240`：理解 step、obs、reward、reset。
7. `legged_robot.py:346-460`：理解 command、PD、随机化和 curriculum。
8. 机器人 config：先读 `my_robot_config.py`，再读 `go2w_config.py`。
9. `rsl_rl/runners/him_on_policy_runner.py`：理解 runner 需要的 env 字段。
10. `rsl_rl/modules/him_actor_critic.py`、`him_estimator.py`：理解 actor input 和 estimator target。
11. `rsl_rl/algorithms/him_ppo.py`、`him_rollout_storage.py`：理解 next critic obs 与额外训练数据。
12. `scripts/play.py` 和 deploy/sim2sim：最后再看部署合同。

### 7.3 目标项目的每个新文件都要回答的五个问题

1. 它属于资产、任务、term、runner 还是入口层？
2. 它依赖哪个 manager/entity/sensor contract？
3. 它输出的 shape、单位、频率和 reset 行为是什么？
4. 它是否会改变共享 state？如果会，谁负责恢复/记录？
5. 它能否被单独阅读、替换和审查？

---

## 8. 观测、动作和 HIM 的合同模板

后续开发应维护三张表，而不是依赖记忆或散落的数字。

### 8.1 Observation contract

| 字段 | 示例 | 必须记录 |
| --- | --- | --- |
| term name | `base_ang_vel` | 唯一名称 |
| source | IMU sensor | entity/sensor/command 来源 |
| actor/critic | actor + critic | 是否进入部署 policy |
| dimension | 3 | shape 和拼接顺序 |
| scale | 0.25 | 单位/缩放 |
| noise | uniform range | 训练和 play 行为 |
| history | 1 或 H | 最新帧位置、reset 行为 |
| frame | body/world | 坐标系 |
| availability | online/privileged | 实机是否可获得 |

### 8.2 Action contract

| 字段 | 必须记录 |
| --- | --- |
| action name/order | 例如 FR/FL/RR/RL + hip/thigh/calf；不能靠猜 |
| dimension | 与 actuator 数量一致 |
| range/clip | policy action 与 actuator target 各自范围 |
| scale/offset | 目标角或目标速度的公式语义 |
| actuator | stiffness/damping/effort/armature |
| decimation | 每个 policy action 对应多少 physics step |
| delay | sampled action 与 executed action 是否不同 |
| deployment mapping | C++/ONNX/MuJoCo 的顺序和符号 |

### 8.3 HIM contract

| 字段 | 示例/要求 |
| --- | --- |
| `D` | 单步 actor observation dimension；由 term 表计算 |
| `H` | history length；reset 和 newest-frame order 固定 |
| `H*D` | estimator/actor history input |
| `velocity_dim` | 当前参考为 3；要写明 frame 和 target time |
| `latent_dim` | 当前参考为 16；目标项目可不同，但必须显式记录 |
| actor input | `D + velocity_dim + latent_dim` |
| critic input | privileged obs dimension |
| target | next critic obs 的哪一段、是否 scaled |
| gradient | estimator 输出是否 detach |
| reset | future recurrent state 或 history 如何清零 |

### 8.4 维度守恒原则

每次增加或删除 observation term，都必须同步更新：

- actor observation contract；
- critic observation contract；
- history length；
- runner/storage shape；
- checkpoint metadata；
- HIM estimator input/target；
- play/export/deploy wrapper。

禁止只改一个 `num_observations` 数字。HIMLoco 中的 45/46/57 与 270/276/342 是不同实验合同，不能互换。

---

## 9. 常见失败模式和排查顺序

### 9.1 任务找不到

先查：

1. 目标项目的 `legged_mjlab/envs/__init__.py` 是否被入口 import；参考项目则检查 `unitree_rl_mjlab/src/tasks/__init__.py`。
2. `legged_mjlab/envs/go2/__init__.py` 是否执行 `register_mjlab_task`。
3. task ID 是否拼写一致。
4. `utils/task_registry.py` 是否只是转发门面，是否把注册错误吞掉。

对照：`unitree_rl_mjlab/src/tasks/__init__.py:1-5` 和 `config/go2/__init__.py:1-24`。

### 9.2 XML 或 mesh 找不到

先查：

1. 包根和 resource root 是否唯一。
2. `get_spec` 是否使用实际 XML 路径。
3. meshdir、asset bytes、XML 中的 mesh 名是否一致。
4. 是否仍有 `/home/...` 或 `from src import SRC_PATH`。

目标当前 `resources/robots/unitree_go2/go2_constants.py:7,18-21` 已经显示出此类风险。

### 9.3 观测维度不一致

按以下顺序对表：

```text
term dimensions
  -> actor group concatenated dimension
  -> history dimension
  -> wrapper observation space
  -> runner policy input
  -> checkpoint metadata
  -> play/export input
```

不要首先修改 network hidden dims；先找出哪个 term 或 history contract 不一致。

### 9.4 actor 能训练但 play 行为异常

检查：

- play cfg 是否关闭了训练时必需的 observation term；
- command scaling/range 是否改变；
- action clip/scale/offset 是否改变；
- history newest frame 顺序是否反了；
- HIM estimator 是否使用了不同的 `D`；
- XML 中 joint order 是否和训练时一致；
- checkpoint 是否来自另一个机器人/任务。

### 9.5 HIM loss 下降但动作抖动

不要只看 estimator loss。至少同时记录：

- predicted velocity 与 target velocity 的 RMSE/偏差/延迟；
- contact/swing/touchdown/liftoff 分桶误差；
- action jerk、wheel/leg speed、torque/power；
- tracking、fall、terrain success；
- sampled action 与 delayed/executed action 的差异。

现有 `HIMLoco/docs/himloco_wheel_leg_training_evaluation.md:7-175` 已经把这些训练阶段风险列成了建议矩阵；它是分析文档，不是已经跑完的实验结果。

---

## 10. 文档级验收清单

### 10.1 参考项目阅读完成条件

- [ ] 能从 `unitree_rl_mjlab/scripts/train.py` 追到 task registry、env cfg、wrapper、runner 和日志。
- [ ] 能从 `unitree_rl_mjlab/src/tasks/velocity/velocity_env_cfg.py` 解释所有 manager 区块。
- [ ] 能从 `unitree_rl_mjlab/src/assets/robots/unitree_go2/go2_constants.py` 解释 XML、mesh、actuator、collision 和 initial state。
- [ ] 能从 HIMLoco `task_registry.py` 解释 env 和 HIM runner 创建。
- [ ] 能从 HIMLoco `legged_robot.py` 解释 action delay、decimation、termination、reward、obs history 和 curriculum。
- [ ] 能从 HIMLoco `him_actor_critic.py`/`him_estimator.py`/`him_ppo.py` 解释 D、H、latent、next critic obs 和 estimator update。
- [ ] 已把 Isaac Gym 专有 API 与 mjlab manager API 分开。

### 10.2 目标架构完成条件（未来工程 gate）

- [ ] 有唯一包根和安装元数据。
- [ ] Go2 XML/mesh 路径不依赖个人绝对路径。
- [ ] 一个 baseline flat task 可以被注册和列举。
- [ ] 一个 rough task 可以独立启用 terrain scan/contact/curriculum。
- [ ] actor/critic/action/command/reward/termination 的 contract 有表格。
- [ ] train/play 使用同一个 task ID 体系，play 只做显式 override。
- [ ] checkpoint 保存 env/agent/contract metadata。
- [ ] HIM adapter 只有在标准 PPO baseline 合同稳定后才启用。

### 10.3 本轮实际完成条件

- [x] 只写入/修改 Markdown 文档。
- [x] 没有安装依赖。
- [x] 没有修改现有业务代码、测试、构建脚本或架构记忆。
- [x] 已记录当前目标工程的空文件、空目录、缺少 packaging 和资源路径冲突。
- [x] 已记录参考项目事实与设计建议的区别。
- [x] 已完成独立覆盖审查；其结论是当前只能交付静态设计文档，不能宣称目标环境完成或输出 `[ALL_TESTS_PASSED]`。
- [x] 已完成独立构建/安装审查；其结论是目标 packaging、Unitree 多 GPU 依赖和 HIMLoco 双 setup.py 仍需未来验证，不能宣称安装/构建通过。
- [x] 已完成数学/API 风险审查；其结论是 `spec.assets`、actuator 解析、batch shape、reward/termination shape、timeout/reset 仍需版本和运行时验证。
- [x] 硬件安全审查：本轮安全审查已返回风险阻断意见，结论为未通过（REJECT），不能视为安全签字。
- [ ] 训练/仿真/安装验证：本轮明确不执行，不能打勾。

---

## 11. 下一轮实际开发顺序（仍然不在本轮实施）

1. 按已确定的 `legged_mjlab` 包根、`robots/` wrapper 和包内 `resources/` 资源策略建立文件契约。
2. 以 `envs/go2` 的 Go2 flat 为第一任务，只使用标准 mjlab manager 和标准 RSL runner。
3. 让任务注册、配置加载、单环境 play、少量环境训练形成最小闭环。
4. 加入 contact sensor 和 terrain scan，再加入 rough terrain。
5. 将 terrain randomization、friction、COM、push、encoder noise 拆成事件并逐项记录。
6. 加入 curriculum 和评估指标。
7. 再定义 `envs/him_go2` history adapter，先复现目标自定义维度的静态 shape 合同。
8. 再实现 estimator/latent/next critic obs 的训练链路，并做 baseline 对照。
9. 最后处理 motion tracking、sim2sim、ONNX 和 deployment。

每一步都先更新文档中的 contract 和验收证据，再进入实现。这样项目的目录清晰度来自边界清晰，而不是来自把代码机械地拆成更多文件。

---

## 附录 A：本次引用的关键文件索引

### unitree_rl_mjlab

- `setup.py`
- `doc/setup_zh.md`
- `src/__init__.py`
- `src/tasks/__init__.py`
- `src/tasks/velocity/velocity_env_cfg.py`
- `src/tasks/velocity/config/go2/__init__.py`
- `src/tasks/velocity/config/go2/env_cfgs.py`
- `src/tasks/velocity/config/go2/rl_cfg.py`
- `src/tasks/velocity/mdp/observations.py`
- `src/tasks/velocity/mdp/rewards.py`
- `src/tasks/velocity/mdp/terminations.py`
- `src/tasks/velocity/mdp/velocity_command.py`
- `src/tasks/velocity/mdp/curriculums.py`
- `src/tasks/velocity/rl/runner.py`
- `src/tasks/tracking/tracking_env_cfg.py`
- `src/tasks/tracking/config/g1/env_cfgs.py`
- `scripts/train.py`
- `scripts/play.py`
- `scripts/list_envs.py`

### HIMLoco

- `legged_gym/setup.py`
- `ARCHITECTURE_CONTEXT.md`
- `legged_gym/legged_gym/envs/__init__.py`
- `legged_gym/legged_gym/envs/base/base_config.py`
- `legged_gym/legged_gym/envs/base/base_task.py`
- `legged_gym/legged_gym/envs/base/legged_robot_config.py`
- `legged_gym/legged_gym/envs/base/legged_robot.py`
- `legged_gym/legged_gym/envs/my_robot/my_robot_config.py`
- `legged_gym/legged_gym/envs/go2w/go2w_config.py`
- `legged_gym/legged_gym/envs/go2w/go2w_legged_robot.py`
- `legged_gym/legged_gym/utils/task_registry.py`
- `legged_gym/legged_gym/scripts/train.py`
- `legged_gym/legged_gym/scripts/play.py`
- `rsl_rl/rsl_rl/modules/him_actor_critic.py`
- `rsl_rl/rsl_rl/modules/him_estimator.py`
- `rsl_rl/rsl_rl/algorithms/him_ppo.py`
- `rsl_rl/rsl_rl/runners/him_on_policy_runner.py`
- `rsl_rl/rsl_rl/storage/him_rollout_storage.py`
- `docs/himloco_wheel_leg_training_evaluation.md`

### 当前目标工程

- `README.md`
- `setup.py`（当前 0 字节占位）
- `legged_mjlab/__init__.py`
- `legged_mjlab/envs/__init__.py`
- `legged_mjlab/envs/base/base_config.py`
- `legged_mjlab/envs/base/legged_robot_config.py`
- `legged_mjlab/envs/base/base_task.py`
- `legged_mjlab/envs/base/legged_robot.py`
- `legged_mjlab/test/test_env.py`（当前 0 字节占位）
- `legged_mjlab/utils/helpers.py`、`legged_mjlab/utils/logger.py`（当前 0 字节占位）
- `docs/setup.md`、`docs/设计.md`（当前 0 字节占位）
- `docs/前言.md`（已有设计前言，尚非安装或训练实现文档）
- `resources/robots/unitree_go2/go2_constants.py`
- `resources/robots/unitree_go2/xmls/go2.xml`
- `resources/robots/unitree_go2/xmls/scene_go2.xml`
- `rsl_rl/setup.py`

## 附录 B：代理审查记录

本轮按项目协议派出了 `codebase_explorer`、`robotics_architect`、`ai_slam_dev`、`devops_build_engineer`、`math_verifier`，并在文档编辑后再次派出 `safety_reviewer` 与 `test_sim_generator`。代码库探索、架构、HIM 合同、构建边界和数学/API 分支已返回只读结论；数学分支第一次请求 429，重试后返回风险拒绝意见。安全审查已返回风险阻断意见，结论为未通过（REJECT），不能视为安全签字；coverage 重试也没有返回新增报告。已有 coverage 摘要只用于补齐文档验收清单，不等于测试执行。由此本文不能宣称运行时完成，也不能输出 `[ALL_TESTS_PASSED]`。后续若服务恢复，应优先让独立审查者复核：

1. 所有文件路径和行号；
2. unitree 与 HIMLoco 的维度/调用链描述；
3. mjlab 与 Isaac Gym 的迁移边界；
4. 安装命令是否被错误地表述为已执行；
5. 文档是否误写了目标工程已经可运行。

构建审查还确认：Unitree 的 `mjlab/mujoco-warp`、Torch/CUDA/GPU driver、C++ 部署依赖和动态 `torchrunx` 应分层记录；HIMLoco 的 `legged_gym` 与本地 `rsl_rl` 应分别处理；HIMLoco 的 play 示例包含个人绝对 checkpoint 路径，不能当成通用命令。

### 附录 C：独立覆盖审查摘要

`test_sim_generator` 以只读方式复核了本文的验收覆盖，结论如下：

- baseline 的有效 packaging、唯一包根/资源根、实际 task contract、asset/spec、actuator/contact、manager cfg、观测/动作/奖励/终止、task registry、train/play、checkpoint、reset/timeout 等仍是阻断项。
- 参考项目的 45/6/16、57/6/16 等维度只能作为参考值；目标项目必须从实际 term 表和 runner contract 推导并运行时确认。
- HIM adapter 至少要验证 `[B,D] -> [B,H*D]`、最新帧顺序、partial reset、timeout、action history 和 estimator target；当前目标没有 adapter，也没有运行时证据。
- 未来最小证据包应包含 contract 表/hash、资源构造结果、任务注册/加载 smoke、reset/timeout/HIM history 测试、checkpoint 同配置恢复与错配拒绝、flat/rough 评估指标。
- 本轮没有测试代码、mock、replay、日志或 checkpoint 产物；coverage 审查本身不等于运行时验证。
