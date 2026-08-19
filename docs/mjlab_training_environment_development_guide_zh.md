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

## 4. 目标 `legged_mjlab` 的推荐目录

下面是设计建议，不是当前目录快照。建议保留已有 `resources/robots/` 资产位置，先通过 Python 包装器引用它们，避免第一阶段做无关的资源搬家。

```text
legged_mjlab/
├── pyproject.toml 或 setup.py             # 目标项目的唯一安装入口
├── README.md
├── docs/
│   ├── mjlab_training_environment_development_guide_zh.md
│   ├── architecture.md                    # 后续可拆出稳定架构
│   └── contracts/                         # 后续保存 obs/action/asset 表
├── legged_mjlab/
│   ├── __init__.py
│   ├── assets/
│   │   ├── __init__.py
│   │   └── robots/
│   │       └── unitree_go2/
│   │           ├── __init__.py
│   │           └── go2_constants.py        # EntityCfg/spec/actuator/碰撞
│   ├── tasks/
│   │   ├── __init__.py                     # 导入任务，触发注册
│   │   ├── velocity/
│   │   │   ├── __init__.py
│   │   │   ├── velocity_env_cfg.py         # 共享 manager cfg 工厂
│   │   │   ├── mdp/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── observations.py
│   │   │   │   ├── rewards.py
│   │   │   │   ├── terminations.py
│   │   │   │   ├── commands.py
│   │   │   │   └── curriculums.py
│   │   │   ├── config/go2/
│   │   │   │   ├── __init__.py              # register_mjlab_task
│   │   │   │   ├── env_cfgs.py
│   │   │   │   └── rl_cfg.py
│   │   │   └── rl/
│   │   │       └── runner.py                # 仅任务确有差异时保留
│   │   └── him_velocity/                    # 第二阶段再添加
│   │       ├── history_adapter.py
│   │       ├── him_env_cfg.py
│   │       ├── config/go2/
│   │       └── rl/runner.py
│   └── scripts/
│       ├── train.py
│       ├── play.py
│       └── list_envs.py
├── resources/robots/unitree_go2/            # 当前已有 XML/mesh，作为原始资源
└── rsl_rl/                                  # 暂时保留为独立依赖/参考，先不要复制修改
```

### 4.1 目录规则

1. `assets/robots/<robot>/` 只负责“模型是什么、怎么编译、怎么驱动”；不放 reward 和 command。
2. `tasks/<task>/mdp/` 只放 term 函数；不在里面偷偷注册 task。
3. `tasks/<task>/config/<robot>/env_cfgs.py` 只做组合和机器人命名匹配；不实现物理循环。
4. `config/<robot>/__init__.py` 是 task ID 的唯一注册点；不要在多个模块重复注册同一 ID。
5. `scripts/` 只负责 CLI、设备、日志、checkpoint 和 runner；不要把 reward/observation 实现塞进脚本。
6. HIM 只作为第二阶段扩展；先把普通 actor/critic env 合同跑通。
7. `resources/` 的原始 XML、mesh 与 Python 配置不混放；Python 通过稳定的路径函数引用资源。

### 4.2 当前目标项目需要先解决的结构阻塞

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

---

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

明确以下选择之一，并全项目统一：

- 选择 `legged_mjlab` 作为 Python 包根，资源 wrapper 位于 `legged_mjlab/assets/robots`；或
- 选择 `src` 作为包根，所有任务和资源 wrapper 都位于 `src`。

[设计建议] 对当前目标更建议第一种，因为根包名已经是 `legged_mjlab`，同时将原始 MJCF/mesh 保留在 `resources/robots`。不要让 `resources/robots/.../go2_constants.py` 继续引用不存在的 `src.SRC_PATH`。

### 必须固定

```text
PROJECT_ROOT
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
tasks/velocity/config/go2/env_cfgs.py
  -> 调用共享工厂
  -> 填 EntityCfg
  -> 填 base/feet/contact/joint/site/geom 名称
  -> 配置 rough/flat 差异
  -> 配置 play overrides

tasks/velocity/config/go2/rl_cfg.py
  -> actor/critic network
  -> PPO hyperparameters
  -> experiment name / save interval / rollout length

tasks/velocity/config/go2/__init__.py
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

1. `tasks/__init__.py` 是否被入口 import。
2. `config/<robot>/__init__.py` 是否执行注册。
3. task ID 是否拼写一致。
4. 是否把 `mdp` 或 `utils` 错误加入/排除 import black list。

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
- [ ] 数学、硬件安全审查：对应代理分支本轮因 429 未返回，不能打勾。
- [ ] 训练/仿真/安装验证：本轮明确不执行，不能打勾。

---

## 11. 下一轮实际开发顺序（仍然不在本轮实施）

1. 先决定目标包根：建议 `legged_mjlab`，并把资源路径策略写成单独契约。
2. 以 Go2 flat 为第一任务，只使用标准 mjlab manager 和标准 RSL runner。
3. 让任务注册、配置加载、单环境 play、少量环境训练形成最小闭环。
4. 加入 contact sensor 和 terrain scan，再加入 rough terrain。
5. 将 terrain randomization、friction、COM、push、encoder noise 拆成事件并逐项记录。
6. 加入 curriculum 和评估指标。
7. 再定义 HIM history adapter，先复现 45/6 或目标自定义维度的静态 shape 合同。
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

本轮按项目协议并发派出了 `codebase_explorer`、两个 `ai_slam_dev` 阅读分支、架构分支，以及 safety/math/coverage/build 四个审查分支。探索、阅读、架构、安全和数学分支受到 `429 Too Many Requests` 限流；coverage 与 build 分支成功返回只读审查，确认本文不能宣称运行时完成，也不能输出 `[ALL_TESTS_PASSED]`。因此本文没有伪造失败分支结论，也没有把未运行的测试标成通过。后续若服务恢复，应优先让独立审查者复核：

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
