# 训练问题三栈对比报告

这份文档要解决的问题很直接：现在训练问题很大时，先不要急着改 reward、加随机化或换网络，而是先把三条训练链路的接口边界对齐。我们对比的是 `/home/kk/github/unitree_rl_mjlab`、`/home/kk/legged_mjlab/.venv/lib64/python3.11/site-packages/mjlab`、`/home/kk/github/HIMLoco`。它们表面上都在做 legged locomotion，但训练入口、环境 API、观测布局、动作空间、PPO runner、terminal obs 语义和部署边界都不同；如果混用，症状通常不是“效果差一点”，而是启动失败、shape 错、能跑但 estimator 学错、或者 sim2sim/真机动作完全错位。

代码观察：三个路径都存在。`unitree_rl_mjlab` 是基于旧版 `mjlab` 的 Unitree 任务扩展；安装态 `mjlab==1.6.0` 是通用 MuJoCo-Warp manager-based 框架；`HIMLoco` 是 Isaac Gym / `legged_gym` + 自定义 HIM-RSL-RL 体系。当前 `/home/kk/legged_mjlab` 还叠加了一个关键事实：`rsl-rl-lib==5.4.2` 虽然在 venv 中存在，但 Python 实际 import 到的顶层 `rsl_rl` 是当前项目有意使用的 editable 本地分叉 `/home/kk/legged_mjlab/rsl_rl`，版本显示为 `1.0.2`。这对当前 `him_go2` 是预期选择；风险只出现在把安装版 `mjlab` runner/console script 和这个本地 old/HIM-style `rsl_rl` 混跑时。

未执行/未验证：本文没有启动长训练、没有跑 MuJoCo/Isaac Gym rollout、没有做 GPU 性能验证、没有改任何训练代码。文中的“代码观察”来自静态源码与短导入/版本检查；“这里是推断”用于说明训练表现层面的风险。

## 一页结论

| 优先级 | 结论 | 对训练的影响 |
| --- | --- | --- |
| P0 | `unitree_rl_mjlab` 声明依赖 `mjlab==1.2.0`、`mujoco-warp==3.5.0`，当前 venv 是 `mjlab==1.6.0`、`mujoco-warp==3.11.0` | 直接用当前 venv 跑 unitree 任务会遇到旧 API 缺失，例如 `mjlab.utils.os.update_assets` |
| P0 | 当前项目明确使用本地 editable `rsl_rl==1.0.2`；仅当使用安装版 `mjlab.scripts.train` / `MjlabOnPolicyRunner` 时，才会和 `mjlab==1.6.0` 期望的新式算法 API 发生兼容风险 | 对当前 `him_go2` 不是错误；对 stock `mjlab` runner，`alg.save/load/get_policy/as_onnx` 可能不存在 |
| P0 | `mjlab.rl.RslRlVecEnvWrapper` 返回 `TensorDict` 风格 obs group；HIMLoco old runner 期待 flat tensor 和 privileged tensor | 不能把 HIMLoco 的 `HIMOnPolicyRunner/HIMPPO/HIMActorCritic` 直接接到默认 mjlab wrapper |
| P0 | HIMLoco GO2W 是 16 action，当前项目 `him_go2` 是 12 action，unitree Go2 也是 12 action 但 obs/reward/runner 语义不同 | GO2W policy、GO2 policy、当前 `him_go2` policy 不能互换 |
| P1 | unitree velocity actor 不含 `base_lin_vel`，安装版 mjlab velocity actor 含 `base_lin_vel`，HIMLoco actor 通过 HIM estimator 间接估计速度 | 观测维度可能能凑齐，但语义和切片会错 |
| P1 | HIMLoco 的 history 是 frame-major，安装版 mjlab ObservationManager 的 history 更接近 term-major | 给 HIMEstimator 喂错 history 排列时，网络维度可能对，但时间语义错 |
| P1 | reward 语义不同：mjlab 默认按 `step_dt` 缩放 reward；HIMLoco 可启用 `only_positive_rewards=True`；unitree 重写了部分 reward 公式 | 不能直接复制 reward scale，也不能用曲线数值直接横向比较 |
| P1 | terminal/timeout 语义不同：mjlab raw env 分离 `terminated/truncated` 且默认 auto reset；HIMLoco HIM runner 需要 reset 前 terminal privileged obs | 如果 terminal critic obs 被 reset 后 obs 污染，estimator target 会错 |
| P2 | HIMLoco sim2sim GO2W 配置存在绝对路径 `/home/kk/HIMLoco/...`，当前代码路径是 `/home/kk/github/HIMLoco/...`；`wheel_vel_scale=10.0` 与训练 `vel_scale=20.0` 不一致 | sim2sim 结果不可信，严重时直接加载错误文件或轮速输出缩放错 |
| P2 | unitree deploy 的 `bad_orientation()` 当前直接 `return false`，而 RL 状态机注册了姿态检查 | 真机跌倒/大姿态异常保护边界不足，不能当安全 stop 依赖 |

所以，当前最像根因的不是某一项 reward，而是三套训练接口混用：旧 `unitree_rl_mjlab` 任务代码、安装版 `mjlab==1.6.0` 框架、HIMLoco 旧式 `legged_gym/rsl_rl` 算法，被放在同一个调试语境里了。

## 三个路径分别是什么

| 路径 | 类型 | 主要入口 | 训练栈 |
| --- | --- | --- | --- |
| `/home/kk/github/unitree_rl_mjlab` | Unitree 机器人任务扩展源码 | `scripts/train.py`、`scripts/play.py`、`scripts/list_envs.py` | MuJoCo/MuJoCo-Warp + `mjlab.ManagerBasedRlEnv` + `mjlab.rl.MjlabOnPolicyRunner` |
| `/home/kk/legged_mjlab/.venv/lib64/python3.11/site-packages/mjlab` | 已安装 `mjlab==1.6.0` 框架包 | `.venv/bin/train`、`.venv/bin/play`、`.venv/bin/list-envs` | MuJoCo-Warp + manager-based env + 新式 RSL-RL config / TensorDict obs groups |
| `/home/kk/github/HIMLoco` | Isaac Gym/legged_gym 项目 | `legged_gym/legged_gym/scripts/train.py`、`play.py` | Isaac Gym PhysX + `legged_gym` imperative env + old `rsl_rl` + HIM estimator |

关键点：这三个路径不是“同一套代码的三个版本”。`unitree_rl_mjlab` 与安装版 `mjlab` 同属 MuJoCo-Warp manager-based 思路，但版本 API 已经拉开；`HIMLoco` 则是另一条 Isaac Gym/PhysX 训练链，环境 step contract 和算法输入布局完全不同。

当前项目 `/home/kk/legged_mjlab` 还提供了自定义入口 `legged_mjlab/scripts/train.py`，它会先 `import legged_mjlab.envs`，再通过本地 `task_registry` 创建 `him_go2` 等任务。相反，`.venv/bin/train` 是安装版 `mjlab.scripts.train:main` 的 console script，不会自动注册当前项目里的 `him_go2`。如果目标是跑当前项目任务，应该优先使用：

```bash
cd /home/kk/legged_mjlab
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python legged_mjlab/scripts/train.py --task him_go2 --headless
```

而不是裸跑：

```bash
.venv/bin/train him_go2
```

## 入口和注册链路

### unitree_rl_mjlab

```text
/home/kk/github/unitree_rl_mjlab/scripts/train.py
  -> import src.tasks
  -> src/tasks/**/config/**/__init__.py 注册 Unitree-* task
  -> load_env_cfg/load_rl_cfg
  -> ManagerBasedRlEnv
  -> RslRlVecEnvWrapper / MjlabOnPolicyRunner
  -> VelocityOnPolicyRunner 或 MotionTrackingOnPolicyRunner
```

代码观察：`setup.py` 明确写着 `mjlab==1.2.0` 和 `mujoco-warp==3.5.0`。多个机器人 asset constants 从 `mjlab.utils.os import update_assets`，但安装版 `mjlab==1.6.0` 中没有观察到该 symbol。这里不是训练超参问题，而是导入期 API 断裂。

建议：如果要复现 unitree 原始任务，先建立匹配它的独立环境；如果要把它迁到 `mjlab==1.6.0`，要先处理 `update_assets`、训练脚本参数、runner save/export API、task registry 命名，再谈 reward。

### 安装版 mjlab 1.6.0

```text
.venv/bin/train
  -> mjlab.scripts.train:main
  -> mjlab.tasks 自动导入内置任务
  -> register_mjlab_task / load_env_cfg / load_rl_cfg
  -> ManagerBasedRlEnv
  -> RslRlVecEnvWrapper
  -> MjlabOnPolicyRunner
```

代码观察：`mjlab-1.6.0.dist-info/entry_points.txt` 注册了 `train`、`play`、`list-envs`、`viz-nan` 等 console scripts。`METADATA` 里要求 `rsl-rl-lib==5.4.2`，该 dist-info 确实存在；但当前 Python 顶层 `import rsl_rl` 解析到 `/home/kk/legged_mjlab/rsl_rl/__init__.py`，也就是当前项目有意使用的 editable HIM/old-style 分叉。

这意味着两个事实同时成立：

1. 新版 RSL-RL 发行包在 venv 里存在；
2. 当前项目实际导入的 `rsl_rl` 是本地 old/HIM-style tree，这是 `him_go2` 路线的预期选择。

需要注意的不是“本地 `rsl_rl` 用错了”，而是入口边界：当前项目脚本应该继续使用本地 `rsl_rl`；如果改用 `.venv/bin/train` 或安装版 `mjlab/rl/runner.py`，该 runner 会按新版算法 API 调用 `self.alg.save()`、`self.alg.load()` 和 `self.alg.get_policy().as_onnx()`。短检查显示本地 `PPO` class 没有 `save` 和 `load` 方法，但本地 `OnPolicyRunner/HIMOnPolicyRunner` 自己实现了 runner 级 `save/load`，所以当前项目 `him_go2` 路线不需要 `PPO.save()`。

### HIMLoco

```text
/home/kk/github/HIMLoco/legged_gym/legged_gym/scripts/train.py
  -> import legged_gym.envs
  -> task_registry.register("go2w", Go2w, GO2WRoughCfg, Go2wRoughCfgPPO)
  -> TaskRegistry.make_env
  -> Go2w(LeggedRobot)
  -> HIMOnPolicyRunner
  -> HIMPPO / HIMActorCritic / HIMEstimator
```

代码观察：HIMLoco 的 `go2w_config.py` 写明 `num_one_step_observations = 3 + 3 + 3 + 16 + 16 + 16 = 57`，`num_observations = 57 * 6 = 342`，`num_actions = 16`。它依赖 `isaacgym`，不能直接当成 `mjlab` 任务在当前 MuJoCo-Warp venv 中跑。

注意：HIMLoco 的 `Go2wRoughCfgPPO.runner.resume = True`。如果没有有效 checkpoint 路径，默认训练路径可能走 resume/load 分支；这和从零训练的期望不一致。

## 环境 API 和 step contract

| 维度 | unitree_rl_mjlab | mjlab 1.6.0 | HIMLoco |
| --- | --- | --- | --- |
| raw env 类型 | `ManagerBasedRlEnv` | `ManagerBasedRlEnv` | `LeggedRobot` / Isaac Gym task |
| raw step 返回 | 继承 mjlab 风格 | `(obs_dict, reward, terminated, truncated, extras)` | legged_gym 风格，HIM runner 需要更多信息 |
| RSL wrapper 返回 | 依赖当前 `mjlab.rl` | `(TensorDict(obs_dict), rew, dones, extras)` | flat obs / privileged obs / infos |
| HIM runner 需要 | 不原生支持 | 默认 wrapper 不支持 | `(obs, privileged_obs, rewards, dones, infos, termination_ids, termination_privileged_obs)` |
| auto reset | mjlab 默认 `auto_reset=True` | `auto_reset=True`，可配置 false | reset 逻辑在 legged_gym env 内部管理 |
| timeout 信息 | wrapper 可能写 `extras["time_outs"]` | `is_finite_horizon=False` 时用 `truncated` | `send_timeouts=True`，PPO 在 timeout bootstrap |

这里是推断：如果当前训练表现是启动即报错、runner 初始化失败、保存时报错，优先查 `rsl_rl` 版本和 wrapper contract；如果训练能跑但 estimator loss、速度估计或动作震荡异常，优先查 obs layout、history 顺序和 terminal privileged obs。

当前项目的 `HIMRslRlWrapper` 已经在补 HIM 适配：它强制 6 帧历史、单帧 45 维、总 actor obs 270 维；它会把 `env.cfg.auto_reset=False`，手动保存 terminal privileged obs，再返回 legacy HIM runner 需要的 7 元组。这是正确方向，但它也说明一个边界：默认安装版 `mjlab.rl.RslRlVecEnvWrapper` 不是 HIMLoco runner 的直接替代品。

## 观测空间：最大的错位来源

### 三者 actor obs 对比

| 栈 | actor 当前帧内容 | history | 维度倾向 | 关键风险 |
| --- | --- | --- | --- | --- |
| unitree Go2 velocity | `base_ang_vel`、`projected_gravity`、`command`、`phase`、`joint_pos`、`joint_vel`、`actions`、`height_scan` | `history_length=1` | rough 带 height scan，flat 删除 height scan | 不含 `base_lin_vel`，但加了 gait phase |
| mjlab 1.6 velocity | `base_lin_vel`、`base_ang_vel`、`projected_gravity`、`joint_pos`、`joint_vel`、`actions`、`command`、`height_scan` | 默认 group history，可配置 | 默认 actor 开头含 `base_lin_vel` | 和 unitree actor 前几段直接错位 |
| HIMLoco GO2W | `commands[:3]`、`base_ang_vel`、`projected_gravity`、`dof_err(16)`、`dof_vel(16)`、`actions(16)` | 6 帧 frame-major | 57 * 6 = 342 | 依赖 16 action 和 frame-major 历史 |
| 当前项目 `him_go2` | `commands/base_ang_vel/gravity/dof_pos/dof_vel/actions` 的 12-DOF HIM 风格 | 6 帧 frame-major | 45 * 6 = 270 | 不是 HIMLoco GO2W；没有 wheel 维度 |

关键点：观测维度不是唯一问题，顺序才是更大的问题。举例：假设两个策略输入都是 270 维，只要前 3 维在一个环境里是 `commands`，另一个环境里是 `base_ang_vel`，策略看到的物理意义就完全变了。此时训练不收敛不是网络太小，而是输入合同已经错。

### privileged / critic obs

unitree 和 mjlab 的 critic group 是 manager term 拼接。unitree critic 在 actor terms 上加 `base_lin_vel`、`foot_height`、`foot_air_time`、`foot_contact`、`foot_contact_forces`；安装版 mjlab 也使用 critic group，但 foot height 通过 `TerrainHeightSensorCfg` 更明确地表达地形相对高度。

HIMLoco 的 privileged obs 更特殊。GO2W 里当前观测先拼出 57 维 actor frame，再追加 `base_lin_vel(3)`、`disturbance(3)`、187 个 height samples、12 维 contact force，形成 `num_one_step_privileged_obs = 57 + 3 + 3 + 187 + 12 = 262`。`HIMEstimator.update()` 硬编码从 `next_critic_obs[:, num_one_step_obs:num_one_step_obs+3]` 取下一步 base velocity target，又从 `next_critic_obs[:, 3:num_one_step_obs+3]` 取 next obs target。也就是说，它不是“随便给 critic obs 都行”，它依赖具体切片位置。

所以，如果要把 HIMLoco 思路迁到 mjlab，不能只把 class name 改成 `HIMActorCritic`。必须显式定义：

```text
actor_history = [frame_t, frame_t-1, ..., frame_t-5]
one_step_obs = [command, base_ang_vel, projected_gravity, dof_pos_rel, dof_vel, last_action]
critic_one_step = [one_step_obs, base_lin_vel, disturbance_or_zero, height_samples, contact_forces]
estimator_vel_target = critic_one_step[num_one_step_obs : num_one_step_obs + 3]
```

如果其中任一段缺失或位置变化，HIM estimator 会学到错误目标。

## 动作空间、关节顺序和控制

| 维度 | unitree Go2 | mjlab 1.6 velocity base | HIMLoco GO2W | 当前项目 `him_go2` |
| --- | --- | --- | --- | --- |
| 动作维度 | 12 | 由机器人 actuator 决定，base cfg 默认 `JointPositionActionCfg` | 16 | 12 |
| 控制类型 | 关节位置 target | 关节位置 target | 腿关节位置 + 轮关节速度 | 关节位置 target |
| action scale | Go2 base cfg 为 `0.25` | base 默认 `0.5`，机器人可覆盖 | 腿 `0.25`，轮速度 `vel_scale=20.0` | `action_scale=0.25`，hip reduction 后 hip 为 `0.125` |
| decimation / dt | `0.005 * 4 = 0.02s` | 常见 `0.005 * 4 = 0.02s` | `0.005 * 4 = 0.02s` | `0.005 * 4 = 0.02s` |
| 顺序保护 | unitree deploy YAML 有 joint map | manager 根据 actuator target 顺序 | 训练/部署/sim2sim 都依赖 16 维顺序 | `_build_actions(... preserve_order=True)` |

HIMLoco GO2W 的核心差异是轮子：`Go2w._compute_torques()` 对腿关节使用位置目标，对 wheel indices 使用速度目标 `actions * vel_scale`。这不是普通四足 12-DOF 的动作空间扩展一点点，而是控制语义换了一半。把 GO2W 的 policy 用到 12-DOF Go2，或者把 12-DOF Go2 policy 用到 GO2W，都应该先视为无效。

当前项目 `him_go2` 的动作空间更像 12-DOF 腿式 Go2，并且 `_build_actions()` 对 hip 使用 `action_scale * hip_reduction`，即 hip target scale 为 `0.125`，其他关节为 `0.25`。这与 HIMLoco deploy/policy/himloco 的 12-DOF配置接近，但不是 GO2W。

## reward 与 terrain：不要直接抄 scale

### reward 结构差异

| 栈 | 典型 reward 语义 | 高风险点 |
| --- | --- | --- |
| unitree velocity | `track_linear_velocity`、`track_angular_velocity`、`body_orientation_l2`、`variable_posture`、`feet_gait`、`feet_clearance`、`stand_still`、`joint_pos_limits` 等 | `phase/feet_gait` 强加步态先验；`foot_height` 使用世界系 z；termination 权重大 |
| mjlab 1.6 velocity | `base_lin_vel` actor 可见；有 `upright`、terrain-relative `foot_height`、`out_of_terrain_bounds` 等更完整 terms | actor obs 与 unitree 不同；reward 默认按 dt 缩放 |
| HIMLoco GO2W | tracking、orientation、base height、collision、stand still、terrain crossing、action rate、torque 等 | `only_positive_rewards=True` 会裁剪总 reward；轮速和 action delay 影响 reward 可解释性 |
| 当前 `him_go2` | 更接近 12-DOF HIM 风格，`only_positive_rewards=True`，大量 DR 默认关闭 | `foot_clearance=-0.01` 是 penalty，terrain 默认 plane 但 `measure_heights=True` |

这里是推断：如果训练前期频繁摔倒，unitree 的 `is_terminated=-200` 在 dt 缩放后仍会给明显终止惩罚；HIMLoco 的 `only_positive_rewards=True` 会先避免总 reward 被大量负项压穿，再追加 termination 类惩罚。两者 early learning 的 reward 分布不一样，所以不能看某个 scale 数值相同就认为训练信号相同。

### terrain-relative foot height

unitree Go2 的 `foot_height()` 使用 site world z。安装版 mjlab velocity 引入 `TerrainHeightSensorCfg`，critic/reward 可以使用 terrain-relative foot height。在平地上这两个值可能近似可用；到了 rough terrain，世界 z 高不代表脚相对地面高，特别是在坡、台阶、障碍附近。若训练问题表现为抬脚、踩边、楼梯上不去，建议优先查这一项。

建议的小步实验不是一次性搬完整 reward，而是先验证：

1. 平地任务关闭 height scan 和 terrain curriculum，确认 12-DOF policy 能稳定站/走；
2. rough terrain 只打开 height scan，观察 obs finite 和 critic height term；
3. 再把 foot clearance 从 world z 改成 terrain-relative 定义；
4. 最后才加复杂 gait/terrain curriculum。

## domain randomization 差异

| 随机化项 | unitree_rl_mjlab | mjlab 1.6 | HIMLoco GO2W | 当前 `him_go2` |
| --- | --- | --- | --- | --- |
| friction | foot friction startup，常见范围 0.3-1.2 | manager event，可支持更细粒度 geom friction | `randomize_friction=True`，0.8-1.2 | 默认 False，配置范围 0.3-1.6 |
| COM/payload | base COM，Go2 常见 ±0.05 | manager event | payload [-1,2] kg、COM ±0.15 | 默认 False，payload [-1,3]、COM ±0.05 |
| link mass | 部分任务配置 | 可通过 event 支持 | `randomize_link_mass=True` | 默认 False |
| motor strength / gains | unitree 侧较轻 | 可通过 actuator/event 扩展 | motor strength、Kp/Kd 都随机化 | 默认 False |
| observation/action latency | unitree 基础没有 HIMLoco 那么重 | 需要自定义 buffer/event | action delay=True | 当前配置有字段但默认 False |
| push/disturbance | interval push | interval event | disturbance + push | 默认 False |

HIMLoco 的 DR 明显更重，特别是 action delay、motor strength、Kp/Kd 和持续外部扰动。这里是推断：如果直接把 HIMLoco 的 DR 强度搬到当前 `him_go2`，在基础接口还没稳定时只会让问题不可诊断。训练排查应该从最小任务开始：先关掉 DR，证明 reset/step/obs/reward/runner/save 都对，再一次只打开一个扰动源。

## PPO、runner 和 HIM estimator

### 普通 PPO 与 HIM PPO 的接口差异

HIMLoco 的 `HIMOnPolicyRunner` 不是普通 `OnPolicyRunner` 换个类名。它在 rollout 中做了几件额外事情：

```text
obs = env.get_observations()
privileged_obs = env.get_privileged_observations()
actions = HIMPPO.act(obs, critic_obs)
obs, privileged_obs, rewards, dones, infos, termination_ids, termination_privileged_obs = env.step(actions)
next_critic_obs = critic_obs.clone()
next_critic_obs[termination_ids] = termination_privileged_obs
HIMPPO.process_env_step(rewards, dones, infos, next_critic_obs)
```

`HIMPPO` 的 storage 比普通 rollout storage 多了 `next_critic_observations`，训练时 `HIMEstimator.update(obs_batch, next_critic_obs_batch)` 先更新 estimator，再计算 PPO surrogate/value loss。这个设计的重点是：estimator target 需要下一步 critic obs，尤其 terminal env 需要 reset 前的 terminal privileged obs。

### 当前最大算法接口风险

| 风险 | 触发条件 | 表现 |
| --- | --- | --- |
| 安装态 `MjlabOnPolicyRunner` 与本地 old-style PPO API 不匹配 | 使用 `.venv/bin/train` 或安装版 `mjlab/rl/runner.py`，同时 `import rsl_rl` 解析到当前项目本地分叉 | 对 stock `mjlab` runner，保存、加载或导出可能报 `AttributeError`；对当前 `him_go2` 入口不是问题 |
| TensorDict 喂给 old storage | 直接用安装版 `RslRlVecEnvWrapper` 接 old `OnPolicyRunner/HIMOnPolicyRunner` | storage 创建或 actor forward shape/type 错 |
| terminal obs 污染 | `auto_reset=True` 且没有 reset 前 terminal privileged obs | estimator 学到下一 episode 初始状态，loss 和策略都异常 |
| history 排列错 | 直接用 mjlab ObservationManager history 给 HIMEstimator | 维度可能正确，但 temporal order 和 term order 错 |
| critic 切片错 | critic group term 顺序不同，却沿用 `next_critic_obs[:, 45:48]` 或 `[:,57:60]` | velocity target 错，latent 学歪 |

当前项目的 `HIMRslRlWrapper` 已经处理了部分问题：强制 actor frame 宽度、检查 finite、禁用 native auto reset、返回 7 元组、写 `infos["time_outs"]`、记录 `termination_privileged_obs`。这说明当前修复方向是围绕 HIM contract 做 adapter，而不是让 HIM runner 直接吃安装版 mjlab wrapper。

## 版本与依赖风险

### 已核对事实

```text
当前 venv metadata:
  mjlab          1.6.0
  legged_mjlab  0.0.1 editable -> /home/kk/legged_mjlab
  rsl_rl        1.0.2 editable -> /home/kk/legged_mjlab/rsl_rl
  rsl-rl-lib    5.4.2 installed
  mujoco        3.11.0
  mujoco-warp   3.11.0
  torch         2.13.0

运行时 import:
  rsl_rl.__file__ = /home/kk/legged_mjlab/rsl_rl/__init__.py
  PPO has save/load = False/False
```

这不是“缺少新版 `rsl-rl-lib`”，也不是“当前项目不该用本地 `rsl_rl`”。更准确地说：当前项目按设计解析到本地 editable old/HIM tree；如果用 `.venv/bin/train` 跑安装版 `mjlab` 内置任务，它会 import `mjlab.scripts.train`，再进入 `mjlab.rl`，然后 `mjlab.rl` 里 `from rsl_rl...` 很可能拿到本地 `rsl_rl`。这条 stock `mjlab` 链路需要单独隔离或适配。

### 建议隔离方式

| 目标 | 建议环境 | 原因 |
| --- | --- | --- |
| 跑 unitree 原始任务 | 单独 venv，按 `unitree_rl_mjlab/setup.py` 锁到 `mjlab==1.2.0`、`mujoco-warp==3.5.0` | 避免旧 API 与 1.6.0 断裂 |
| 跑安装版 mjlab 内置任务 | 单独 venv，或确认安装版 `MjlabOnPolicyRunner` 已适配当前本地 `rsl_rl` API | stock `mjlab` runner 默认按新式算法 API 工作 |
| 跑当前项目 `him_go2` | 当前项目 venv + `python legged_mjlab/scripts/train.py --task him_go2` | 继续使用本地 `/home/kk/legged_mjlab/rsl_rl`、本地 HIM wrapper 与 old HIM runner 适配，这是正确路径 |
| 跑 HIMLoco `go2w` | Isaac Gym 专用环境，从 `/home/kk/github/HIMLoco/legged_gym` 启动 | `HIMLoco` 依赖 Isaac Gym/PhysX，不能用 mjlab venv 直接替代 |

## sim2sim 和部署安全边界

### Unitree deploy

代码观察：`unitree_rl_mjlab/deploy/include/isaaclab/envs/mdp/terminations.h` 中 `bad_orientation()` 当前直接返回 false；Go2/G1/H1_2/R1/A2 等 RL 状态里又注册了 `bad_orientation(env, 1.0)` 作为切 Passive 条件。结果是：状态机看起来有姿态保护，但该保护函数实际失效。

真机前必须补的检查：

1. policy ONNX 输入维度、obs 顺序、action 输出维度离线检查；
2. joint id map 与 SDK 电机顺序逐关节确认；
3. default pose、PD gains、action scale 与训练 metadata/YAML 对齐；
4. 恢复 roll/pitch 或 projected gravity 姿态停止逻辑；
5. 对动作 target 加离线位置边界检查，而不是只依赖 MuJoCo 训练时 actuator limit。

### HIMLoco sim2sim / deploy

代码观察：`sim2sim/go2w/config.yaml` 中路径写为 `/home/kk/HIMLoco/...`，但本次对比代码实际在 `/home/kk/github/HIMLoco/...`。同一文件里 `wheel_vel_scale: 10.0`，而训练 `GO2WRoughCfg.control.vel_scale = 20.0`。这两个差异足以让 sim2sim 结论不可靠。

代码观察：`deploy/policy/himloco/config.yaml` 有 `joint_direction`，但 C++ `RefreshStateForControl()` 和 `SetCommand()` 路径主要读取 `joint_mapping`；如果底层 adapter 或电机固件没有处理方向符号，那么观测和命令方向都可能错。

上硬件前最低限度要做：

1. 空载逐关节正方向检查；
2. `joint_mapping` 与 `joint_direction` 同时验证，不只看 mapping；
3. policy 输入 `[1, history_dim]`、输出 `[1, num_actions]` 离线跑一次；
4. torque pre-clamp 与实际硬件 PD 输出差异评估；
5. SafeStop、姿态保护、失联保护和 NaN 保护独立触发测试。

## 最小验证矩阵

这些命令用于定位接口问题，不用于长训练。建议加 `PYTHONDONTWRITEBYTECODE=1`，避免只读检查生成新的 `__pycache__`。

| 验证项 | 命令 / 伪命令 | 预期 | 失败时说明 |
| --- | --- | --- | --- |
| 当前 venv 包版本 | `cd /home/kk/legged_mjlab && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import importlib.metadata as md; [print(n, md.version(n)) for n in ["mjlab","legged_mjlab","rsl_rl","rsl-rl-lib","mujoco","mujoco-warp","torch"]]'` | 看见 mjlab/rsl_rl/rsl-rl-lib 版本 | 确认 metadata，不代表 import 解析正确 |
| 当前 import 路径 | `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import mjlab, legged_mjlab, rsl_rl; print(mjlab.__file__); print(legged_mjlab.__file__); print(rsl_rl.__file__)'` | 对当前 `him_go2`，`rsl_rl` 指向本地 editable 是预期结果；对 stock `mjlab==1.6.0` runner，则需确认 API 兼容或隔离环境 | 目标入口与 `rsl_rl` API 不匹配才是 P0 |
| console script 指向 | `sed -n '1,20p' /home/kk/legged_mjlab/.venv/bin/train` | 应显示 `from mjlab.scripts.train import main` | 说明裸 `train` 不会注册当前 `him_go2` |
| unitree 注册检查 | `cd /home/kk/github/unitree_rl_mjlab && PYTHONDONTWRITEBYTECODE=1 /home/kk/legged_mjlab/.venv/bin/python scripts/list_envs.py` | 在兼容环境应列出 `Unitree-*` 任务 | 当前 venv 若报 `update_assets` 缺失，是版本不兼容，不要继续训练 |
| 当前项目帮助 | `cd /home/kk/legged_mjlab && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python legged_mjlab/scripts/train.py --help` | 能看到当前项目 argparse/任务参数 | 如果这里失败，先修当前入口 |
| `him_go2` reset/step shape | 写一个 2 env、3 step smoke：检查 actor `[2,270]`、critic `[2,235]`、action `[2,12]`、reward finite | shape/finite 都通过 | 任一 shape 不符先停，不跑长训练 |
| HIM terminal obs | 人为设置短 episode，检查 7 元组返回、`termination_ids`、`termination_privileged_obs`、`infos["time_outs"]` | timeout bootstrap 信息存在，terminal privileged obs 来自 reset 前 | 缺失会污染 estimator target |
| plain PPO smoke | 不接 HIM，跑一个最小内置/简化任务 1 iteration、2 steps/env | 验证 obs -> storage -> update -> save | 如果 save 报错，说明 runner/alg 版本仍错 |
| HIM estimator 切片 assert | 在 wrapper/runner 前加断言：`one_step_obs_dim=45`、`history=6`、`critic_dim=235`；GO2W 则是 57/342/262 | 所有维度都显式等于配置 | 维度靠推断会掩盖切片错 |
| HIMLoco GO2W 注册 | 在 Isaac Gym 环境：`cd /home/kk/github/HIMLoco/legged_gym && PYTHONDONTWRITEBYTECODE=1 python legged_gym/scripts/train.py --help` | 能 import `isaacgym` 并显示 `--task` | 当前 mjlab venv 缺 Isaac Gym 时不要继续 |
| sim2sim 配置静态检查 | 解析 `/home/kk/github/HIMLoco/sim2sim/go2w/config.yaml`，检查路径存在、`num_actions=16`、`num_obs_encoder=342`、`wheel_vel_scale=20.0` 是否与训练一致 | 所有路径和 scale 对齐 | 当前观察到路径和 wheel scale 不一致 |
| policy export shape | 对 ONNX/JIT 模型离线输入 dummy tensor，检查输出 action dim | GO2: 12；GO2W: 16；当前 `him_go2`: 12 | 不一致不能 sim2sim/上硬件 |

## 建议排查顺序

### 第一步：先把要跑的栈说死

不要同时问“unitree 怎么调”“mjlab 1.6 怎么跑”“HIMLoco 怎么迁移”。先选一个目标：

| 目标 | 应使用入口 | 不应使用 |
| --- | --- | --- |
| 当前项目 `him_go2` | `/home/kk/legged_mjlab/.venv/bin/python legged_mjlab/scripts/train.py --task him_go2` | `.venv/bin/train him_go2` |
| unitree Go2/G1 | `/home/kk/github/unitree_rl_mjlab/scripts/train.py Unitree-Go2-Flat/Rough`，且环境版本匹配 1.2.0 | 当前 `mjlab==1.6.0` venv 直接跑 |
| 安装版 mjlab 内置任务 | `.venv/bin/train Mjlab-Velocity-*`，并使用干净 stock mjlab API 环境，或显式适配本地 `rsl_rl` | 当前项目 HIM runner |
| HIMLoco GO2W | Isaac Gym 环境下 `legged_gym/scripts/train.py --task go2w` | MuJoCo-Warp venv |

### 第二步：先验证接口，不调 reward

最小检查顺序：

1. `import` 路径：`mjlab.__file__`、`rsl_rl.__file__`、入口脚本是否是预期的；
2. task 注册：能列出目标 task；
3. env reset：obs dict/flat tensor 是否符合 runner 预期；
4. env step：action shape、reward shape、terminated/truncated/dones shape 是否正确；
5. terminal obs：done env 是否能拿到 reset 前 privileged obs；
6. 1 iteration dry-run：只跑极少步，确认 storage/update/save/export 不崩；
7. 再开始 reward、curriculum、domain randomization。

### 第三步：如果目标是当前 `him_go2`

当前 `him_go2` 的合同应当被写成硬断言：

```text
num_actions = 12
num_one_step_observations = 45
history_length = 6
num_observations = 270
num_privileged_obs = 45 + 3 + 187 = 235
policy_dt = sim.dt * decimation = 0.005 * 4 = 0.02s
```

建议先跑 plane、DR 全关、`num_envs=2/16` 的 smoke，再扩大到 1024 env。不要一上来打开 rough terrain、push、latency、mass/friction randomization。每次只改一个因素，并记录以下指标：

```text
obs finite / critic finite / reward finite
termination rate / timeout rate
mean episode length
predicted velocity vs target velocity
action mean/std and clipping ratio
reward terms: tracking, orientation, stand_still, collision, dof_pos_limits
```

### 第四步：如果目标是迁移 HIMLoco 思路到 mjlab

必改项不是 reward，而是 adapter contract：

1. 当前项目已选择本地 old HIM flat tensor 路线；迁移 HIMLoco 思路时应围绕该 contract 做 adapter，不要混入新版 TensorDict runner contract；
2. old HIM 路线要保留 `num_obs`、`num_privileged_obs`、`num_one_step_obs`、`num_actions` 属性；
3. `env.step()` 必须返回 HIM runner 需要的 7 元组，或同步改 runner；
4. actor one-step layout 固定为 `[command, base_ang_vel, projected_gravity, dof_pos_rel, dof_vel, last_action]`；
5. critic layout 中 `base_lin_vel` 的位置必须与 `HIMEstimator.update()` 切片一致；
6. history 必须是 frame-major：`[t, t-1, ..., t-5]`；
7. terminal privileged obs 必须来自 reset 前；
8. 训练接口稳定后再恢复 ONNX/JIT 导出和 sim2sim。

## 最值得优先修的点

1. **守住入口边界**：当前 venv 同时包含安装版 `mjlab==1.6.0`、新版 `rsl-rl-lib==5.4.2`、editable 本地 `rsl_rl==1.0.2`。这对当前 `him_go2` 是预期配置；只有当你把安装版 `mjlab` runner/console script 和本地 `rsl_rl` 混跑时，才变成 P0 API 兼容风险。
2. **禁止裸 `train` 跑当前项目 task**：`.venv/bin/train` 是 `mjlab.scripts.train`，当前 `him_go2` 要走 `legged_mjlab/scripts/train.py`。
3. **给 obs/action 加启动断言**：`him_go2` 必须断言 45/270/235/12；GO2W 必须断言 57/342/262/16。断言要在 runner 创建前失败。
4. **先证明 terminal obs 正确**：HIM estimator 对 `next_critic_obs` 很敏感，`auto_reset=True` 会破坏这个语义；当前 `HIMRslRlWrapper` 的手动 reset 是关键路径。
5. **把 sim2sim 配置修到和训练一致**：HIMLoco GO2W 的路径和 `wheel_vel_scale` 需要先对齐，否则 sim2sim 不能作为训练质量证据。
6. **上真机前修复姿态 stop**：unitree deploy 的 `bad_orientation=false` 不能作为跌倒保护；HIMLoco 也要确认 `joint_direction` 是否实际生效。

## 关键文件索引

### unitree_rl_mjlab

- `/home/kk/github/unitree_rl_mjlab/setup.py`：声明 `mjlab==1.2.0`、`mujoco-warp==3.5.0`。
- `/home/kk/github/unitree_rl_mjlab/scripts/train.py`：训练入口，导入 `src.tasks` 并创建 runner。
- `/home/kk/github/unitree_rl_mjlab/scripts/play.py`：回放/导出入口，存在配置字段兼容风险。
- `/home/kk/github/unitree_rl_mjlab/src/tasks/velocity/velocity_env_cfg.py`：velocity 基础 env cfg，actor obs 不含 `base_lin_vel`，包含 `phase`。
- `/home/kk/github/unitree_rl_mjlab/src/tasks/velocity/config/go2/env_cfgs.py`：Go2 rough/flat 配置、12-DOF feet/contact/termination。
- `/home/kk/github/unitree_rl_mjlab/src/tasks/velocity/config/go2/rl_cfg.py`：PPO 超参，常见 `num_steps_per_env=24`。
- `/home/kk/github/unitree_rl_mjlab/src/tasks/velocity/mdp/observations.py`：`phase()`、`foot_height()` 等关键 obs。
- `/home/kk/github/unitree_rl_mjlab/src/tasks/velocity/mdp/rewards.py`：重写 tracking/orientation/gait/clearance 等 reward。
- `/home/kk/github/unitree_rl_mjlab/deploy/include/isaaclab/envs/mdp/terminations.h`：`bad_orientation()` 当前返回 false。

### installed mjlab 1.6.0

- `/home/kk/legged_mjlab/.venv/lib64/python3.11/site-packages/mjlab-1.6.0.dist-info/METADATA`：依赖版本，含 `rsl-rl-lib==5.4.2`。
- `/home/kk/legged_mjlab/.venv/lib64/python3.11/site-packages/mjlab-1.6.0.dist-info/entry_points.txt`：console scripts 指向 `mjlab.scripts.*`。
- `/home/kk/legged_mjlab/.venv/lib64/python3.11/site-packages/mjlab/envs/manager_based_rl_env.py`：`auto_reset`、`scale_rewards_by_dt`、terminated/truncated 语义。
- `/home/kk/legged_mjlab/.venv/lib64/python3.11/site-packages/mjlab/rl/vecenv_wrapper.py`：返回 `TensorDict` obs groups，并写 `extras["time_outs"]`。
- `/home/kk/legged_mjlab/.venv/lib64/python3.11/site-packages/mjlab/rl/runner.py`：新版保存/加载/ONNX 导出接口。
- `/home/kk/legged_mjlab/.venv/lib64/python3.11/site-packages/mjlab/tasks/velocity/velocity_env_cfg.py`：安装版 velocity MDP，actor obs 含 `base_lin_vel`，action scale base 默认 `0.5`。
- `/home/kk/legged_mjlab/.venv/lib64/python3.11/site-packages/mjlab/managers/observation_manager.py`：obs group 和 history 拼接语义。
- `/home/kk/legged_mjlab/.venv/lib64/python3.11/site-packages/mjlab/managers/reward_manager.py`：reward dt scaling 和 `nan_to_num`。

### HIMLoco

- `/home/kk/github/HIMLoco/legged_gym/legged_gym/scripts/train.py`：Isaac Gym 训练入口。
- `/home/kk/github/HIMLoco/legged_gym/legged_gym/envs/__init__.py`：注册 `a1`、`my_robot`、`go2w`。
- `/home/kk/github/HIMLoco/legged_gym/legged_gym/envs/go2w/go2w_config.py`：GO2W 57/342/262/16 维度、DR、reward、PPO 配置。
- `/home/kk/github/HIMLoco/legged_gym/legged_gym/envs/go2w/go2w_legged_robot.py`：GO2W obs 拼接、wheel torque/velocity 控制、terminal obs。
- `/home/kk/github/HIMLoco/rsl_rl/rsl_rl/runners/him_on_policy_runner.py`：HIM runner 的 7 元组 step contract。
- `/home/kk/github/HIMLoco/rsl_rl/rsl_rl/algorithms/him_ppo.py`：`next_critic_obs` storage、timeout bootstrap、estimator update。
- `/home/kk/github/HIMLoco/rsl_rl/rsl_rl/modules/him_actor_critic.py`：actor 输入为 one-step obs + estimated velocity + latent。
- `/home/kk/github/HIMLoco/rsl_rl/rsl_rl/modules/him_estimator.py`：硬编码 estimator target 切片。
- `/home/kk/github/HIMLoco/sim2sim/go2w/config.yaml`：GO2W sim2sim 路径、wheel scale、obs/action 维度。
- `/home/kk/github/HIMLoco/deploy/policy/himloco/config.yaml`：12-DOF deploy config、joint mapping/direction/action scale。

### 当前项目 `/home/kk/legged_mjlab`

- `/home/kk/legged_mjlab/legged_mjlab/scripts/train.py`：当前项目训练入口，会注册本地 env。
- `/home/kk/legged_mjlab/legged_mjlab/utils/task_registry.py`：根据 runner class 选择 `HIMRslRlWrapper` 或普通 wrapper。
- `/home/kk/legged_mjlab/legged_mjlab/wrappers/him_wrapper.py`：HIM history、manual reset、terminal privileged obs 适配。
- `/home/kk/legged_mjlab/legged_mjlab/envs/him_go2/him_go2_config.py`：当前 12-DOF `him_go2` 的 45/270/235/12 配置。
- `/home/kk/legged_mjlab/legged_mjlab/envs/him_go2/him_go2_env.py`：把 `HimGo2RoughCfg` 转为 `ManagerBasedRlEnvCfg`。
- `/home/kk/legged_mjlab/rsl_rl`：当前实际 import 到的 old/HIM 风格 RSL-RL 分叉。

## 最后判断

当前训练问题应先按“接口错配”处理，而不是按“策略还没调好”处理。最小闭环是：选定一个训练栈，隔离它的 venv 和入口，跑 2-16 个 env 的 reset/step shape + finite 检查，再跑 1 iteration dry-run。只有当 `import path -> task registry -> env step -> wrapper -> storage -> update -> save/export` 都稳定后，reward scale、terrain curriculum、domain randomization 和网络结构才值得调。

如果目标是当前 `/home/kk/legged_mjlab` 的 `him_go2`，建议下一步只做一件事：给启动阶段加硬断言并跑 smoke，证明 `obs=[N,270]`、`critic=[N,235]`、`action=[N,12]`、`step()` 训练态 7 元组、`time_outs`、`termination_privileged_obs` 都正确。这个通过之前，不建议扩大环境数或打开复杂地形。
