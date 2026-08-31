# `himloco_lab` 旧版 `rsl_rl` 迁移使用说明

本文回答一个边界明确的问题：在 `/home/kk/legged_mjlab` 里继续使用旧版 HIM `rsl_rl==1.0.2`，应怎样参考 `himloco_lab` 的迁移方式组织包来源、task 注册、wrapper、runner、PPO/storage 和训练入口。本文是迁移/使用方案，**不代表源码已按下面代码块修改**；下面所有 Python 代码均是候选实现或可直接替换片段，只供后续手工落地与审查。

未执行/未验证：本文没有运行训练、没有改 `.py`、没有改 `setup.py`、没有改测试或构建脚本，也没有更新 `ARCHITECTURE_CONTEXT.md`。

## 1. 边界与当前结论

用户目标不是接入 MJLab 内部新版 `rsl-rl-lib` 训练栈，而是继续使用旧版 HIM fork。这里的关键边界是：旧 HIM 代码长期假设顶层包名就是 `rsl_rl`，而 MJLab 1.6.0 的依赖也提供同名顶层包 `rsl_rl`。所以迁移不能只说“安装旧版”，必须在启动前固定：

| 项 | 本文约定 |
| --- | --- |
| 源码边界 | 本文只新增文档，不写入业务代码、测试代码、构建脚本或 release/memory 文件。 |
| 使用目标 | 训练 `him_go2` 时使用项目内旧 HIM `rsl_rl==1.0.2` 的 `HIMOnPolicyRunner`、`HIMPPO`、`HIMActorCritic`、`HIMRolloutStorage`。 |
| 推荐先手 | 先保留旧 `rsl_rl` 的 canonical namespace，并通过 `task_registry.load_project_rsl()` 在训练入口显式加载项目源码；等 wrapper/runner/storage 契约稳定后，再考虑迁到项目命名空间。 |
| 不做事项 | 不声明已完成训练兼容，不声明 `[ALL_TESTS_PASSED]`，不把部署安全、checkpoint/export、sim2real 当作已验证。 |

为什么先做 wrapper/runner/storage 契约：`rsl_rl` 包来源只是第一层问题。真正会影响训练正确性的，是 HIM 的七元组 step 返回、terminal critic、`time_outs` bootstrap、历史观测 `[N,270]`、native critic `[N,235]` 到 runner critic `[N,48]` 的切片，以及 storage 是否把 `next_critic_observations` 从 rollout 一直送到 estimator update。

### 1.1 Profile 边界：禁止混接 Legacy HIM 与新版 MJLab ABI

数学审查结论先放在这里：旧 HIM profile 和新版 `rsl-rl-lib` profile 是两套 ABI，不是同一个 runner 的两个小配置。只要 runner、storage、step 返回值或 checkpoint 任一层混用，就会出现“shape 看似对、语义已经错”的状态。

| Profile | 包来源 | step ABI | runner / algorithm / storage | observation 约定 | checkpoint 约定 | 禁止事项 |
| --- | --- | --- | --- | --- | --- | --- |
| Legacy HIM profile | `/home/kk/legged_mjlab/rsl_rl/rsl_rl`，旧 HIM fork `rsl_rl==1.0.2` | HIM 七元组：`history, privileged, rewards, dones, infos, done_ids, terminal_privileged` | `HIMOnPolicyRunner`、`HIMPPO`、`HIMRolloutStorage` | actor history `[N,270]`，native critic `[N,235]`，runner critic `[N,48]` | 旧 HIM 单体 `HIMActorCritic`、PPO optimizer、estimator optimizer | 不得接入 TensorDict 四元 step，不得加载新版 `OnPolicyRunner` checkpoint。 |
| MJLab ABI migration profile | `.venv/site-packages/rsl_rl`，安装版 `rsl-rl-lib==5.4.2` | 新版 TensorDict / 四元 step 风格，按 MJLab `obs_groups` 组织 | `OnPolicyRunner` 及新版 storage/obs group 机制 | policy/critic 由新版 wrapper 和 `obs_groups` 管理 | 新版 runner/checkpoint schema | 不得消费 Legacy HIM 七元组，不得复用 HIM storage 的 `next_critic_observations` 语义。 |

所以本文后续所有候选代码都属于 **Legacy HIM profile**。如果后续要迁到 MJLab ABI profile，应另开文档和代码分支，把 wrapper、runner、storage、checkpoint/export 一次性切到新版接口；不能在同一训练入口里让旧 HIM runner 吃新版 TensorDict，也不能让新版 `OnPolicyRunner` 吃旧 HIM 七元组。

## 2. `himloco_lab` 的迁移方式

参考项目路径为 `/tmp/himloco_lab_zip/himloco_lab-master`，对应外部指针是 `https://github.com/IsaacZH/himloco_lab/tree/master`。本文使用本地目录取证，没有从网络重新拉取。

### 2.1 结构：自定义 `rsl_rl` 放在项目命名空间内

`himloco_lab` 没有把自定义 HIM 算法作为顶层 `rsl_rl` 包安装，而是放在：

```text
/tmp/himloco_lab_zip/himloco_lab-master/source/himloco_lab/himloco_lab/rsl_rl/
  algorithms/him_ppo.py
  config/rl_cfg.py
  env/vec_env.py
  modules/him_actor_critic.py
  modules/him_estimator.py
  runners/him_on_policy_runner.py
  storage/him_rollout_storage.py
  wrappers/himloco_vec_env_wrapper.py
```

路径证据：

| 证据 | 观察 |
| --- | --- |
| `/tmp/himloco_lab_zip/himloco_lab-master/source/himloco_lab/setup.py:25-37` | 安装包名是 `himloco_lab`，`packages=["himloco_lab"]`，不是单独安装顶层 `rsl_rl`。 |
| `/tmp/himloco_lab_zip/himloco_lab-master/source/himloco_lab/himloco_lab/rsl_rl/__init__.py:35-41` | `himloco_lab.rsl_rl` 统一导出 algorithms、modules、storage、runners、wrappers。 |
| `/tmp/himloco_lab_zip/himloco_lab-master/scripts/himloco_rsl_rl/train.py:56-58` | 训练脚本显式 `from himloco_lab.rsl_rl import HIMOnPolicyRunner, HimlocoVecEnvWrapper`，不会误拿外部 `rsl_rl`。 |

这给本项目两个可选方向：

| 路线 | 包名 | 优点 | 代价 |
| --- | --- | --- | --- |
| 先保留 canonical `rsl_rl` | 顶层仍叫 `rsl_rl`，但启动前强制加载 `/home/kk/legged_mjlab/rsl_rl/rsl_rl` | 最小改造，兼容现有旧 HIM 的绝对 import 和 checkpoint 习惯 | 仍要防止同进程混入 `rsl-rl-lib` 子模块 |
| 迁到项目命名空间 | 例如 `legged_mjlab.rsl_rl` 或 `legged_mjlab.algorithms.him_rsl_rl` | 像 `himloco_lab` 一样彻底避开同名包冲突 | 需要系统替换 import、checkpoint/export 元数据和脚本入口 |

本文推荐先走第一条：对当前仓库最小、可审查，也符合用户“不要写代码”的当前边界。第二条应作为稳定后的后续迁移，不要在 wrapper/runner/storage 契约还没有验收前一次性全搬。

### 2.2 task 注册：Gym registry 同时挂 env cfg 和 HIM agent cfg

`himloco_lab` 的 Go2 task 注册在：

```text
/tmp/himloco_lab_zip/himloco_lab-master/source/himloco_lab/himloco_lab/tasks/locomotion/robots/go2/__init__.py
```

代码观察：

| 行号 | 作用 |
| ---: | --- |
| `:3-12` | `gym.register(id="Unitree-Go2-Velocity", entry_point="himloco_lab.envs:HimlocoManagerBasedRLEnv", kwargs={...})`。 |
| `:9-11` | 同一个 task kwargs 里同时给 `env_cfg_entry_point`、`himloco_rsl_rl_cfg`、`rsl_rl_cfg_entry_point`。 |
| `:15-23` | Play task 继续使用同一个 `HimlocoManagerBasedRLEnv`，但换 play cfg 和 agent cfg。 |

本项目当前不是 Gym/Hydra registry，而是 `legged_mjlab.utils.task_registry`。等价迁移点是：`legged_mjlab/envs/him_go2/__init__.py:6-12` 已注册 `task_id="him_go2"`、`wrapper_name="him"`，训练入口再通过 `task_registry.make_env()` 和 `make_alg_runner()` 创建 wrapper/runner。

### 2.3 Isaac Lab wrapper：先拿到 reset 前 critic，再给旧 runner 七元组

`himloco_lab` 先扩展 Isaac Lab manager env，再包一层 HIM vec env wrapper：

| 文件 | 路径证据 | 作用 |
| --- | --- | --- |
| manager env | `/tmp/himloco_lab_zip/himloco_lab-master/source/himloco_lab/himloco_lab/envs/him_manager_based_rl_env.py:20-120` | `step()` 返回 `(obs_after_reset, obs_before_reset, reward, terminated, time_outs, extras)`，核心是 `:73-120` 先 compute `obs_buf_before_reset`，再 reset done env，最后 compute reset 后 obs。 |
| HIM wrapper | `/tmp/himloco_lab_zip/himloco_lab-master/source/himloco_lab/himloco_lab/rsl_rl/wrappers/himloco_vec_env_wrapper.py:237-327` | 把 manager env 的 6 元组转成旧 HIM runner 需要的 7 元组：`obs, privileged_obs, rewards, dones, infos, termination_ids, termination_privileged_obs`。 |
| 维度读取 | 同 wrapper `:90-113` | 从 observation manager 读取 `policy`/`critic` group 维度，建立 history buffer。 |

本项目当前 wrapper 已经走了相似路线，但针对 MJLab/MuJoCo：

| 文件 | 当前事实 |
| --- | --- |
| `/home/kk/legged_mjlab/legged_mjlab/wrappers/him_wrapper.py:26-34` | 强制 6 帧、单帧 45 维，得到 actor history 270 维。 |
| `/home/kk/legged_mjlab/legged_mjlab/wrappers/him_wrapper.py:388-469` | `step()` 接收 native 5 元组，输出旧 HIM 7 元组，并写入 `terminated`、`truncated`、`time_outs`、`timeout_bootstrap`、`termination_privileged_obs`。 |
| `/home/kk/legged_mjlab/legged_mjlab/wrappers/him_wrapper.py:49-79` | 尝试关闭 native `auto_reset`，使 done 时能拿到 reset 前 terminal frame。 |

注意：`himloco_lab` 里 `infos["time_outs"] = truncated` 的前提是 `not cfg.is_finite_horizon`，见 wrapper `:265-268`。本项目当前 wrapper 写成 `truncated & ~terminated`，见 `/home/kk/legged_mjlab/legged_mjlab/wrappers/him_wrapper.py:448-452`，但这还不够：后续候选必须再加 finite-horizon 条件。只有 `is_finite_horizon=False` 的延续任务才允许 timeout bootstrap；有限时域任务即使 `truncated=True`，bootstrap mask 也必须全 false。

### 2.4 runner / algorithm / module / config 入口

`himloco_lab` 的入口链很清楚：

```text
scripts/himloco_rsl_rl/train.py
  -> gym.make(task, cfg=env_cfg)
  -> HimlocoVecEnvWrapper(...)
  -> HIMOnPolicyRunner(env, agent_cfg.to_dict(), ...)
  -> HIMPPO(actor_critic, ...)
  -> HIMRolloutStorage(...)
  -> HIMEstimator.update(obs_batch, next_critic_obs_batch)
```

路径证据：

| 层 | 证据 |
| --- | --- |
| train | `/tmp/himloco_lab_zip/himloco_lab-master/scripts/himloco_rsl_rl/train.py:110-137`：创建 env、包 wrapper、创建 `HIMOnPolicyRunner`。 |
| runner 构造 | `/tmp/himloco_lab_zip/himloco_lab-master/source/himloco_lab/himloco_lab/rsl_rl/runners/him_on_policy_runner.py:74-98`：按 `policy_class_name`、`algorithm_class_name` 构造 actor critic、PPO 和 storage。 |
| runner 训练循环 | 同文件 `:137-167`：`obs = env.get_observations()`，step 后用 `termination_ids` 把 `next_critic_obs` 的 done 行替换为 `termination_privileged_obs`。 |
| PPO | `/tmp/himloco_lab_zip/himloco_lab-master/source/himloco_lab/himloco_lab/rsl_rl/algorithms/him_ppo.py:90-108`：`act(obs, critic_obs)` 保存旧 transition；`process_env_step(..., next_critic_obs)` 写入 next critic，并用 `time_outs` 做旧式 bootstrap。 |
| storage | `/tmp/himloco_lab_zip/himloco_lab-master/source/himloco_lab/himloco_lab/rsl_rl/storage/him_rollout_storage.py:37-48,61-68,86-99,128-167`：Transition 和 storage schema 里显式有 `next_critic_observations`，mini-batch 也返回它。 |
| module | `/tmp/himloco_lab_zip/himloco_lab-master/source/himloco_lab/himloco_lab/rsl_rl/modules/him_actor_critic.py:73-99,165-207`：actor 使用 history estimator 输出 `vel + latent`；critic 单独评估 privileged obs。 |
| estimator | `/tmp/himloco_lab_zip/himloco_lab-master/source/himloco_lab/himloco_lab/rsl_rl/modules/him_estimator.py:76-84`：`update(obs_history, next_critic_obs)` 从 next critic 切速度和下一帧 observation target。 |
| config | `/tmp/himloco_lab_zip/himloco_lab-master/source/himloco_lab/himloco_lab/rsl_rl/config/rl_cfg.py:157-180` 与 `tasks/locomotion/agents/himloco_rsl_rl_cfg.py:11-37`：runner cfg 固定 `HIMOnPolicyRunner`、`HIMActorCritic`、`HIMPPO` 和 history length。 |

## 3. 本项目当前事实

### 3.1 版本和包冲突

代码观察：

| 位置 | 当前事实 | 影响 |
| --- | --- | --- |
| `/home/kk/legged_mjlab/setup.py:13-18` | 根项目依赖 `mjlab==1.6.0`、`mujoco-warp==3.11.0`、`scipy==1.17.0`。 | MJLab 版本固定为 1.6.0。 |
| `/home/kk/legged_mjlab/.venv/lib/python3.11/site-packages/mjlab-1.6.0.dist-info/METADATA:1-3,38` | 已安装 `mjlab==1.6.0`，并声明 `Requires-Dist: rsl-rl-lib==5.4.2`。 | 新版 `rsl-rl-lib` 会提供同名顶层包 `rsl_rl`。 |
| `/home/kk/legged_mjlab/rsl_rl/setup.py:3-15` | 本地旧 fork 元数据为 `rsl_rl==1.0.2`。 | 这是用户要使用的 HIM 旧版实现。 |
| `/home/kk/legged_mjlab/setup.py:7-24` | 根项目把 `rsl_rl/rsl_rl` 映射成顶层包 `rsl_rl` 一起打包。 | 安装时会和 `rsl-rl-lib` 的顶层 `rsl_rl` 竞争。 |
| `/home/kk/legged_mjlab/.venv/lib/python3.11/site-packages/rsl_rl_lib-5.4.2.dist-info/top_level.txt:1` | `rsl-rl-lib` 的 top-level 也是 `rsl_rl`。 | 不能只看 distribution 名称，必须看 `rsl_rl.__file__`。 |
| 当前来源检查 | `.venv/bin/python` 直接 `import rsl_rl` 指向 `.venv/lib/python3.11/site-packages/rsl_rl/__init__.py`，`rsl_rl.runners.__all__` 只有 `DistillationRunner`、`OnPolicyRunner`，没有 `HIMOnPolicyRunner`。 | 训练前必须调用 `load_project_rsl()`，否则会拿到 MJLab 依赖的新包。 |

这里的冲突不是“两个版本号哪个更新”的问题，而是同一进程里不能混用两套同名 `rsl_rl.*` 子模块。旧 HIM runner 内部有绝对 import，例如 `/home/kk/legged_mjlab/rsl_rl/rsl_rl/runners/him_on_policy_runner.py:53-55` 从 `rsl_rl.algorithms`、`rsl_rl.modules`、`rsl_rl.env` 导入；如果只替换顶层包而留下已安装子模块，会得到不可审查的混合状态。

### 3.2 `task_registry.load_project_rsl()` fallback

本项目已有 fallback：

| 位置 | 行为 |
| --- | --- |
| `/home/kk/legged_mjlab/legged_mjlab/utils/task_registry.py:59-63` | 项目旧 `rsl_rl` 源码目录固定为 `Path(...)/rsl_rl/rsl_rl`。 |
| `:74-77` | 移除 `sys.modules` 中已有的 `rsl_rl` 与所有 `rsl_rl.*`。 |
| `:80-132` | 用 canonical name `rsl_rl` 加载项目源码，并检查 `rsl_rl.runners.HIMOnPolicyRunner`。 |
| `:135-171` | 先尝试已导入/已安装包；若 `rsl_rl.runners` 缺少 `HIMOnPolicyRunner`，再 fallback 到项目源码。 |
| `:257-266` | `make_alg_runner()` 里再次调用 `load_project_rsl()`，再按训练配置里的 runner 名创建 runner。 |

这和 `himloco_lab` 的命名空间方案不同：`himloco_lab` 通过 `himloco_lab.rsl_rl` 避开冲突，本项目当前通过 runtime fallback 保留 canonical `rsl_rl`。

并发边界也必须写进契约：`load_project_rsl()` 会触碰 `sys.modules`，只能在单线程进程启动早期调用，且必须发生在训练、play、导出、日志线程或任何会 import `rsl_rl.*` 的后台线程启动前。如果进程里已经导入了来自 `.venv/site-packages` 的不兼容 `rsl_rl.*` 子模块，安全做法不是继续清理后混跑，而是 fail closed 并重启一个干净 Python 进程。第 6 节来源验证也因此必须使用独立新 Python 进程，不能在已经跑过训练/import smoke 的交互式解释器里复测。

### 3.3 HIM wrapper 七元组与 shape

当前 HIM-Go2 的 shape 契约来自配置和 wrapper：

| 数据 | shape | 证据 |
| --- | ---: | --- |
| one-step actor | `45` | `/home/kk/legged_mjlab/legged_mjlab/envs/him_go2/him_go2_config.py:9-14`。 |
| history actor | `6 * 45 = 270` | 同配置 `:10-12`，wrapper `:26-34`。 |
| native critic / privileged | `45 + 3 + 187 = 235` | 同配置 `:13`，其中 187 对应 17×11 height scan。 |
| runner critic | `45 + 3 = 48` | `/home/kk/legged_mjlab/rsl_rl/rsl_rl/runners/him_on_policy_runner.py:165-168`。 |
| actions | `12` | `/home/kk/legged_mjlab/legged_mjlab/envs/him_go2/him_go2_config.py:14`。 |
| wrapper step | 7 元组 | `/home/kk/legged_mjlab/legged_mjlab/wrappers/him_wrapper.py:461-469`。 |

七元组是：

```text
(
  history,                  # [N,270]
  privileged,               # [N,235] reset 后下一回合 critic source
  rewards,                  # [N]
  dones,                    # [N] = terminated | truncated
  infos,                    # dict，含 time_outs/timeout_bootstrap/terminal critic/action masks
  done_ids,                 # [K]
  terminal_privileged,      # [K,235] 或空 [0,235]
)
```

runner 再把 `privileged[..., :48]` 作为 HIM critic。代码观察：`_prepare_critic_obs()` 在 `/home/kk/legged_mjlab/rsl_rl/rsl_rl/runners/him_on_policy_runner.py:469-492` 会截断到 `num_critic_obs=48`；`_apply_terminal_critic_obs()` 在 `:509-536` 对 done 行用 terminal privileged 替换 next critic，再交给 `HIMPPO.process_env_step()`。

## 4. 推荐路线

### 4.1 当前推荐：保留旧 `rsl_rl` canonical namespace，先把契约写死

推荐最小路线如下：

```text
/home/kk/legged_mjlab
  setup.py                          # 允许本地旧 rsl_rl 源码存在，但不把来源检查交给 pip 猜
  legged_mjlab/utils/task_registry.py
      load_project_rsl()             # 训练前强制得到项目旧 HIM backend
  legged_mjlab/wrappers/him_wrapper.py
      HIMRslRlWrapper                # native 5 元组 -> HIM 七元组
  rsl_rl/rsl_rl/
      runners/him_on_policy_runner.py
      algorithms/him_ppo.py
      storage/him_rollout_storage.py
      modules/him_actor_critic.py
      modules/him_estimator.py
  legged_mjlab/scripts/train.py      # 入口先 load_project_rsl，再建 env/runner
  legged_mjlab/scripts/play.py       # 当前为空；后续 play 入口也必须先走同一 source/profile gate
```

这样做的理由：

| 决策 | 原因 |
| --- | --- |
| 不先迁到新版 `rsl-rl-lib==5.4.2` | 用户明确使用旧版 `rsl_rl`；新版 ABI 是另一条迁移线。 |
| 不先改成 `legged_mjlab.rsl_rl` | 这会牵动所有绝对 import、checkpoint key、export 和脚本，当前收益不如先验收 HIM 训练契约。 |
| 先修 wrapper/runner/storage | terminal auto-reset、`time_outs`、`next_critic_observations` 和 `[45,270,235,48]` shape 是训练正确性边界。 |
| 来源检查必须在入口执行 | 当前解释器直接 import 可能拿到 `.venv/site-packages/rsl_rl` 新包；训练、play、export 都必须在创建 runner 或加载 checkpoint 前确认 `HIMOnPolicyRunner`。 |

### 4.2 后续可选：迁到项目命名空间

如果要完全参考 `himloco_lab`，后续可以把旧 HIM fork 移到项目命名空间，例如：

```text
legged_mjlab/rsl_rl/
  algorithms/
  config/
  env/
  modules/
  runners/
  storage/
  wrappers/
```

然后把所有旧 HIM 内部 import 从 `from rsl_rl...` 改为 `from legged_mjlab.rsl_rl...`，训练入口显式 `from legged_mjlab.rsl_rl.runners import HIMOnPolicyRunner`。这条路线更干净，但不是本文建议的第一步，因为会扩大改动面。

## 5. 完整候选实现过程

下面代码块是候选实现。它们展示后续手工修改时的目标形状，**没有写入源码**。如果落地，建议一次只改一个边界，改完就跑对应 gate。

### 5.1 `setup.py`：包来源策略候选

候选目标：保留旧 `rsl_rl` canonical namespace 的源码包映射，但把注释说清楚：`setup.py` 不能单独保证运行时来源，训练入口仍必须调用 `load_project_rsl()`。

```python
# /home/kk/legged_mjlab/setup.py
from setuptools import find_packages, setup


PROJECT_PACKAGES = find_packages(include=["legged_mjlab", "legged_mjlab.*"])

# Keep the old HIM fork importable as canonical `rsl_rl`.
# This preserves the legacy absolute imports inside rsl_rl/rsl_rl/*.  It does
# not by itself prove that runtime import resolution will prefer this source
# over `rsl-rl-lib`; task_registry.load_project_rsl() is still the source gate.
LOCAL_RSL_RL_SOURCE = "rsl_rl/rsl_rl"
LOCAL_RSL_RL_SUBPACKAGES = find_packages(where=LOCAL_RSL_RL_SOURCE)
LOCAL_RSL_RL_PACKAGES = ["rsl_rl"] + [
    f"rsl_rl.{name}" for name in LOCAL_RSL_RL_SUBPACKAGES
]

INSTALL_REQUIRES = [
    "mjlab==1.6.0",
    "mujoco-warp==3.11.0",
    "scipy==1.17.0",
]


setup(
    name="legged_mjlab",
    version="0.0.1",
    packages=PROJECT_PACKAGES + LOCAL_RSL_RL_PACKAGES,
    package_dir={"rsl_rl": LOCAL_RSL_RL_SOURCE},
    install_requires=INSTALL_REQUIRES,
)
```

如果后续决定走 `himloco_lab` 风格的项目命名空间，不应继续把本地 HIM fork 打包成顶层 `rsl_rl`。那时 `setup.py` 候选应改为只打包 `legged_mjlab.*`，并把 fork 移到 `legged_mjlab/rsl_rl` 后统一改 import。本文不建议当前一步直接跳过去。

### 5.2 `task_registry.load_project_rsl()`：来源 gate 候选

候选目标：任何训练/play/export 前先拿到项目内旧 HIM backend；`load_project_rsl()` 进入后首先执行启动期上下文断言，候选最小检查是主线程且 `threading.active_count() == 1`。随后先扫描已导入的 `rsl_rl.*` 子模块；只要发现任一非项目源码子模块，就 fail closed 并要求重启干净进程。即使顶层 `rsl_rl` 已经来自项目源码，也不能绕过启动期断言和不兼容子模块扫描。`sys.modules` 切换只允许发生在单线程进程启动早期，不在训练中途清理模块继续跑。

```python
# /home/kk/legged_mjlab/legged_mjlab/utils/task_registry.py
# 可直接替换 load_project_rsl 相关 helper 的候选片段。
import importlib
import importlib.util
import sys
import threading
import warnings
from pathlib import Path


def _rsl_module_source(module):
    origin = getattr(module, "__file__", None)
    if origin:
        return str(origin)
    locations = getattr(module, "__path__", None)
    if locations:
        return "namespace package (" + ", ".join(map(str, locations)) + ")"
    return "unknown source"


def _project_rsl_package():
    package_dir = Path(__file__).resolve().parents[2] / "rsl_rl" / "rsl_rl"
    return package_dir, package_dir / "__init__.py"


def _is_under(path, root):
    if path in (None, "", "unknown source"):
        return False
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (OSError, ValueError):
        return False


def _rsl_modules_snapshot():
    return {
        name: module
        for name, module in sys.modules.items()
        if name == "rsl_rl" or name.startswith("rsl_rl.")
    }


def _assert_startup_context():
    # Candidate minimum startup check; stricter entrypoints may add process gates.
    current = threading.current_thread()
    if current is not threading.main_thread() or threading.active_count() != 1:
        raise RuntimeError(
            "load_project_rsl() must run on the main thread in a single-threaded "
            "startup context, before training/play/export workers are created"
        )


def _incompatible_imported_submodules(package_dir):
    bad = []
    for name, module in _rsl_modules_snapshot().items():
        if name == "rsl_rl":
            continue
        source = _rsl_module_source(module)
        if not _is_under(source, package_dir):
            bad.append(f"{name} from {source}")
    return bad


def _assert_no_incompatible_imported_submodules(package_dir):
    bad_submodules = _incompatible_imported_submodules(package_dir)
    if bad_submodules:
        raise RuntimeError(
            "rsl_rl submodules are already imported from an incompatible "
            "backend; restart a fresh Python process before importing training "
            "code. Incompatible submodules: " + "; ".join(bad_submodules)
        )


def _remove_rsl_modules():
    for name in list(sys.modules):
        if name == "rsl_rl" or name.startswith("rsl_rl."):
            sys.modules.pop(name, None)


def _load_project_rsl_source(reason):
    package_dir, init_file = _project_rsl_package()
    original_modules = _rsl_modules_snapshot()
    _assert_startup_context()
    _assert_no_incompatible_imported_submodules(package_dir)

    _remove_rsl_modules()

    try:
        if not init_file.is_file():
            raise FileNotFoundError(f"project rsl_rl initializer does not exist: {init_file}")

        spec = importlib.util.spec_from_file_location(
            "rsl_rl",
            init_file,
            submodule_search_locations=[str(package_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"could not create import spec for {init_file}")

        project_module = importlib.util.module_from_spec(spec)
        sys.modules["rsl_rl"] = project_module
        spec.loader.exec_module(project_module)

        project_runners = importlib.import_module("rsl_rl.runners")
        if getattr(project_runners, "HIMOnPolicyRunner", None) is None:
            raise AttributeError(
                f"project rsl_rl runners at {project_runners.__file__} "
                "does not export HIMOnPolicyRunner"
            )
    except Exception as exc:
        _remove_rsl_modules()
        sys.modules.update(original_modules)
        raise RuntimeError(
            "unable to load the repository rsl_rl backend after rejecting "
            f"the currently imported backend ({reason}); expected package at "
            f"{package_dir}; {type(exc).__name__}: {exc}"
        ) from exc

    warnings.warn(
        "the imported rsl_rl backend is incompatible with this project "
        f"({reason}); using the repository backend at {package_dir}",
        RuntimeWarning,
        stacklevel=3,
    )
    return project_runners


def load_project_rsl():
    _assert_startup_context()
    package_dir, _ = _project_rsl_package()
    _assert_no_incompatible_imported_submodules(package_dir)

    imported = sys.modules.get("rsl_rl")
    if imported is not None and _is_under(_rsl_module_source(imported), package_dir):
        runners = importlib.import_module("rsl_rl.runners")
        if getattr(runners, "HIMOnPolicyRunner", None) is None:
            raise RuntimeError("project rsl_rl does not expose HIMOnPolicyRunner")
        return runners

    imported_source = _rsl_module_source(imported) if imported is not None else "not imported"
    reason = f"active rsl_rl source is {imported_source}; forcing project HIM backend"
    return _load_project_rsl_source(reason)
```

验收点：必须在独立新 Python 进程中调用。调用后 `import rsl_rl; print(rsl_rl.__file__)` 应指向 `/home/kk/legged_mjlab/rsl_rl/rsl_rl/__init__.py` 或同一源码目录，而不是 `.venv/lib/python3.11/site-packages/rsl_rl/__init__.py`。如果同一进程已经 import 过不兼容 `rsl_rl.runners`、`rsl_rl.storage` 等子模块，期望结果是 fail closed 并提示重启，而不是静默清理继续训练。

### 5.3 `HIMRslRlWrapper`：七元组与 terminal critic 候选

候选目标：native MJLab env 返回 5 元组，wrapper 输出旧 HIM runner 七元组；shape 固定为 `[N,45] -> [N,270]`、native critic `[N,235]`、terminal critic 保持 reset 前状态。这里同时把 action 入口契约写死：policy raw action 先经过 finite/shape 检查、强制幅值 clamp 和可选动作速率限制，只有 accepted action 能进入 native env。

```python
# /home/kk/legged_mjlab/legged_mjlab/wrappers/him_wrapper.py
# 候选核心类，省略 VecEnvWrapper 内部公共转发实现。
from collections.abc import Mapping
import math

import torch

from legged_mjlab.wrappers.vec_env_wrapper import VecEnvWrapper


class HIMRslRlWrapper(VecEnvWrapper):
    def __init__(
        self,
        env,
        history_length=6,
        one_step_obs_dim=45,
        expected_privileged_obs_dim=235,
        action_dim=12,
        action_clip=1.0,
        max_action_rate=None,
        is_finite_horizon=None,
    ):
        super().__init__(env)
        if int(history_length) != 6:
            raise ValueError("HIM Go2 requires 6 history frames")
        if int(one_step_obs_dim) != 45:
            raise ValueError("HIM Go2 requires one-step actor obs width 45")
        if expected_privileged_obs_dim is not None and int(expected_privileged_obs_dim) != 235:
            raise ValueError("HIM Go2 native critic obs width must be 235")
        if action_dim is not None and int(action_dim) != 12:
            raise ValueError("HIM Go2 action width must be 12")
        try:
            action_clip = float(action_clip)
        except (TypeError, ValueError) as exc:
            raise ValueError("action_clip must be a finite positive float") from exc
        if not math.isfinite(action_clip) or action_clip <= 0.0:
            raise ValueError("action_clip must be a finite positive float")
        if max_action_rate is not None:
            try:
                max_action_rate = float(max_action_rate)
            except (TypeError, ValueError) as exc:
                raise ValueError("max_action_rate must be a finite positive float or None") from exc
            if not math.isfinite(max_action_rate) or max_action_rate <= 0.0:
                raise ValueError("max_action_rate must be a finite positive float or None")

        self.history_length = 6
        self.one_step_obs_dim = 45
        self._num_obs = 270
        self._num_one_step_obs = 45
        self._num_privileged_obs = expected_privileged_obs_dim
        self._num_actions = action_dim
        self._action_clip = action_clip
        self._max_action_rate = max_action_rate
        self._is_finite_horizon = self._resolve_finite_horizon(is_finite_horizon)
        self._previous_accepted_actions = None
        self.obs_history_buf = None
        self._privileged_obs = None
        self.termination_ids = None
        self.termination_privileged_obs = None
        self._manual_reset = self._disable_auto_reset_if_available()

    @staticmethod
    def _cfg_get(obj, name, default=None):
        if obj is None:
            return default
        if isinstance(obj, Mapping):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def _resolve_finite_horizon(self, explicit_value):
        if explicit_value is not None:
            return bool(explicit_value)
        cfg = getattr(self.env, "cfg", None)
        for source in (self._cfg_get(cfg, "env"), cfg):
            value = self._cfg_get(source, "is_finite_horizon", None)
            if value is not None:
                return bool(value)
        return False

    def _disable_auto_reset_if_available(self):
        cfg = getattr(self.env, "cfg", None)
        if cfg is None:
            return False
        if isinstance(cfg, Mapping):
            if "auto_reset" not in cfg:
                raise RuntimeError("env.cfg has no auto_reset field")
            cfg["auto_reset"] = False
            configured = cfg["auto_reset"]
        else:
            if not hasattr(cfg, "auto_reset"):
                raise RuntimeError("env.cfg has no auto_reset field")
            cfg.auto_reset = False
            configured = cfg.auto_reset
        if configured is not False:
            raise RuntimeError("failed to disable native auto_reset")
        return True

    @staticmethod
    def _split_obs(obs_dict):
        if not isinstance(obs_dict, Mapping):
            raise TypeError("native observation must be a mapping")
        actor = obs_dict.get("actor", obs_dict.get("policy"))
        critic = obs_dict.get("critic", obs_dict.get("privileged"))
        if actor is None:
            raise KeyError("observation has no actor/policy group")
        if critic is None:
            raise KeyError("observation has no critic/privileged group")
        return actor, critic

    def _validate_frame(self, value, name, width=None, batch=True):
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape [N,F], got {tuple(value.shape)}")
        if batch and value.shape[0] != self.num_envs:
            raise ValueError(f"{name} batch mismatch: {value.shape[0]} vs {self.num_envs}")
        if width is not None and value.shape[-1] != width:
            raise ValueError(f"{name} width mismatch: {value.shape[-1]} vs {width}")
        if not torch.is_floating_point(value):
            raise TypeError(f"{name} must be floating point")
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"{name} contains NaN or Inf")
        return value

    def _split_and_validate(self, obs_dict):
        actor, critic = self._split_obs(obs_dict)
        actor = self._validate_frame(actor, "actor", width=45)
        critic = self._validate_frame(critic, "critic", width=self.num_privileged_obs)
        if actor.device != critic.device:
            raise ValueError("actor and critic must share device")
        return actor, critic

    def _ensure_history(self, actor):
        shape = (self.num_envs, 6, 45)
        if self.obs_history_buf is None:
            self.obs_history_buf = torch.zeros(shape, dtype=actor.dtype, device=actor.device)
        if tuple(self.obs_history_buf.shape) != shape:
            raise ValueError("history buffer shape drift")
        if self.obs_history_buf.device != actor.device or self.obs_history_buf.dtype != actor.dtype:
            raise ValueError("history buffer device/dtype drift")

    def _append_history(self, actor):
        self._ensure_history(actor)
        self.obs_history_buf[:, 1:].copy_(self.obs_history_buf[:, :-1].clone())
        self.obs_history_buf[:, 0].copy_(actor)
        return self.obs_history_buf.reshape(self.num_envs, 270)

    def _normalize_env_ids(self, env_ids, device):
        ids = torch.as_tensor(env_ids, device=device)
        if ids.dtype == torch.bool:
            if ids.numel() != self.num_envs:
                raise ValueError("boolean env_ids must have one value per env")
            ids = torch.nonzero(ids.reshape(-1), as_tuple=False).flatten()
        else:
            ids = ids.reshape(-1).long()
        if ids.numel() and (ids.min() < 0 or ids.max() >= self.num_envs):
            raise IndexError("env id out of range")
        return ids

    def _ensure_action_state(self, reference):
        shape = (self.num_envs, 12)
        if self._previous_accepted_actions is None:
            self._previous_accepted_actions = torch.zeros(
                shape,
                dtype=reference.dtype,
                device=reference.device,
            )
        if tuple(self._previous_accepted_actions.shape) != shape:
            raise ValueError("previous action buffer shape drift")
        if (
            self._previous_accepted_actions.device != reference.device
            or self._previous_accepted_actions.dtype != reference.dtype
        ):
            self._previous_accepted_actions = torch.zeros(
                shape,
                dtype=reference.dtype,
                device=reference.device,
            )

    def _reset_action_state(self, reference, env_ids=None):
        self._ensure_action_state(reference)
        if env_ids is None:
            self._previous_accepted_actions.zero_()
        else:
            ids = self._normalize_env_ids(env_ids, reference.device)
            self._previous_accepted_actions[ids] = 0.0

    def reset(self, env_ids=None):
        result = self.env.reset(env_ids=env_ids) if env_ids is not None else self.env.reset()
        obs_dict = result[0] if isinstance(result, tuple) else result
        actor, critic = self._split_and_validate(obs_dict)
        self._ensure_history(actor)
        if env_ids is None:
            self.obs_history_buf.zero_()
            self.obs_history_buf[:, 0].copy_(actor)
        else:
            ids = self._normalize_env_ids(env_ids, actor.device)
            self.obs_history_buf[ids] = 0.0
            self.obs_history_buf[ids, 0] = actor[ids]
        history = self.obs_history_buf.reshape(self.num_envs, 270)
        self._last_obs = history
        self._last_privileged_obs = critic
        self._privileged_obs = critic
        self.termination_ids = torch.empty(0, dtype=torch.long, device=actor.device)
        self.termination_privileged_obs = critic.new_empty((0, critic.shape[-1]))
        self._reset_action_state(actor, env_ids)
        return history, critic

    @staticmethod
    def _as_batch_bool(value, name, device, num_envs):
        tensor = torch.as_tensor(value, device=device)
        if tensor.numel() != num_envs:
            raise ValueError(f"{name} must contain {num_envs} values")
        return tensor.reshape(-1).bool()

    @staticmethod
    def _terminal_candidate(infos):
        for key in (
            "termination_privileged_obs",
            "terminal_privileged_obs",
            "final_privileged_obs",
            "terminal_critic_obs",
            "final_critic_obs",
        ):
            if key in infos and infos[key] is not None:
                return infos[key], key
        return None, "unavailable"

    def _select_terminal_privileged(self, candidate, done_ids):
        terminal = self._validate_frame(candidate, "terminal privileged", width=235, batch=False)
        if terminal.shape[0] == self.num_envs:
            return terminal.index_select(0, done_ids)
        if terminal.shape[0] == done_ids.numel():
            return terminal
        raise ValueError(
            "terminal privileged obs must be full [N,235] or compact [K,235]"
        )

    def _sanitize_actions(self, actions):
        raw_actions = self._validate_frame(actions, "policy raw actions", width=12)
        clipped_actions = raw_actions.clamp(-self._action_clip, self._action_clip)
        clip_mask = (clipped_actions != raw_actions).any(dim=-1)

        accepted_actions = clipped_actions
        rate_mask = torch.zeros(raw_actions.shape[0], dtype=torch.bool, device=raw_actions.device)
        if self._max_action_rate is not None:
            self._ensure_action_state(raw_actions)
            delta = (clipped_actions - self._previous_accepted_actions).clamp(
                -self._max_action_rate,
                self._max_action_rate,
            )
            accepted_actions = self._previous_accepted_actions + delta
            rate_mask = (accepted_actions != clipped_actions).any(dim=-1)

        self._previous_accepted_actions = accepted_actions.detach().clone()
        return (
            raw_actions.detach().clone(),
            clipped_actions.detach().clone(),
            accepted_actions,
            clip_mask,
            rate_mask,
        )

    def step(self, actions):
        raw_actions, clipped_actions, accepted_actions, clip_mask, rate_mask = self._sanitize_actions(actions)
        result = self.env.step(accepted_actions)
        if not isinstance(result, tuple) or len(result) != 5:
            raise ValueError("native env.step must return (obs, rewards, terminated, truncated, infos)")

        obs_dict, rewards, terminated, truncated, infos = result
        actor, critic = self._split_and_validate(obs_dict)
        rewards = torch.as_tensor(rewards, device=actor.device, dtype=torch.float32).reshape(-1)
        if rewards.numel() != self.num_envs:
            raise ValueError("reward batch mismatch")
        terminated = self._as_batch_bool(terminated, "terminated", actor.device, self.num_envs)
        truncated = self._as_batch_bool(truncated, "truncated", actor.device, self.num_envs)
        infos = dict(infos or {})

        dones = terminated | truncated
        done_ids = torch.nonzero(dones, as_tuple=False).flatten()

        terminal_candidate, source = self._terminal_candidate(infos)
        if terminal_candidate is None and self._manual_reset and done_ids.numel() > 0:
            # With auto_reset disabled, the current critic is still reset-before terminal critic.
            terminal_candidate, source = critic, "step_terminal_observation"

        terminal_available = torch.zeros_like(dones)
        if terminal_candidate is None:
            terminal_privileged = critic.new_empty((0, critic.shape[-1]))
        else:
            terminal_privileged = self._select_terminal_privileged(terminal_candidate, done_ids)
            terminal_available[done_ids] = True

        if self._manual_reset and done_ids.numel() > 0:
            reset_result = self.env.reset(env_ids=done_ids)
            reset_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            reset_actor, reset_critic = self._split_and_validate(reset_obs)
            actor = actor.clone()
            critic = critic.clone()
            actor[done_ids] = reset_actor[done_ids]
            critic[done_ids] = reset_critic[done_ids]
            self._reset_action_state(actor, done_ids)

        self._ensure_history(actor)
        if done_ids.numel() > 0:
            self.obs_history_buf[done_ids] = 0.0
        history = self._append_history(actor)

        if self._is_finite_horizon:
            time_outs = torch.zeros_like(dones)
        else:
            time_outs = truncated & ~terminated
        infos["terminated"] = terminated
        infos["truncated"] = truncated
        infos["time_outs"] = time_outs
        infos["timeout_bootstrap"] = time_outs & terminal_available
        infos["termination_privileged_obs"] = terminal_privileged
        infos["termination_privileged_obs_source"] = source
        infos["policy_raw_actions"] = raw_actions
        infos["clipped_actions"] = clipped_actions
        infos["accepted_actions"] = accepted_actions.detach().clone()
        infos["action_clip_mask"] = clip_mask
        infos["action_rate_limit_mask"] = rate_mask
        infos["is_finite_horizon"] = self._is_finite_horizon

        self._last_obs = history
        self._last_privileged_obs = critic
        self._privileged_obs = critic
        self.termination_ids = done_ids
        self.termination_privileged_obs = terminal_privileged

        return history, critic, rewards, dones, infos, done_ids, terminal_privileged
```

注意：如果 native env 无法关闭 auto-reset，也没有在 `infos` 里提供 reset 前 terminal critic，本 wrapper 不能用 reset 后 critic 冒充 terminal critic。那是 `BLOCKED`，不是 warning。

动作边界也要分层看：`HIMPPO.act()` 和 `HIMActorCritic.act()` 返回的是用于 PPO log-prob 的 policy raw action；wrapper 只在进入仿真 env 前把它变成 accepted/clipped action，并在 `infos` 里暴露 `policy_raw_actions`、`clipped_actions`、`accepted_actions`、`action_clip_mask`、`action_rate_limit_mask`。幅值 clamp 强制存在：`action_clip` 默认 `1.0`，必须是 finite positive float，不允许 `None`；仅 `max_action_rate` 动作速率限制可选。这只是仿真入口保护，不等于硬件安全。真实硬件仍必须有独立的动作幅值、目标角、速度、torque/effort、watchdog 和异常安全态 gate。

finite-horizon 边界不能省略：`is_finite_horizon=True` 时，`time_outs` 与 `timeout_bootstrap` 必须全 false；只有 `is_finite_horizon=False`、`truncated=True`、`terminated=False` 且 terminal critic 可用的行，才允许旧 HIM bootstrap。

代码观察：当前 `him_go2_config.py` 片段里没有直接出现 `is_finite_horizon` 字段。落地时建议在 env cfg 或 wrapper 参数中显式写出该值，并把有限/非有限两类任务都纳入单测；不要只依赖隐式默认。

### 5.4 `HIMOnPolicyRunner`：关键训练循环与 terminal critic 候选

候选目标：旧 runner 继续消费七元组，但把 `[N,235]` native critic 明确压到 `[N,48]`，并且只在 `timeout_bootstrap` 为真时用 terminal critic bootstrap。

```python
# /home/kk/legged_mjlab/rsl_rl/rsl_rl/runners/him_on_policy_runner.py
# 候选关键片段：__init__ 中的 shape 解析、critic 准备、terminal 替换和 learn 主循环。
import os
import time
from collections import deque
from collections.abc import Mapping

import torch

from rsl_rl.algorithms import HIMPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import HIMActorCritic


class HIMOnPolicyRunner:
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.env = env
        self.device = torch.device(device)
        self.log_dir = log_dir
        self.cfg = dict(train_cfg["runner"])
        self.alg_cfg = dict(train_cfg["algorithm"])
        self.policy_cfg = dict(train_cfg["policy"])

        reset_obs, reset_critic = self.env.reset()
        reset_obs = self._as_2d(reset_obs, "reset actor").to(self.device)
        reset_critic = self._prepare_critic_obs(reset_critic, reset_obs).to(self.device)

        self.num_actor_obs = reset_obs.shape[-1]       # 270
        self.num_one_step_obs = 45
        self.num_critic_obs = 48                       # 45 actor frame + 3 base lin vel
        self.num_actions = int(getattr(env, "num_actions", 12))
        if self.num_actor_obs != 270:
            raise ValueError(f"HIM actor history must be 270, got {self.num_actor_obs}")
        if self.num_actions != 12:
            raise ValueError(f"HIM action dim must be 12, got {self.num_actions}")

        actor_critic = HIMActorCritic(
            self.num_actor_obs,
            self.num_critic_obs,
            self.num_one_step_obs,
            self.num_actions,
            **self._strip_metadata(self.policy_cfg),
        ).to(self.device)
        self.alg = HIMPPO(
            actor_critic,
            device=self.device,
            **self._strip_metadata(self.alg_cfg),
        )

        self.num_steps_per_env = int(self.cfg["num_steps_per_env"])
        self.save_interval = int(self.cfg.get("save_interval", 500))
        self.alg.init_storage(
            int(getattr(self.env, "num_envs", reset_obs.shape[0])),
            self.num_steps_per_env,
            [self.num_actor_obs],
            [self.num_critic_obs],
            [self.num_actions],
        )

        self._initial_obs = reset_obs
        self._initial_critic_obs = reset_critic
        self.current_learning_iteration = 0
        self.tot_timesteps = 0
        self.tot_time = 0
        self.writer = None

    @staticmethod
    def _strip_metadata(section):
        ignored = {"runner_class_name", "policy_class_name", "algorithm_class_name", "metadata"}
        return {key: value for key, value in section.items() if key not in ignored}

    @staticmethod
    def _as_2d(value, name):
        if value is None:
            raise ValueError(f"{name} is required")
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        if value.ndim > 2:
            value = value.reshape(value.shape[0], -1)
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape [N,F]")
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"{name} contains NaN or Inf")
        return value

    def _prepare_critic_obs(self, critic_obs, actor_obs):
        critic_obs = self._as_2d(critic_obs, "critic observation")
        if critic_obs.shape[-1] < 48:
            raise ValueError(f"critic obs narrower than 48: {critic_obs.shape[-1]}")
        # Native critic is [N,235] = actor45 + base_lin_vel3 + height_scan187.
        # HIM runner consumes only [N,48] for critic/value/estimator target.
        return critic_obs[:, :48]

    @staticmethod
    def _termination_indices(termination_ids, num_envs, device):
        ids = torch.as_tensor(termination_ids, device=device)
        if ids.dtype == torch.bool:
            ids = torch.nonzero(ids.reshape(-1), as_tuple=False).flatten()
        else:
            ids = ids.reshape(-1).long()
        if ids.numel() and (ids.min() < 0 or ids.max() >= num_envs):
            raise IndexError("termination index out of range")
        return ids

    def _apply_terminal_critic_obs(self, next_critic_obs, termination_ids, terminal_privileged_obs):
        ids = self._termination_indices(
            termination_ids,
            next_critic_obs.shape[0],
            next_critic_obs.device,
        )
        if ids.numel() == 0:
            return next_critic_obs
        if terminal_privileged_obs is None:
            return next_critic_obs

        terminal = self._as_2d(terminal_privileged_obs, "terminal privileged").to(
            device=next_critic_obs.device,
            dtype=next_critic_obs.dtype,
        )
        if terminal.shape[0] == next_critic_obs.shape[0]:
            terminal = terminal.index_select(0, ids)
        elif terminal.shape[0] != ids.numel():
            raise ValueError("terminal privileged batch must be full [N,P] or compact [K,P]")
        terminal_critic = self._prepare_critic_obs(terminal, None)
        next_critic_obs = next_critic_obs.clone()
        next_critic_obs.index_copy_(0, ids, terminal_critic)
        return next_critic_obs

    @staticmethod
    def _unpack_step(result):
        if not isinstance(result, (tuple, list)) or len(result) != 7:
            raise ValueError("HIM wrapper step must return 7 values")
        return result

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if init_at_random_ep_len and hasattr(self.env, "episode_length_buf"):
            max_len = int(getattr(self.env, "max_episode_length", 1))
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=max_len)

        obs = self._initial_obs.to(self.device)
        critic_obs = self._initial_critic_obs.to(self.device)
        self.alg.actor_critic.train()

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        num_envs = int(getattr(self.env, "num_envs", obs.shape[0]))
        cur_reward_sum = torch.zeros(num_envs, device=self.device)
        cur_episode_length = torch.zeros(num_envs, device=self.device)

        target_iter = self.current_learning_iteration + int(num_learning_iterations)
        for it in range(self.current_learning_iteration, target_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    (
                        next_obs,
                        next_native_critic,
                        rewards,
                        dones,
                        infos,
                        termination_ids,
                        termination_privileged_obs,
                    ) = self._unpack_step(self.env.step(actions))

                    obs = self._as_2d(next_obs, "next actor").to(self.device)
                    critic_obs = self._prepare_critic_obs(next_native_critic, obs).to(self.device)
                    rewards = torch.as_tensor(rewards, device=self.device, dtype=torch.float32).reshape(-1)
                    dones = torch.as_tensor(dones, device=self.device, dtype=torch.bool).reshape(-1)
                    infos = dict(infos or {})

                    next_critic_obs = self._apply_terminal_critic_obs(
                        critic_obs.detach(),
                        termination_ids,
                        termination_privileged_obs,
                    )
                    self.alg.process_env_step(rewards, dones, infos, next_critic_obs)

                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    done_ids = torch.nonzero(dones, as_tuple=False).flatten()
                    if done_ids.numel() > 0:
                        rewbuffer.extend(cur_reward_sum[done_ids].detach().cpu().tolist())
                        lenbuffer.extend(cur_episode_length[done_ids].detach().cpu().tolist())
                        cur_reward_sum[done_ids] = 0.0
                        cur_episode_length[done_ids] = 0.0

                collection_time = time.time() - start
                learn_start = time.time()
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss = self.alg.update()
            learn_time = time.time() - learn_start
            self.tot_timesteps += self.num_steps_per_env * num_envs
            self.tot_time += collection_time + learn_time

            if self.log_dir is not None and it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

        self.current_learning_iteration += int(num_learning_iterations)
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
```

### 5.5 `HIMPPO` 与 storage schema：`next_critic_observations` 候选

候选目标：storage schema 先声明 `next_privileged_observations`，PPO 只按显式 `timeout_bootstrap` mask 做 bootstrap，不把普通 `time_outs` 直接等价为可 bootstrap。

```python
# /home/kk/legged_mjlab/rsl_rl/rsl_rl/storage/him_rollout_storage.py
import torch


class HIMRolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.next_critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        device="cpu",
    ):
        self.device = device
        self.num_envs = int(num_envs)
        self.num_transitions_per_env = int(num_transitions_per_env)
        self.obs_shape = tuple(obs_shape)
        self.privileged_obs_shape = tuple(privileged_obs_shape)
        self.actions_shape = tuple(actions_shape)

        self.observations = torch.zeros(
            self.num_transitions_per_env, self.num_envs, *self.obs_shape, device=device
        )
        self.privileged_observations = torch.zeros(
            self.num_transitions_per_env, self.num_envs, *self.privileged_obs_shape, device=device
        )
        self.next_privileged_observations = torch.zeros(
            self.num_transitions_per_env, self.num_envs, *self.privileged_obs_shape, device=device
        )
        self.actions = torch.zeros(
            self.num_transitions_per_env, self.num_envs, *self.actions_shape, device=device
        )
        self.rewards = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=device)
        self.dones = torch.zeros(
            self.num_transitions_per_env, self.num_envs, 1, device=device, dtype=torch.bool
        )
        self.values = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=device)
        self.returns = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=device)
        self.advantages = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=device)
        self.actions_log_prob = torch.zeros(
            self.num_transitions_per_env, self.num_envs, 1, device=device
        )
        self.mu = torch.zeros(
            self.num_transitions_per_env, self.num_envs, *self.actions_shape, device=device
        )
        self.sigma = torch.zeros(
            self.num_transitions_per_env, self.num_envs, *self.actions_shape, device=device
        )
        self.step = 0

    def add_transitions(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.next_privileged_observations[self.step].copy_(transition.next_critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1).bool())
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.step += 1

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]
        self.advantages = self.returns - self.values
        advantage_std = self.advantages.std(unbiased=False)
        self.advantages = (self.advantages - self.advantages.mean()) / (advantage_std + 1e-8)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // int(num_mini_batches)
        if mini_batch_size < 1:
            raise ValueError("mini_batch_size must be positive")
        indices = torch.randperm(
            int(num_mini_batches) * mini_batch_size,
            requires_grad=False,
            device=self.device,
        )

        observations = self.observations.flatten(0, 1)
        critic_observations = self.privileged_observations.flatten(0, 1)
        next_critic_observations = self.next_privileged_observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for _ in range(int(num_epochs)):
            for i in range(int(num_mini_batches)):
                batch_idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
                yield (
                    observations[batch_idx],
                    critic_observations[batch_idx],
                    actions[batch_idx],
                    next_critic_observations[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                )

    def clear(self):
        self.step = 0
```

```python
# /home/kk/legged_mjlab/rsl_rl/rsl_rl/algorithms/him_ppo.py
# 候选关键片段：act/process_env_step/update。
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.storage import HIMRolloutStorage


class HIMPPO:
    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.actor_critic = actor_critic.to(self.device)
        self.storage = None
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = HIMRolloutStorage.Transition()
        self.num_learning_epochs = int(num_learning_epochs)
        self.num_mini_batches = int(num_mini_batches)
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = HIMRolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    def act(self, obs, critic_obs):
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs.detach()
        self.transition.critic_observations = critic_obs.detach()
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_critic_obs):
        self.transition.next_critic_observations = next_critic_obs.detach().clone()
        self.transition.rewards = rewards.detach().clone()
        self.transition.dones = dones.detach().clone()

        bootstrap_mask = None
        if infos is not None:
            bootstrap_mask = infos.get("timeout_bootstrap")
            if bootstrap_mask is None:
                bootstrap_mask = infos.get("timeout_bootstrap_mask")
        if bootstrap_mask is None:
            bootstrap_mask = torch.zeros_like(self.transition.rewards, dtype=torch.bool)
        else:
            bootstrap_mask = torch.as_tensor(
                bootstrap_mask,
                device=self.device,
                dtype=torch.bool,
            ).reshape(-1)
        if bootstrap_mask.numel() != self.transition.rewards.numel():
            raise ValueError("timeout_bootstrap mask does not match reward batch")

        if bool(bootstrap_mask.any().item()):
            with torch.no_grad():
                next_values = self.actor_critic.evaluate(next_critic_obs).reshape(-1)
            reward_batch = self.transition.rewards.reshape(-1)
            reward_batch += self.gamma * next_values * bootstrap_mask.to(next_values.dtype)
            self.transition.rewards = reward_batch.reshape_as(self.transition.rewards)

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_estimation_loss = 0.0
        mean_swap_loss = 0.0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs,
        )
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            next_critic_obs_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            estimation_loss, swap_loss = self.actor_critic.estimator.update(
                obs_batch,
                next_critic_obs_batch,
                lr=self.learning_rate,
            )

            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze(-1))
            surrogate = -advantages_batch.squeeze(-1) * ratio
            surrogate_clipped = -advantages_batch.squeeze(-1) * torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param,
                    self.clip_param,
                )
                value_loss = torch.max(
                    (value_batch - returns_batch).pow(2),
                    (value_clipped - returns_batch).pow(2),
                ).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_estimation_loss += float(estimation_loss)
            mean_swap_loss += float(swap_loss)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return (
            mean_value_loss / num_updates,
            mean_surrogate_loss / num_updates,
            mean_estimation_loss / num_updates,
            mean_swap_loss / num_updates,
        )
```

注意：这里的 `self.transition.actions` 仍然是 policy raw action，用来和 `actions_log_prob`、`action_mean`、`action_sigma` 保持 PPO 分布一致。幅值 clamp 和动作速率限制不应偷偷塞进 PPO log-prob 之后，而应在 wrapper 的仿真入口形成 `accepted_actions` 并可审查地记录 mask。

### 5.6 `train.py`：使用入口候选

候选目标：入口先 import task 注册，再显式 source gate，再创建 env/runner。训练命令只依赖本项目 task id，不走 `himloco_lab` 的 Hydra/Gym CLI。

```python
# /home/kk/legged_mjlab/legged_mjlab/scripts/train.py
import argparse
from pathlib import Path


def _positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="him_go2")
    parser.add_argument("--device", "--sim-device", dest="device", default=None)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=_positive_int)
    parser.add_argument("--max-iterations", type=_positive_int)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import legged_mjlab.envs  # noqa: F401
    from legged_mjlab.utils.task_registry import load_project_rsl, task_registry

    runners = load_project_rsl()
    if getattr(runners, "HIMOnPolicyRunner", None) is None:
        raise RuntimeError("project old rsl_rl backend did not expose HIMOnPolicyRunner")

    spec = task_registry.get(args.task)
    train_cfg = spec.train_cfg_cls()
    if args.max_iterations is not None:
        train_cfg.runner.max_iterations = args.max_iterations

    env, _ = task_registry.make_env(
        args.task,
        device=args.device,
        play=False,
        num_envs=args.num_envs,
    )
    train_cfg_dict = train_cfg.to_dict()
    runner_cfg = train_cfg_dict["runner"]
    experiment_name = runner_cfg.get("experiment_name", args.task)
    log_dir = Path(args.log_dir) / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    runner = task_registry.make_alg_runner(
        args.task,
        env,
        train_cfg_dict,
        str(log_dir),
    )
    runner.learn(
        num_learning_iterations=runner_cfg["max_iterations"],
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    main()
```

### 5.7 `HIMActorCritic`：actor/critic 模块候选

候选目标：旧 HIM actor 不直接吃完整 history，而是从 history 取当前 actor frame `[N,45]`，拼上 estimator 输出的 base velocity `[N,3]` 和 latent `[N,16]`，得到 actor 输入 `[N,64]`；critic 单独消费 runner 裁好的 `[N,48]`。这里的 `act()` 返回 policy raw action，不在模块内 clamp。

```python
# /home/kk/legged_mjlab/rsl_rl/rsl_rl/modules/him_actor_critic.py
# 候选关键完整类；依赖同目录 actor_critic.py 的 get_activation。
import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.him_estimator import HIMEstimator
from .actor_critic import get_activation


class HIMActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_one_step_obs,
        num_actions,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=1.0,
        min_log_std=-20.0,
        max_log_std=5.0,
        estimator_kwargs=None,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            print(
                "HIMActorCritic.__init__ ignored unexpected arguments: "
                + str(sorted(kwargs))
            )

        self.num_actor_obs = int(num_actor_obs)
        self.num_critic_obs = int(num_critic_obs)
        self.num_one_step_obs = int(num_one_step_obs)
        self.num_actions = int(num_actions)
        if self.num_one_step_obs <= 0:
            raise ValueError("num_one_step_obs must be positive")
        self.history_size = self.num_actor_obs // self.num_one_step_obs
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        if self.num_actor_obs != 270:
            raise ValueError(f"HIM actor history must be 270, got {self.num_actor_obs}")
        if self.num_one_step_obs != 45:
            raise ValueError(f"HIM one-step actor obs must be 45, got {self.num_one_step_obs}")
        if self.history_size != 6:
            raise ValueError(f"HIM history length must be 6, got {self.history_size}")
        if self.num_critic_obs != 48:
            raise ValueError(f"HIM critic obs must be 48, got {self.num_critic_obs}")
        if self.num_actions != 12:
            raise ValueError(f"HIM action dim must be 12, got {self.num_actions}")
        if not math.isfinite(float(init_noise_std)) or float(init_noise_std) <= 0.0:
            raise ValueError("init_noise_std must be finite and positive")

        estimator_kwargs = dict(estimator_kwargs or {})
        self.estimator = HIMEstimator(
            temporal_steps=6,
            num_one_step_obs=45,
            **estimator_kwargs,
        )
        if self.estimator.num_latent != 16:
            raise ValueError(f"HIM latent dim must be 16, got {self.estimator.num_latent}")

        actor_input_dim = 45 + 3 + self.estimator.num_latent
        critic_input_dim = 48
        self.actor = self._make_mlp(actor_input_dim, actor_hidden_dims, 12, activation)
        self.critic = self._make_mlp(critic_input_dim, critic_hidden_dims, 1, activation)

        self.log_std = nn.Parameter(
            torch.full((12,), math.log(float(init_noise_std)))
        )
        self.distribution = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _make_mlp(input_dim, hidden_dims, output_dim, activation_name):
        layers = []
        last_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            layers.append(get_activation(activation_name))
            last_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, int(output_dim)))
        return nn.Sequential(*layers)

    @staticmethod
    def _validate_2d(value, name, width):
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape [N,{width}], got {tuple(value.shape)}")
        if value.shape[-1] != width:
            raise ValueError(f"{name} width mismatch: {value.shape[-1]} vs {width}")
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"{name} contains NaN or Inf")
        return value

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def std(self):
        log_std = torch.nan_to_num(
            self.log_std,
            nan=0.0,
            posinf=self.max_log_std,
            neginf=self.min_log_std,
        ).clamp(self.min_log_std, self.max_log_std)
        return torch.exp(log_std)

    @property
    def action_mean(self):
        if self.distribution is None:
            raise RuntimeError("action distribution has not been built")
        return self.distribution.mean

    @property
    def action_std(self):
        if self.distribution is None:
            return self.std
        return self.distribution.stddev

    @property
    def entropy(self):
        if self.distribution is None:
            raise RuntimeError("action distribution has not been built")
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs_history):
        obs_history = self._validate_2d(obs_history, "actor history", 270)
        with torch.no_grad():
            vel, latent = self.estimator(obs_history)
        current_actor_obs = obs_history[:, :45]
        actor_input = torch.cat((current_actor_obs, vel, latent), dim=-1)
        mean = self.actor(actor_input)
        if not bool(torch.isfinite(mean).all().item()):
            raise FloatingPointError("policy action mean contains NaN or Inf")
        std = self.std.to(device=mean.device, dtype=mean.dtype)
        self.distribution = Normal(mean, torch.ones_like(mean) * std)

    def act(self, obs_history=None, **kwargs):
        self.update_distribution(obs_history)
        actions = self.distribution.sample()
        if not bool(torch.isfinite(actions).all().item()):
            raise FloatingPointError("sampled policy raw actions contain NaN or Inf")
        return actions

    def act_inference(self, obs_history, observations=None):
        obs_history = self._validate_2d(obs_history, "actor history", 270)
        vel, latent = self.estimator(obs_history)
        current_actor_obs = obs_history[:, :45]
        actions_mean = self.actor(torch.cat((current_actor_obs, vel, latent), dim=-1))
        if not bool(torch.isfinite(actions_mean).all().item()):
            raise FloatingPointError("inference policy raw actions contain NaN or Inf")
        return actions_mean

    def get_actions_log_prob(self, actions):
        if self.distribution is None:
            raise RuntimeError("call act() or update_distribution() before log_prob")
        actions = self._validate_2d(actions, "policy raw actions", 12)
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations, **kwargs):
        critic_observations = self._validate_2d(
            critic_observations,
            "critic observations",
            48,
        )
        return self.critic(critic_observations)
```

### 5.8 `HIMEstimator`：history 到 velocity/latent 候选

候选目标：从 actor history `[N,270]` 估计 base velocity `[N,3]` 与 latent `[N,16]`。`update(history, next_critic)` 使用本项目当前 critic layout：`next_obs = next_critic[:, :45]`，`target_vel = next_critic[:, 45:48]`。不要复制 `himloco_lab` 旧切片 `3:48`，因为本项目 runner critic 的前 45 维就是 current actor frame，后 3 维才是 base velocity。

```python
# /home/kk/legged_mjlab/rsl_rl/rsl_rl/modules/him_estimator.py
# 候选关键完整类；保留 SwAV/Sinkhorn 风格的 latent target。
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .actor_critic import get_activation


class HIMEstimator(nn.Module):
    def __init__(
        self,
        temporal_steps,
        num_one_step_obs,
        enc_hidden_dims=(128, 64, 16),
        tar_hidden_dims=(128, 64),
        activation="elu",
        learning_rate=1e-3,
        max_grad_norm=10.0,
        num_prototype=32,
        temperature=3.0,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            print("HIMEstimator.__init__ ignored unexpected arguments: " + str(sorted(kwargs)))

        self.temporal_steps = int(temporal_steps)
        self.num_one_step_obs = int(num_one_step_obs)
        self.num_latent = int(enc_hidden_dims[-1])
        self.learning_rate = float(learning_rate)
        self.max_grad_norm = float(max_grad_norm)
        self.temperature = float(temperature)

        if self.temporal_steps != 6:
            raise ValueError(f"HIM estimator temporal_steps must be 6, got {self.temporal_steps}")
        if self.num_one_step_obs != 45:
            raise ValueError(f"HIM estimator one-step obs must be 45, got {self.num_one_step_obs}")
        if self.num_latent != 16:
            raise ValueError(f"HIM estimator latent dim must be 16, got {self.num_latent}")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")

        self.encoder = self._make_mlp(
            input_dim=270,
            hidden_dims=enc_hidden_dims[:-1],
            output_dim=3 + self.num_latent,
            activation_name=activation,
        )
        self.target = self._make_mlp(
            input_dim=45,
            hidden_dims=tar_hidden_dims,
            output_dim=self.num_latent,
            activation_name=activation,
        )
        self.proto = nn.Embedding(int(num_prototype), self.num_latent)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def _make_mlp(input_dim, hidden_dims, output_dim, activation_name):
        layers = []
        last_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            layers.append(get_activation(activation_name))
            last_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, int(output_dim)))
        return nn.Sequential(*layers)

    @staticmethod
    def _validate_2d(value, name, min_width=None, exact_width=None):
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape [N,F], got {tuple(value.shape)}")
        if exact_width is not None and value.shape[-1] != exact_width:
            raise ValueError(f"{name} width mismatch: {value.shape[-1]} vs {exact_width}")
        if min_width is not None and value.shape[-1] < min_width:
            raise ValueError(f"{name} is narrower than {min_width}: {value.shape[-1]}")
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"{name} contains NaN or Inf")
        return value

    def _history_input(self, obs_history):
        return self._validate_2d(obs_history, "HIM history", exact_width=270)

    def _critic_input(self, next_critic):
        return self._validate_2d(next_critic, "next critic", min_width=48)

    def encode(self, obs_history):
        obs_history = self._history_input(obs_history)
        parts = self.encoder(obs_history.detach())
        pred_vel, latent = parts[:, :3], parts[:, 3:]
        latent = F.normalize(latent, dim=-1, p=2, eps=1e-6)
        return pred_vel, latent

    def forward(self, obs_history):
        pred_vel, latent = self.encode(obs_history)
        return pred_vel.detach(), latent.detach()

    def get_latent(self, obs_history):
        return self.forward(obs_history)

    def update(self, obs_history, next_critic, lr=None):
        if lr is not None:
            self.learning_rate = float(lr)
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate

        obs_history = self._history_input(obs_history)
        next_critic = self._critic_input(next_critic)

        next_obs = next_critic[:, :45].detach()
        target_vel = next_critic[:, 45:48].detach()

        pred = self.encoder(obs_history.detach())
        pred_vel, source_latent = pred[:, :3], pred[:, 3:]
        target_latent = self.target(next_obs)

        source_latent = F.normalize(source_latent, dim=-1, p=2, eps=1e-6)
        target_latent = F.normalize(target_latent, dim=-1, p=2, eps=1e-6)

        with torch.no_grad():
            self.proto.weight.copy_(F.normalize(self.proto.weight, dim=-1, p=2, eps=1e-6))
            q_source = sinkhorn(source_latent @ self.proto.weight.T)
            q_target = sinkhorn(target_latent @ self.proto.weight.T)

        log_p_source = F.log_softmax(
            (source_latent @ self.proto.weight.T) / self.temperature,
            dim=-1,
        )
        log_p_target = F.log_softmax(
            (target_latent @ self.proto.weight.T) / self.temperature,
            dim=-1,
        )

        swap_loss = -0.5 * (q_source * log_p_target + q_target * log_p_source).mean()
        estimation_loss = F.mse_loss(pred_vel, target_vel)
        loss = estimation_loss + swap_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), swap_loss.item()


@torch.no_grad()
def sinkhorn(scores, eps=0.05, iters=3):
    if eps <= 0.0:
        raise ValueError("sinkhorn eps must be positive")
    if scores.ndim != 2:
        raise ValueError("sinkhorn scores must have shape [batch, prototype]")

    q = torch.softmax(scores / eps, dim=-1).T
    num_proto, batch = q.shape
    tiny = torch.finfo(q.dtype).tiny
    q /= q.sum().clamp_min(tiny)
    for _ in range(int(iters)):
        q /= q.sum(dim=1, keepdim=True).clamp_min(tiny)
        q /= num_proto
        q /= q.sum(dim=0, keepdim=True).clamp_min(tiny)
        q /= batch
    return (q * batch).T
```

### 5.9 `play.py`：Legacy HIM play 入口候选

代码观察：`/home/kk/legged_mjlab/legged_mjlab/scripts/play.py` 当前是空文件。下面候选只是入口形状，不等于已落地。它属于 Legacy HIM profile：启动早期调用 `load_project_rsl()`，`make_env(..., play=True)` 创建 HIM wrapper，加载 runner/checkpoint，然后每一步显式消费七元组。

```python
# /home/kk/legged_mjlab/legged_mjlab/scripts/play.py
import argparse
from pathlib import Path

import torch


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="him_go2")
    parser.add_argument("--device", "--sim-device", dest="device", default="cuda:0")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=_positive_int, default=1)
    parser.add_argument("--steps", type=_positive_int, default=1000)
    return parser.parse_args(argv)


def _unpack_him_step(result):
    if not isinstance(result, (tuple, list)) or len(result) != 7:
        raise ValueError("Legacy HIM play expects wrapper step to return 7 values")
    return result


def main(argv=None):
    args = parse_args(argv)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")

    import legged_mjlab.envs  # noqa: F401
    from legged_mjlab.utils.task_registry import load_project_rsl, task_registry

    runners = load_project_rsl()
    if getattr(runners, "HIMOnPolicyRunner", None) is None:
        raise RuntimeError("project old rsl_rl backend did not expose HIMOnPolicyRunner")

    spec = task_registry.get(args.task)
    train_cfg = spec.train_cfg_cls().to_dict()
    env, _ = task_registry.make_env(
        args.task,
        device=args.device,
        play=True,
        num_envs=args.num_envs,
    )
    runner = task_registry.make_alg_runner(args.task, env, train_cfg, log_dir=None)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=getattr(runner, "device", args.device))

    obs, critic = env.reset()
    obs = obs.to(getattr(runner, "device", obs.device))

    with torch.inference_mode():
        for step_id in range(args.steps):
            raw_actions = policy(obs)
            (
                obs,
                critic,
                rewards,
                dones,
                infos,
                done_ids,
                terminal_privileged,
            ) = _unpack_him_step(env.step(raw_actions))
            obs = obs.to(getattr(runner, "device", obs.device))
            if step_id % 100 == 0:
                print(
                    "step",
                    step_id,
                    "mean_reward",
                    torch.as_tensor(rewards).float().mean().item(),
                    "num_done",
                    torch.as_tensor(dones).bool().sum().item(),
                    "num_action_clipped",
                    torch.as_tensor(infos.get("action_clip_mask", [])).bool().sum().item()
                    if isinstance(infos, dict) and "action_clip_mask" in infos
                    else "unknown",
                )

    close = getattr(env, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main()
```

### 5.10 仿真到部署的 action/PD/torque 接口契约

这一节只定义审查契约，不声明可以上真机。F1/F2 的核心是不要把“限幅”写成一个笼统词，而要能沿着数据链审计：policy raw action、accepted/clipped action、delayed target、executed torque 分别是什么，来自哪里，谁负责验证。

| 层 | 数据/配置 | 来源证据 | 契约 | 当前验证状态 |
| --- | --- | --- | --- | --- |
| policy raw action | `[N,12]`，`HIMActorCritic.act()` / `act_inference()` 输出 | 候选第 5.7 节；当前配置 `num_actions=12` 见 `/home/kk/legged_mjlab/legged_mjlab/envs/him_go2/him_go2_config.py:14` | 只要求 shape 正确且 finite；这是 PPO 分布样本/均值，不保证幅值安全，不得直接发硬件。 | NOT_RUN |
| accepted/clipped action | `[N,12]`，进入 native env 的动作 | 候选第 5.3 节；MJLab action 构建见 `/home/kk/legged_mjlab/legged_mjlab/envs/him_go2/him_go2_env.py:200-224` | wrapper 先 clamp raw action 到 `[-action_clip, action_clip]`，再按 `max_action_rate` 可选限制相邻 policy step 的增量；`infos` 必须记录 raw/clipped/accepted 及 mask。 | NOT_RUN |
| delayed target | 关节位置目标，通常是 `default_dof_pos + scale * accepted_action`，再经过仿真动作延迟/保持 | default joint 与 action scale 见 `him_go2_config.py:61-87`；动作延迟随机化见 `him_go2_config.py:166-170` | 训练和 play 必须明确 hip/非 hip scale、decimation、action latency 是否启用；延迟后的 target 不等于 raw action。 | NOT_RUN |
| executed torque / effort | PD 执行后的 actuator force/torque | PD/effort 标称值见 `him_go2_config.py:78-85`；PD 和 effort DR 见 `him_go2_config.py:143-155`、`him_go2_env.py:424-445`；torque limit reward 读取 actuator force/limit 见 `him_go2_env.py:910-934` | torque/effort limit 由 actuator/MJLab 执行层负责，不能由 policy action clamp 替代；需要验证实际 actuator `force_limit` 与 reward/gate 使用同一 joint order。 | NOT_RUN |
| deploy YAML | `rl_kp`、`rl_kd`、`action_scale`、`hip_scale`、`torque_limits` | `/home/kk/legged_mjlab/deploy/deploy_mujoco/him_go2/policy/config.yaml:14-34` | 部署前必须把 sim config 和 deploy YAML 做一致性验证；例如 sim calf effort limit `45` 与 deploy YAML calf `35.55` 这种差异必须解释、修正或显式接受。 | BLOCKED |
| 真实硬件 gating | 急停、watchdog、传感器新鲜度、通信断连、分阶段上电、低速/低增益首跑 | 不在本文执行 | 硬件层必须再次检查 action、target、velocity、torque/effort、温度/电流和状态机；仿真 wrapper clamp 不能作为硬件安全证明。 | BLOCKED |

一致性 gate 的最低要求：从 sim 配置导出一张 `joint_order, default_angle, action_scale, kp, kd, effort_limit` 表，再从 deploy YAML 读出同样字段，逐 joint 对比。差异只能有三种处理：改配置、写明工程理由并签审、或禁止部署。当前还没有这张对比表，所以 deploy/sim2real 仍是 `BLOCKED`。

额外注意：当前 `_build_actions()` 读取 `self.robot_cfg.control.action_clip`，见 `/home/kk/legged_mjlab/legged_mjlab/envs/him_go2/him_go2_env.py:211-224`；而基础配置里可直接看到的是 `normalization.clip_actions`，见 `/home/kk/legged_mjlab/legged_mjlab/envs/base/legged_mjlab_config.py:194-203`。这意味着 native action clip 的配置来源也要进入 gate，不能只因为 wrapper 候选有 `action_clip=1.0` 就认为执行层已经一致。

## 6. 在 `/home/kk/legged_mjlab` 中使用

下面命令是使用说明，不是本文已执行的成功记录，并且全部默认处在 Legacy HIM profile 内。先从最小来源检查开始，不要一上来跑 4096 env 长训练；`cuda:0` 只是有可用 GPU 时的示例，当前文档修订没有验证 CUDA、MuJoCo Warp 或显存余量。

### 6.1 安装/环境前置

```bash
cd /home/kk/legged_mjlab
source .venv/bin/activate
python -m pip install -e .
```

判断点：

| 检查 | 预期 |
| --- | --- |
| `python -m pip show mjlab` | 版本应为 `1.6.0`。 |
| `python -m pip show rsl-rl-lib` | 可能存在 `5.4.2`，这是 MJLab 依赖，不代表当前训练会使用它。 |
| `python -m pip show rsl-rl` 或 `python -m pip show rsl_rl` | 可能显示旧 fork `1.0.2`，但仍不能替代 `rsl_rl.__file__` 来源检查。 |

### 6.2 来源检查：确认训练将使用本地旧 `rsl_rl`

```bash
cd /home/kk/legged_mjlab
source .venv/bin/activate
python - <<'PY'
import importlib
from importlib import metadata

from legged_mjlab.utils.task_registry import load_project_rsl

runners = load_project_rsl()
import rsl_rl

print("active rsl_rl:", rsl_rl.__file__)
print("active runners:", runners.__file__)
print("has HIMOnPolicyRunner:", hasattr(runners, "HIMOnPolicyRunner"))
for dist_name in ("mjlab", "rsl-rl-lib", "rsl_rl"):
    try:
        print(dist_name, metadata.version(dist_name))
    except metadata.PackageNotFoundError:
        print(dist_name, "<not installed>")
PY
```

预期判断：

| 输出 | 解释 |
| --- | --- |
| `active rsl_rl` 指向 `/home/kk/legged_mjlab/rsl_rl/rsl_rl/__init__.py` 或同一源码目录 | PASS，可以继续 wrapper/import smoke。 |
| `active rsl_rl` 指向 `.venv/lib/python3.11/site-packages/rsl_rl/__init__.py` 且 `has HIMOnPolicyRunner: False` | BLOCKED，入口没有成功切到旧 HIM backend。 |
| `rsl-rl-lib 5.4.2` 存在 | 正常；它是 MJLab 依赖，但不应成为 HIM runner 的 active backend。 |

### 6.3 静态 import smoke

```bash
cd /home/kk/legged_mjlab
source .venv/bin/activate
python - <<'PY'
import legged_mjlab.envs
from legged_mjlab.utils.task_registry import load_project_rsl, task_registry

runners = load_project_rsl()
print("tasks:", task_registry.list_tasks())
print("runner file:", runners.__file__)
print("HIM runner:", runners.HIMOnPolicyRunner)
PY
```

这一步只验证 registry/import/source gate，不验证 MJLab env 构造、reset、step 或训练。

### 6.4 最小 reset/step smoke

如果机器没有可用 GPU，MJLab/MuJoCo Warp 可能无法真实构造环境。下面的 `cuda:0` 是 GPU 环境示例；CPU/无 CUDA 的机器不能因为文档里有命令就断言 PASS。优先用最小 env 数和最短迭代，不要直接开默认 4096 env。

```bash
cd /home/kk/legged_mjlab
source .venv/bin/activate
python - <<'PY'
import torch
import legged_mjlab.envs
from legged_mjlab.utils.task_registry import load_project_rsl, task_registry

load_project_rsl()
env, env_cfg = task_registry.make_env("him_go2", device="cuda:0", num_envs=1)
obs, critic = env.reset()
print("reset obs:", tuple(obs.shape))
print("reset critic:", tuple(critic.shape))
actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
result = env.step(actions)
print("step tuple len:", len(result))
print("step shapes:", tuple(result[0].shape), tuple(result[1].shape), tuple(result[2].shape), tuple(result[3].shape))
env.close()
PY
```

预期：

| 项 | 预期 |
| --- | --- |
| reset obs | `(1, 270)` |
| reset critic | `(1, 235)` |
| step tuple len | `7` |
| step actor/critic/reward/done | `(1,270)`、`(1,235)`、`(1,)`、`(1,)` |

### 6.5 短训练 smoke

```bash
cd /home/kk/legged_mjlab
source .venv/bin/activate
python -m legged_mjlab.scripts.train \
  --task him_go2 \
  --device cuda:0 \
  --num-envs 1 \
  --max-iterations 1 \
  --log-dir /tmp/legged_mjlab_him_smoke
```

若这一步通过，只能说明最小 rollout/update 跑通；还不能说明 reward、domain randomization、长期稳定、checkpoint/export 或部署可用。

显存边界：`him_go2_config.py:9` 默认 `num_envs=4096`，`him_go2_config.py:228` 默认 `num_steps_per_env=100`。仅 rollout storage 中 actor history `[4096,100,270]` 就是数百 MB 级 float32 张量，再加 critic/next_critic/actions/mu/sigma、环境状态、地形、模型和 optimizer，不能把默认规模作为首个 smoke。当前环境没有完成 CUDA/显存验证，所以第 8 节仍保持 `NOT_RUN`。

### 6.6 正式训练命令

```bash
cd /home/kk/legged_mjlab
source .venv/bin/activate
python -m legged_mjlab.scripts.train \
  --task him_go2 \
  --device cuda:0 \
  --log-dir logs
```

默认配置会使用 `/home/kk/legged_mjlab/legged_mjlab/envs/him_go2/him_go2_config.py:9` 的 `num_envs=4096` 和 `:225-230` 的 runner 设置：`num_steps_per_env=100`、`max_iterations=10000`。在 smoke 未通过前，不建议直接跑这个规模。

## 7. 风险/审查表

| 项 | 状态 | 风险 | 必要审查 |
| --- | --- | --- | --- |
| Profile/ABI 混接 | BLOCKER | Legacy HIM 七元组与新版 `rsl-rl-lib` TensorDict/四元 step ABI 不可混接；runner、storage、checkpoint 任一层混用都会破坏语义。 | 按第 1.1 节选择单一 profile；Legacy HIM 只能使用项目内旧 `rsl_rl`、`HIMOnPolicyRunner`、`HIMPPO`、`HIMRolloutStorage` 和七元组。 |
| 包来源 | PENDING | `rsl-rl-lib==5.4.2` 与本地 `rsl_rl==1.0.2` 同名顶层包冲突；直接 import 可能拿到新包。 | 每次训练前在独立新 Python 进程、单线程启动期调用 `load_project_rsl()`，打印 `rsl_rl.__file__`、`runners.__file__`、`hasattr(HIMOnPolicyRunner)`。 |
| `load_project_rsl()` 并发/source gate | PENDING | 候选会触碰 `sys.modules`；若训练/export 线程已启动，或不兼容 `rsl_rl.*` 子模块已导入，继续替换会产生混合命名空间；即使顶层 `rsl_rl` 已来自项目源码，也不能跳过这些检查。 | `load_project_rsl()` 开头必须做启动早期单线程断言，候选最小检查为 `threading.active_count() == 1`；先扫描 `_incompatible_imported_submodules(package_dir)`，发现非项目源码子模块时 fail closed 并要求重启，随后才允许已加载项目包快速返回；`_load_project_rsl_source()` 也必须依赖同一前置检查。 |
| terminal auto-reset | PENDING | native env 若 auto-reset 后才返回 obs，done 行 `next_obs["critic"]` 是 reset 后状态，不能作为 terminal critic。 | wrapper 必须关闭 auto-reset 或读取 `infos` 中 reset 前 terminal critic；缺失则 BLOCKED。 |
| finite-horizon timeout | BLOCKER | 旧 HIM/`himloco_lab` 用 `time_outs` bootstrap；有限时域下如果仍对 `truncated` bootstrap，会改变任务数学定义。 | 使用等价逻辑：`is_finite_horizon=True` 时 `time_outs=False[N]`、`timeout_bootstrap=False[N]`；只有非有限时域才允许 `truncated & ~terminated & terminal_available`。 |
| shape | PENDING | `[45,270,235,48]` 任一漂移都会导致 estimator、critic 或 actor 输入错位。 | reset/step 单测断言 actor `[N,270]`、native critic `[N,235]`、runner critic `[N,48]`、action `[N,12]`。 |
| storage schema | PENDING | 如果 `next_critic_observations` 没有预分配/写入/mini-batch 传出，estimator update 实际拿不到目标。 | 检查 Transition、storage tensor、`add_transitions()`、feedforward/recurrent generator、`HIMPPO.update()`。 |
| action clamp / rate | PENDING | policy raw action 只做 shape/finite 检查不足；如果候选实现保留可关闭幅值 clamp 的分支，异常幅值会直接进入仿真控制。 | wrapper 必须在 native env 前生成 `accepted_actions`；`action_clip` 默认 `1.0`，必须是 finite positive float 且不允许 `None`；`_sanitize_actions()` 无条件幅值 clamp，仅动作速率限制可选，并记录 raw/clipped/accepted 与 mask。 |
| PD/effort/torque 契约 | BLOCKED | “限幅”不是单一接口；policy action、关节位置 target、PD 后 torque/effort 是不同层。 | 明确 raw action、accepted action、delayed target、executed torque 的 shape、单位、joint order 和来源；验证 actuator `force_limit` 与 reward/gate 同源。 |
| sim/deploy YAML 一致性 | BLOCKED | deploy YAML 的 `rl_kp/rl_kd/action_scale/hip_scale/torque_limits` 可能与训练配置不一致；例如 calf torque limit 差异需要解释。 | 生成逐 joint 对比表；不一致必须修正、签审或禁止部署。 |
| CUDA/显存 | NOT_RUN | 默认 `4096 env * 100 steps` 的 storage 和 MJLab env 状态显存压力高；当前环境未验证 CUDA。 | smoke 使用 `num_envs=1`、`max_iterations=1`；默认规模训练前记录 GPU 型号、可用显存、实际峰值和退出码。 |
| checkpoint/export | NOT_RUN | 旧 checkpoint 保存单体 `HIMActorCritic`、optimizer、estimator optimizer；迁移后键可能不兼容。 | 加 schema version、键映射、shape 检查；不可转换时显式拒绝。 |
| 部署安全 | BLOCKED | 训练/仿真成功不能证明真实 Go2 部署安全；仿真 wrapper clamp 不是硬件安全层。 | 真实硬件前必须另行确认急停、硬件侧 action/target/torque 限制、watchdog、NaN/Inf、断连安全态、传感器新鲜度和分阶段上电。 |
| 未运行项 | NOT_RUN | 本文只写文档，没有运行 wrapper 单测、reset/step 或训练。 | 按第 8 节 gate 逐项执行并记录实际命令、退出码和日志。 |

## 8. 验收 gate

| Gate | 状态 | 命令/检查 | 通过条件 |
| --- | --- | --- | --- |
| 文档验证 | PASS | 本文已包含边界声明、Profile 禁止混接、`himloco_lab` 路径证据、当前事实、推荐路线、候选代码块、使用步骤、风险表和 gate。 | 文档文件存在且 `Legacy HIM profile`、`TensorDict`、`action_clip_mask`、`is_finite_horizon`、`HIMEstimator`、`play.py` 等 required keywords 可检索。 |
| Profile ABI | NOT_RUN | 静态检查训练/play 入口、runner、storage 和 wrapper。 | Legacy HIM profile 内不得出现新版 TensorDict 四元 step、`OnPolicyRunner` storage 或新版 checkpoint schema；MJLab ABI migration profile 另开分支。 |
| source gate 静态 import | NOT_RUN | 第 6.3 节命令，且必须是独立新 Python 进程和启动早期单线程上下文。 | task 注册可列出 `him_go2`；`load_project_rsl()` 开头先通过 `threading.active_count() == 1` 最小断言，再扫描不兼容 `rsl_rl.*` 子模块，随后 runner 指向本地旧 HIM backend；若已污染非项目源码子模块，必须 fail closed。 |
| wrapper 单测 | NOT_RUN | 运行现有 `legged_mjlab/test/test_him_wrapper_terminal_privileged.py` 或补充同等测试。 | terminal full batch、compact batch、no done、shape mismatch 均按预期；finite-horizon 与非 finite-horizon 两类 timeout case 都覆盖。 |
| action clamp/rate 单测 | NOT_RUN | 构造空值、`0`、`inf` 等非法 `action_clip` 参数，以及超幅、NaN/Inf、shape mismatch、相邻 step 大跳变动作。 | 非法 `action_clip` fail closed；NaN/Inf 和 shape mismatch fail closed；超幅动作进入 env 前始终被 clamp；启用 rate limit 时 `accepted_actions` 增量受限；`infos` 中 raw/clipped/accepted/mask 可审查。 |
| 最小 reset/step | NOT_RUN | 第 6.4 节命令；`cuda:0` 仅在 GPU 可用时执行。 | reset/step shape 与七元组契约一致，无 NaN/Inf、无 source drift；CPU/无 CUDA 环境不能标记 PASS。 |
| 短训练 | NOT_RUN | 第 6.5 节命令；优先 `num_envs=1`、`max_iterations=1`。 | 1 env、1 iteration 完成一次 rollout/update/save，不出现 shape/device/NaN 错误；记录 GPU/CPU 环境、峰值显存和退出码。 |
| finite-horizon 数学 gate | NOT_RUN | 单测构造 `terminated/truncated/is_finite_horizon/terminal_available` 四维组合。 | `is_finite_horizon=True` 时 bootstrap 全 false；非有限时域仅 `truncated & ~terminated & terminal_available` 为 true。 |
| HIM module gate | NOT_RUN | 单测 `HIMActorCritic` 和 `HIMEstimator`。 | actor 输入拼接 `45 + 3 + 16`，critic 输入 `48`；estimator update 使用 `next_critic[:, :45]` 与 `next_critic[:, 45:48]`，不出现 `3:48` 旧切片。 |
| play 入口 | NOT_RUN | 落地候选 `play.py` 后加载一个合法 checkpoint。 | 入口早期单线程调用 `load_project_rsl()`，`make_env(play=True)`，`runner.load(..., load_optimizer=False)`，每步消费七元组，且不绕过 wrapper 强制 action clamp。 |
| sim/deploy 一致性 | BLOCKED | 对比 sim 配置和 `deploy/deploy_mujoco/him_go2/policy/config.yaml`。 | 逐 joint 对齐 default angle、action scale、hip scale、kp、kd、effort/torque limit、joint order；差异有签审记录，否则禁止部署。 |
| checkpoint/export | NOT_RUN | 加载旧 checkpoint、保存新 checkpoint、导出 policy。 | schema version 和 key/shape 显式检查；不能静默加载不兼容权重。 |
| 部署安全 | BLOCKED | 不在本文执行。 | 未满足硬件急停、watchdog、硬件侧限幅/限速/限 torque、传感器新鲜度、断连安全态和分阶段上电门槛前禁止声明可部署。 |

当前不能发出 `[ALL_TESTS_PASSED]`：本文只完成文档方案，没有完成运行时验证。
