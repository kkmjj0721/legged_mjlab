# 将本地 `rsl_rl` fork 完整改造成 MJLab 风格架构

本文给出 `/home/kk/legged_mjlab` 的目标架构、文件处置、接口骨架和迁移顺序。这里的“完整转换”不是把几个类名替换成新名字，而是把训练运行时的顶层 ABI、环境返回值、观测分组、算法构造、rollout storage、checkpoint 和导出路径统一到 MJLab 支持的 `rsl-rl-lib==5.4.2`。

先说结论：最终运行进程中的 canonical `rsl_rl` 必须由 MJLab 支持的 `rsl-rl-lib==5.4.2` 提供。HIM 的历史编码、estimator、专用 PPO 损失和终止状态处理应迁移到 `legged_mjlab.rl.him` 这类显式项目命名空间，并通过 qualified import 接入；不能再由 `task_registry.load_project_rsl()` 在 import 时删除、重载或替换 `rsl_rl.*`。这样做既保留 MJLab 的 `TensorDict`、`obs_groups`、`OnPolicyRunner`、`PPO`、`RolloutStorage`、`MLPModel`、`VecEnv` 契约，也让 HIM 的扩展点可定位、可测试、可序列化。

本文是设计和实施骨架，不是已完成的代码变更。当前回合只新增本 Markdown 文件；运行时训练、rollout、checkpoint 转换、JIT 和 ONNX 导出仍需后续执行并记录结果。

## 1. 目标、边界与当前事实

### 1.1 目标运行链

最终数据流应当是单向且可追踪的：

```text
MJLab ManagerBasedRlEnv
    native observation groups + native 5-return step
                    │
                    ▼
RslRlVecEnvWrapper
    TensorDict + 4-return VecEnv.step()
                    │
                    ▼
legged_mjlab.rl.him.HIMVecEnvWrapper   (仅 HIM task 使用)
    history / critic projection / terminal metadata
                    │
                    ▼
MJLab MjlabOnPolicyRunner
    alg.act(obs)
    alg.process_env_step(obs, rewards, dones, extras)
    alg.compute_returns(obs)
                    │
                    ▼
legged_mjlab.rl.him.HIMPPO
    installed PPO math + explicit HIM estimator/storage extension
```

普通任务可以在 `RslRlVecEnvWrapper` 后直接进入 `MjlabOnPolicyRunner`；HIM 只增加显式的 `HIMVecEnvWrapper` 和 `HIMPPO`，不改变全局 `rsl_rl` 所属包。

因此，“转换本地 fork”在这里指迁移其中仍有价值的 HIM 行为和参数语义，而不是保留一份与上游完全平行的通用 RL 实现。如果确实需要发布独立分发包，也应使用项目专属分发名和 import path；它可以依赖 `rsl-rl-lib`，但不能再导出顶层 `rsl_rl` 来竞争 MJLab 的依赖。

### 1.2 已知的 ABI 差异

| 边界 | MJLab / `rsl-rl-lib==5.4.2` | 当前本地 fork | 完整改造的方向 |
| --- | --- | --- | --- |
| 顶层包 | 安装包提供 canonical `rsl_rl` | 仓库内 `rsl_rl/rsl_rl` 也声明同名包 | 安装包拥有顶层名称；项目扩展使用 `legged_mjlab.rl.him`。 |
| 观测 | `TensorDict`，由具名 group 组成 | 位置式 `num_obs`、`num_privileged_obs` 和独立 buffer | 先定义 group，再由 `obs_groups` 选择模型输入。 |
| 环境 step | `VecEnv.step()` 返回 `(obs, rewards, dones, extras)` 四项 | HIM wrapper 路径向 runner 暴露七项 | 七项内部信息压入 `extras`，对外固定四项。 |
| runner | `OnPolicyRunner` 调用 `construct_algorithm()`、`act(obs)` 和四参 `process_env_step()` | `HIMOnPolicyRunner` 自己编排 actor/critic 和专用参数 | 使用 MJLab 的 `MjlabOnPolicyRunner` 主循环；HIM 编排下沉到算法和 adapter。 |
| 算法 | `PPO(actor, critic, storage, ...)`，模型按 `MLPModel` 接口工作 | `HIMPPO` 持有单体 `HIMActorCritic` | `HIMPPO` 继承或严格实现新式 `PPO` 合约，并显式扩展 target 数据。 |
| storage | `RolloutStorage` 用 `TensorDict` 和命名 `Batch` | `HIMRolloutStorage` 使用平行 tensor 和旧字段名 | 在 MJLab storage schema 上添加 `next_critic`、terminal provenance 和 mask。 |
| 模型 | `MLPModel` 分离 actor / critic，支持 `as_jit()`、`as_onnx()` | `HIMActorCritic` 将 actor、critic、estimator、分布合为单体 | 拆成可解析的 actor、critic、estimator；为 actor 提供 export surface。 |
| 配置 | `RslRlOnPolicyRunnerCfg`、`RslRlModelCfg`、`RslRlPpoAlgorithmCfg` | 旧式 `runner_class_name`、`policy_class_name`、`algorithm_class_name` | 使用 dataclass 配置和 qualified `class_name`，显式填写 `obs_groups`。 |
| 包加载 | 一次导入即可确定来源 | `load_project_rsl()` 动态换源 | 删除运行时 namespace swap；来源检查只读、失败即报错。 |

安装侧的参考文件是 `.venv/lib/python3.11/site-packages/mjlab/rl/vecenv_wrapper.py`、`.venv/lib/python3.11/site-packages/mjlab/rl/{runner.py,config.py}` 以及同一环境中的 `rsl_rl/env/vec_env.py`、`algorithms/ppo.py`、`storage/rollout_storage.py`。它们是 ABI 参考，不应被复制成第二套 canonical 包。

### 1.3 非目标

本迁移只处理 RL runtime 的包边界和 HIM 接入。它不重新设计 MJLab 场景、奖励、传感器、控制器或任务物理语义；现有任务语义应在独立的环境迁移中冻结并回归。不要把未在本契约中的新观测通道或新动力学扰动作为本转换的验收条件。

## 2. 冻结的接口契约

在移动网络代码之前先冻结以下契约。任何字段若要改名或改 shape，必须同步更新 config、storage、checkpoint manifest 和 export wrapper。

### 2.1 包与导入所有权

```python
# 合法：项目代码显式依赖已安装 ABI，并以 qualified name 引入 HIM。
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from legged_mjlab.rl.him.estimator import HIMEstimator
from legged_mjlab.rl.him.ppo import HIMPPO

# 非目标：不要在任何任务入口执行 sys.modules 删除、canonical package 重载，
# 也不要把项目目录安装成另一个顶层 rsl_rl。
```

启动时可以记录 `rsl_rl.__file__`、版本和 `mjlab.__file__`，但这只是诊断信息。诊断失败应停止构造 runner，而不是寻找另一个同名包作为 fallback。

### 2.2 环境和 TensorDict

`RslRlVecEnvWrapper` 是标准边界。其外部方法必须满足：

| 方法 | 输入 | 输出 | shape / 时序约束 |
| --- | --- | --- | --- |
| `get_observations()` | 无 | `TensorDict` | 每个叶子至少有 batch `[N, F]`；group 在同一 device。 |
| `reset()` | 无，或由 wrapper 内部处理 partial reset | `(TensorDict, extras)` | reset 后提供当前 episode 的第一帧；不返回 actor/critic 位置元组。 |
| `step(actions)` | `torch.Tensor [N, A]` | `(obs, rewards, dones, extras)` | 恰好四项；`rewards` 和 `dones` 与 `N` 对齐。 |
| `extras` | 字典 | 终止、日志和可选训练元数据 | 所有扩展字段具名；不得依赖第 5、6、7 个返回位置。 |

底层 MJLab 环境仍可能产生 `(obs_dict, reward, terminated, truncated, extras)` 五项；五项只存在于 wrapper 内部解包。对外的 `dones` 定义为：

```text
dones = terminated | truncated
```

如果任务不是 finite horizon，供 bootstrap 使用的 mask 应为 `truncated & ~terminated`。原始 `terminated` 和 `truncated` 必须同时放入 `extras`，以便 HIM 算法区分“结束原因”和“可否 bootstrap”。有限时域任务不得因为存在 `truncated` 就自动取得下一状态 value。

### 2.3 观测 group 和 shape

当前 HIM-Go2 迁移候选的 shape 层次保持如下；它们是不同边界，不能互相覆盖：

| 层 | group / 字段 | shape | 含义 |
| --- | --- | --- | --- |
| native policy frame | `policy_frame` | `[N, 45]` | manager 每个 policy step 产生的一帧 actor 输入。 |
| native critic source | `critic_native` | `[N, 235]` | 完整 critic 源，保留给 adapter、诊断或将来的 value 配置。 |
| HIM actor input | `policy` | `[N, 270]` | 6 帧 × 45，current-first，历史由 HIM adapter 维护。 |
| HIM estimator / PPO critic view | `critic` | `[N, 48]` | 明确投影为 45 维当前帧加 3 维 base velocity。 |
| action | — | `[N, 12]` | 与环境 action manager 顺序一致。 |

`critic_native` 到 `critic` 的投影必须是有名 helper，而不是在 runner 中散落 `[:, :48]`。如果未来要让 value model 使用完整 235 维，应新建配置和模型输入契约；不能静默把 187 个额外值混入既有 48 维 estimator 输入。

HIM adapter 的输出建议为：

```text
TensorDict(
    {
        "policy":       [N, 270],
        "critic":       [N, 48],
        "critic_native": [N, 235],
    },
    batch_size=[N],
)
```

因此最小的 `obs_groups` 是：

```python
obs_groups = {
    "actor": ("policy",),
    "critic": ("critic",),
}
```

安装版 `resolve_obs_groups()` 会检查 group 是否存在，但仍可能为缺失配置提供 fallback。迁移后的配置必须显式写出上述两个键，不能把 fallback 当作契约。

### 2.4 terminal critic 的语义

底层环境若开启 auto-reset，done 行的普通 `obs["critic"]` 可能已经是新 episode 的 reset frame。它不能冒充当前 transition 的 terminal frame。HIM adapter 必须在 reset 前捕获真实 terminal critic，或者明确标记该行不可用。

建议的 `extras` 扩展如下：

```text
extras["terminated"]         : bool [N]
extras["truncated"]          : bool [N]
extras["time_outs"]          : bool [N]   # 只在可 bootstrap 的时域语义下提供
extras["terminal_critic"]    : float [N, 48]
extras["terminal_available"] : bool [N]
```

如果生产端只有紧凑的 `[K, 48]` terminal tensor，必须同时提供唯一且范围正确的 `terminal_env_ids[K]`，在 adapter 内展开为 `[N, 48]` 和 `[N]`。没有 env-id 对齐证明时，`terminal_available` 必须为 false。

算法使用的中间 mask 固定为：

```text
timeout_bootstrap = (~terminated) & time_outs & terminal_available
```

这条 mask 只影响 timeout reward correction；普通 GAE 仍按 `dones` 处理 episode 边界。任何 `terminated=True` 的行，即使同时 `truncated=True`，都不能 bootstrap。

## 3. 文件处置矩阵

下表是“完整转换”后的目标处置。`move` 表示保留语义但改到显式 namespace；`replace` 表示当前文件名可以保留，但实现必须完全换成新契约；`retire` 表示不再参与安装或运行时导入；`keep` 表示内容仍有价值，但不承担 canonical ABI。

| 当前路径 | 处置 | 最终位置 / 处理 | 说明 |
| --- | --- | --- | --- |
| `rsl_rl/`（整个本地 fork） | `retire` | 不再作为顶层 `rsl_rl` 安装；只在迁移期间作为来源 | 先移动 HIM 专有实现，再从根包的 `packages` 和 `package_dir` 中移除。 |
| `rsl_rl/rsl_rl/algorithms/ppo.py` | `retire` | 使用安装包 `rsl_rl.algorithms.PPO` | 不维护第二份通用 PPO。 |
| `rsl_rl/rsl_rl/runners/on_policy_runner.py` | `retire` | 使用安装包 `rsl_rl.runners.OnPolicyRunner`，由 `mjlab.rl.runner.MjlabOnPolicyRunner` 扩展 | 旧位置式训练循环不再是入口。 |
| `rsl_rl/rsl_rl/runners/him_on_policy_runner.py` | `move` → `retire` | 把终止处理、日志或 shape 校验拆到 `legged_mjlab/rl/him/ppo.py` 和 `wrapper.py`；最终删除专用 runner | `HIMOnPolicyRunner` 只作为迁移参考，不作为最终 class name。 |
| `rsl_rl/rsl_rl/algorithms/him_ppo.py` | `move` | `legged_mjlab/rl/him/ppo.py` | 改为 `HIMPPO.construct_algorithm(obs, env, cfg, device)`、`act(obs)`、四参 `process_env_step()`。 |
| `rsl_rl/rsl_rl/storage/him_rollout_storage.py` | `move` | `legged_mjlab/rl/him/storage.py` | 继承安装版 `RolloutStorage` 的 schema，并增加显式 `next_critic` 及 terminal 字段。 |
| `rsl_rl/rsl_rl/modules/him_actor_critic.py` | `move` → `replace` | `legged_mjlab/rl/him/models.py` | 拆出 `HIMActorModel`、critic adapter 和 export module；单体类只能保留为临时转换器。 |
| `rsl_rl/rsl_rl/modules/him_estimator.py` | `move` | `legged_mjlab/rl/him/estimator.py` | 去掉对 `rsl_rl.modules` 相对路径的依赖；把 `loss()` 与 `optimizer.step()` 分开。 |
| `rsl_rl/rsl_rl/modules/actor_critic*.py`、`utils/`、`setup.py` | `retire` | 由安装包对应模块替代 | 避免旧 API 通过同名 import 被重新带入。 |
| `mjlab_rsl_rl/` | `retire` | 不进入最终安装；可在迁移期间作为差异清单 | 当前只有 `rsl_rl/env/vec_env.py` 和空 `setup.py`，不是完整目标包，不能作为 fallback。 |
| `legged_mjlab/utils/task_registry.py` | `replace` | 保留 registry 职责，删除 `load_project_rsl()` 及 `sys.modules` 操作 | 通过显式 `runner_factory`、`wrapper_factory` 和 qualified algorithm name 组装任务。 |
| `legged_mjlab/envs/him_go2/him_go2_config.py` | `replace` | 使用 `RslRlOnPolicyRunnerCfg` 及其嵌套 model/algorithm config | 旧 `runner_class_name`、`policy_class_name` 等字段只在迁移转换器中读取。 |
| `legged_mjlab/wrappers/rsl_rl_wrapper.py` | `replace` | 薄的项目兼容入口，委托或等价实现 `mjlab.rl.vecenv_wrapper.RslRlVecEnvWrapper` | 对外必须是 `TensorDict`、4-return、标准 `VecEnv`。 |
| `legged_mjlab/wrappers/him_wrapper.py` | `move` → `replace` | `legged_mjlab/rl/him/wrapper.py` | 保留 history 和 terminal 语义，但不再返回七项或依赖旧 runner。 |
| `legged_mjlab/wrappers/vec_env_wrapper.py` | `replace` | 若仍被其他 task 使用，改为普通项目 wrapper 基类；不得伪装成 `rsl_rl` ABI | 需要逐个调用方确认，不能让旧基类隐式决定返回值。 |
| `legged_mjlab/scripts/train.py`、`play.py`、`utils/exporter.py` | `replace` | 直接构造安装版 runner 和显式 HIM adapter | 删除 `load_project_rsl()` 调用，并统一新 checkpoint/export 入口。 |
| `docs/rsl_rl_mjlab_兼容迁移方案.md` | `keep` | 标记为早期迁移记录，并链接本文 | 其中的 ABI 观察仍有用；不要把它当成最终文件布局的唯一规范。 |
| `docs/him_go2_env_续写与外部_rsl_rl_接入.md` | `keep` | 作为环境迁移和审计档案 | 其大段候选代码不是本次转换的已落地实现。 |
| `docs/目录结构对比.md`、`docs/设计.md` | `replace` | 更新为最终目录和包所有权说明 | 删除会暗示本地 fork 是 canonical backend 的旧描述。 |
| `docs/setup.md`、`docs/uv使用.md` | `replace` | 统一依赖安装、版本锁定和来源检查 | 安装路径只能产生一个顶层 `rsl_rl`。 |
| `docs/前言.md` 及新增本文 | `keep` | 本文作为最终转换设计入口 | 仍需在实际实现后补充版本、测试和发布记录。 |

文件矩阵中的 `move` 和 `replace` 都是迁移阶段动作，不代表当前工作树已经完成。尤其是 `rsl_rl/` 和 `mjlab_rsl_rl/` 在删除前必须先完成 import、checkpoint 和导出回归。

## 4. Final target code layout

下面的树是目标 end-state，而不是要求本回合立即创建的文件清单。安装的第三方包位于虚拟环境外部的依赖层；项目只保存显式扩展。

```text
.
├── pyproject.toml                         # 唯一项目元数据和依赖入口（目标）
├── legged_mjlab/
│   ├── __init__.py
│   ├── envs/
│   │   └── him_go2/
│   │       ├── __init__.py
│   │       ├── him_go2_config.py           # MJLab env cfg + RSL-RL cfg
│   │       └── him_go2_env.py
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── contracts.py                    # group names, shapes, mask validators
│   │   ├── mjlab_wrapper.py                # optional project re-export/thin shim
│   │   └── him/
│   │       ├── __init__.py
│   │       ├── config.py                   # HIM-only serializable options
│   │       ├── wrapper.py                  # HIMVecEnvWrapper, 4-return ABI
│   │       ├── models.py                   # HIMActorModel + export adapter
│   │       ├── estimator.py                # HIMEstimator and loss surface
│   │       ├── storage.py                  # HIMRolloutStorage extension
│   │       └── ppo.py                      # HIMPPO for MjlabOnPolicyRunner
│   ├── wrappers/
│   │   ├── __init__.py
│   │   └── rsl_rl_wrapper.py               # thin alias to MJLab wrapper
│   ├── utils/
│   │   └── task_registry.py               # explicit factories, no namespace swap
│   └── scripts/
│       ├── train.py
│       ├── play.py
│       └── export.py
├── docs/
│   ├── rsl_rl_mjlab_full_conversion.md
│   └── ...
└── .venv/lib/python3.11/site-packages/
    ├── mjlab/rl/
    │   ├── config.py
    │   ├── runner.py
    │   └── vecenv_wrapper.py
    └── rsl_rl/                            # rsl-rl-lib==5.4.2, canonical runtime
        ├── algorithms/
        ├── env/
        ├── models/
        ├── runners/
        └── storage/
```

`legged_mjlab.rl.him` 可以依赖 `rsl_rl`，但 `rsl_rl` 不能反向依赖项目路径。这样 `resolve_callable("legged_mjlab.rl.him.ppo:HIMPPO")` 是一条明确的扩展边，而不是把扩展类伪装成 `rsl_rl.algorithms.HIMPPO`。

## 5. 最终架构的代码骨架

以下代码块是小型、可实现的骨架，用来固定接口和数据方向；它们不是从安装包逐段复制的源码，也不会在本回合写入 Python 文件。

### 5.1 最终 MJLab-style `RslRlVecEnvWrapper`

标准 wrapper 最好直接复用安装版 `mjlab.rl.vecenv_wrapper.RslRlVecEnvWrapper`。如果项目需要本地 finite-horizon 或诊断 shim，必须保持同一接口。下面展示该 shim 的关键行为：

```python
from __future__ import annotations

from collections.abc import Mapping

import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from rsl_rl.env import VecEnv


class RslRlVecEnvWrapper(VecEnv):
    """Expose a ManagerBasedRlEnv through the rsl-rl-lib VecEnv ABI."""

    def __init__(self, env: ManagerBasedRlEnv, clip_actions: float | None = None):
        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = env.unwrapped.num_envs
        self.device = torch.device(env.unwrapped.device)
        self.num_actions = env.unwrapped.action_manager.total_action_dim
        self.max_episode_length = env.unwrapped.max_episode_length
        self._validate_spaces_if_needed()
        # The installed OnPolicyRunner expects an initialized environment.
        self.env.reset()

    @property
    def unwrapped(self) -> ManagerBasedRlEnv:
        return self.env.unwrapped

    @property
    def cfg(self) -> ManagerBasedRlEnvCfg:
        return self.unwrapped.cfg

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.unwrapped.episode_length_buf = value

    def _pack(self, obs: Mapping[str, torch.Tensor]) -> TensorDict:
        if not isinstance(obs, Mapping):
            raise TypeError("MJLab observations must be a mapping of named groups")
        groups = {}
        for name, value in obs.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"observation group {name!r} is not a tensor")
            if value.shape[0] != self.num_envs:
                raise ValueError(f"group {name!r} has an invalid batch dimension")
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"observation group {name!r} is non-finite")
            groups[name] = value
        return TensorDict(groups, batch_size=[self.num_envs], device=self.device)

    def get_observations(self) -> TensorDict:
        return self._pack(self.unwrapped.observation_manager.compute())

    def reset(self) -> tuple[TensorDict, dict]:
        obs, extras = self.env.reset()
        return self._pack(obs), dict(extras or {})

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"expected actions [{self.num_envs}, {self.num_actions}]")
        if not torch.isfinite(actions).all():
            raise FloatingPointError("actions contain NaN or Inf")
        if self.clip_actions is not None:
            actions = actions.clamp(-self.clip_actions, self.clip_actions)

        obs, reward, terminated, truncated, extras = self.env.step(actions)
        obs_td = self._pack(obs)
        reward = torch.as_tensor(reward, device=self.device, dtype=torch.float32).reshape(-1)
        terminated = torch.as_tensor(terminated, device=self.device, dtype=torch.bool).reshape(-1)
        truncated = torch.as_tensor(truncated, device=self.device, dtype=torch.bool).reshape(-1)
        if reward.numel() != self.num_envs or terminated.numel() != self.num_envs:
            raise ValueError("environment outputs do not match num_envs")

        info = dict(extras or {})
        info["terminated"] = terminated
        info["truncated"] = truncated
        if not self.cfg.is_finite_horizon:
            info["time_outs"] = truncated & ~terminated
        dones = (terminated | truncated).to(torch.long)
        return obs_td, reward, dones, info

    def close(self) -> None:
        self.env.close()

    def _validate_spaces_if_needed(self) -> None:
        # Keep action-space mutation and Space validation in one small helper.
        # It must not alter the observation or return-value contract.
        return None
```

关键点有三个：

1. `TensorDict` 的 batch size 是 `[N]`，group 的 feature 轴留在最后；不要把 group 名称转换成位置参数。
2. `dones` 只给 runner 一个合并 mask，但原始两个 mask 和 timeout 规则仍在 `extras` 中可追踪。
3. wrapper 不承担 HIM estimator 或 rollout storage 逻辑。这样普通 `PPO` 也能使用同一环境边界。

### 5.2 HIM observation adapter

HIM 专用 wrapper 作为第二层 adapter，输入和输出仍满足 `VecEnv` 四项返回。它只负责 history、明确的 48 维 critic projection 和 terminal metadata：

```python
import torch

from tensordict import TensorDict
from rsl_rl.env import VecEnv


class HIMVecEnvWrapper(VecEnv):
    def __init__(self, base: VecEnv, history_length: int = 6):
        if history_length != 6:
            raise ValueError("the frozen HIM contract uses six frames")
        self.base = base
        self.num_envs = base.num_envs
        self.num_actions = base.num_actions
        self.device = base.device
        self.max_episode_length = base.max_episode_length
        self._history = None  # [N, 6, 45], current frame at index zero
        self._current = None   # cache: runner may query observations twice

    @property
    def cfg(self):
        return self.base.cfg

    @property
    def episode_length_buf(self):
        return self.base.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.base.episode_length_buf = value

    def _project_critic(self, native: torch.Tensor) -> torch.Tensor:
        if native.ndim != 2 or native.shape[-1] != 235:
            raise ValueError("native critic must be [N, 235]")
        # The first 45 columns are the current actor frame; the next 3 are velocity.
        return torch.cat((native[:, :45], native[:, 45:48]), dim=-1)

    def _adapt(self, native: TensorDict) -> TensorDict:
        frame = native["policy_frame"]
        critic_native = native["critic_native"]
        if frame.shape != (self.num_envs, 45):
            raise ValueError("policy_frame must be [N, 45]")
        if self._history is None:
            self._history = frame.new_zeros(self.num_envs, 6, 45)
        self._history[:, 1:].copy_(self._history[:, :-1])
        self._history[:, 0].copy_(frame)
        return TensorDict(
            {
                "policy": self._history.reshape(self.num_envs, 270),
                "critic": self._project_critic(critic_native),
                "critic_native": critic_native,
            },
            batch_size=[self.num_envs],
            device=frame.device,
        )

    def get_observations(self):
        # OnPolicyRunner queries once during construction and again at learn().
        # A read must not append the same frame twice to the temporal history.
        if self._current is None:
            self._current = self._adapt(self.base.get_observations())
        return self._current

    def reset(self):
        native, extras = self.base.reset()
        self._history = None
        self._current = self._adapt(native)
        return self._current, extras

    def step(self, actions):
        native, rewards, dones, extras = self.base.step(actions)
        self._current = self._adapt(native)
        # The base wrapper/manager must have captured these before reset.
        terminal = extras.get("terminal_critic")
        if terminal is not None:
            extras["terminal_critic"] = self._project_critic(terminal)
        return self._current, rewards, dones, extras

    def close(self):
        return self.base.close()
```

实际实现还必须支持 done 行的 partial history reset。上面的 `_history = None` 只表达全量 reset 的骨架；生产实现应接收 base wrapper 提供的 done ids，在对应行清零并写入新 episode 第一帧。这个细节需要专项 rollout 测试。

### 5.3 runner 和 config

安装版 `OnPolicyRunner` 的构造入口是 `algorithm.class_name -> construct_algorithm(obs, env, cfg, device)`。`MjlabOnPolicyRunner` 只应负责 MJLab 的 runner 持久化和导出增强；HIM 不应重新复制主循环。

```python
from dataclasses import asdict

from mjlab.rl.config import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


HIM_RL_CFG = RslRlOnPolicyRunnerCfg(
    class_name="MjlabOnPolicyRunner",
    num_steps_per_env=100,
    max_iterations=10_000,
    save_interval=500,
    experiment_name="him_go2",
    obs_groups={
        "actor": ("policy",),
        "critic": ("critic",),
    },
    actor=RslRlModelCfg(
        class_name="legged_mjlab.rl.him.models:HIMActorModel",
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        class_name="MLPModel",
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        class_name="legged_mjlab.rl.him.ppo:HIMPPO",
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1e-3,
        schedule="fixed",
        gamma=0.998,
        lam=0.95,
        entropy_coef=0.01,
        clip_param=0.2,
    ),
)


def make_train_dict() -> dict:
    train_cfg = asdict(HIM_RL_CFG)
    # The dataclass has no arbitrary HIM fields; keep them in an explicit,
    # serializable extension section consumed by HIMPPO.construct_algorithm().
    train_cfg["him"] = {
        "history_length": 6,
        "one_step_obs_dim": 45,
        "critic_dim": 48,
        "native_critic_dim": 235,
        "latent_dim": 16,
    }
    return train_cfg


RUNNER_TYPES = {"MjlabOnPolicyRunner": MjlabOnPolicyRunner}
```

注意安装版 config 的字段名是 `class_name`，不是旧配置的 `runner_class_name`、`policy_class_name` 或 `algorithm_class_name`。任务 registry 可以把 `HIM_RL_CFG.class_name` 映射到 `RUNNER_TYPES`，然后把 `make_train_dict()` 传给 runner；具体的 dataclass flattening 只能在一个明确的边界函数中完成。

### 5.4 `HIMPPO`：适配 MJLab runner 的算法骨架

`HIMPPO` 的最重要变化是签名，而不是把旧循环原样搬家：

安装版 `PPO.process_env_step()` 的 timeout 分支使用 transition 开始时记录的 value；它不会根据 HIM 的 terminal critic 或 `next_critic` 再评估一次 critic。因此，单纯把 terminal tensor 塞进 `extras`、再调用原生 `PPO.process_env_step()`，并不能实现本迁移要求的 target 语义。`HIMPPO` 必须在自己的四参入口中选择并写入 aligned `next_critic`。

```python
from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups

from .storage import HIMRolloutStorage


class HIMPPO(PPO):
    """PPO extension consumed by the installed MJLab OnPolicyRunner."""

    @classmethod
    def construct_algorithm(cls, obs, env, cfg, device):
        groups = resolve_obs_groups(
            obs,
            cfg["obs_groups"],
            default_sets=["actor", "critic"],
        )

        actor_cfg = dict(cfg["actor"])
        critic_cfg = dict(cfg["critic"])
        actor_cls = resolve_callable(actor_cfg.pop("class_name"))
        critic_cls = resolve_callable(critic_cfg.pop("class_name"))
        actor_cfg["him_cfg"] = dict(cfg.get("him", {}))

        actor: MLPModel = actor_cls(
            obs, groups, "actor", env.num_actions, **actor_cfg
        ).to(device)
        critic: MLPModel = critic_cls(
            obs, groups, "critic", 1, **critic_cfg
        ).to(device)

        storage = HIMRolloutStorage(
            training_type="rl",
            num_envs=env.num_envs,
            num_transitions_per_env=cfg["num_steps_per_env"],
            obs=obs,
            actions_shape=(env.num_actions,),
            critic_shape=(cfg.get("him", {}).get("critic_dim", 48),),
            device=device,
        )

        algorithm_cfg = dict(cfg["algorithm"])
        algorithm_cfg.pop("class_name", None)
        return cls(
            actor,
            critic,
            storage,
            him_cfg=cfg.get("him", {}),
            device=device,
            multi_gpu_cfg=cfg.get("multi_gpu"),
            **algorithm_cfg,
        )

    def __init__(self, actor, critic, storage, him_cfg=None, **kwargs):
        super().__init__(actor, critic, storage, **kwargs)
        self.him_cfg = dict(him_cfg or {})
        self.estimator = getattr(self.actor, "estimator", None)
        if self.estimator is None:
            raise TypeError("HIMActorModel must expose an estimator")

    @staticmethod
    def _critic_view(obs: TensorDict, critic: torch.Tensor) -> TensorDict:
        view = obs.clone()
        view["critic"] = critic
        return view

    def act(self, obs: TensorDict) -> torch.Tensor:
        # This is the exact signature called by OnPolicyRunner.
        self.transition.hidden_states = (
            self.actor.get_hidden_state(),
            self.critic.get_hidden_state(),
        )
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(
            self.transition.actions
        ).detach()
        self.transition.distribution_params = tuple(
            p.detach() for p in self.actor.output_distribution_params
        )
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
    ) -> None:
        """Record the four-return step and construct an aligned next critic."""
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        next_critic = obs["critic"].detach()
        available = extras.get("terminal_available")
        terminal = extras.get("terminal_critic")
        if terminal is not None:
            if available is None:
                raise ValueError("terminal_critic requires terminal_available")
            available = torch.as_tensor(
                available, device=next_critic.device, dtype=torch.bool
            ).reshape(-1)
            terminal = torch.as_tensor(
                terminal, device=next_critic.device, dtype=next_critic.dtype
            )
            if terminal.shape != next_critic.shape:
                raise ValueError("terminal critic and critic group must have equal shape")
            next_critic = next_critic.clone()
            next_critic[available] = terminal[available]
        else:
            available = torch.zeros(
                next_critic.shape[0], device=next_critic.device, dtype=torch.bool
            )

        self.transition.next_critic = next_critic
        self.transition.terminal_critic = terminal
        self.transition.terminal_available = available
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones.to(dtype=torch.bool)

        terminated = torch.as_tensor(
            extras.get("terminated", dones), device=self.device, dtype=torch.bool
        ).reshape(-1)
        time_outs = torch.as_tensor(
            extras.get(
                "time_outs",
                torch.zeros_like(terminated, dtype=torch.bool),
            ),
            device=self.device,
            dtype=torch.bool,
        ).reshape(-1)
        bootstrap_mask = time_outs & ~terminated & available
        if bootstrap_mask.any():
            with torch.no_grad():
                next_values = self.critic(
                    self._critic_view(obs, next_critic)
                ).reshape(-1)
            self.transition.rewards += self.gamma * next_values * bootstrap_mask

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        # The inherited GAE loop remains valid because timeout correction was
        # applied above and dones are stored in the base schema.
        super().compute_returns(obs)

    def update(self) -> dict[str, float]:
        # Use the installed PPO ratio/value-loss equations, but consume the
        # HIMBatch fields below and add estimator loss before optimizer.step().
        for batch in self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        ):
            policy_loss, value_loss, entropy_loss = self._ppo_terms(batch)
            estimate_loss, swap_loss = self.estimator.loss(
                batch.observations["policy"],
                batch.next_critic,
            )
            total = (
                policy_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_loss
                + estimate_loss
                + swap_loss
            )
            self.optimizer.zero_grad()
            total.backward()
            # PPO is not an nn.Module; clip the modules owned by the algorithm.
            # HIMActorModel owns the estimator in this target design.
            trainable = list(self.actor.parameters()) + list(self.critic.parameters())
            torch.nn.utils.clip_grad_norm_(trainable, self.max_grad_norm)
            self.optimizer.step()
        self.storage.clear()
        return self._loss_summary()

    def _ppo_terms(self, batch):
        raise NotImplementedError  # implement with installed PPO math and HIMBatch

    def _loss_summary(self) -> dict[str, float]:
        raise NotImplementedError
```

这里的 `_ppo_terms()` 不是让旧 runner 继续存在的借口；它是一个窄的内部 helper，用来复用安装版 PPO 的 ratio、clipped value 和 entropy 计算。实现时应保留安装版的 `actor`、`critic`、distribution 和 normalizer surface。`HIMEstimator` 应提供无副作用的 `loss(history, next_critic)`；若它继续在自己的 `update()` 中执行 optimizer step，就会和 `HIMPPO` 的 optimizer 重复更新同一组参数，必须先拆开。

`HIMActorModel` 至少要实现 `MLPModel` 被 runner、PPO 和 exporter 使用的表面：`forward(obs, stochastic_output=...)`、`get_output_log_prob()`、`output_distribution_params`、`reset()`、`get_hidden_state()`、`update_normalization()`、`as_jit()` 和 `as_onnx()`。它内部可以把 `[N,270]` history 编码成 `45 + 3 + latent_dim`，但这个内部拼接不应泄漏到 runner 的位置参数接口。

### 5.5 storage：扩展 MJLab `RolloutStorage` schema

标准 `RolloutStorage` 没有 `next_critic` 或 terminal provenance。扩展必须在 transition、预分配 tensor、feedforward batch 和 recurrent batch 四个层面同时声明。下面是核心骨架：

```python
from __future__ import annotations

from collections.abc import Generator

import torch
from tensordict import TensorDict

from rsl_rl.storage import RolloutStorage


class HIMRolloutStorage(RolloutStorage):
    class Transition(RolloutStorage.Transition):
        def __init__(self) -> None:
            super().__init__()
            self.next_critic: torch.Tensor | None = None
            self.terminal_critic: torch.Tensor | None = None
            self.terminal_available: torch.Tensor | None = None

    class Batch(RolloutStorage.Batch):
        def __init__(self, *args, next_critic=None,
                     terminal_critic=None, terminal_available=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.next_critic = next_critic
            self.terminal_critic = terminal_critic
            self.terminal_available = terminal_available

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...],
        critic_shape: tuple[int, ...],
        device: str,
    ) -> None:
        super().__init__(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            device,
        )
        self.critic_shape = critic_shape
        self.next_critic = torch.zeros(
            num_transitions_per_env, num_envs, *critic_shape, device=device
        )
        self.terminal_critic = torch.zeros_like(self.next_critic)
        self.terminal_available = torch.zeros(
            num_transitions_per_env, num_envs, dtype=torch.bool, device=device
        )

    def add_transition(self, transition: Transition) -> None:
        if transition.next_critic is None:
            raise ValueError("next_critic is required for every HIM transition")
        slot = self.step
        if transition.next_critic.shape != self.next_critic[slot].shape:
            raise ValueError("next_critic shape does not match storage schema")
        super().add_transition(transition)
        self.next_critic[slot].copy_(transition.next_critic)
        if transition.terminal_critic is not None:
            self.terminal_critic[slot].copy_(transition.terminal_critic)
        if transition.terminal_available is not None:
            self.terminal_available[slot].copy_(
                transition.terminal_available.reshape(-1)
            )

    def mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[Batch, None, None]:
        if self.training_type != "rl":
            raise ValueError("HIM storage is only defined for RL training")
        total = self.num_envs * self.num_transitions_per_env
        mini = total // num_mini_batches
        permutation = torch.randperm(
            mini * num_mini_batches, device=self.device
        )

        obs = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        log_prob = self.actions_log_prob.flatten(0, 1)
        next_critic = self.next_critic.flatten(0, 1)
        terminal_critic = self.terminal_critic.flatten(0, 1)
        terminal_available = self.terminal_available.flatten(0, 1)
        if self.distribution_params is None:
            raise RuntimeError("distribution parameters were not recorded")
        old_params = tuple(p.flatten(0, 1) for p in self.distribution_params)

        for _ in range(num_epochs):
            for batch_id in range(num_mini_batches):
                begin = batch_id * mini
                ids = permutation[begin : begin + mini]
                yield self.Batch(
                    observations=obs[ids],
                    actions=actions[ids],
                    values=values[ids],
                    advantages=advantages[ids],
                    returns=returns[ids],
                    old_actions_log_prob=log_prob[ids],
                    old_distribution_params=tuple(p[ids] for p in old_params),
                    next_critic=next_critic[ids],
                    terminal_critic=terminal_critic[ids],
                    terminal_available=terminal_available[ids],
                )

    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        # Use the exact trajectory split, padding, masks, and hidden-state
        # indices from RolloutStorage, then index next_critic and its two
        # terminal fields with the same [time, env] selection.
        for base_batch, time_env_ids in self._aligned_recurrent_batches(
            num_mini_batches, num_epochs
        ):
            yield self.Batch(
                observations=base_batch.observations,
                actions=base_batch.actions,
                values=base_batch.values,
                advantages=base_batch.advantages,
                returns=base_batch.returns,
                old_actions_log_prob=base_batch.old_actions_log_prob,
                old_distribution_params=base_batch.old_distribution_params,
                hidden_states=base_batch.hidden_states,
                masks=base_batch.masks,
                next_critic=self.next_critic[time_env_ids],
                terminal_critic=self.terminal_critic[time_env_ids],
                terminal_available=self.terminal_available[time_env_ids],
            )

    def _aligned_recurrent_batches(self, num_mini_batches, num_epochs):
        raise NotImplementedError
```

这段骨架刻意把 `next_critic` 预分配为 `[T, N, C]`，其中 `C=48`。实现 `_aligned_recurrent_batches()` 时不能重新生成一套随机排列；必须复用 base storage 的 trajectory/mask/hidden-state 对齐。否则 estimator 的 target 可能来自另一个环境或另一个时间步，训练不会立刻报 shape 错，但语义已经损坏。

如果迁移后不需要审计 provenance，可以在验证完成后只保留 `next_critic` 和 `terminal_available`，省掉重复的 `terminal_critic` buffer；在转换初期保留三者更容易定位 auto-reset 和 env-id 错位。

## 6. task registry：停止 runtime namespace swap

最终 registry 只做三件事：登记 task、创建 native env/wrapper、按显式配置选择 runner。它不导入或重载另一个 `rsl_rl`。

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

from legged_mjlab.rl.him.wrapper import HIMVecEnvWrapper


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    env_cls: type
    env_cfg_cls: type
    rl_cfg_factory: Callable[[], Any]
    wrapper_factory: Callable[[Any], Any]


RUNNERS = {
    "MjlabOnPolicyRunner": MjlabOnPolicyRunner,
}


def _standard_wrapper(native_env):
    return RslRlVecEnvWrapper(native_env)


def _him_wrapper(native_env):
    return HIMVecEnvWrapper(RslRlVecEnvWrapper(native_env))


def _as_train_dict(rl_cfg) -> tuple[dict, str]:
    data = asdict(rl_cfg)
    runner_name = data.pop("class_name")
    if not isinstance(runner_name, str) or runner_name not in RUNNERS:
        raise ValueError(f"unsupported explicit runner: {runner_name!r}")
    return data, runner_name


class TaskRegistry:
    def __init__(self):
        self._tasks: dict[str, TaskSpec] = {}

    def register(self, spec: TaskSpec) -> None:
        if spec.task_id in self._tasks:
            raise KeyError(spec.task_id)
        self._tasks[spec.task_id] = spec

    def make_env(self, task_id: str, *, device=None, play=False):
        spec = self._tasks[task_id]
        env_cfg = spec.env_cfg_cls(play=play) if callable(spec.env_cfg_cls) else spec.env_cfg_cls()
        native = spec.env_cls(cfg=env_cfg, device=device, play=play)
        return spec.wrapper_factory(native), env_cfg

    def make_runner(self, task_id: str, env, log_dir: str):
        spec = self._tasks[task_id]
        train_cfg, runner_name = _as_train_dict(spec.rl_cfg_factory())
        # HIMPPO is resolved by rsl-rl-lib's qualified callable resolver from
        # train_cfg["algorithm"]["class_name"]. It is not registered under rsl_rl.
        runner_cls = RUNNERS[runner_name]
        return runner_cls(env, train_cfg, log_dir=log_dir, device=str(env.device))


task_registry = TaskRegistry()
```

实际项目中 `env_cfg_cls(play=...)` 的调用方式需按现有环境构造器调整；示例重点是所有 ownership 都是显式的。`train.py`、`play.py` 应直接调用 `task_registry.make_env()` 和 `make_runner()`，不再调用 `load_project_rsl()`。如果旧函数名必须暂时保留，建议让它抛出“removed; use installed rsl-rl-lib”错误，而不是执行 fallback。

## 7. 迁移顺序与 transitional-only 组件

每个阶段只切换一个边界；阶段完成前不要把旧路径和新路径混在同一个 Python 进程中。

| 阶段 | 从当前文件 / 行为 | 迁移到 | 完成条件 | transitional-only 内容 |
| --- | --- | --- | --- | --- |
| 0. 依赖基线 | 根 `setup.py` 把本地 `rsl_rl` 打包为顶层包；安装环境同时存在同名候选 | 唯一依赖入口，固定 `rsl-rl-lib==5.4.2`，记录 `rsl_rl.__file__` | 新解释器只能解析支持的 canonical 包 | 根包中旧 `package_dir` 和独立 `rsl_rl/setup.py`。 |
| 1. 标准 wrapper | `legged_mjlab/wrappers/rsl_rl_wrapper.py` 的位置式返回 | `mjlab.rl.vecenv_wrapper.RslRlVecEnvWrapper` 或等价薄 shim | reset/step/get_observations 的类型和四项返回全部固定 | 旧 actor/privileged tuple 和五项 runner 适配。 |
| 2. HIM adapter | `him_wrapper.py` 内部 history、terminal 字段和旧 base class | `legged_mjlab/rl/him/wrapper.py` | group、history、partial reset、terminal env-id 对齐有测试 | 七项 HIM wrapper path。 |
| 3. 配置 | `HimGo2CfgPPO` 的旧 class-name 字段 | `RslRlOnPolicyRunnerCfg` + 三个 nested config | `obs_groups` 显式解析，qualified names 可导入 | `runner_class_name` 等兼容读取器。 |
| 4. 模型 | 单体 `HIMActorCritic` | `HIMActorModel`、标准 `MLPModel` critic、独立 `HIMEstimator` | actor/critic/estimator 的 parameter ownership 和 export surface 明确 | 单体类只用于 checkpoint 转换。 |
| 5. 算法 | `HIMPPO.act(obs, critic_obs)`、位置式 process | `HIMPPO.construct_algorithm()`、`act(obs)`、四参 process | 可由 `OnPolicyRunner` 完整调用，timeout 和 terminal mask 正确 | `HIMOnPolicyRunner`。 |
| 6. storage | `HIMRolloutStorage` 平行 tensor 和 `next_critic_observations` | `RolloutStorage` 子类及 named `Batch` | feedforward/recurrent 均保持 `(time, env)` 对齐 | 旧 generator 返回的长 tuple。 |
| 7. registry | `load_project_rsl()` 删除并重载 `sys.modules` | 显式 runner/wrapper factory | 同进程只存在一个 `rsl_rl`，算法通过 qualified path 解析 | `load_project_rsl` 兼容入口。 |
| 8. checkpoint/export | 旧单体 key 和旧 exporter | versioned manifest、映射器、actor export module | strict load、JIT、ONNX 和 eager 数值对照完成 | 直接 `strict=False` 加载旧 checkpoint。 |
| 9. 清理 | `rsl_rl/`、`mjlab_rsl_rl/` 和旧文档中的活动指令 | 删除或归档，文档只保留最终入口 | 打包、import、训练、回放和部署脚本不再引用旧路径 | 所有旧目录只能作为 git 历史，不可被 import。 |

迁移时最容易犯的错误是先删掉本地 fork，再发现旧 checkpoint 仍依赖它的 key 命名。正确顺序是先建立 explicit namespace 和转换器，完成加载/导出回归，再 retire canonical fork。

## 8. checkpoint、`state_dict`、JIT 和 ONNX

### 8.1 versioned manifest

新 checkpoint 不应只有裸 `state_dict`。建议在 runner 的 `save()` 结果旁边加入 manifest，至少包含：

```python
manifest = {
    "schema_version": 2,
    "backend": "rsl-rl-lib",
    "rsl_rl_version": "5.4.2",
    "mjlab_version": "1.6.0",
    "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
    "shapes": {
        "policy_frame": 45,
        "policy_history": 270,
        "critic": 48,
        "critic_native": 235,
        "actions": 12,
    },
    "history_order": "current_first",
    "model_layout": {
        "actor": "legged_mjlab.rl.him.models:HIMActorModel",
        "critic": "rsl_rl.models:MLPModel",
        "estimator": "legged_mjlab.rl.him.estimator:HIMEstimator",
    },
    "normalization": {"actor": True, "critic": True},
}
```

manifest 是校验输入契约的依据，不是对旧模型兼容的保证。加载时必须先检查 schema、group、shape、history order 和 action order，再决定是否进入 key mapping。

### 8.2 从旧 `HIMActorCritic` 映射

旧单体模型通常把 actor、critic、estimator 和分布参数放在同一个 `state_dict`。转换器应：

1. 建立显式表：旧 key → 新 `actor.*`、`critic.*`、`estimator.*` key；每一条映射都检查 dtype 和完整 shape。
2. 对 normalizer、distribution 参数和线性层分别处理，不使用模糊的前缀替换覆盖未知 key。
3. 对安装版 5.x 的 distribution key 进行版本化映射；例如旧 `std` / `log_std` 是否对应新 distribution module，必须以实际 shape 和模块 state dict 为准。
4. 输出 `missing_keys`、`unexpected_keys`、`shape_mismatches` 和 observation-contract diff；有任何未解释项就拒绝加载。
5. 只有 actor、critic、estimator 参数 ownership 完整后才尝试 optimizer state。若参数组顺序或数量发生变化，宁可重建 optimizer，也不要伪造可恢复状态。

禁止用 `strict=False` 把旧 checkpoint 静默塞进新模型。允许的兼容路径是“读取旧 schema → 显式映射 → strict load → 另存新 schema”。不可转换的 checkpoint 应明确报错并保留原文件。

### 8.3 JIT

安装版 `OnPolicyRunner.export_policy_to_jit()` 通过 `alg.get_policy().as_jit()` 导出。要保持兼容，`HIMActorModel` 必须提供一个 TorchScript-friendly export module：

```python
class HIMPolicyJit(torch.nn.Module):
    def __init__(self, actor, estimator):
        super().__init__()
        self.actor = actor
        self.estimator = estimator

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # history: [B, 270], current frame first
        velocity, latent = self.estimator.encode_for_export(history)
        features = torch.cat((history[:, :45], velocity, latent), dim=-1)
        return self.actor.deterministic_head(features)

    @torch.jit.export
    def reset(self) -> None:
        return None
```

JIT 输入应是部署所需的 flat history tensor，而不是训练期 `TensorDict` 或 storage object。导出前冻结 normalizer、distribution 的 deterministic path 和 history ordering；导出后用同一批有限输入比较 eager 与 scripted 输出，至少检查 shape、最大绝对误差和异常值。

### 8.4 ONNX

安装版 runner 的 ONNX 路径通过 `alg.get_policy().as_onnx()`，默认 dummy input 是 `[1, input_size]`。HIM 的 export wrapper 必须明确：

```text
input name  = obs
input shape = [B, 270]
output name = actions
output      = [B, 12]
```

ONNX graph 不能依赖 `next_critic`、terminal metadata、optimizer 或训练期 TensorDict key。需要按实际 Torch 版本选择 legacy export / `dynamo=False`，固定 opset 后执行 ONNX runtime 对照；如果部署要求动态 batch，应在 manifest 和 graph 轴定义中显式记录，不能让导出工具默认猜测。

JIT 和 ONNX 都只导出 actor inference path。critic、estimator 的训练 target 和 terminal 字段必须留在训练 runtime，不得因为把它们塞进 checkpoint 就让部署输入契约膨胀。

## 9. 资源、复杂度与实现注意事项

对于当前候选的 `N=4096`、`T=100`、float32：

| buffer | 单个时间平面近似大小 | `T` 个时间步近似大小 |
| --- | ---: | ---: |
| `next_critic [N,48]` | 0.75 MiB | 75 MiB |
| policy history `[N,270]` | 4.22 MiB | 若逐步复制进 storage 会显著增加占用，应按需要决定是否存完整 history |
| terminal critic `[N,48]` | 0.75 MiB | 75 MiB（可在审计完成后压缩或移除） |

真实峰值还包括 observations、actions、values、advantages、distribution parameters、normalizer、梯度和 TensorDict 元数据。实现建议：

- 所有 wrapper tensor 留在环境/runner device，避免每个 policy step 的 CPU 往返。
- history 更新用批量 slice/copy；不要按环境 Python 循环。
- `terminal_critic` 只在需要时复制，先用 `terminal_available` 做 mask，再避免无意义的全量 clone。
- storage 的 `[T,N]` 顺序和 flatten 顺序要与安装版 generator 一致；任何自定义索引都要记录 time/env 对应关系。
- 先用很小的 `N`、短 rollout 和单次 update 观察 shape，再扩大到配置中的并发量；一次只改变一个变量。

计算量的主项是 `O(T·N·(actor + critic + estimator))`；新增 storage 的空间是 `O(T·N·C)`，其中 `C=48`。如果选择保留完整 235 维 target，内存和 estimator 带宽会按比例上升，必须单独评估，不能在配置中无提示切换。

## 10. 最终验证矩阵

下表是“转换可宣布完成”所需的检查，不是当前已通过的结果。当前状态统一记为 `NOT_RUN`，除非后续子任务提供可复现日志和 artifact。

| 编号 | 检查 | 必须证明的结果 | 当前状态 |
| --- | --- | --- | --- |
| V1 | 依赖来源 | `rsl_rl.__file__` 指向 `rsl-rl-lib==5.4.2`；项目路径不提供 canonical 同名包 | `NOT_RUN` |
| V2 | import 图 | 没有 `load_project_rsl()`、`sys.modules` namespace swap 或隐式 fallback | `NOT_RUN` |
| V3 | wrapper reset | `reset()` 返回 `(TensorDict, extras)`，batch `[N]`，group device/dtype 正确 | `NOT_RUN` |
| V4 | wrapper step | `step()` 恰好四项；reward/done shape、finite 检查和 action clip 正确 | `NOT_RUN` |
| V5 | observation groups | `obs_groups` 显式包含 actor/critic，`resolve_obs_groups()` 无 warning/fallback | `NOT_RUN` |
| V6 | shape contract | native `[45,235]`、HIM `[270,48]`、action `12` 在 reset/step/runner 全链一致 | `NOT_RUN` |
| V7 | history | current-first 顺序、全量 reset、partial reset 和连续 step 都得到预期帧 | `NOT_RUN` |
| V8 | terminal alignment | compact/full terminal 数据按 env id 展开；reset frame 不被当作 terminal frame | `NOT_RUN` |
| V9 | mask truth table | `terminated`、仅 `truncated`、二者同时为真、finite horizon 四种组合符合 `timeout_bootstrap` 规则 | `NOT_RUN` |
| V10 | storage write | `Transition`、预分配 buffer、feedforward batch 和 recurrent batch 的 `next_critic` 同索引 | `NOT_RUN` |
| V11 | algorithm construction | 安装版 `OnPolicyRunner` 能调用 `HIMPPO.construct_algorithm()`、`act(obs)`、四参 process、`compute_returns(obs)` | `NOT_RUN` |
| V12 | one-update smoke | 小规模环境完成 reset、短 rollout、一次 update，无 shape/device/NaN 错误 | `NOT_RUN` |
| V13 | rollout regression | 多个 episode 边界后 history、reward、done、terminal metadata 仍稳定 | `NOT_RUN` |
| V14 | checkpoint conversion | 旧 schema 只能经显式映射加载；新 checkpoint strict load 并恢复 manifest | `NOT_RUN` |
| V15 | `state_dict` parity | 映射后的 actor/critic/estimator 参数和 normalizer 与预期 shape 一致 | `NOT_RUN` |
| V16 | JIT export | `as_jit()` 可 script/save/load；eager 与 JIT action 输出在容差内一致 | `NOT_RUN` |
| V17 | ONNX export | `as_onnx()` 生成预期输入/输出；ONNX runtime 与 eager 输出在容差内一致 | `NOT_RUN` |
| V18 | play/inference | 新 runner 可加载 checkpoint，history 初始化和 deterministic action 正确 | `NOT_RUN` |
| V19 | scale-up training | 从小规模逐步扩大到目标 `N/T` 后，吞吐、显存和日志仍可接受 | `NOT_RUN` |
| V20 | package/release | 构建产物不含可导入的旧顶层 fork；文档、脚本和 manifest 指向同一 ABI | `NOT_RUN` |

只有 V1–V20 的证据齐全，才可以把“完整转换”标为完成。尤其是 runtime training、rollout、export 和 inference validation 目前仍是待执行事项，本文不宣称它们已经通过。

## 11. 收束

最终架构的判断标准不是“本地 HIM 类还能不能被某个 import 找到”，而是：一个进程只拥有一个由 MJLab 支持的 `rsl_rl` ABI；所有环境交互都走 `TensorDict` 和四项 `VecEnv.step()`；HIM 的差异通过 `legged_mjlab.rl.him` 的显式算法、模型、storage 和 wrapper 接入；checkpoint、JIT、ONNX 都能从同一份版本化观测契约重建。

本回合只新增本文档，未修改任何 Python 文件，也未执行端到端训练、rollout、checkpoint 转换或导出验证。
