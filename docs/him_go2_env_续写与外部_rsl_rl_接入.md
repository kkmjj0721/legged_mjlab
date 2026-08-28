# `him_go2` 环境续写与外部 `rsl_rl` 接入说明

> 本文是 `/home/kk/legged_mjlab` 的续写与审计方案。
>
> 本文只新增文档，不直接修改任何 `.py` 文件。下面的 Python 代码块是后续手工替换 `legged_mjlab/envs/him_go2/him_go2_env.py` 时使用的完整参考实现。

## 1. 目标与边界

目标是把现有的旧式 `LeggedMjlabCfg` 配置，转换成一个真正由 mjlab manager 驱动的 Go2 rough-terrain 环境，再通过项目目录内的 HIM 版 `rsl_rl` 训练。整体边界如下：

```text
HimGo2RoughCfg
        │ 旧项目配置对象
        ▼
HimGo2Env._build_mjlab_managercfg()
        │ flat ManagerBasedRlEnvCfg
        ▼
mjlab ManagerBasedRlEnv
        │ native 观测：actor[45]、critic[235]
        ▼
HIMRslRlWrapper
        │ actor history[6, 45] -> [270]
        ▼
项目内 rsl_rl.runners.HIMOnPolicyRunner
```

这里的 `rsl_rl` 不是官方 mjlab runner。当前仓库已经提供了自己的 `HIMOnPolicyRunner`、`HIMActorCritic`、`HIMPPO` 和 wrapper；`task_registry.load_project_rsl()` 会在已安装包缺失或不兼容时加载仓库内的实现。

需要注意：当前 loader 的策略是“兼容的已安装 `rsl_rl` 优先，项目实现作为 fallback”。因此，若要求绝对只使用项目内实现，必须在运行时检查 `rsl_rl.__file__`，而不能只看 `setup.py`。

## 2. 当前文件的明确缺口

当前 `legged_mjlab/legged_mjlab/envs/him_go2/him_go2_env.py` 还不能被 Python 解析。当前真实阻断是 manager-config builder `:71-102` 中的 `metrics = ,`（`:101`）语法错误，以及 registry/构造器、builder 调用/定义和 action 字段不匹配；`ViewerConfig` 的 `:87-94` 是合法构造，不是错误。主要缺口是：

| 位置 | 当前问题 | 处理方式 |
| --- | --- | --- |
| `_build_mjlab_managercfg` | builder `:71-102` 中 `metrics = ,`（`:101`）为语法错误；`ViewerConfig` `:87-94` 不是错误；`recorders` 可省略并使用 mjlab 1.6.0 默认空 dict | 统一构造 manager 配置 |
| `MujocoCfg` | 把重力向量传给了 `integrator` | `integrator="implicitfast"`，另传 `gravity=tuple(cfg.sim.gravity)` |
| `_build_scene` | 依赖空的地形与传感器实现 | 使用合法的 generator、接触传感器和 raycast 传感器 |
| `_build_terrain` | generator 的 `sub_terrains={}` 无法生成地形，也没有非法类型报错 | 深拷贝 `ROUGH_TERRAINS_CFG` 后显式更新字段 |
| `_joint_names` | 缺少 `self` | 改成静态方法或普通实例方法 |
| `_build_actions` | `actuator_name`、`hip_scale_reduction`、`action_clip` 均不符合当前项目接口 | 使用 `actuator_names`、`hip_reduction`，不把旧 clip 字段硬塞给 mjlab |
| `_reward_scale` | `BaseConfig` 子类不是 Mapping，不能调用 `.get()` | 使用 `getattr(cfg.rewards.scales, name, 0.0)` |
| `_build_observations` | 空实现 | 严格构造 actor 45 维和当前 scope 的 critic 235 维；若另行选择把足端接触加入 critic，才同步升级到 239 |
| `_build_events` | 空实现 | 加入 reset、push 和可用的 domain randomization |
| `_build_commands` | 空实现 | 注册名为 `twist` 的 `UniformVelocityCommandCfg` |
| `_build_rewards` | `_add_reward` 调用不完整且无返回值；上一版混用了官方 velocity task 函数与项目奖励 | 按 `unitree_rl_mjlab` 的本地 `mdp` 模式注册任务特有 reward，并覆盖所有非零旧 reward |
| `_build_terminations` / `_build_curriculum` | 空实现 | 增加 time-out、姿态、base 接触和 terrain curriculum |
| `__init__` | `render_mode` 没有默认值，但现有 `train.py` 不传它 | 将 `render_mode` 默认设为 `None` |

`Go2Asset.sensor` 里还保留了一套早期 API 风格的传感器 helper，例如 `prim_path`、`target_names_expr` 和 `ContactMatch.GEOM_NAME`。它与当前 mjlab 1.6.0 的 scene-level sensor API 不兼容。本方案不调用那套 helper，而是在 `SceneCfg.sensors` 中使用当前 API；后续如果要清理 stale helper，应另开一次代码变更。

## 3. HIM 的观测契约

### 3.1 actor 45 维

actor group 必须只产生一帧 45 维观测。文档候选固定唯一的 native/policy actor 顺序为
`base_ang_vel → projected_gravity → command → joint_pos → joint_vel → actions`；这个顺序
由第 4 节 `actor_terms` 的插入顺序定义，不能再把部署 YAML 的字段列表当成 native 顺序。

| term | 维度 | 来源 |
| --- | ---: | --- |
| `base_ang_vel` | 3 | `go2/imu_ang_vel` |
| `projected_gravity` | 3 | mjlab 内置函数 |
| `command` | 3 | `twist` command 的 `vx, vy, wz` |
| `joint_pos` | 12 | 12 个 Go2 关节相对默认角 |
| `joint_vel` | 12 | 12 个 Go2 关节速度 |
| `actions` | 12 | 上一帧 **policy action**（归一化、未延迟、未执行的动作） |
| **合计** | **45** | |

HIM-Go2 的 policy canonical joint order 契约为：
`FL(hip,thigh,calf), RL(hip,thigh,calf), FR(hip,thigh,calf), RR(hip,thigh,calf)`。
这是文档候选的 policy/deployment canonical leg-major 顺序；它不同于当前配置字典按 hip/thigh/calf 分组的插入顺序，也不同于 XML 的 `FL,FR,RL,RR` 全局腿顺序。部署 YAML 的字段/关节数组顺序属于外部布局，不能直接视为 native actor/action 顺序，必须用零动作与单位脉冲做 deployment↔native round-trip 后才可接线。候选必须使用显式有序 leg-major joint/action 列表，并在 resolve 后按名称逐项断言；dict 插入顺序、MJCF/XML 自然顺序和部署 YAML 注释都不是已验证映射，必须通过零动作与单位脉冲 round-trip。

旧配置里的 `num_commands = 4` 包含 heading 这个命令配置字段，但 `heading_command=False` 时 heading 不进入 actor 观测；因此 actor 中必须使用 `generated_commands("twist")` 的 3 维输出。

### 3.2 critic 235 维

critic 以同样的 45 维 actor 一帧开头，再拼接：

| term | 维度 |
| --- | ---: |
| actor terms | 45 |
| `base_lin_vel` | 3 |
| `height_scan` | 17 × 11 = 187 |
| **合计** | **235** |

这里必须区分两套 critic layout。upstream HIMLoco 的 layout 是 `[3 维 base velocity, 45 维 actor]`；依据 `/home/kk/github/HIMLoco/rsl_rl/rsl_rl/modules/him_estimator.py:76-84`，其 `vel` 是 `next_critic_obs[:, 45:48]`，`next_obs` 是 `next_critic_obs[:, 3:48]`。本地 HIM runner 的 layout 则是 `[45 维 actor, 3 维 velocity]`；依据 `rsl_rl/rsl_rl/modules/him_estimator.py:103-109`，本地 `next_obs` 是 `next_critic_obs[..., :45]`，`vel` 是 `next_critic_obs[..., 45:48]`。这是布局差异，不是语义矛盾；两套切片不能互换。wrapper 仍然必须保留当前 scope 的完整 235 维 privileged observation，这样既满足 estimator 的窄输入约定，也保留 height scan 信息供后续扩展。`foot_contact` 的 4 维可作为可选 critic 扩展，但只有在同步修改 `num_privileged_obs`、wrapper shape gate、smoke test 和 runner 截断说明后，才允许把 native critic 写成 239。

原生 mjlab observation group 的 `history_length` 必须保持为 `1`。6 帧历史由已有的 `HIMRslRlWrapper` 维护，否则会把 45 维再次历史展开，导致 actor 维度重复。

`actions` 这个 actor 字段固定表示 policy 在上一个 policy step 发出的归一化 action；它不是 actuator delay、PD、effort clip 之后的 executed action。若部署侧要观测或记录 executed/delayed action，必须另设字段，不能复用这 12 维而不改 shape/训练语义。

动作状态必须按四层记录：raw policy action → accepted/clipped action → delayed target → executed torque。第4节候选中 native actor 的 actions 固定是上一帧 raw policy action；环境候选对输入做 [-1,1] clamp，wrapper 负责 finite 检查，部署 adapter 对越界/NaN/Inf 的 reject、输出抑制和 safe path 均没有 runtime 证据。observation delay 按 policy-step 计，custom ScaledIdealPdActuator 的 delay 按 physics-step 计；不能把普通 IdealPd 与 custom actuator 描述成共用 fused buffer。shape 仍固定为 actor 45、native critic 235、wrapper history 270、runner estimator 48；本地 HIMActorCritic 的输入是 45+3+16=64，upstream HIMLoco 的 76 不能直接复用。

History 的时序顺序必须与 critic 的 feature 切片分开记录。HIMLoco 在 `/home/kk/github/HIMLoco/legged_gym/legged_gym/envs/go2w/go2w_legged_robot.py:51-52` 先放入 `current_obs`，再接 `obs_buf` 中的旧帧；当前 wrapper 在 `/home/kk/legged_mjlab/legged_mjlab/wrappers/him_wrapper.py:161-180` 先将旧的 `obs_history_buf` 向后移动，再把当前 actor 写入 `obs_history_buf[:, 0]`，flatten 后的 270 维顺序是 `[current frame, older frames...]`。这个 current-first 约定不能由 upstream estimator 的 critic 切片反推。

## 4. `him_go2_env.py` 完整替换代码

下面的代码块是单文件替换版本。复制时覆盖目标文件的全部内容；本文本身不会自动覆盖它。

第 4 节候选遵守 `RewardManager` 的统一契约：每个 `RewardTermCfg(func=...)` 在一次
policy step 中按 `func(env, **params)` 调用；自定义函数虽然写在 `HimGo2Env` 类内，
仍必须把 `env` 作为第一个显式参数，并由 `_build_rewards` 传入其余参数。每个 term 最终
返回位于 `env.device`、通常为 `torch.float32` 的一维 tensor `[N]`，其中 `N` 是并行环境
数；关节、足端、接触项都必须在函数内部先归约到环境维，不能返回 `[N, J]` 或
`[N, foot]`。调用时刻是完成本次 action 的 decimation 后、manager 汇总 reward 的当前
状态；本候选 `sim.dt=0.005`、`decimation=4`，policy `step_dt=0.02 s`。速度单位为
`m/s` 或 `rad/s`，角度为 `rad`，高度为 `m`，力为 `N`，力矩为 `N·m`，功率为
`N·m·rad/s`；tracking 输出无量纲，penalty 输出是对应量经过平方/绝对值及关节或足端
归约后的标量。`RewardTermCfg.weight` 直接使用 legacy raw scale，并由
`scale_rewards_by_dt=True` 统一乘 policy `step_dt`，不得在函数内再次乘 dt。

最终候选的自定义 reward 计算全部属于 `HimGo2Env` 的 `@staticmethod`；代码块不保留
同名模块级 reward helper。`envs_mdp.*` 只代表 mjlab 通用 term（不是本项目自定义
reward），可继续按同一 `func(env, **params) -> [N]` 契约注册。

```python
import math
import copy

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
import mjlab.terrains as terrain_gen
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensor,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from legged_mjlab.envs.him_go2.go2_asset import Go2Asset
from legged_mjlab.envs.him_go2.him_go2_config import HimGo2RoughCfg


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _foot_contact_observation(env, sensor_name: str) -> torch.Tensor:
    """Return one binary contact value for each of the four feet."""

    sensor = env.scene[sensor_name]
    found = sensor.data.found
    if found is None:
        raise RuntimeError(f"contact sensor {sensor_name!r} did not expose found")
    canonical_names = tuple(
        f"{leg}_foot_collision" for leg in ("FL", "RL", "FR", "RR")
    )
    primary_names = tuple(sensor.primary_names)
    if set(primary_names) != set(canonical_names):
        raise RuntimeError(
            f"{sensor_name!r} must resolve exactly to {canonical_names}, "
            f"got {primary_names}"
        )
    indices = [primary_names.index(name) for name in canonical_names]
    # ContactSensor resolves primary names in entity/model order because its
    # internal find_geoms call does not preserve the input tuple order.  Apply
    # the policy canonical order explicitly before reducing contact slots.
    if found.ndim == 3:
        found = found[:, indices, :]
    elif found.ndim == 2:
        found = found[:, indices]
    else:
        raise RuntimeError(
            f"{sensor_name!r} returned unsupported found shape {tuple(found.shape)}"
        )
    if found.ndim == 3:
        found = found.any(dim=-1)
    if found.ndim != 2 or found.shape[1] != len(canonical_names):
        raise RuntimeError(
            f"{sensor_name!r} must resolve to four feet, got {tuple(found.shape)}"
        )
    return (found > 0).float()


def randomize_motor_strength(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    strength_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Scale computed actuator torque before the actuator effort clamp.

    This is deliberately a custom event: ``dr.effort_limits`` randomizes the
    saturation limit and is not equivalent to legacy motor-strength DR.
    """

    asset: Entity = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    low, high = (float(strength_range[0]), float(strength_range[1]))
    for actuator in asset.actuators:
        setter = getattr(actuator, "set_motor_strength", None)
        if setter is None:
            raise TypeError(
                "motor-strength DR requires the documentation candidate's "
                "ScaledIdealPdActuator"
            )
        shape = (env_ids.numel(), actuator.target_ids.numel())
        samples = torch.empty(shape, device=env.device).uniform_(low, high)
        setter(env_ids, samples)


class HimGo2Env(ManagerBasedRlEnv):
    """Go2 rough-terrain manager environment for the repository HIM runner."""

    def __init__(
        self,
        cfg: HimGo2RoughCfg,
        device=None,
        render_mode: str | None = None,
        play: bool = False,
        debug_vis: bool = False,
    ):
        self.cfg = cfg
        self.play = bool(play)
        self.managercfg = self._build_mjlab_managercfg(
            cfg,
            play=self.play,
            debug_vis=debug_vis,
        )
        self.only_positive_rewards = bool(
            getattr(cfg.rewards, "only_positive_rewards", False)
        )
        super().__init__(
            cfg=self.managercfg,
            device=device or getattr(cfg.env, "device", "cpu"),
            render_mode=render_mode,
        )

    def step(self, action):
        if action.shape[-1] != 12:
            raise ValueError(f"HIM Go2 expects 12 actions, got {action.shape}")
        if not bool(torch.isfinite(action).all().item()):
            raise FloatingPointError("policy action contains NaN or Inf")
        action = torch.clamp(action, -1.0, 1.0)
        result = super().step(action)
        if not self.only_positive_rewards:
            return result
        obs, reward, terminated, truncated, extras = result
        # mjlab does not consume the legacy `only_positive_rewards` field.
        # Apply the legacy post-sum clamp only after all reward terms have been
        # accumulated; termination remains a separate manager signal.
        return obs, torch.clamp_min(reward, 0.0), terminated, truncated, extras

    @staticmethod
    def _reward_tracking_lin_vel(
        env: ManagerBasedRlEnv,
        std: float,
        command_name: str,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        """Return xy velocity tracking reward as a finite ``[N]`` tensor."""

        asset: Entity = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        assert command is not None, f"Command '{command_name}' not found."
        actual = asset.data.root_link_lin_vel_b
        error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
        return torch.exp(-error / std**2)

    @staticmethod
    def _reward_tracking_ang_vel(
        env: ManagerBasedRlEnv,
        std: float,
        command_name: str,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        """Return yaw velocity tracking reward as a finite ``[N]`` tensor."""

        asset: Entity = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        assert command is not None, f"Command '{command_name}' not found."
        error = torch.square(
            command[:, 2] - asset.data.root_link_ang_vel_b[:, 2]
        )
        return torch.exp(-error / std**2)

    @staticmethod
    def _reward_orientation(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        """Return projected-gravity xy square error as ``[N]``."""

        asset: Entity = env.scene[asset_cfg.name]
        body_ids = asset_cfg.body_ids
        if not isinstance(body_ids, slice):
            body_quat_w = asset.data.body_link_quat_w[:, body_ids, :]
            if body_quat_w.shape[1] != 1:
                raise ValueError("body_orientation_l2 expects one selected body")
            projected_gravity_b = quat_apply_inverse(
                body_quat_w.squeeze(1), asset.data.gravity_vec_w
            )
            return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
        return torch.sum(
            torch.square(asset.data.projected_gravity_b[:, :2]), dim=1
        )

    @staticmethod
    def _reward_lin_vel_z(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        return torch.square(asset.data.root_link_lin_vel_b[:, 2])

    @staticmethod
    def _reward_ang_vel_xy(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)

    @staticmethod
    def _reward_dof_acc(
        env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        joint_acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
        return torch.sum(torch.square(joint_acc), dim=1)

    @staticmethod
    def _reward_action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
        action_manager = env.action_manager
        action_rate = action_manager.action - action_manager.prev_action
        return torch.sum(torch.square(action_rate), dim=1)

    @staticmethod
    def _reward_smoothness(env: ManagerBasedRlEnv) -> torch.Tensor:
        action_manager = env.action_manager
        smoothness = (
            action_manager.action
            - 2.0 * action_manager.prev_action
            + action_manager.prev_prev_action
        )
        return torch.sum(torch.square(smoothness), dim=1)

    @staticmethod
    def _reward_torques(
        env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        actuator_force = asset.data.actuator_force[:, asset_cfg.actuator_ids]
        return torch.sum(torch.square(actuator_force), dim=1)

    @staticmethod
    def _reward_dof_vel(
        env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
        return torch.sum(torch.square(joint_vel), dim=1)

    @staticmethod
    def _reward_dof_pos_limits(
        env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        soft_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
        lower = soft_limits[..., 0]
        upper = soft_limits[..., 1]
        violation = torch.relu(lower - joint_pos) + torch.relu(joint_pos - upper)
        return torch.sum(violation, dim=1)

    @staticmethod
    def _reward_dof_vel_limits(
        env: ManagerBasedRlEnv,
        velocity_limits: float | torch.Tensor | tuple[float, ...],
        soft_limit: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
        limits = torch.as_tensor(
            velocity_limits, device=joint_vel.device, dtype=joint_vel.dtype
        )
        if limits.ndim == 0:
            limits = limits.expand_as(joint_vel)
        elif limits.ndim == 1 and limits.shape[0] == joint_vel.shape[1]:
            limits = limits.unsqueeze(0)
        elif limits.shape != joint_vel.shape:
            raise ValueError(
                "velocity_limits must be a scalar, one value per selected joint, "
                f"or have shape {tuple(joint_vel.shape)}"
            )
        soft_limit_tensor = torch.as_tensor(
            soft_limit, device=joint_vel.device, dtype=joint_vel.dtype
        )
        violation = torch.relu(
            torch.abs(joint_vel) - limits * soft_limit_tensor
        )
        return torch.sum(torch.clamp(violation, max=1.0), dim=1)

    @staticmethod
    def _reward_feet_stumble(
        env: ManagerBasedRlEnv, sensor_name: str
    ) -> torch.Tensor:
        force = env.scene[sensor_name].data.force
        if force is None or force.ndim != 3 or force.shape[1:] != (4, 3):
            raise ValueError("feet force must have shape [N, 4, 3]")
        horizontal_force = torch.linalg.vector_norm(force[..., :2], dim=-1)
        stumble = horizontal_force > 5.0 * torch.abs(force[..., 2])
        return stumble.any(dim=1).to(dtype=force.dtype)

    @staticmethod
    def _reward_termination(env: ManagerBasedRlEnv) -> torch.Tensor:
        return env.termination_manager.terminated.to(dtype=torch.float32)

    @staticmethod
    def _reward_feet_air_time(
        env: ManagerBasedRlEnv,
        sensor_name: str,
        threshold: float = 0.4,
        command_name: str | None = None,
        command_threshold: float = 0.1,
    ) -> torch.Tensor:
        """Return mode-time air/contact reward as ``[N]``.

        With four feet, ``mean(in_contact) == 0.5`` means exactly two feet are
        in contact.  It is a two-contact mode, not a single-stance condition.
        """

        sensor: ContactSensor = env.scene[sensor_name]
        sensor_data = sensor.data
        air_time = sensor_data.current_air_time
        contact_time = sensor_data.current_contact_time
        in_contact = contact_time > 0.0
        in_mode_time = torch.where(in_contact, contact_time, air_time)
        two_contact_mode = torch.mean(in_contact.float(), dim=1) == 0.5
        mode_time = torch.min(
            torch.where(two_contact_mode.unsqueeze(-1), in_mode_time, 0.0), dim=1
        )[0]
        reward = torch.clamp(
            threshold - torch.abs(mode_time - threshold), min=0.0
        )
        if command_name is not None:
            command = env.command_manager.get_command(command_name)
            if command is not None:
                command_norm = torch.linalg.vector_norm(command[:, :2], dim=1)
                command_norm = command_norm + torch.abs(command[:, 2])
                reward *= (command_norm > command_threshold).to(reward.dtype)
        return reward

    @staticmethod
    def _reward_collision(
        env: ManagerBasedRlEnv,
        sensor_name: str,
        force_threshold: float = 10.0,
    ) -> torch.Tensor:
        """Count hit history samples, not the total number of contacts."""

        data = env.scene[sensor_name].data
        if data.force_history is not None:
            # Explicit sensor contract: [N, primary, history, 3] = batch,
            # matched geom/body, history sample, force vector in N.
            history = data.force_history
            if history.ndim != 4 or history.shape[-1] != 3:
                raise ValueError(
                    "force_history must have shape [N, primary, history, 3]"
                )
            hit_per_slot = (
                torch.linalg.vector_norm(history, dim=-1) > force_threshold
            ).any(dim=1)
            # Reduce primary items first, then count hit history slots.  The
            # result is [N]; a history slot is not a contact-count unit.
            return hit_per_slot.sum(dim=-1).to(dtype=torch.float32)
        assert data.found is not None
        found = data.found > 0
        if found.ndim == 3:
            found = found.any(dim=-1)
        if found.ndim != 2:
            raise ValueError("found must have shape [N, primary] or [N, primary, slot]")
        return found.any(dim=-1).to(dtype=torch.float32)

    @staticmethod
    def _reward_illegal_contact(
        env: ManagerBasedRlEnv,
        sensor_name: str,
        force_threshold: float = 10.0,
    ) -> torch.Tensor:
        """Return a boolean ``[N]`` termination mask for forbidden contact."""

        data = env.scene[sensor_name].data
        if data.force_history is not None:
            history = data.force_history
            if history.ndim != 4 or history.shape[-1] != 3:
                raise ValueError(
                    "force_history must have shape [N, primary, history, 3]"
                )
            hit = torch.linalg.vector_norm(history, dim=-1) > force_threshold
            return hit.any(dim=1).any(dim=1)
        assert data.found is not None
        found = data.found > 0
        if found.ndim == 3:
            found = found.any(dim=-1)
        if found.ndim != 2:
            raise ValueError("found must have shape [N, primary] or [N, primary, slot]")
        return found.any(dim=-1)

    @staticmethod
    def _reward_hip_pos(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        default_joint_pos = asset.data.default_joint_pos
        assert default_joint_pos is not None
        diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - default_joint_pos[
            :, asset_cfg.joint_ids
        ]
        return torch.sum(torch.square(diff), dim=1)

    @staticmethod
    def _reward_base_height(
        env: ManagerBasedRlEnv,
        target_height: float,
        sensor_name: str,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        data = env.scene[sensor_name].data
        valid = data.distances >= 0
        relative_height = (
            asset.data.root_link_pose_w[:, 2].unsqueeze(-1)
            - data.hit_pos_w[..., 2]
        )
        relative_height = torch.where(
            valid, relative_height, torch.zeros_like(relative_height)
        )
        measured_height = relative_height.sum(dim=-1) / valid.sum(
            dim=-1
        ).clamp_min(1)
        return torch.square(measured_height - target_height)

    @staticmethod
    def _reward_foot_clearance(
        env: ManagerBasedRlEnv,
        target_height: float,
        command_name: str | None = None,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        """Return world/site absolute-height clearance cost as ``[N]``."""

        asset: Entity = env.scene[asset_cfg.name]
        foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
        foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
        cost = torch.sum(
            torch.abs(foot_z - target_height)
            * torch.linalg.vector_norm(foot_vel_xy, dim=-1),
            dim=1,
        )
        if command_name is not None:
            command = env.command_manager.get_command(command_name)
            if command is not None:
                command_norm = torch.linalg.vector_norm(command[:, :2], dim=1)
                command_norm = command_norm + torch.abs(command[:, 2])
                cost *= (command_norm > command_threshold).to(cost.dtype)
        return cost

    @staticmethod
    def _reward_joint_power(
        env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        torque = asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
        velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
        return torch.sum(torch.abs(torque * velocity), dim=1)

    @staticmethod
    def _reward_stand_still(
        env: ManagerBasedRlEnv,
        command_name: str,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[
            :, asset_cfg.joint_ids
        ]
        reward = torch.sum(torch.square(diff), dim=1)
        command = env.command_manager.get_command(command_name)
        if command is not None:
            command_norm = torch.linalg.vector_norm(command[:, :2], dim=1)
            command_norm = command_norm + torch.abs(command[:, 2])
            reward *= (command_norm <= command_threshold).to(reward.dtype)
        return reward

    @staticmethod
    def _reward_torque_limits(
        env: ManagerBasedRlEnv,
        soft_limit: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Return normalized squared force-limit cost as ``[N]``."""

        asset = env.scene[asset_cfg.name]
        # Caveat: actuator_force may already be post-effort-clamp.  In that
        # case no value above the hard limit survives for this term to see;
        # soft_limit=1.0 also leaves no margin and can make the cost identically
        # zero.  S14 must verify whether a pre-clamp signal is available.
        force = asset.data.actuator_force
        limits = torch.full_like(force, float("inf"))
        for actuator in asset.actuators:
            force_limit = getattr(actuator, "force_limit", None)
            if force_limit is not None:
                limits[:, actuator.ctrl_ids] = force_limit
        force = force[:, asset_cfg.actuator_ids]
        limits = limits[:, asset_cfg.actuator_ids]
        denominator = torch.clamp(
            limits * max(float(soft_limit), 1.0e-6), min=1.0e-6
        )
        excess = torch.relu(torch.abs(force) / denominator - 1.0)
        return torch.sum(torch.square(excess), dim=1)

    @staticmethod
    def _joint_names(cfg) -> tuple[str, ...]:
        """Return the fixed policy/deployment order, not dict insertion order."""

        return tuple(
            f"{leg}_{part}_joint"
            for leg in ("FL", "RL", "FR", "RR")
            for part in ("hip", "thigh", "calf")
        )

    @staticmethod
    def _noise(cfg, field: str, enabled: bool):
        if not enabled or not cfg.noise.add_noise:
            return None
        scale = float(getattr(cfg.noise.noise_scales, field))
        level = float(cfg.noise.noise_level)
        return Unoise(n_min=-level * scale, n_max=level * scale)

    @staticmethod
    def _obs_delay(cfg, range_field: str, enabled: bool):
        """Convert a legacy integer-lag range to mjlab policy-step delay kwargs."""

        if not enabled:
            return {}
        low, high = getattr(cfg.domain_rand, range_field)
        return {
            "delay_min_lag": int(low),
            "delay_max_lag": int(high),
        }

    def _build_mjlab_managercfg(
        self,
        cfg,
        play: bool = False,
        debug_vis: bool = False,
    ) -> ManagerBasedRlEnvCfg:
        entity_name = cfg.asset.name
        # Pass play through so actuator latency is disabled for evaluation.
        asset = Go2Asset(cfg, play=play)

        if not cfg.terrain.measure_heights:
            raise ValueError(
                "HIM Go2 requires terrain.measure_heights=True so critic keeps "
                "the 17x11=187 height-scan contract"
            )

        return ManagerBasedRlEnvCfg(
            decimation=int(cfg.control.decimation),
            scene=self._build_scene(cfg, asset, play, debug_vis),
            observations=self._build_observations(cfg, play),
            actions=self._build_actions(cfg),
            events=self._build_events(cfg, play),
            seed=int(cfg.env.seed),
            sim=SimulationCfg(
                nconmax=35,
                njmax=1500,
                contact_sensor_maxmatch=500,
                mujoco=MujocoCfg(
                    timestep=float(cfg.sim.dt),
                    integrator="implicitfast",
                    gravity=tuple(float(x) for x in cfg.sim.gravity),
                    iterations=10,
                    ls_iterations=20,
                    ccd_iterations=500,
                ),
            ),
            viewer=ViewerConfig(
                origin_type=ViewerConfig.OriginType.ASSET_BODY,
                entity_name=entity_name,
                body_name="base_link",
                distance=1.5,
                elevation=-10.0,
                azimuth=90.0,
            ),
            episode_length_s=1.0e9 if play else float(cfg.env.episode_length_s),
            rewards=self._build_rewards(cfg),
            terminations=self._build_terminations(cfg),
            commands=self._build_commands(cfg, debug_vis),
            curriculum=self._build_curriculum(cfg, play),
            metrics={},
            recorders={},
            is_finite_horizon=False,
            # The candidate environment must expose the real terminal frame.
            # The wrapper may assert/compatibly adapt it and reset done envs
            # only after capture; it must not infer terminal state from an
            # already auto-reset observation.
            auto_reset=False,
            # Legacy legged_gym reward scales are multiplied by the policy
            # step_dt=0.005*4=0.02; keep mjlab's dt scaling enabled.
            scale_rewards_by_dt=True,
        )

    def _build_scene(self, cfg, asset: Go2Asset, play: bool, debug_vis: bool):
        entity_name = cfg.asset.name
        return SceneCfg(
            num_envs=int(cfg.env.num_envs),
            env_spacing=float(cfg.env.env_spacing),
            terrain=self._build_terrain(cfg, play),
            entities={entity_name: asset.entity.get_robot_cfg()},
            sensors=tuple(
                self._build_sensors(
                    cfg,
                    entity_name=entity_name,
                    debug_vis=debug_vis,
                )
            ),
            extent=2.0,
        )

    def _build_sensors(self, cfg, entity_name: str, debug_vis: bool):
        foot_geoms = tuple(
            f"{leg}_foot_collision" for leg in ("FL", "FR", "RL", "RR")
        )
        sensors = [
            ContactSensorCfg(
                name="feet_ground_contact",
                primary=ContactMatch(
                    mode="geom",
                    pattern=foot_geoms,
                    entity=entity_name,
                ),
                secondary=ContactMatch(mode="body", pattern="terrain"),
                fields=("found", "force"),
                reduce="netforce",
                num_slots=1,
                track_air_time=True,
            ),
            ContactSensorCfg(
                name="nonfoot_ground_touch",
                primary=ContactMatch(
                    mode="geom",
                    # Explicit XML names are intentional: a broad
                    # `.*_collision` pattern would include the four foot
                    # geoms and turn normal stance contacts into failures.
                    pattern=(
                        "base1_collision",
                        "base2_collision",
                        "base3_collision",
                        "FL_thigh_collision",
                        "FR_thigh_collision",
                        "RL_thigh_collision",
                        "RR_thigh_collision",
                        "FL_calf1_collision",
                        "FL_calf2_collision",
                        "FR_calf1_collision",
                        "FR_calf2_collision",
                        "RL_calf1_collision",
                        "RL_calf2_collision",
                        "RR_calf1_collision",
                        "RR_calf2_collision",
                    ),
                    entity=entity_name,
                ),
                secondary=ContactMatch(mode="body", pattern="terrain"),
                fields=("found", "force"),
                reduce="none",
                num_slots=1,
                history_length=4,
            ),
            ContactSensorCfg(
                name="base_ground_touch",
                primary=ContactMatch(
                    mode="geom",
                    pattern=(
                        "base1_collision",
                        "base2_collision",
                        "base3_collision",
                    ),
                    entity=entity_name,
                ),
                secondary=ContactMatch(mode="body", pattern="terrain"),
                fields=("found", "force"),
                reduce="none",
                num_slots=1,
                history_length=4,
            ),
        ]

        x_points = tuple(float(x) for x in cfg.terrain.measured_points_x)
        y_points = tuple(float(y) for y in cfg.terrain.measured_points_y)
        sensors.append(
            RayCastSensorCfg(
                name="height_scan",
                frame=ObjRef(
                    type="body",
                    name="base_link",
                    entity=entity_name,
                ),
                pattern=GridPatternCfg(
                    size=(max(x_points) - min(x_points), max(y_points) - min(y_points)),
                    resolution=float(cfg.terrain.horizontal_scale),
                ),
                ray_alignment="yaw",
                max_distance=2.0,
                exclude_parent_body=True,
                include_geom_groups=(0,),
                debug_vis=bool(debug_vis),
            )
        )
        return sensors

    def _build_terrain(self, cfg, play: bool):
        mesh_type = cfg.terrain.mesh_type
        if mesh_type == "plane":
            return TerrainEntityCfg(
                terrain_type="plane",
                terrain_generator=None,
                env_spacing=float(cfg.env.env_spacing),
                debug_vis=False,
            )

        if mesh_type != "generator":
            raise ValueError(
                f"HIM Go2 supports mesh_type='plane' or 'generator', got {mesh_type!r}"
            )

        proportions = tuple(float(p) for p in cfg.terrain.terrain_proportions)
        if len(proportions) != 5 or any(p < 0.0 for p in proportions):
            raise ValueError(
                "terrain_proportions must contain five non-negative values"
            )
        if sum(proportions) <= 0.0:
            raise ValueError("terrain_proportions must not sum to zero")
        terrain_size = (
            float(cfg.terrain.terrain_length),
            float(cfg.terrain.terrain_width),
        )
        hscale = float(cfg.terrain.horizontal_scale)
        vscale = float(cfg.terrain.vertical_scale)
        # Preserve the legacy five-way proportion order while using mjlab's
        # native terrain config classes. In curriculum mode each key becomes
        # one column; in random mode proportions are sampling weights.
        sub_terrains = {
            "smooth_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=proportions[0],
                size=terrain_size,
                slope_range=(0.0, 0.7),
                platform_width=2.0,
                horizontal_scale=hscale,
                vertical_scale=vscale,
            ),
            "rough_slope": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=proportions[1],
                size=terrain_size,
                noise_range=(0.02, 0.10),
                noise_step=0.02,
                horizontal_scale=hscale,
                vertical_scale=vscale,
            ),
            "stairs_up": terrain_gen.BoxPyramidStairsTerrainCfg(
                proportion=proportions[2],
                size=terrain_size,
                step_height_range=(0.0, 0.2),
                step_width=0.3,
                platform_width=2.0,
            ),
            "stairs_down": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
                proportion=proportions[3],
                size=terrain_size,
                step_height_range=(0.0, 0.2),
                step_width=0.3,
                platform_width=2.0,
            ),
            "discrete": terrain_gen.HfDiscreteObstaclesTerrainCfg(
                proportion=proportions[4],
                size=terrain_size,
                obstacle_width_range=(0.4, 0.8),
                obstacle_height_range=(0.05, 0.2),
                num_obstacles=12,
                platform_width=1.5,
                horizontal_scale=hscale,
                vertical_scale=vscale,
            ),
        }

        # ROUGH_TERRAINS_CFG supplies the remaining valid generator defaults.
        # Deep-copy it before updating legacy fields so this environment never
        # mutates the shared preset for a later environment construction.
        generator = copy.deepcopy(ROUGH_TERRAINS_CFG)
        generator.sub_terrains = sub_terrains
        generator.curriculum = bool(cfg.terrain.curriculum and not play)
        generator.size = terrain_size
        generator.border_width = float(cfg.terrain.border_size)
        generator.num_rows = 5 if play else int(cfg.terrain.num_rows)
        generator.num_cols = 5 if play else int(cfg.terrain.num_cols)
        return TerrainEntityCfg(
            terrain_type="generator",
            terrain_generator=generator,
            env_spacing=float(cfg.env.env_spacing),
            max_init_terrain_level=(
                None if play else int(cfg.terrain.max_init_terrain_level)
            ),
            debug_vis=False,
        )

    def _build_actions(self, cfg):
        action_scale = float(cfg.control.action_scale)
        hip_scale = action_scale * float(cfg.control.hip_reduction)
        actions = {}
        # JointPositionAction resolves actuator targets in entity-natural
        # order.  Separate leg terms make the policy/deployment order explicit
        # even though the XML leg order is FL, FR, RL, RR.
        for leg in ("FL", "RL", "FR", "RR"):
            leg_joint_names = tuple(
                f"{leg}_{part}_joint" for part in ("hip", "thigh", "calf")
            )
            leg_scales = {
                name: (
                    hip_scale if name.endswith("_hip_joint") else action_scale
                )
                for name in leg_joint_names
            }
            default_angles = cfg.init_state.default_joint_angles
            # mjlab applies clip after scale and default offset.  Therefore
            # this target-space interval is the exact equivalent of raw
            # action clip [-1, 1], without shifting the -1.5 calf pose.
            target_clip = {
                name: (
                    float(default_angles[name]) - leg_scales[name],
                    float(default_angles[name]) + leg_scales[name],
                )
                for name in leg_joint_names
            }
            actions[f"{leg.lower()}_joint_position"] = JointPositionActionCfg(
                entity_name=cfg.asset.name,
                actuator_names=(rf"^{leg}_(?:hip|thigh|calf)_joint$",),
                scale=leg_scales,
                use_default_offset=True,
                preserve_order=True,
                clip=target_clip,
            )
        return actions

    def _build_events(self, cfg, play: bool):
        entity_name = cfg.asset.name
        joint_cfg = SceneEntityCfg(
            entity_name,
            joint_names=self._joint_names(cfg),
            preserve_order=True,
        )
        actuator_cfg = SceneEntityCfg(
            entity_name,
            actuator_names=(".*",),
            preserve_order=True,
        )
        foot_cfg = SceneEntityCfg(
            entity_name,
            geom_names=tuple(
                f"{leg}_foot_collision" for leg in ("FL", "FR", "RL", "RR")
            ),
            preserve_order=True,
        )
        body_cfg = SceneEntityCfg(entity_name, body_names=("base_link",))
        all_body_cfg = SceneEntityCfg(entity_name, body_names=(".*",))

        reset_pose_range = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.0, 0.0),
            "yaw": (-math.pi, math.pi),
        }
        if play:
            # Evaluation should be reproducible by default.  Reset events are
            # still needed for the wrapper's partial-reset path, but their
            # pose must not introduce an unreported random initial condition.
            reset_pose_range = {
                key: (0.0, 0.0) for key in reset_pose_range
            }

        events = {
            "reset_base": EventTermCfg(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": reset_pose_range,
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg(entity_name),
                },
            ),
            "reset_joints": EventTermCfg(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (-0.0, 0.0),
                    "velocity_range": (-0.0, 0.0),
                    "asset_cfg": joint_cfg,
                },
            ),
        }

        # All domain-randomization events are disabled in play mode, including
        # startup events. Returning here prevents a later DR addition from
        # accidentally leaking into evaluation.
        if play:
            return events

        if cfg.domain_rand.push_robots:
            max_push = float(cfg.domain_rand.max_push_vel_xy)
            events["push_robot"] = EventTermCfg(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(
                    float(cfg.domain_rand.push_interval_s),
                    float(cfg.domain_rand.push_interval_s),
                ),
                params={
                    "asset_cfg": SceneEntityCfg(entity_name),
                    "velocity_range": {
                        "x": (-max_push, max_push),
                        "y": (-max_push, max_push),
                        "z": (0.0, 0.0),
                        "roll": (0.0, 0.0),
                        "pitch": (0.0, 0.0),
                        "yaw": (0.0, 0.0),
                    }
                },
            )

        if cfg.domain_rand.randomize_payload_mass:
            events["payload_mass"] = EventTermCfg(
                mode="startup",
                func=dr.body_mass,
                params={
                    "asset_cfg": body_cfg,
                    "operation": "add",
                    "ranges": tuple(cfg.domain_rand.payload_mass_range),
                },
            )

        if cfg.domain_rand.randomize_friction:
            events["foot_friction"] = EventTermCfg(
                mode="startup",
                func=dr.geom_friction,
                params={
                    "asset_cfg": foot_cfg,
                    "operation": "abs",
                    "ranges": tuple(cfg.domain_rand.friction_range),
                    "shared_random": True,
                },
            )

        if cfg.domain_rand.randomize_motor_zero_offset:
            zero_offset_range = getattr(
                cfg.domain_rand,
                "motor_zero_pos_offset_range",
                getattr(cfg.domain_rand, "motor_zero_offset_range", (0.0, 0.0)),
            )
            events["encoder_bias"] = EventTermCfg(
                mode="startup",
                func=dr.encoder_bias,
                params={
                    "asset_cfg": joint_cfg,
                    "bias_range": tuple(zero_offset_range),
                },
            )

        if cfg.domain_rand.randomize_com_displacement:
            events["base_com"] = EventTermCfg(
                mode="startup",
                func=dr.body_com_offset,
                params={
                    "asset_cfg": body_cfg,
                    "operation": "add",
                    "ranges": {
                        0: tuple(cfg.domain_rand.com_displacement_range),
                        1: tuple(cfg.domain_rand.com_displacement_range),
                        2: tuple(cfg.domain_rand.com_displacement_range),
                    },
                },
            )

        if cfg.domain_rand.randomize_link_mass:
            events["link_mass"] = EventTermCfg(
                mode="startup",
                func=dr.body_mass,
                params={
                    "asset_cfg": all_body_cfg,
                    "operation": "scale",
                    "ranges": tuple(cfg.domain_rand.link_mass_range),
                },
            )

        if cfg.domain_rand.randomize_joint_friction:
            events["joint_friction"] = EventTermCfg(
                mode="startup",
                func=dr.joint_friction,
                params={
                    "asset_cfg": joint_cfg,
                    "operation": "abs",
                    "ranges": tuple(cfg.domain_rand.joint_friction_range),
                },
            )

        if cfg.domain_rand.randomize_joint_damping:
            events["joint_damping"] = EventTermCfg(
                mode="startup",
                func=dr.joint_damping,
                params={
                    "asset_cfg": joint_cfg,
                    "operation": "abs",
                    "ranges": tuple(cfg.domain_rand.joint_damping_range),
                },
            )

        if cfg.domain_rand.randomize_joint_armature:
            events["joint_armature"] = EventTermCfg(
                mode="startup",
                func=dr.dof_armature,
                params={
                    "asset_cfg": joint_cfg,
                    "operation": "abs",
                    "ranges": tuple(cfg.domain_rand.joint_armature_range),
                },
            )

        if cfg.domain_rand.randomize_pd_gains:
            events["pd_gains"] = EventTermCfg(
                mode="startup",
                func=dr.pd_gains,
                params={
                    "asset_cfg": actuator_cfg,
                    "operation": "scale",
                    "kp_range": tuple(cfg.domain_rand.stiffness_multiplier_range),
                    "kd_range": tuple(cfg.domain_rand.damping_multiplier_range),
                },
            )

        if cfg.domain_rand.randomize_motor_strength:
            events["motor_strength"] = EventTermCfg(
                mode="startup",
                func=randomize_motor_strength,
                params={
                    "asset_cfg": actuator_cfg,
                    "strength_range": tuple(cfg.domain_rand.motor_strength_range),
                },
            )

        return events

    def _build_commands(self, cfg, debug_vis: bool):
        ranges = cfg.commands.ranges
        return {
            "twist": UniformVelocityCommandCfg(
                entity_name=cfg.asset.name,
                resampling_time_range=(
                    float(cfg.commands.resampling_time),
                    float(cfg.commands.resampling_time),
                ),
                heading_command=bool(cfg.commands.heading_command),
                heading_control_stiffness=0.5,
                rel_standing_envs=0.05,
                debug_vis=bool(debug_vis),
                ranges=UniformVelocityCommandCfg.Ranges(
                    lin_vel_x=tuple(ranges.lin_vel_x),
                    lin_vel_y=tuple(ranges.lin_vel_y),
                    ang_vel_z=tuple(ranges.ang_vel_yaw),
                    heading=(
                        tuple(ranges.heading)
                        if bool(cfg.commands.heading_command)
                        else None
                    ),
                ),
            )
        }

    def _scale_for_reward(self, cfg, name: str) -> float:
        return float(getattr(cfg.rewards.scales, name, 0.0))

    def _add_reward(self, terms, cfg, name: str, func, params=None):
        weight = self._scale_for_reward(cfg, name)
        if weight == 0.0:
            return
        terms[name] = RewardTermCfg(
            func=func,
            weight=weight,
            params=dict(params or {}),
        )

    def _build_rewards(self, cfg):
        entity_name = cfg.asset.name
        joint_cfg = SceneEntityCfg(
            entity_name,
            joint_names=self._joint_names(cfg),
            preserve_order=True,
        )
        actuator_cfg = SceneEntityCfg(
            entity_name,
            actuator_names=(".*",),
            preserve_order=True,
        )
        foot_cfg = SceneEntityCfg(
            entity_name,
            site_names=("FL", "FR", "RL", "RR"),
            preserve_order=True,
        )
        hip_cfg = SceneEntityCfg(
            entity_name,
            joint_names=(
                "FL_hip_joint",
                "FR_hip_joint",
                "RL_hip_joint",
                "RR_hip_joint",
            ),
            preserve_order=True,
        )

        terms = {}
        std = math.sqrt(float(cfg.rewards.tracking_sigma))
        self._add_reward(
            terms,
            cfg,
            "tracking_lin_vel",
            HimGo2Env._reward_tracking_lin_vel,
            {"command_name": "twist", "std": std, "asset_cfg": SceneEntityCfg(entity_name)},
        )
        self._add_reward(
            terms,
            cfg,
            "tracking_ang_vel",
            HimGo2Env._reward_tracking_ang_vel,
            {"command_name": "twist", "std": std, "asset_cfg": SceneEntityCfg(entity_name)},
        )
        self._add_reward(
            terms,
            cfg,
            "lin_vel_z",
            HimGo2Env._reward_lin_vel_z,
            {"asset_cfg": SceneEntityCfg(entity_name)},
        )
        self._add_reward(
            terms,
            cfg,
            "ang_vel_xy",
            HimGo2Env._reward_ang_vel_xy,
            {"asset_cfg": SceneEntityCfg(entity_name)},
        )
        self._add_reward(
            terms,
            cfg,
            "orientation",
            HimGo2Env._reward_orientation,
            {"asset_cfg": SceneEntityCfg(entity_name)},
        )
        self._add_reward(
            terms,
            cfg,
            "dof_acc",
            HimGo2Env._reward_dof_acc,
            {"asset_cfg": joint_cfg},
        )
        self._add_reward(
            terms,
            cfg,
            "joint_power",
            HimGo2Env._reward_joint_power,
            {"asset_cfg": joint_cfg},
        )
        self._add_reward(
            terms,
            cfg,
            "base_height",
            HimGo2Env._reward_base_height,
            {
                "target_height": float(cfg.rewards.base_height_target),
                "sensor_name": "height_scan",
                "asset_cfg": SceneEntityCfg(entity_name),
            },
        )
        self._add_reward(
            terms,
            cfg,
            "foot_clearance",
            HimGo2Env._reward_foot_clearance,
            {
                # The old config stores -0.2; the Unitree implementation uses
                # site_pos_w directly, so keep the converted positive target explicit.
                "target_height": max(
                    0.05, float(abs(cfg.rewards.clearance_height_target))
                ),
                "command_name": "twist",
                "command_threshold": 0.1,
                "asset_cfg": foot_cfg,
            },
        )
        self._add_reward(
            terms,
            cfg,
            "action_rate",
            HimGo2Env._reward_action_rate,
        )
        self._add_reward(
            terms,
            cfg,
            "smoothness",
            HimGo2Env._reward_smoothness,
        )
        self._add_reward(
            terms,
            cfg,
            "feet_air_time",
            HimGo2Env._reward_feet_air_time,
            {
                "sensor_name": "feet_ground_contact",
                "threshold": 0.4,
                "command_name": "twist",
                "command_threshold": 0.1,
            },
        )
        self._add_reward(
            terms,
            cfg,
            "collision",
            HimGo2Env._reward_collision,
            {
                "sensor_name": "nonfoot_ground_touch",
                "force_threshold": float(cfg.rewards.max_contact_force),
            },
        )
        self._add_reward(
            terms,
            cfg,
            "stand_still",
            HimGo2Env._reward_stand_still,
            {
                "command_name": "twist",
                "command_threshold": 0.1,
                "asset_cfg": joint_cfg,
            },
        )
        self._add_reward(
            terms,
            cfg,
            "torques",
            HimGo2Env._reward_torques,
            {"asset_cfg": actuator_cfg},
        )
        self._add_reward(
            terms,
            cfg,
            "dof_vel",
            HimGo2Env._reward_dof_vel,
            {"asset_cfg": joint_cfg},
        )
        self._add_reward(
            terms,
            cfg,
            "dof_pos_limits",
            HimGo2Env._reward_dof_pos_limits,
            {"asset_cfg": joint_cfg},
        )
        self._add_reward(
            terms,
            cfg,
            "torque_limits",
            HimGo2Env._reward_torque_limits,
            {
                "soft_limit": float(cfg.rewards.soft_torque_limit),
                "asset_cfg": actuator_cfg,
            },
        )
        self._add_reward(
            terms,
            cfg,
            "hip_pos",
            HimGo2Env._reward_hip_pos,
            {"asset_cfg": hip_cfg},
        )
        return terms

    def _build_observations(self, cfg, play: bool):
        entity_name = cfg.asset.name
        joint_cfg = SceneEntityCfg(
            entity_name,
            joint_names=self._joint_names(cfg),
            preserve_order=True,
        )
        noise_enabled = not play

        actor_terms = {
            "base_ang_vel": ObservationTermCfg(
                func=envs_mdp.builtin_sensor,
                params={"sensor_name": f"{entity_name}/imu_ang_vel"},
                noise=self._noise(cfg, "ang_vel", noise_enabled),
                scale=float(cfg.normalization.obs_scales.ang_vel),
                **self._obs_delay(
                    cfg,
                    "range_obs_imu_latency",
                    cfg.domain_rand.randomize_obs_imu_latency and not play,
                ),
            ),
            "projected_gravity": ObservationTermCfg(
                func=envs_mdp.projected_gravity,
                params={"asset_cfg": SceneEntityCfg(entity_name)},
                noise=self._noise(cfg, "gravity", noise_enabled),
                **self._obs_delay(
                    cfg,
                    "range_obs_imu_latency",
                    cfg.domain_rand.randomize_obs_imu_latency and not play,
                ),
            ),
            "command": ObservationTermCfg(
                func=envs_mdp.generated_commands,
                params={"command_name": "twist"},
            ),
            "joint_pos": ObservationTermCfg(
                func=envs_mdp.joint_pos_rel,
                params={"biased": True, "asset_cfg": joint_cfg},
                noise=self._noise(cfg, "dof_pos", noise_enabled),
                scale=float(cfg.normalization.obs_scales.dof_pos),
                **self._obs_delay(
                    cfg,
                    "range_obs_motor_latency",
                    cfg.domain_rand.randomize_obs_motor_latency and not play,
                ),
            ),
            "joint_vel": ObservationTermCfg(
                func=envs_mdp.joint_vel_rel,
                params={"asset_cfg": joint_cfg},
                noise=self._noise(cfg, "dof_vel", noise_enabled),
                scale=float(cfg.normalization.obs_scales.dof_vel),
                **self._obs_delay(
                    cfg,
                    "range_obs_motor_latency",
                    cfg.domain_rand.randomize_obs_motor_latency and not play,
                ),
            ),
            "actions": ObservationTermCfg(func=envs_mdp.last_action),
        }

        # ``enable_corruption=False`` disables noise models, but the
        # ObservationManager still applies a term's delay buffer.  Build a
        # genuinely clean privileged copy instead of shallow-copying delayed
        # actor terms.  Encoder bias is an actor-side observation effect too.
        critic_terms = {}
        for name, term in actor_terms.items():
            critic_term = copy.deepcopy(term)
            critic_term.noise = None
            critic_term.delay_min_lag = 0
            critic_term.delay_max_lag = 0
            critic_term.delay_hold_prob = 0.0
            critic_term.delay_update_period = 0
            if name == "joint_pos":
                critic_term.params = {
                    "biased": False,
                    "asset_cfg": joint_cfg,
                }
            critic_terms[name] = critic_term
        critic_terms.update({
            "base_lin_vel": ObservationTermCfg(
                # Use root-link velocity in the body frame.  The XML
                # velocimeter at the IMU site is not automatically identical
                # to the legacy root-state velocity used by HIM.
                func=envs_mdp.base_lin_vel,
                params={"asset_cfg": SceneEntityCfg(entity_name)},
            ),
            "height_scan": ObservationTermCfg(
                func=envs_mdp.height_scan,
                params={"sensor_name": "height_scan"},
                scale=float(cfg.normalization.obs_scales.height_measurements),
            ),
        })

        return {
            "actor": ObservationGroupCfg(
                terms=actor_terms,
                concatenate_terms=True,
                enable_corruption=bool(noise_enabled and cfg.noise.add_noise),
                history_length=1,
            ),
            "critic": ObservationGroupCfg(
                terms=critic_terms,
                concatenate_terms=True,
                enable_corruption=False,
                history_length=1,
            ),
        }

    def _build_terminations(self, cfg):
        entity_name = cfg.asset.name
        body_cfg = SceneEntityCfg(entity_name, body_names=("base_link",))
        return {
            "time_out": TerminationTermCfg(
                func=envs_mdp.time_out,
                time_out=True,
            ),
            "bad_orientation": TerminationTermCfg(
                func=envs_mdp.bad_orientation,
                params={
                    "limit_angle": math.radians(70.0),
                    "asset_cfg": body_cfg,
                },
            ),
            "base_contact": TerminationTermCfg(
                func=HimGo2Env._reward_illegal_contact,
                params={
                    "sensor_name": "base_ground_touch",
                    "force_threshold": 10.0,
                },
            ),
            "nonfoot_contact": TerminationTermCfg(
                func=HimGo2Env._reward_illegal_contact,
                params={
                    "sensor_name": "nonfoot_ground_touch",
                    "force_threshold": 10.0,
                },
            ),
        }

    def _build_curriculum(self, cfg, play: bool):
        if play or not cfg.terrain.curriculum:
            return {}
        return {
            "terrain_levels": CurriculumTermCfg(
                func=mdp.terrain_levels_vel,
                params={
                    "command_name": "twist",
                    "asset_cfg": SceneEntityCfg(cfg.asset.name),
                },
            )
        }
```

### 4.1 奖励函数的来源：`mdp` 不是 `velocity_mdp`

这里需要把上一版文档中的命名和来源说清楚：`velocity_mdp` 不是官方 API，也不是一个需要安装的模块名。它只是 Python 导入语句

```python
from mjlab.tasks.velocity import mdp as velocity_mdp
```

人为创建的本地别名。官方包路径是 `mjlab.tasks.velocity.mdp`；写成 `mdp`、`velocity_mdp` 或其他名字都只是变量命名差异。上一版直接用这个别名注册奖励，容易让人误以为这些函数就是 `unitree_rl_mjlab` 的奖励实现，这一点是错误的。

`unitree_rl_mjlab` 的实际做法是先导入官方模块，再用项目自己的模块覆盖同名变量：

```python
# unitree_rl_mjlab/src/tasks/velocity/velocity_env_cfg.py
from mjlab.tasks.velocity import mdp
from mjlab.envs import mdp as envs_mdp

# 这一行会覆盖上面的 mdp；后续 mdp.track_linear_velocity 等来自 src。
import src.tasks.velocity.mdp as mdp
```

`src/tasks/velocity/mdp/__init__.py` 先重新导出 `mjlab.envs.mdp` 的通用项，再导出本项目的 `rewards.py`、`observations.py`、`terminations.py` 和 command/curriculum 实现。因此 Unitree 的配置才会这样注册奖励：

```python
rewards = {
    "track_linear_velocity": RewardTermCfg(
        func=mdp.track_linear_velocity,
        weight=1.0,
        params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "feet_clearance": RewardTermCfg(
        func=mdp.feet_clearance,
        weight=-1.0,
        params={
            "target_height": 0.10,
            "command_name": "twist",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", site_names=()),
        },
    ),
    # 通过 __init__.py 重新导出的 mjlab 通用项也可以继续使用。
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
}
```

本文件的完整替换代码为了保持“单文件可复制”，借用了 Unitree 的函数命名和函数签名，例如 `track_linear_velocity`、`feet_air_time`、`feet_clearance`、`self_collision_cost` 和 `stand_still`；这只说明注册接口和部分非 tracking term 的参考来源，不能视为所有实现的数值语义都与 Unitree 对齐。最终候选的这些自定义 reward 都定义为 `HimGo2Env` 内的 `@staticmethod`，并由 `HimGo2Env.<method>` 注册；不要在同一候选代码块再添加同名模块级 helper。tracking 两项明确按 `/home/kk/github/2027_RC_legged_robot/legged_gym/envs/legged_gym_go1` 的 legacy 轴语义：线速度只跟踪 xy，角速度只跟踪 yaw。Unitree mjlab 的原 `track_linear_velocity` 还把 z 项合并到 tracking，原 `track_angular_velocity` 还把 xy 项合并到 tracking；本候选则将 z/xy 分别保留为独立的 `lin_vel_z`/`ang_vel_xy` reward term。只有 `envs_mdp.*` 是可保留的 mjlab 通用 reward term，不属于本项目自定义实现。

本次修订的关键映射如下：

| 旧 reward 名 | 修订后的函数 | 实现来源/语义 |
| --- | --- | --- |
| `tracking_lin_vel` | `track_linear_velocity` | 按 `/home/kk/github/2027_RC_legged_robot/legged_gym/envs/legged_gym_go1` 的 legacy 轴语义，只计算 command/actual 的 xy 误差；Unitree mjlab 原函数还把 z 项并入 tracking，本候选由独立的 `lin_vel_z` term 承担 |
| `tracking_ang_vel` | `track_angular_velocity` | 按 `/home/kk/github/2027_RC_legged_robot/legged_gym/envs/legged_gym_go1` 的 legacy 轴语义，只计算 command/actual 的 yaw 误差；Unitree mjlab 原函数还把 xy 项并入 tracking，本候选由独立的 `ang_vel_xy` term 承担 |
| `lin_vel_z` / `ang_vel_xy` | `lin_vel_z` / `ang_vel_xy` | 补回旧配置中原本非零、但上一版漏注册的两项 |
| `orientation` | `body_orientation_l2` | 用 projected gravity 的 xy 分量衡量身体倾斜 |
| `feet_air_time` | `feet_air_time` | 候选选择 Unitree 风格 `threshold=0.4` 与 mode-time `min`；`mean(contact)==0.5` 是四足中恰好两足接触，不是 single stance；legacy 另有 `threshold=0.5`、per-foot first-contact 累加和 xy gate |
| `foot_clearance` | `feet_clearance` | 直接采用 Unitree 的 foot site 高度与水平速度实现，不把 `height_scan` 误传给官方同名函数 |
| `collision` | `self_collision_cost` | 使用 `nonfoot_ground_touch` 的 `force_history=[N, primary, history, 3]`；先按 primary 聚合，再按 history 槽计数，槽数不是接触总数；`max_contact_force=100` 仅是候选策略值，不等于 legacy `>0.1` 或 Unitree 默认 `10`，runtime 仍为 `NOT_RUN` |
| `stand_still` | `stand_still` | 低速命令时惩罚相对默认姿态的关节误差 |
| `hip_pos` | `hip_pos` | 修正为 hip 相对默认关节姿态的 L2，不再错误复用 `joint_pos_limits` |

第4节现已把这些项也封装为 `HimGo2Env._reward_*` 静态方法；它们的公式参考 mjlab 通用 term，但 callable 归属仍在本类。特别是 legacy `dof_acc` 是相邻 policy step 的 velocity difference derivative，而 `envs_mdp.joint_acc_l2` 使用 mjlab physics acceleration；二者在时间采样和数值上不等价，候选选择必须记录为语义差异。

### 4.2 代码中的有意映射

有几个旧配置字段不能机械地一对一搬运：

- `clearance_height_target=-0.2` 是旧项目的 signed、root-relative 约定；候选 `feet_clearance` 使用 world/site 的绝对高度与水平速度，并用绝对高度误差乘速度，不同于 legacy body-frame、root-relative 的平方高度误差。代码取 `abs(-0.2)` 只是一次明确的语义重映射，不是单位转换；目标仍须通过落地数据校准。
- 旧的 `torque_limits` 不是一个标准 mjlab reward term；代码使用每个 `IdealPdActuator` 的 `force_limit` 做超限惩罚。`actuator_force` 可能已经是 effort clamp 后的信号，读不到超限值；`soft_limit=1.0` 又没有安全裕量，可能使该项恒为零，必须由 S14 runtime 核验。物理 actuator 限幅本身仍由 `Go2Asset` 完成。
- `tracking_sigma=0.25` 表示指数跟踪函数的 `std=sqrt(tracking_sigma)`；这与 Unitree 配置中显式写 `math.sqrt(0.25)` 的语义一致。
- `feet_air_time` 依赖 `feet_ground_contact` 开启 `track_air_time=True`，`collision` 与 `base_contact` 分别依赖接触传感器的 `force_history`。候选 force history 的轴契约是 `[N, primary, history, 3]`；`primary` 是匹配的 geom/body，`history` 是历史样本槽，最后一轴是三维力，不能把 history 槽数量说成接触总数。因此这些 reward/termination 不能只注册函数而不配置对应 sensor field/history。
- candidate 配置中的 `max_contact_force=100.0` 仅是 collision 的候选策略值，不等于 legacy contact force `>0.1`，也不等于 Unitree `self_collision` 默认 `10`；collision runtime 仍为 `NOT_RUN`。
- `terrain_proportions` 没有直接传进 `ROUGH_TERRAINS_CFG`。官方 rough preset 已经提供非空的 `sub_terrains`；curriculum 模式按这些子地形类型生成列。若必须复现旧的五比例地形，需要在代码中显式构造每个 `SubTerrainCfg`，不能保留空字典。
- 当前仓库配置中的 restitution、payload mass 和部分 latency 字段没有在本文件中伪造为不存在的事件。它们应分别映射到 `dr` 中对应的 model field 或显式 sensor/action delay；先保证主环境和 shape contract 可运行。

## 5. 外部/本地 `rsl_rl` 接入

### 5.1 当前 loader 的行为

`legged_mjlab/legged_mjlab/utils/task_registry.py` 的边界是：

1. 导入当前解释器能找到的 `rsl_rl`。
2. 检查 `rsl_rl.runners.HIMOnPolicyRunner` 是否存在。
3. 若不存在，移除 `rsl_rl.*` 模块并从仓库目录 `legged_mjlab/rsl_rl/rsl_rl` 以 canonical namespace `rsl_rl` 加载。
4. `make_alg_runner()` 根据 `train_cfg["runner"]["runner_class_name"]` 查找 `HIMOnPolicyRunner`。

因此 `HimGo2CfgPPO` 中的三个类名必须保持：

```python
runner_class_name = "HIMOnPolicyRunner"
policy_class_name = "HIMActorCritic"
algorithm_class_name = "HIMPPO"
```

### 5.2 安装与来源验证

在项目虚拟环境中执行：

```bash
cd /home/kk/legged_mjlab
source .venv/bin/activate
uv pip install --python .venv/bin/python -e .
```

然后先验证 runner 来源，不要直接启动 4096 个环境：

```bash
cd /home/kk/legged_mjlab
source .venv/bin/activate
PYTHONPATH=/home/kk/legged_mjlab python - <<'PY'
from pathlib import Path

from legged_mjlab.utils.task_registry import load_project_rsl

runners = load_project_rsl()
import rsl_rl

print("rsl_rl:", Path(rsl_rl.__file__).resolve())
print("HIMOnPolicyRunner:", runners.HIMOnPolicyRunner)
assert hasattr(runners, "HIMOnPolicyRunner")
assert "legged_mjlab/rsl_rl/rsl_rl" in str(Path(rsl_rl.__file__).resolve())
PY
```

如果断言失败，说明解释器先找到了另一个兼容 `rsl_rl`。这时应使用一个没有官方 `rsl_rl` 的项目虚拟环境，或调整启动时的包搜索路径；不要把一个已加载的官方 `rsl_rl` 子模块与仓库实现混用。

### 5.3 训练入口

现有 `scripts/train.py` 已经使用 `task_registry.make_env()` 和 `make_alg_runner()`，建议先用小批量和 1 个 iteration 做 shape smoke test：

```bash
cd /home/kk/legged_mjlab
source .venv/bin/activate
PYTHONPATH=/home/kk/legged_mjlab python -m legged_mjlab.scripts.train \
  --task him_go2 \
  --device cpu \
  --num-envs 4 \
  --max-iterations 1 \
  --log-dir /tmp/legged_mjlab_logs
```

确认 CPU smoke test 通过后，再切换到 GPU 和正式数量：

资源预检必须单独记录。critic 的 height scan 是每个环境 17×11=187 条 ray，4096 个环境会同时放大 raycast、contact history、HIM history、CPU 调度和 GPU/warp 显存压力；当前工作树没有 NVIDIA driver 可用的证据。因此先完成 4-env CPU，再按 16→64→256 等小规模阶梯做 GPU（若设备可用）验证，记录峰值显存、耗时和 reset/step 是否稳定，最后才尝试 4096。GPU/显存不足属于资源未验证或资源阻断，不得被记为环境接口或训练代码已通过。

```bash
PYTHONPATH=/home/kk/legged_mjlab python -m legged_mjlab.scripts.train \
  --task him_go2 \
  --device cuda:0 \
  --num-envs 4096 \
  --log-dir logs
```

`HIMRslRlWrapper` 的 `step()` 返回 7 项：

```python
(
    history_obs,              # [N, 270]
    privileged_obs,           # [N, 235]
    rewards,                  # [N]
    dones,                    # [N]
    infos,
    termination_ids,          # [K]
    termination_privileged,   # [K, 235] 或空 tensor
)
```

这正是项目内 `HIMOnPolicyRunner._unpack_step()` 支持的 terminal privileged observation 路径。候选 `ManagerBasedRlEnvCfg.auto_reset` 必须直接设为 `False`，由候选环境在 `step()` 中先保留真实 terminal critic frame；`HIMRslRlWrapper` 只负责断言 frame/shape/索引并做兼容适配，确认捕获后才 reset 已结束的环境。不能先 auto-reset，再依赖 wrapper 从 reset 后 observation 猜测 terminal frame，否则 timeout bootstrap 会使用错误状态。

## 6. 训练前的 smoke test

### 6.1 静态检查

手工把代码复制到目标 `.py` 后执行：

```bash
cd /home/kk/legged_mjlab
source .venv/bin/activate
python -m py_compile legged_mjlab/envs/him_go2/him_go2_env.py
```

### 6.2 manager 配置检查

```bash
cd /home/kk/legged_mjlab
source .venv/bin/activate
PYTHONPATH=/home/kk/legged_mjlab python - <<'PY'
from legged_mjlab.envs.him_go2.him_go2_config import HimGo2RoughCfg
from legged_mjlab.envs.him_go2.him_go2_env import HimGo2Env

cfg = HimGo2RoughCfg()
manager_cfg = HimGo2Env._build_mjlab_managercfg(
    object.__new__(HimGo2Env),
    cfg,
    play=True,
    debug_vis=False,
)

assert manager_cfg.decimation == 4
assert manager_cfg.sim.mujoco.integrator == "implicitfast"
assert manager_cfg.sim.mujoco.gravity == (0.0, 0.0, -9.81)
assert manager_cfg.scene.terrain.terrain_generator is not None
assert manager_cfg.observations["actor"].history_length == 1
assert all(
    manager_cfg.actions[name].actuator_names
    for name in (
        "fl_joint_position",
        "rl_joint_position",
        "fr_joint_position",
        "rr_joint_position",
    )
)
print("manager config construction: OK")
PY
```

### 6.3 真正环境的 shape 检查

```bash
cd /home/kk/legged_mjlab
source .venv/bin/activate
PYTHONPATH=/home/kk/legged_mjlab python - <<'PY'
import torch

import legged_mjlab.envs  # registers him_go2
from legged_mjlab.utils.task_registry import task_registry, load_project_rsl

load_project_rsl()
env, env_cfg = task_registry.make_env(
    "him_go2",
    device="cpu",
    play=True,
    num_envs=4,
)
history, privileged = env.reset()
assert tuple(history.shape) == (4, 270), history.shape
assert tuple(privileged.shape) == (4, 235), privileged.shape
assert env.num_actions == 12

action = torch.zeros((4, 12), device=history.device, dtype=history.dtype)
step_result = env.step(action)
next_history, next_privileged = step_result[:2]
assert tuple(next_history.shape) == (4, 270), next_history.shape
assert tuple(next_privileged.shape) == (4, 235), next_privileged.shape
print("HIM environment shape contract: OK")
PY
```

### 6.4 失败时按顺序定位

1. `SyntaxError`：目标文件没有完整替换，或把 Markdown fence 一并复制进了 Python 文件。
2. `TypeError: unexpected keyword actuator_name`：动作配置仍是旧代码，必须使用 `actuator_names`。
3. `ValueError: no matching names`：先打印 `Go2Asset` 的 `entity.joint_names`、`entity.actuator_names`、`entity.site_names`，核对 XML 名称；当前 XML 的足端 site 是 `FL/FR/RL/RR`，接触 geom 是 `*_foot_collision`。
4. critic 维度不是 235：检查 actor term 顺序、height scan 网格是否为 17×11，以及是否把 history 放进了原生 mjlab group。若你显式启用 4 维 `foot_contact` 扩展，则这里才应改为 239，并同步修改 `num_privileged_obs` 与 wrapper 断言。
5. wrapper 报 `auto_reset=False`：说明底层环境配置没有暴露 `ManagerBasedRlEnvCfg.auto_reset`，不能继续训练，必须先解决 terminal observation 保留问题。
6. `rsl_rl` 来源断言失败：运行时加载的是另一个兼容包；先处理包来源，再排查 PPO 参数。
7. GPU 启动失败但 CPU 通过：先降低 `num_envs`，确认 mujoco-warp、CUDA 和 `mujoco_warp` 版本，再恢复 4096。

## 7. 与参考实现的对应关系

- `unitree_rl_mjlab/src/tasks/velocity/velocity_env_cfg.py` 的 pinned 依赖是 `mjlab==1.2.0`、`mujoco-warp==3.5.0`；其 scene、sensor、observation、action、command、event、reward、termination、curriculum 的 flat `ManagerBasedRlEnvCfg` 组织方式在本文仅作迁移组织参考，不是当前 mjlab v1.6.0 的 API 或运行证据。
- 官方 mjlab v1.6.0 的 API 与迁移规则另见第 9.3 节。
- `unitree_rl_mjlab/src/tasks/velocity/mdp/rewards.py` 才是 Unitree velocity task 的任务特有奖励实现；其中的 `track_linear_velocity`、`feet_clearance`、`feet_air_time` 和 `stand_still` 通过 `RewardTermCfg(func=mdp.<name>, ...)` 注册。
- `unitree_rl_mjlab/src/tasks/velocity/config/go2/env_cfgs.py` 提供了 Go2 的实际接触 geom、site、raycast frame、rough-terrain play override 和 `illegal_contact` 组织方式。
- `AMP_mjlab/src/tasks/amp_loco/amp_env_cfg.py` 与其 MDP term 展示了 `RewardTermCfg`、`TerminationTermCfg`、`SceneEntityCfg` 的可复制写法。
- `legged_gym` 的 legacy 配置仍然适合用来核对 observation/reward 的语义和维度，但不能直接把它的 manager、clip、reset 或旧 sensor 字段搬进 mjlab。
- 官方 mjlab 文档中的 [environment config](https://mujocolab.github.io/mjlab/main/source/environment_config.html)、[observations](https://mujocolab.github.io/mjlab/main/source/observations.html)、[actions](https://mujocolab.github.io/mjlab/main/source/actions.html)、[events](https://mujocolab.github.io/mjlab/main/source/events.html)、[rewards](https://mujocolab.github.io/mjlab/main/source/rewards.html)、[terminations](https://mujocolab.github.io/mjlab/main/source/terminations.html)、[terrain](https://mujocolab.github.io/mjlab/main/source/terrain.html) 是当前 API 对照入口。

## 8. 完成判据

这部分手工代码落地后，`him_go2` 才算达到“可训练”而不是“文件能导入”：

- Python 文件可编译；
- `manager_cfg` 可以创建，且 `MujocoCfg.integrator`、`gravity` 类型正确；
- XML 的 actuator、joint、site、geom 全部匹配；
- actor 为 `[N, 45]`，wrapper history 为 `[N, 270]`；
- critic 为 `[N, 235]`，其中前 48 维符合 HIM estimator；
- 候选环境在 `auto_reset=False` 时直接提供真实 terminal privileged observation，wrapper 只做断言/兼容后供 runner 使用；
- runtime `rsl_rl.__file__` 指向项目内 `legged_mjlab/rsl_rl/rsl_rl`；
- 4 环境 CPU、1 iteration smoke test 通过后，才启动 GPU/4096 环境正式训练。

## 9. 2026-08-28 审计勘误与最终接线矩阵

本节是对前文方案的强制勘误。它按当前工作树重新核对，不能把“配置类里出现了字段”解释成“运行时已经生效”。本轮只修改本文档；源码仍保持审计开始时的状态。

### 9.1 当前结论

| 层 | 当前事实 | 状态 |
|---|---|---|
| Python 入口 | `him_go2_env.py:101` 为 `metrics = ,` | 阻断 import |
| Asset | `Go2Asset._parse_cfg()` 在 `self.entity` 创建前访问 `self.robot_cfg`/`self.asset` | 阻断构造 |
| Manager | builder 的参数签名、`MujocoCfg.integrator`、actions 字段、command range 均有错配；当前定义锚点为 scene `:101`、events `:228`、commands `:349`、observations `:410`、terminations `:449`、curriculum `:461`、rewards `:399-408` | 未对齐 |
| Registry → Env | `task_registry.py:201` 传 `device=...`，而当前 `HimGo2Env.__init__` 仍只接收 `sim_device` | P0 构造阻断 |
| Action order | 当前配置 dict 是“按关节类型分组”的插入顺序；部署是 leg-major `FL, RL, FR, RR`；XML runtime 是 leg-major `FL, FR, RL, RR`；当前源码没有可直接复用的唯一重排表 | 文档候选以 canonical joint order 为契约，通过 `_joint_names()` 与四个分腿 action term 固定 `FL,RL,FR,RR × hip,thigh,calf` 的策略顺序；仍需 P0 round-trip 验证 |
| Effort/clip | 仿真 calf 为 `45 N*m`，部署为 `35.55 N*m`；当前 action clip 字段也不合法 | P0 安全阻断 |
| Delay/fail-safe | action latency 的 physics/policy 单位未闭环；部署无已验证 watchdog、安全停机和断连处理 | P0 安全阻断 |
| Actor observation | 目标单帧 45，但当前 builder 只写了两个 term 且没有返回字典 | 未实现 |
| Native critic | 当前 scope 目标可构造为 `45+3+187=235`；`45+3+4+187=239` 仅是可选足端接触扩展 | 文档目标，源码未实现 |
| HIM runner critic | `rsl_rl/rsl_rl/runners/him_on_policy_runner.py:165-168` 固定为 `45+3=48`，并在 `:469-492` 截断更宽 critic | 已实现为 48，和 native 235 是两层契约 |
| Reward | `him_go2_config.py:174-202` 有非零 scale，但 `_build_rewards()` 返回空字典 | 实际注册 0 项 |
| Domain randomization | payload、mass/inertia、armature、friction、restitution、motor strength、latency、push 等未形成可验证接线 | 未放行 |
| Termination/safety | `terminate_after_contacts_on=[]`，当前仅 timeout；初始姿态、落地冲击、危险接触均未验证 | P0 安全阻断 |
| Task/训练 | 本地 registry、HIM wrapper、HIM runner 已存在；`play.py` 与 `test_env.py` 为空 | 入口未闭环 |
| 安装 | 当前 `.venv` 没有 `pip`；根目录也没有 `pyproject.toml`/`uv.lock`/console script | 文档必须使用 `uv pip` |

因此当前项目的准确描述是：**架构方向正确，但环境不是“差几项配置”，而是尚未通过 import、asset、manager、shape 和 runner contract 的连续验收。**

### 9.2 配置字段逐项接线状态

#### Reward scale

| legacy scale | 当前声明 | 文档中应接到的 manager term | 依赖与状态 |
|---|---:|---|---|
| `tracking_lin_vel` | `1.0` | `track_linear_velocity` | 按 `/home/kk/github/2027_RC_legged_robot/legged_gym/envs/legged_gym_go1` 的 legacy 轴语义使用 `twist`/base body velocity 的 xy 误差；z 惩罚由独立 `lin_vel_z` term 承担；当前未注册 |
| `tracking_ang_vel` | `0.5` | `track_angular_velocity` | 按 `/home/kk/github/2027_RC_legged_robot/legged_gym/envs/legged_gym_go1` 的 legacy 轴语义使用 `twist`/base angular velocity 的 yaw 误差；xy 惩罚由独立 `ang_vel_xy` term 承担；当前未注册 |
| `lin_vel_z` | `-1.5` | `lin_vel_z` | robot root body z 速度的独立惩罚；当前未注册 |
| `ang_vel_xy` | `-0.05` | `ang_vel_xy` | robot root body xy 角速度的独立惩罚；当前未注册 |
| `orientation` | `-0.2` | `body_orientation_l2`/`flat_orientation_l2` | projected gravity；当前未注册 |
| `dof_acc` | `-2.5e-7` | `HimGo2Env._reward_dof_acc` | 12 joint order；当前未注册 |
| `joint_power` | `-2e-5` | 自定义 `_joint_power_l1` | 需要 actuator force 与 joint velocity；当前未注册 |
| `base_height` | `-1.0` | 自定义 `_base_height_l2` | `height_scan` 有效命中的 terrain-relative measured height 均值与 target `0.30`；当前未注册 |
| `foot_clearance` | `-0.01` | 自定义 `feet_clearance` | 四个 foot site；当前未注册 |
| `action_rate` | `-0.01` | `HimGo2Env._reward_action_rate` | action manager history；当前未注册 |
| `smoothness` | `-0.01` | `HimGo2Env._reward_smoothness` | 二阶 action history；当前未注册 |
| `feet_air_time` | `+0.1` | `feet_air_time` | `track_air_time=True` contact sensor；当前未注册 |
| `collision` | `-0.5` | `self_collision_cost`/illegal contact cost | non-foot force history；当前未注册 |
| `feet_stumble` | `-0.0` | 不注册 | 当前权重为零；没有可宣称的活动 term |
| `stand_still` | `-1.0` | `stand_still` | low-command gate + default pose；当前未注册 |
| `torques` | `-0.0` | `joint_torques_l2` | 当前权重为零，按约定不注册 |
| `dof_vel` | `-0.0` | `joint_vel_l2` | 当前权重为零，按约定不注册 |
| `dof_pos_limits` | `-10.0` | `joint_pos_limits` | soft joint limits；当前未注册 |
| `dof_vel_limits` | `-0.0` | 自定义/不注册 | 当前权重为零 |
| `torque_limits` | `-1.0` | 自定义 force-limit cost | 不是普通 torque L2；需要 actuator force limit |
| `hip_pos` | `-1.0` | `hip_pos` | hip 四关节独立 `SceneEntityCfg`；当前未注册 |
| `termination` | `-0.0` | `is_terminated` 或自定义 terminal term | 当前权重为零；不能混淆 termination manager |

#### Reward 参数字段（不是 scale）

这些字段也存在于 legacy 配置，但不会因为出现在 `rewards` 类中就自动进入 mjlab manager。表中“候选使用”只描述文档候选代码显式读取了该字段；当前仓库的 `HimGo2Env` 仍未完成可导入/运行闭环。

| legacy 参数 | 当前声明 | 候选接线 | 当前状态 |
|---|---:|---|---|
| `tracking_sigma` | `0.25` | 跟踪 term 使用 `std=sqrt(tracking_sigma)` | 候选使用，未完成真实 manager/reward 数值验证 |
| `soft_dof_pos_limit` | `0.85` | 传给 `EntityArticulationInfoCfg.soft_joint_pos_limit_factor`，由 `joint_pos_limits` 使用 | 当前源码未接入；候选显式使用，未运行证明 |
| `soft_dof_vel_limit` | `1.0` | 无活动映射；`dof_vel_limits` scale 为 `-0.0`，且没有额外 velocity-limit term | **声明但不生效**；不得写成已接入的 soft velocity limit |
| `soft_torque_limit` | `1.0` | 候选自定义 `_torque_limit_cost(soft_limit=...)` | 当前源码未接入；候选显式使用，未运行证明 |
| `base_height_target` | `0.30` | 候选自定义 `_base_height_l2(target_height=..., sensor_name="height_scan")` | 当前源码未接入；候选显式使用，未运行证明 |
| `max_contact_force` | `100.0` | candidate collision 的候选策略阈值 `self_collision_cost(force_threshold=...)`；不等于 legacy contact force `>0.1` 或 Unitree `self_collision` 默认 `10` | 当前源码未接入；collision runtime 为 `NOT_RUN`，未运行证明 |
| `clearance_height_target` | `-0.2` | 候选转换为 `feet_clearance` 的正高度目标 `0.2 m` | 当前源码未接入；候选显式使用，仍需数值校准 |
| `only_positive_rewards` | `True` | 候选在 `super().step()` 汇总后执行 `reward.clamp_min(0.0)` | mjlab 不自动读取；当前源码未接入，候选顺序仍需 runtime 验证 |

所有非零 term 都必须满足：函数输出 `[N]`，`RewardTermCfg.weight` 直接使用该表的 raw legacy scale，sensor/command/entity 参数显式传入。候选必须保持 `scale_rewards_by_dt=True`，让 legacy legged_gym 的 reward scale 统一乘 policy `step_dt=0.005*4=0.02`。禁止使用 raw legacy scales + `scale_rewards_by_dt=False`，否则会产生约 `1/0.02=50` 倍的 reward 漂移；也禁止先手工把 scale 乘 `dt` 后再设为 `True`，否则会重复乘一次 `dt`。函数输出和 weight 只能保留这一条 dt 缩放路径。

特别注意：mjlab 的 `RewardManager` 不会读取 `legacy rewards.only_positive_rewards`。如果要保留当前 `True` 的语义，必须在环境 reward 汇总之后显式执行 `reward.clamp_min(0.0)`，并验证 termination penalty 是否应在截断前还是截断后加入；如果不打算改环境生命周期，则应把该字段改为明确的“不生效/待迁移”，不能默认为已接入。

#### Domain randomization

| flag | legacy 范围 | mjlab 1.6.0 映射 | 生命周期 | 当前状态 |
|---|---|---|---|---|
| `randomize_payload_mass` | `[-2.5,2.5] kg` | `dr.body_mass` 对 `base_link`，`operation="add"` | `reset` 或 `startup` | 当前是 `pass`（`him_go2_env.py:283-284`），未接入；候选已显式注册，但未运行证明 |
| `randomize_link_mass` | `[0.9,1.1]` ratio | `dr.body_mass`，`operation="scale"` | `startup`/`reset` | 配置声明，当前未接入；只改 mass，不等价于同步 inertia |
| `randomize_com_displacement` | `[-0.05,0.05] m` | `dr.body_com_offset` 的 x/y/z | `startup`/`reset` | 候选事件，未运行证明，未放行 |
| `randomize_joint_friction` | `[0.01,1.15]` | `dr.joint_friction` | `startup`/`reset` | 候选事件，未运行证明，未放行 |
| `randomize_joint_damping` | `[0.3,1.5]` | `dr.joint_damping` | `startup`/`reset` | 候选事件，未运行证明，未放行 |
| `randomize_joint_armature` | `[0.0001,0.05]` | `dr.joint_armature`/`dr.dof_armature` | `startup`/`reset` | 当前未接入 |
| `randomize_friction` | `[0.2,1.3]` | `dr.geom_friction` 足端 geom；若使用 pair 必须有显式 contact pair | `reset` | 当前未接入 |
| `randomize_restitution` | `[0,0.4]` | 需要对应 geom/contact model 字段的自定义映射 | `reset` | 不能用不存在的 `dr.restitution` 冒充，未接入 |
| `push_robots` | interval `4 s`, max `1 m/s` | `mdp.push_by_setting_velocity` | `interval` | 候选注册，未放行；play 必须关闭 |
| `randomize_pd_gains` | Kp/Kd `[0.8,1.2]` | `dr.pd_gains`，`operation="scale"` | `startup`/`reset` | v1.6.0 API 支持 `IdealPdActuator`；候选事件尚未运行证明，未放行 |
| `randomize_motor_zero_offset` + `motor_zero_pos_offset_range` | pos `[-0.035,0.035]` | `dr.encoder_bias` | `startup`/`reset` | 当前源码未接入；候选只接 position bias，未运行证明 |
| `motor_zero_vel_offset_range` | vel `[-0.01,0.01]` | 无已确认的 mjlab 1.6.0 内置等价 term | — | **声明但不生效**；候选不把 velocity bias 错接到 `dr.encoder_bias`，需自定义 adapter + 测试后才可启用 |
| `randomize_motor_strength` | `[0.8,1.2]` | 输出 torque 后乘 per-env factor，再做 effort clip；`dr.effort_limits` 仅改变饱和上限 | `startup`/`reset` | 当前源码未接入；候选提供 `ScaledIdealPdActuator` + custom event，未运行证明 |
| `randomize_obs_motor_latency` | `[1,4]` | 每个 `ObservationTermCfg.delay_min_lag/max_lag` | policy step | 当前未接入；候选 helper 已接到 joint pos/vel，未运行证明 |
| `randomize_obs_imu_latency` | `[1,3]` | IMU `builtin_sensor` term 的 delay | policy step | 当前未接入；候选 helper 已接到 IMU terms，未运行证明 |
| `randomize_cmd_action_latency` | `[1,4]` | `IdealPdActuatorCfg.delay_min_lag/max_lag`（physics lag）或单独 action queue（policy lag） | actuator 为 physics step；queue 为 policy step | 当前仅 Asset 有意图；候选已做 policy→physics 换算且 play 关闭，未运行证明 |

关键单位规则：mjlab observation delay 使用 **policy step**；`IdealPdActuatorCfg.delay_min_lag/max_lag` 使用 **physics timestep**。当前 `dt=0.005`、decimation=4，因此 actuator `[1,4]` 是 5–20 ms，而不是 1–4 个 policy step；若目标是旧部署中的 1–4 policy steps，应换算为 physics lag `[4,16]` 或另加 policy-level action queue。动作会在每个 decimation physics substep 写入 actuator target。三个 IdealPd actuator group 若传入相同 delay 参数，当前实现会共享一个融合 delay buffer，不等于每个电机独立抽样。

#### 安全硬门禁：动作、力矩、接触与失效路径

以下内容不是“后续优化”，而是动作进入仿真或硬件前必须闭环的契约：

| 契约 | 当前证据 | 文档候选实现的强制规则 | 放行条件 |
|---|---|---|---|
| 关节/动作顺序 | 当前配置 dict 按 `hip` 全组、`thigh` 全组、`calf` 全组插入；部署是 leg-major `FL, RL, FR, RR`；`go2.xml` runtime 是 leg-major `FL, FR, RL, RR` | 候选代码固定 policy/deployment 顺序为 `FL,RL,FR,RR` 且按每腿 `hip,thigh,calf`；四个 action term 显式分腿，禁止隐式依赖 dict 或 XML 全局顺序 | 用零动作、单位脉冲和 round-trip 映射逐关节证明 FL/RL/FR/RR 的 hip/thigh/calf 均一致 |
| 力矩上限 | 当前仿真 hip/thigh/calf 为 `23.5/23.5/45 N*m`，部署为 `23.7/23.7/35.55 N*m` | 候选采用部署硬上限 `23.7/23.7/35.55`；输出必须先 finite 检查、硬 clip，再按硬件 effort limit clip | S12 回放确认策略输出、PD 输出和部署下发值均不超限 |
| action clip | 当前源码使用不存在的旧字段 `control.action_clip`；mjlab `clip` 位于 scale/offset 后 | 候选按 `default_angle ± scale` 设置 target-space clip，且 `hip_reduction=.5` 使 hip scale 为 `.125`；wrapper/部署边界再对 raw action 做 `[-1,1]` finite/range 检查 | S5/S10/S12 均能捕获 NaN/Inf、越界和饱和动作 |
| 延迟单位 | legacy `[1,4]` 被直接送入 physics actuator 的风险 | 明确 policy-step queue 与 physics-step actuator lag 的转换；play 默认关闭随机延迟与 hold | 用时间戳/脉冲输入验证实际延迟为预期毫秒数 |
| 非法接触 | `terminate_after_contacts_on=[]`，旧 broad pattern 会匹配足端 | 非足端传感器只能列出 `base1/2/3`、四条腿的 `thigh`/`calf1/2`；四个 `*_foot_collision` 只进入足端传感器 | base 接触、危险碰撞和跌倒在一个 policy step 内 `terminated=True`，并停止后续危险动作 |
| 初始姿态 | `base z=0.42` 与默认关节角尚未做落地验证 | reset 后必须测足端离地/穿透、零动作 settle 和第一步冲击；失败时不得打开 push/DR | S2/S4 记录姿态、足端高度、接触力和最大初始 impulse |
| 控制器失效 | wrapper 有有限值检查，部署 FSM 尚无可证明 watchdog/安全停机链路 | 部署边界必须覆盖 NaN/Inf、通信超时、传感器断连和 checkpoint 输出异常 | 故障注入后进入定义好的安全停机状态，并留下可审计日志 |

因此动作顺序、effort 上限、clip、延迟单位和安全停机均属于 **P0 blocker**；只要其中一项没有 round-trip 或故障注入证据，就不能把文档候选实现称为可部署实现。

#### 其他 legacy 配置字段接线矩阵

下面补齐 reward/DR 之外的配置字段。这里的“候选使用”只表示完整候选代码读取了该字段；在 S0–S18 运行证据出现前，仍不能把它标记为已生效。

| 配置路径 | 当前/候选语义 | 状态 |
|---|---|---|
| `env.num_envs`、`env.env_spacing` | 传给 `SceneCfg` | 候选使用，未运行证明 |
| `env.num_one_step_observations` | wrapper 的 45 维契约与 runner metadata | 候选固定为 45；应与字段交叉断言 |
| `env.history_length`、`env.num_observations` | 6 帧由 `HIMRslRlWrapper` 生成 270；不进入 native group history | wrapper 使用；native manager 不读取 legacy 字段 |
| `env.num_privileged_obs` | 当前 scope 的 native critic 目标宽度 235；239 仅是 foot-contact 扩展 | 候选使用；当前源码声明/构造尚未闭环 |
| `env.num_actions` | 12 维期望值 | 仅 metadata/验收；真实值应由 action manager 解析 |
| `env.send_timeouts` | legacy runner timeout 元数据 | 外部 runner 读取与否需验证；不等价于 mjlab `truncated` |
| `env.episode_length_s`、`env.seed` | manager episode length 与随机种子 | 候选使用，未运行证明 |
| `terrain.mesh_type` | `plane` 或 `generator` 分支 | 候选使用 |
| `terrain.horizontal_scale`、`vertical_scale` | height scan/terrain generator 分辨率 | 当前源码未接入；候选显式传入 raycast 和支持 heightfield 的 sub-terrain |
| `terrain.border_size`、`terrain.terrain_length/width` | generator 的边界与尺寸 | 候选使用，未运行证明 |
| `terrain.num_rows`、`num_cols` | generator 网格参数；curriculum 时列数受子地形类型影响 | 候选使用；不能宣称复现 legacy 五比例 |
| `terrain.terrain_proportions` | legacy 五类地形比例 | 当前源码未被 mjlab 自动读取；候选将五个值显式映射为 `sub_terrains` |
| `terrain.static_friction`、`dynamic_friction`、`restitution` | legacy terrain material 参数 | 当前候选未接入；特别是 restitution 没有确认的内置 DR term |
| `terrain.slope_treshold` | legacy trimesh 可行走坡度阈值 | generator/plane 路径未使用；仅 trimesh 专用 |
| `terrain.measure_heights`、`measured_points_x/y` | 是否启用 height scan 与 17×11 网格 | 候选使用；必须断言实际 ray 数为 187 |
| `terrain.selected`、`terrain.terrain_kwargs` | legacy terrain 选择/透传字典 | 当前未使用；不能假设 manager 自动消费 |
| `terrain.max_init_terrain_level` | `TerrainEntityCfg.max_init_terrain_level` | 候选使用，未运行证明 |
| `commands.curriculum`、`max_curriculum` | legacy command curriculum | 当前候选只实现 terrain curriculum，command curriculum 未接入 |
| `commands.num_commands` | legacy 写成 4（含 heading） | 不作为 manager 维度来源；`heading_command=False` 时实际是 3 |
| `commands.resampling_time`、`heading_command`、`ranges.*` | `UniformVelocityCommandCfg` | 候选使用；标量时间必须转为二元组 |
| `init_state.pos`、`default_joint_angles` | entity initial state 与 zero-action pose | 候选使用，但初始姿态/足端高度尚未验收 |
| `control.control_type` | legacy P/V/T 标记 | 当前候选固定 `IdealPdActuatorCfg`；非 P 模式未实现 |
| `control.stiffness`、`damping` | Ideal-PD actuator 参数 | 候选使用，未运行证明 |
| `control.effort_limit` | actuator force limit | 当前源码为 `23.5/23.5/45`，部署为 `23.7/23.7/35.55`；候选按部署值对齐 |
| `control.action_scale`、`hip_reduction`、`decimation` | action scale、hip scale、policy/physics 频率 | 候选使用 `0.25×0.5=0.125` 对齐部署 hip scale；必须做动作与时间 round-trip |
| `asset.file`、`asset.name` | XML 路径与 Scene entity key | 候选使用，未运行证明 |
| `asset.foot_name` | legacy 单一 foot 名称 | 当前 XML 有四个 `FL/FR/RL/RR` site/geom；该字段未直接使用 |
| `asset.penalize_contacts_on` | legacy body 名称筛选 | 候选用显式 contact sensor 替代；字段自身不会自动注册 reward |
| `asset.terminate_after_contacts_on` | legacy termination body 列表 | 当前为空且不满足安全终止；候选必须显式注册 base/危险碰撞/姿态终止 |
| `asset.privileged_contacts_on` | legacy privileged contact 列表 | 候选用四足 contact + 187 height scan 组成 native critic；字段自身未被 manager 自动读取 |
| `asset.self_collisions` | legacy bitwise collision 开关 | 当前实体 collision editor 未由该字段驱动；需显式核对 `CollisionCfg` |
| `rewards.feet_stumble` | legacy reward scale，当前为 `-0.0` | 不会由字段名自动注册 stumble term | 声明存在但权重为零；当前/候选均无活动 term |
| `rewards.soft_dof_vel_limit` | `1.0` | 没有已确认的自动映射；`rewards.scales.dof_vel_limits=-0.0` | **声明但不生效**；须先定义并验证 velocity-limit term，不能仅保留配置字段 |
| `rewards.only_positive_rewards` | legacy 总 reward clamp | mjlab 不自动读取；当前源码未接入，候选在 `Env.step()` 的 reward 汇总后显式 clamp |

结论：当前最容易造成错觉的字段是 `terrain_proportions`、terrain material 参数、`commands.curriculum`、`control.control_type`、`asset.*contacts`、`self_collisions`、`rewards.soft_dof_vel_limit`、`domain_rand.motor_zero_vel_offset_range` 和 `only_positive_rewards`。它们在配置文件里存在，并不意味着 mjlab manager 会自动使用；其中 `feet_stumble=-0.0` 还明确表示当前没有活动奖励项。

### 9.3 官方 mjlab v1.6.0 规则清单

官方 v1.6.0 tag 是提交 `0fb8a68`，发布说明明确包含 MuJoCo/MuJoCo-Warp 3.11、partial reset 修复、灯光/纹理视觉随机化、`GeomCfg`，以及三个迁移性 breaking changes：`CollisionCfg` 的 `contype/conaffinity/condim/priority` 必须显式设置；自定义 command 的 `_update_command` 增加 `env_ids`；`ViewerConfig` 改为 keyword-only。见 [v1.6.0 release](https://github.com/mujocolab/mjlab/releases/tag/v1.6.0)。

实现时按以下顺序检查官方 API：

1. `ManagerBasedRlEnvCfg` 是 flat dataclass；`decimation` 和 `scene` 是核心输入，manager 字典显式挂载。原生 `reset()` 返回 observation dictionary 与 extras，`step()` 返回 `(obs, reward, terminated, truncated, extras)`。
2. `SceneCfg.entities` 的值是 `EntityCfg`；MJCF 可由 `MjSpec.from_file()` 加载，再在 Python 配置 actuator、collision、initial state 和 sensor。
3. `SceneEntityCfg` 的目标选择字段是复数 `actuator_names`、`joint_names`、`body_names`、`geom_names`，并可 `preserve_order=True`；不能使用 `actuator_name`。
4. observation pipeline 是 `compute → noise → clip → scale → delay → history`。actor noisy、critic clean 的 asymmetric 配置应分别建 group；原生 group history 在 HIM 方案中保持 `1`，6 帧由 `HIMRslRlWrapper` 维护。
5. reward term 必须返回 `[N]`；负权重是 penalty。dt 是否缩放必须显式固定，不能把 legacy scale 和 mjlab 默认混用。
6. command 是 class-based `CommandTerm`，`resampling_time_range` 为秒，reset 时必定 resample；标量 `10.0` 必须转换为 `(10.0,10.0)`。
7. events 通过 `EventTermCfg(mode="startup"|"reset"|"interval"|"step", func=...)` 注册；domain randomization 不会因为 legacy `domain_rand` 类存在而自动执行。
8. timeout 应注册为 `TerminationTermCfg(func=..., time_out=True)`，这样它进入 `truncated` 并允许 value bootstrap；跌倒/危险接触进入 `terminated`。
9. RSL-RL task registry 的标准形态是 `register_mjlab_task(task_id, env_cfg, play_env_cfg, rl_cfg, runner_cls)`；当前项目自定义 registry 可以保留，但必须清楚说明它不是官方 registry。

官方参考页：[architecture](https://mujocolab.github.io/mjlab/main/source/architecture_overview.html)、[environment config](https://mujocolab.github.io/mjlab/main/source/environment_config.html)、[observations](https://mujocolab.github.io/mjlab/main/source/observations.html)、[actions](https://mujocolab.github.io/mjlab/main/source/actions.html)、[rewards](https://mujocolab.github.io/mjlab/main/source/rewards.html)、[terminations](https://mujocolab.github.io/mjlab/main/source/terminations.html)、[commands](https://mujocolab.github.io/mjlab/main/source/commands.html)、[events](https://mujocolab.github.io/mjlab/main/source/events.html)、[domain randomization](https://mujocolab.github.io/mjlab/main/source/randomization.html)、[curriculum](https://mujocolab.github.io/mjlab/main/source/curriculum.html)、[RSL-RL](https://mujocolab.github.io/mjlab/main/source/training/rsl_rl.html)。这些页面的当前发布日期与 v1.6.0 tag 对齐到 2026-08-09；具体 breaking change 以 release 页为准。

### 9.3.1 固定证据与可复现登记

以下只登记固定来源与入口；`PASS` 仅表示静态来源/路径已核对，**静态核对≠运行通过**。

| 证据项 | 状态 | 固定登记（HEAD、版本、入口） | 复现边界 |
|---|---|---|---|
| 当前项目 | `PASS` | `/home/kk/legged_mjlab` HEAD `8dbfbd66a57a495cb249b634b259c2d91e6aecfd`（旧值仅作为历史审计基线记录） | 只固定源码锚点，不替代运行验证 |
| `AMP_mjlab` | `PASS` | HEAD `6c7a2947fccc973e4af8e6d90e550400f1b6fcfc`；root setup pin：`mjlab==1.2.0`、`mujoco-warp==3.8.1`、`warp-lang==1.12.0`、`rsl-rl-lib==2.3.1`；注册 `src/tasks/amp_loco/config/g1/__init__.py:1-23`；runner `src/tasks/amp_loco/rl/runner.py:12,65-115` | 参考入口，不是当前 mjlab v1.6.0 运行证据 |
| `unitree_rl_mjlab` | `PASS` | HEAD `1425b15f73bd4095f0df53709d7c389c3eb9e790`；pin：`mjlab==1.2.0`、`mujoco-warp==3.5.0`；Go2 注册 `src/tasks/velocity/config/go2/__init__.py:1-23` | 不是 v1.6 证据，只能作迁移组织参考 |
| `HIMLoco` | `BLOCKED` | 无可验证 `.git` commit；`legged_gym==1.0.0`、本地 `rsl_rl==1.0.2`；训练入口 `legged_gym/legged_gym/scripts/train.py`，play 入口 `legged_gym/legged_gym/scripts/play.py:27-30` | play 在 `:29` 硬编码 checkpoint，来源与结果不可复现，风险未解除 |
| `2027_RC_legged_robot` | `BLOCKED` | HEAD `37c83664418b27f552074d2e697d8ee3e72dd3eb`，工作树 dirty；root 包 `1.0.0`、本地 `rsl_rl==1.0.2`；实际 HIM reward 配置为 `legged_gym/envs/him_go1/him_go1_config.py` | dirty 快照不可作为干净复现基线 |
| 2027 Go1 reward 路径核对 | `FAIL` | 用户给出的 `legged_gym/envs/legged_gym_go1/him_go1_config.py` 不存在；只能使用上方实际路径 | 禁止继续引用不存在的路径 |
| 官方 v1.6.0 release | `PASS` | [固定 release](https://github.com/mujocolab/mjlab/releases/tag/v1.6.0)，对应 commit `0fb8a68` | release 是固定证据 |
| 官方正文与安全入口 | `BLOCKED` | 正文使用 `https://mujocolab.github.io/mjlab/main/...` 滚动页面；补记 [recorders](https://mujocolab.github.io/mjlab/main/source/recorders.html) 与 [NaN guard](https://mujocolab.github.io/mjlab/main/source/debugging/nan_guard.html) | 滚动页不能声称固定到 v1.6.0；`NaN guard`/`nan_detection` 不是硬件 fail-safe；S18 为 `UNSAFE`，仍须完成 S18 |
| 运行验证 | `NOT_RUN` | 本登记未运行 reset/step、训练、play 或部署检查 | 保留现有 9.9、9.10 与 S0-S18 结论；没有 `[ALL_TESTS_PASSED]` |

```python
import ast
from pathlib import Path

documented_groups = {'env', 'terrain', 'commands', 'init_state', 'control', 'asset', 'domain_rand', 'rewards', 'ppo'}
class_to_group = {'HimGo2RoughCfg': 'env', 'HimGo2CfgPPO': 'ppo'}
tree = ast.parse(Path('legged_mjlab/envs/him_go2/him_go2_config.py').read_text())  # source anchor: 8-226
source_groups = set()

def collect_class(node, parent_group=None):
    group = class_to_group.get(node.name)
    if node.name in documented_groups:
        group = node.name
    if group is None:
        group = parent_group
    if group is not None:
        source_groups.add(group)
    for item in node.body:
        if isinstance(item, ast.ClassDef):
            # Nested config classes inherit the top-level/parent group unless
            # their own name is an explicit documented group anchor.
            collect_class(item, group)

for node in tree.body:
    if isinstance(node, ast.ClassDef):
        collect_class(node)
missing_groups = documented_groups - source_groups
print('group anchors:', sorted(source_groups), 'missing groups:', sorted(missing_groups))
# This script only checks group anchors; it cannot reliably count inherited or
# deduplicated leaf fields. Review all 130 leaf fields against the manual 9.2
# table; do not treat this smoke check as a 130/130 runtime result.
```

这是文档覆盖的最小只读 smoke，不证明精确运行时字段语义、manager registration 或 runtime 生效。

### 9.4 四个参考项目的边界

| 项目 | 可复用模式 | 不能直接复制到本项目的部分 |
|---|---|---|
| `AMP_mjlab` | `ManagerBasedRlEnvCfg` 组装、`register_mjlab_task`、actor/critic/amp 三组、外部 AMP runner/ONNX | 该项目 pin `mjlab==1.2.0`；AMP observation 为 13 bodies×15=195，runner 将相邻两帧拼接为 390 维 discriminator 输入（不是 210 维）；`AMPPPO` 和 patch 后的 history ordering 不是 HIM-Go2 的 45/270/235 契约，239 只对应可选 foot-contact 扩展 |
| `unitree_rl_mjlab` | `manager`、`terrain`、`reward` 的组织方式 | 参考仓库 pin `mjlab==1.2.0`、`mujoco-warp==3.5.0`；它不是当前 mjlab v1.6.0 的 API 证据，只能借鉴 manager/terrain/reward 组织；其标准 RSL runner 不是仓库内 `HIMOnPolicyRunner` |
| `HIMLoco` | 6 帧 history、estimator 19 输出、terminal privileged obs、policy action 与 delayed executed action 的区分 | Isaac Gym tensor、七元 step 约定、硬编码 16 DOF/57 obs、C++ SDK/FSM 不能直接放入 mjlab manager |
| `legged_gym_go1` | reward 数值基线是实际存在的 `/home/kk/github/2027_RC_legged_robot/legged_gym/envs/him_go1/him_go1_config.py`（用户指定的 `/home/kk/github/2027_RC_legged_robot/legged_gym/envs/legged_gym_go1/him_go1_config.py` 当前不存在）；普通 `legged_gym_go1_config.py` 仅作结构参考 | 动态 `_reward_<name>` 注册、奖励 dt 语义、domain_rand 字段命名仍属 legacy；Isaac Gym 的 `step()` 五元组、旧 terrain/safety 字段、Gym buffer 和动态属性不能直接作为 mjlab 1.6 term |

关键证据：

- AMP：任务注册 `/home/kk/github/AMP_mjlab/src/tasks/amp_loco/config/g1/__init__.py:10-15`；task runner `/home/kk/github/AMP_mjlab/src/tasks/amp_loco/rl/runner.py:65`；上游基类 `/home/kk/github/AMP_mjlab/rsl_rl/runners/amp_on_policy_runner.py:95`；配置（实际文件名为 `rl_cfg.py`，不是 `rsl_cfg.py`）`/home/kk/github/AMP_mjlab/src/tasks/amp_loco/config/g1/rl_cfg.py:79-94`；loader observation dim `/home/kk/github/AMP_mjlab/rsl_rl/utils/motion_loader.py:148-151`；discriminator 构造 `/home/kk/github/AMP_mjlab/rsl_rl/runners/amp_on_policy_runner.py:174-177`。
- Unitree：pin 证据 `/home/kk/github/unitree_rl_mjlab/setup.py:6-9` 为 `mjlab==1.2.0`、`mujoco-warp==3.5.0`；`/home/kk/github/unitree_rl_mjlab/src/tasks/velocity/velocity_env_cfg.py:36-121,165-387,393-431` 与 Go2 `config/go2/env_cfgs.py:22-162` 仅作为 manager/terrain/reward 组织参考，不是当前 mjlab v1.6.0 API 证据。
- HIM：history 顺序见 `/home/kk/github/HIMLoco/legged_gym/legged_gym/envs/go2w/go2w_legged_robot.py:51-52`；upstream estimator 的切片见 `/home/kk/github/HIMLoco/rsl_rl/rsl_rl/modules/him_estimator.py:76-84`；当前 wrapper 的 history 写入见 `/home/kk/legged_mjlab/legged_mjlab/wrappers/him_wrapper.py:161-180`，当前 estimator 的切片见 `rsl_rl/rsl_rl/modules/him_estimator.py:103-109`，runner 入口见 `rsl_rl/rsl_rl/runners/him_on_policy_runner.py:44-118`。
- Go1：HIM reward 数值基线实际为 `/home/kk/github/2027_RC_legged_robot/legged_gym/envs/him_go1/him_go1_config.py:56-88`（用户指定的 `/home/kk/github/2027_RC_legged_robot/legged_gym/envs/legged_gym_go1/him_go1_config.py` 当前不存在）；普通 `/home/kk/github/2027_RC_legged_robot/legged_gym/envs/legged_gym_go1/legged_gym_go1_config.py:3-48` 仅作结构参考；`legged_gym/envs/base/legged_robot.py:1014-1043,1307` 只作 legacy 运行时结构证据。

### 9.5 HIM 分层 shape 与默认决策

不要在文档或代码中把下列数字写成同一层：

```text
native mjlab actor frame       [N, 45]
HIM wrapper history             [N, 270] = 6 * 45
native mjlab critic frame      [N, 235] = 45 + base_lin_vel(3) + height(187)
optional critic extension      [N, 239] = 235 + foot_contact(4)
local HIM runner critic input  [N, 48]  = 45 + base_lin_vel(3)
HIM estimator output           [N, 19]  = velocity(3) + latent(16)
HIM actor network input        [N, 76]  = one_step_actor(57 in HIMLoco) + velocity(3) + latent(16)
```

最后一行是 HIMLoco 的参考契约，不是当前 Go2 的实际 actor input；当前本地 `HIMActorCritic` 应以 `HIMOnPolicyRunner` 构造参数和 wrapper 的 45/270 为准。保守默认是：**native env/wrapper 保留当前配置声明的完整 235，当前 runner 继续显式取前 48 给 estimator/PPO；若要让 critic 网络学习额外 187 维 height scan 或额外 4 维 foot contact，必须另开 runner/storage/网络接口变更，不能只改 `num_privileged_obs`。**

timeout 的正确时序是：候选底层 env 设置 `auto_reset=False` 并直接保证真实 terminal critic → `step()` 返回该 frame → wrapper 只做断言/兼容并在捕获后 reset done env → 将 terminal critic 和 `truncated` mask 交给 runner → 非 timeout failure 不 bootstrap。mjlab 1.6.0 的 partial reset 修复正是为了避免 commands/history/interval timer 跨环境泄漏；不能先 auto-reset 再让 HIM wrapper 猜测，`termination_privileged_obs` 不能退化为 reset 后 observation。

当前 HEAD 的 `rsl_rl/rsl_rl/modules/him_estimator.py:103-109` 已按本地目标契约使用 `next_critic_obs[..., :45]` 作为 target observation、`next_critic_obs[..., 45:48]` 作为 velocity target，即本地 layout 为 `[45 维 actor, 3 维 velocity]`。upstream HIMLoco 的 `/home/kk/github/HIMLoco/rsl_rl/rsl_rl/modules/him_estimator.py:76-84` 使用的是 `[3 维 base velocity, 45 维 actor]`，所以其 `vel` 仍是 `45:48`，但 `next_obs` 是 `3:48`。这不是矛盾，而是两套 layout 的切片约定不同；两套切片不能直接互换，本地实现不能改写成 upstream 的 `3:48`。仍需在 S10 保留回归断言，防止后续把本地 critic 前 48 维布局改回错误切片。

### 9.6 完整代码实现清单（全部写在本文档，不修改源码）

重要组合约定：第 4 节的 `him_go2_env.py` 环境候选与本节 B 的 `go2_asset.py` Asset 候选是同一套候选，必须整体落地，不能把现有源码和候选片段混用。当前源码的 `Go2Asset(cfg)` 仍不兼容 `play` 参数；只有整体落地候选 B 后，才能使用第 4 节中的 `Go2Asset(cfg, play=play)`，从而兑现 play 模式关闭延迟与随机化的契约。

前文第 4 节的 `him_go2_env.py` 代码块是环境文件的完整候选实现；下面补足两个此前没有完整给出的文件。复制时必须以代码块整体替换目标文件，不能把 Markdown fence 复制到 Python；本轮不执行替换。

#### A. `legged_mjlab/envs/him_go2/him_go2_config.py`

~~~python
"""Legacy-style configuration retained as the single input to the mjlab factory."""

from legged_mjlab.envs.base.legged_mjlab_config import (
    LeggedMjlabCfg,
    LeggedMjlabCfgPPO,
)


class HimGo2RoughCfg(LeggedMjlabCfg):
    class env(LeggedMjlabCfg.env):
        num_envs = 4096
        num_one_step_observations = 45
        history_length = 6
        num_observations = 270
        num_privileged_obs = 235
        num_actions = 12
        env_spacing = 3.0
        send_timeouts = True
        episode_length_s = 20.0
        seed = 42

    class terrain(LeggedMjlabCfg.terrain):
        mesh_type = "generator"
        slope_treshold = 0.75  # legacy 字段仅适用于旧 trimesh 语义；当前 generator 候选不读取，矩阵中已标为未使用。
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25.0
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        measure_heights = True
        measured_points_x = [
            -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        ]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5

    class commands(LeggedMjlabCfg.commands):
        curriculum = True
        max_curriculum = 1.0
        num_commands = 4
        resampling_time = 10.0
        heading_command = False

        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class init_state(LeggedMjlabCfg.init_state):
        pos = (0.0, 0.0, 0.42)
        rot = (1.0, 0.0, 0.0, 0.0)
        default_joint_angles = {
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }

    class control(LeggedMjlabCfg.control):
        control_type = "P"
        stiffness = {"hip": 40.0, "thigh": 40.0, "calf": 40.0}
        damping = {"hip": 1.0, "thigh": 1.0, "calf": 1.0}
        # Match the deployment hard limits: hip/thigh 23.7 N*m, calf 35.55 N*m.
        effort_limit = {"hip": 23.7, "thigh": 23.7, "calf": 35.55}
        action_scale = 0.25
        decimation = 4
        # Deployment config.yaml uses hip_scale=0.125; with action_scale=.25,
        # this explicit reduction keeps simulation and deployment aligned.
        hip_reduction = 0.5

    class asset(LeggedMjlabCfg.asset):
        file = "{LEGGED_MJLAB_ROOT_DIR}/resources/robots/unitree_go2/xmls/go2.xml"
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = []
        privileged_contacts_on = ["base", "thigh", "calf"]
        self_collisions = 1

    class domain_rand(LeggedMjlabCfg.domain_rand):
        randomize_payload_mass = True
        payload_mass_range = [-2.5, 2.5]
        randomize_link_mass = True
        link_mass_range = [0.9, 1.1]
        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]
        randomize_joint_friction = True
        joint_friction_range = [0.01, 1.15]
        randomize_joint_damping = True
        joint_damping_range = [0.3, 1.5]
        randomize_joint_armature = True
        joint_armature_range = [0.0001, 0.05]
        randomize_friction = True
        friction_range = [0.2, 1.3]
        randomize_restitution = True
        restitution_range = [0.0, 0.4]
        push_robots = True
        push_interval_s = 4.0
        max_push_vel_xy = 1.0
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.8, 1.2]
        damping_multiplier_range = [0.8, 1.2]
        randomize_motor_zero_offset = True
        motor_zero_pos_offset_range = [-0.035, 0.035]
        # Velocity encoder bias has no equivalent in the current built-in
        # `dr.encoder_bias` term and is deliberately not silently applied.
        motor_zero_vel_offset_range = [-0.01, 0.01]
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        randomize_obs_motor_latency = True
        range_obs_motor_latency = [1, 4]
        randomize_obs_imu_latency = True
        range_obs_imu_latency = [1, 3]
        randomize_cmd_action_latency = True
        range_cmd_action_latency = [1, 4]
        action_hold_prob = 0.3

    class rewards(LeggedMjlabCfg.rewards):
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -1.5
            ang_vel_xy = -0.05
            orientation = -0.2
            dof_acc = -2.5e-7
            joint_power = -2.0e-5
            base_height = -1.0
            foot_clearance = -0.01
            action_rate = -0.01
            smoothness = -0.01
            feet_air_time = 0.1
            collision = -0.5
            feet_stumble = -0.0
            stand_still = -1.0
            torques = -0.0
            dof_vel = -0.0
            dof_pos_limits = -10.0
            dof_vel_limits = -0.0
            torque_limits = -1.0
            hip_pos = -1.0

        only_positive_rewards = True
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.85
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.30
        max_contact_force = 100.0
        clearance_height_target = -0.2


class HimGo2CfgPPO(LeggedMjlabCfgPPO):
    seed = 42
    runner_class_name = "HIMOnPolicyRunner"

    class policy(LeggedMjlabCfgPPO.policy):
        policy_class_name = "HIMActorCritic"

    class algorithm(LeggedMjlabCfgPPO.algorithm):
        algorithm_class_name = "HIMPPO"
        entropy_coef = 0.01

    class runner(LeggedMjlabCfgPPO.runner):
        policy_class_name = "HIMActorCritic"
        algorithm_class_name = "HIMPPO"
        num_steps_per_env = 100
        max_iterations = 10000
        save_interval = 500
        experiment_name = "him_go2"
~~~

#### B. `legged_mjlab/envs/him_go2/go2_asset.py`

~~~python
"""Go2 MJCF entity factory.  This is a documentation-only replacement candidate."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import mujoco
import torch

from legged_mjlab.utils.paths import PROJECT_ROOT
from mjlab.actuator import ActuatorCmd, IdealPdActuator, IdealPdActuatorCfg
from mjlab.actuator.pd_actuator import pd_torque
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
)
from mjlab.utils.spec_config import CollisionCfg


@dataclass(kw_only=True)
class ScaledIdealPdActuatorCfg(IdealPdActuatorCfg):
    """Ideal-PD actuator with a per-environment motor-strength factor."""

    def build(self, entity, target_ids, target_names):
        return ScaledIdealPdActuator(self, entity, target_ids, target_names)


class ScaledIdealPdActuator(IdealPdActuator):
    """Multiply the unclamped PD torque, then apply the nominal effort limit."""

    def initialize(self, mj_model, model, data, device):
        super().initialize(mj_model, model, data, device)
        assert self.force_limit is not None
        self.motor_strength = torch.ones_like(self.force_limit)

    def reset(self, env_ids=None):
        # Keep per-environment motor strength across partial resets.  The
        # candidate event is registered in ``startup`` mode, matching the
        # other model-field DR terms; resetting this tensor to one here would
        # silently erase the sampled strength after the first episode reset.
        super().reset(env_ids)
        assert hasattr(self, "motor_strength")

    def set_motor_strength(self, env_ids, strength):
        if strength.ndim == 1:
            strength = strength.unsqueeze(-1)
        assert hasattr(self, "motor_strength")
        self.motor_strength[env_ids] = strength

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        assert self.stiffness is not None
        assert self.damping is not None
        assert self.force_limit is not None
        assert hasattr(self, "motor_strength")
        torque = pd_torque(self.stiffness, self.damping, cmd)
        if not bool(torch.isfinite(torque).all().item()):
            raise FloatingPointError("PD torque contains NaN or Inf")
        if not bool(torch.isfinite(self.force_limit).all().item()):
            raise FloatingPointError("force_limit contains NaN or Inf")
        if not bool(torch.isfinite(self.motor_strength).all().item()):
            raise FloatingPointError("motor_strength contains NaN or Inf")
        torque = torque * self.motor_strength
        if not bool(torch.isfinite(torque).all().item()):
            raise FloatingPointError("motor-strength-scaled torque contains NaN or Inf")
        torque = torch.clamp(torque, -self.force_limit, self.force_limit)
        if not bool(torch.isfinite(torque).all().item()):
            raise FloatingPointError("clamped torque contains NaN or Inf")
        return torque


class Go2Asset:
    def __init__(self, cfg, play: bool = False):
        self.cfg = cfg
        self.xml_path = self._resolve_xml_path(cfg.asset.file)
        self.joint_names = tuple(cfg.init_state.default_joint_angles.keys())
        self.effort_limit = cfg.control.effort_limit
        self.stiffness = cfg.control.stiffness
        self.damping = cfg.control.damping
        self.armature = cfg.asset.armature
        self.pos = cfg.init_state.pos
        self.rot = cfg.init_state.rot
        self.default_joint_angles = cfg.init_state.default_joint_angles
        self.action_delay_min = 0
        self.action_delay_max = 0
        self.action_hold_prob = 0.0
        if not play and cfg.domain_rand.randomize_cmd_action_latency:
            # The legacy range is expressed in policy steps; mjlab actuator
            # lag is expressed in physics steps.
            decimation = int(cfg.control.decimation)
            self.action_delay_min = int(
                cfg.domain_rand.range_cmd_action_latency[0] * decimation
            )
            self.action_delay_max = int(
                cfg.domain_rand.range_cmd_action_latency[1] * decimation
            )
            self.action_hold_prob = float(cfg.domain_rand.action_hold_prob)
        self.entity = self._EntityFactory(self)
        self.sensor = self._SensorFactory(self)

    @staticmethod
    def _resolve_xml_path(raw_path):
        path = (
            str(raw_path)
            .replace("{LEGGED_MJLAB_ROOT_DIR}", str(PROJECT_ROOT))
            .replace("{PROJECT_ROOT}", str(PROJECT_ROOT))
        )
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Go2 MJCF file does not exist: {resolved}")
        return resolved

    @staticmethod
    def _value(mapping_or_scalar, key):
        if isinstance(mapping_or_scalar, Mapping):
            return float(mapping_or_scalar[key])
        return float(mapping_or_scalar)

    class _EntityFactory:
        def __init__(self, asset):
            self.asset = asset

        def get_spec(self):
            return mujoco.MjSpec.from_file(str(self.asset.xml_path))

        def actuators(self):
            a = self.asset
            delay = {
                "delay_min_lag": a.action_delay_min,
                "delay_max_lag": a.action_delay_max,
                "delay_hold_prob": a.action_hold_prob,
                "delay_update_period": 1,
            }
            return (
                ScaledIdealPdActuatorCfg(
                    target_names_expr=(r".*_hip_joint",),
                    stiffness=self._value(a.stiffness, "hip"),
                    damping=self._value(a.damping, "hip"),
                    effort_limit=self._value(a.effort_limit, "hip"),
                    armature=a.armature,
                    **delay,
                ),
                ScaledIdealPdActuatorCfg(
                    target_names_expr=(r".*_thigh_joint",),
                    stiffness=self._value(a.stiffness, "thigh"),
                    damping=self._value(a.damping, "thigh"),
                    effort_limit=self._value(a.effort_limit, "thigh"),
                    armature=a.armature,
                    **delay,
                ),
                ScaledIdealPdActuatorCfg(
                    target_names_expr=(r".*_calf_joint",),
                    stiffness=self._value(a.stiffness, "calf"),
                    damping=self._value(a.damping, "calf"),
                    effort_limit=self._value(a.effort_limit, "calf"),
                    armature=a.armature,
                    **delay,
                ),
            )

        def get_robot_cfg(self):
            a = self.asset
            foot_geoms = r"^(FL|FR|RL|RR)_foot_collision$"
            collision = CollisionCfg(
                geom_names_expr=(r".*_collision",),
                contype=1,
                conaffinity=0,
                condim={foot_geoms: 3, r".*": 1},
                priority={foot_geoms: 1, r".*": 0},
            )
            initial = EntityCfg.InitialStateCfg(
                pos=a.pos,
                rot=a.rot,
                joint_pos=a.default_joint_angles,
                joint_vel={r".*": 0.0},
            )
            articulation = EntityArticulationInfoCfg(
                actuators=self.actuators(),
                soft_joint_pos_limit_factor=float(a.cfg.rewards.soft_dof_pos_limit),
            )
            return EntityCfg(
                init_state=initial,
                collisions=(collision,),
                spec_fn=self.get_spec,
                articulation=articulation,
            )

        @staticmethod
        def _value(mapping_or_scalar, key):
            return Go2Asset._value(mapping_or_scalar, key)

    class _SensorFactory:
        def __init__(self, asset):
            self.asset = asset

        def foot_contact(self, entity_name):
            return ContactSensorCfg(
                name="feet_ground_contact",
                primary=ContactMatch(
                    mode="geom",
                    pattern=(
                        "FL_foot_collision", "RL_foot_collision",
                        "FR_foot_collision", "RR_foot_collision",
                    ),
                    entity=entity_name,
                ),
                secondary=ContactMatch(mode="body", pattern="terrain"),
                fields=("found", "force"),
                reduce="netforce",
                num_slots=1,
                track_air_time=True,
            )

        def illegal_contact(self, entity_name):
            return ContactSensorCfg(
                name="nonfoot_ground_touch",
                primary=ContactMatch(
                    mode="geom",
                    pattern=(
                        "base1_collision",
                        "base2_collision",
                        "base3_collision",
                        "FL_thigh_collision",
                        "FR_thigh_collision",
                        "RL_thigh_collision",
                        "RR_thigh_collision",
                        "FL_calf1_collision",
                        "FL_calf2_collision",
                        "FR_calf1_collision",
                        "FR_calf2_collision",
                        "RL_calf1_collision",
                        "RL_calf2_collision",
                        "RR_calf1_collision",
                        "RR_calf2_collision",
                    ),
                    entity=entity_name,
                ),
                secondary=ContactMatch(mode="body", pattern="terrain"),
                fields=("found", "force"),
                reduce="maxforce",
                num_slots=1,
                history_length=4,
            )

        def height_scan(self, entity_name, debug_vis=False):
            cfg = self.asset.cfg
            return RayCastSensorCfg(
                name="height_scan",
                frame=ObjRef(
                    type="body", name="base_link", entity=entity_name
                ),
                pattern=GridPatternCfg(
                    size=(
                        max(cfg.terrain.measured_points_x)
                        - min(cfg.terrain.measured_points_x),
                        max(cfg.terrain.measured_points_y)
                        - min(cfg.terrain.measured_points_y),
                    ),
                    resolution=float(cfg.terrain.horizontal_scale),
                ),
                ray_alignment="yaw",
                max_distance=2.0,
                debug_vis=bool(debug_vis),
            )
~~~

上述两个代码块仍然需要按第 10 节验证矩阵逐项验证，尤其是 `ScaledIdealPdActuator` 的非 fused 执行路径、delay 字段、`ContactMatch` pattern 解析、`CollisionCfg` dict pattern 和 `RayCastSensorCfg` 采样点数量；文档中的“完整”表示不再用 `pass`、省略号或未定义属性隐藏实现，不表示未经运行就已经通过 mjlab 构造。

#### C. `legged_mjlab/scripts/play.py`

~~~python
"""Play a checkpoint with the configured HIM Go2 environment."""

import argparse
from pathlib import Path

import torch


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="him_go2")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-envs", type=_positive_int, default=1)
    parser.add_argument("--steps", type=_positive_int, default=1000)
    parser.add_argument("--log-dir", default="logs/play")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    # Importing the task registry also registers the environment.  Loading the
    # project backend must happen before constructing the runner.
    import legged_mjlab.envs  # noqa: F401
    from legged_mjlab.utils.task_registry import load_project_rsl, task_registry

    load_project_rsl()
    spec = task_registry.get(args.task)
    env, _ = task_registry.make_env(
        args.task,
        device=args.device,
        play=True,
        num_envs=args.num_envs,
    )

    try:
        train_cfg = spec.train_cfg_cls().to_dict()
        runner = task_registry.make_alg_runner(
            args.task,
            env,
            train_cfg,
            args.log_dir,
        )
        runner.load(str(args.checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=args.device)

        history, _ = env.reset()
        expected = (args.num_envs, 270)
        if tuple(history.shape) != expected:
            raise RuntimeError(
                f"HIM history shape mismatch: expected {expected}, "
                f"got {tuple(history.shape)}"
            )

        for _ in range(args.steps):
            with torch.inference_mode():
                actions = policy(history)
            if not bool(torch.isfinite(actions).all().item()):
                raise FloatingPointError("inference produced NaN or Inf actions")
            # Keep the normalized action boundary for this candidate script.
            # This clamp is not a hardware fail-safe; deployment must reject
            # non-finite values, suppress output, and supervise communications.
            actions = torch.clamp(actions, -1.0, 1.0)
            result = env.step(actions)
            if not isinstance(result, tuple) or len(result) != 7:
                raise RuntimeError(
                    "HIM wrapper must return "
                    "(history, critic, reward, done, infos, ids, terminal_critic)"
                )
            history, _, _, dones, _, _, _ = result
            if not bool(torch.isfinite(history).all().item()):
                raise FloatingPointError("environment returned NaN or Inf history")
            if bool(torch.as_tensor(dones).all().item()):
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
~~~

#### D. `legged_mjlab/test/test_env.py`

~~~python
"""Dependency-light HIM Go2 CPU smoke test.

This deliberately uses assertions rather than pytest so that it can run in the
minimal project environment.  It is a post-migration test, not a claim that the
current unmodified source already passes.
"""

import torch


def test_him_go2_cpu_smoke():
    import legged_mjlab.envs  # noqa: F401
    from legged_mjlab.utils.task_registry import load_project_rsl, task_registry

    load_project_rsl()
    env, _ = task_registry.make_env(
        "him_go2",
        device="cpu",
        play=True,
        num_envs=4,
    )
    try:
        history, critic = env.reset()
        assert tuple(history.shape) == (4, 270), history.shape
        assert tuple(critic.shape) == (4, 235), critic.shape
        assert bool(torch.isfinite(history).all())
        assert bool(torch.isfinite(critic).all())

        result = env.step(torch.zeros((4, 12), device=history.device))
        assert isinstance(result, tuple) and len(result) == 7
        history, critic, reward, done, infos, done_ids, terminal_critic = result
        assert tuple(history.shape) == (4, 270), history.shape
        assert tuple(critic.shape) == (4, 235), critic.shape
        assert tuple(reward.shape) == (4,), reward.shape
        assert tuple(done.shape) == (4,), done.shape
        assert done_ids.ndim == 1
        assert isinstance(infos, dict)
        if terminal_critic is not None:
            assert terminal_critic.shape[-1] == 235
    finally:
        env.close()


if __name__ == "__main__":
    test_him_go2_cpu_smoke()
~~~

#### E. `legged_mjlab/utils/exporter.py`

~~~python
"""TorchScript exporter for the local HIM actor and estimator."""

import argparse
from pathlib import Path

import torch
from torch import nn


class HimInferenceModule(nn.Module):
    """Expose only normalized history -> normalized action inference."""

    def __init__(self, actor_critic: nn.Module):
        super().__init__()
        self.actor_critic = actor_critic.eval()

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        raw_actions = self.actor_critic.act_inference(history)
        finite = torch.isfinite(raw_actions)
        clamped_actions = torch.clamp(raw_actions, -1.0, 1.0)
        # Preserve NaN/Inf so export checks and deployment boundaries can
        # reject them; clamping must not turn an invalid action into a valid one.
        return torch.where(finite, clamped_actions, raw_actions)


def export_him_policy(actor_critic, sample_history: torch.Tensor, output: Path):
    if tuple(sample_history.shape[1:]) != (270,):
        raise ValueError(
            f"expected sample history [N,270], got {tuple(sample_history.shape)}"
        )
    if not bool(torch.isfinite(sample_history).all().item()):
        raise FloatingPointError("sample history contains NaN or Inf")

    module = HimInferenceModule(actor_critic).to(sample_history.device).eval()
    with torch.inference_mode():
        sample_action = module(sample_history)
    if not bool(torch.isfinite(sample_action).all().item()):
        raise FloatingPointError("policy produced NaN or Inf during export check")

    output.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.trace(module, sample_history, check_trace=True)
    scripted.save(str(output))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="him_go2")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    import legged_mjlab.envs  # noqa: F401
    from legged_mjlab.utils.task_registry import load_project_rsl, task_registry

    load_project_rsl()
    spec = task_registry.get(args.task)
    env, _ = task_registry.make_env(
        args.task,
        device=args.device,
        play=True,
        num_envs=1,
    )
    try:
        runner = task_registry.make_alg_runner(
            args.task,
            env,
            spec.train_cfg_cls().to_dict(),
            str(args.output.parent),
        )
        runner.load(str(args.checkpoint), load_optimizer=False)
        history, _ = env.reset()
        export_him_policy(
            runner.alg.actor_critic,
            history[:1],
            args.output,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
~~~

这三个代码块仍是文档候选实现：`play.py` 依赖 checkpoint，`test_env.py` 依赖环境先通过 S0–S9。`exporter.py` 的 `HimInferenceModule` 先计算 `torch.isfinite(raw_actions)`，再由 `torch.where(finite, clamped_actions, raw_actions)` 只对有限元素采用 `[-1,1]` clamp；NaN/Inf 元素沿 `raw_actions` 分支原样传播。因此 `export_him_policy()` 的 export-time finite check 会拒绝样本输出中的非有限值；若导出模块在运行时产生 NaN/Inf，下游 deployment adapter 仍必须 reject 并抑制输出，exporter 本身不会把它们变成安全动作。

该 exporter 只导出归一化 `[N,270] -> [N,12]` policy action，不负责硬件关节重排、PD scale、effort clip、通信 watchdog 或安全停机；clamp 和 export-time 静态检查都不是硬件 fail-safe。导出文件只有在 S12、S13、S14、S18 均有证据后才能交给部署链路。

### 9.7 现有完整环境代码块的必要修正

前文第 4 节代码在手工落地前必须再修正以下点：

- `_build_terrain()` 的 generator 必须使用非空的 `ROUGH_TERRAINS_CFG.sub_terrains`；不能把 `terrain_proportions` 写在 legacy 配置里就期待 mjlab 自动读取。
- `ContactSensorCfg` 的 `nonfoot_ground_touch` 必须排除四个足端 geom，否则 `collision` 会重复惩罚正常触地；候选代码已改为只列出 `base1/2/3`、四腿 `thigh`、四腿 `calf1/2` 的显式 XML 名称，不再依赖 broad pattern 或可选 `exclude` 字段。
- 第 4 节完整候选的 `_build_terminations()` 必须保留 `time_out`、`bad_orientation`、`base_contact`，并新增 `nonfoot_contact`，复用 `illegal_contact`，传入 `params={"sensor_name": "nonfoot_ground_touch", "force_threshold": 10.0}`；否则 thigh/calf/base 接触仍然只会进入 `collision` reward，不会在同一 policy step 终止。由于 `nonfoot_ground_touch` 的显式列表也包含 `base1/2/3`，`base_contact` 与 `nonfoot_contact` 可能同时命中；保留这一重叠是安全冗余，验证只断言最终的 `terminated`/`truncated`，不引入未经确认的去重 API。
- `randomize_restitution` 不应调用未在 v1.6.0 `dr` 表中确认存在的函数；先标为未接入或写有测试的自定义 adapter。
- `num_privileged_obs=235` 是当前 scope 的 native/wrapper 完整观测宽度；当前 `HIMOnPolicyRunner` storage/network 仍是 48，shape 测试必须同时断言 `(235,48)` 两层，不得只断言 235。若另行启用 `foot_contact` critic 扩展，再把这一组同步升级为 `(239,48)`。
- 候选 `ManagerBasedRlEnvCfg.auto_reset` 必须显式设置为 `False`。底层先返回/保留真实 terminal frame，`HIMRslRlWrapper` 只能断言 frame/shape/索引并做兼容适配，确认捕获后才 reset done env；wrapper 不可依赖底层先 auto-reset，也不可从 reset 后 observation 猜测 terminal frame。若 `auto_reset=False` 后底层不提供真实 terminal frame，应停止训练。
- 当前源码 `HimGo2Env.__init__` 的参数名仍与 registry 不匹配；第 4 节候选实现已统一为 `device=None, render_mode=None`，但本轮不改源码，因此落地后仍需重新跑 S1/S2。
- `go2.xml` 的 runtime actuator 数量为 0，12 个 actuator 由 `IdealPdActuatorCfg` 补建；不要把 `scene_go2.xml` 的 12 个 motor 与同一套 IdealPd actuator 同时使用。
- 当前 `go2.xml` 的有效足端 site/geom 名称是 `FL/FR/RL/RR` 与 `FL_foot_collision/FR_foot_collision/RL_foot_collision/RR_foot_collision`；`foot_name="foot"` 只是 legacy 字段，并没有对应单一实体。
- 当前配置的 `default_joint_angles` 是按关节类型分组的插入顺序（每组为 FL、RL、FR、RR），部署数组是按腿交错的 FL、RL、FR、RR，而 XML runtime 是按腿交错的 FL、FR、RL、RR；候选代码必须通过 `_joint_names()` 与四个分腿 action term 固定策略顺序，并记录到硬件重排表。
- `randomize_motor_strength` 若要保持 HIMLoco/legged_gym 语义，必须在 torque 输出后乘 per-env strength factor 再做 effort clip；候选代码已在文档 B 中提供 `ScaledIdealPdActuator` 与 custom event，`dr.effort_limits` 仅随机化饱和上限，不是同一语义。
- 文档 B 的 `ScaledIdealPdActuator` 候选已写入 finite guard：PD torque、`force_limit`、`motor_strength`、缩放后 torque 和 clamp 后 torque 均检查 NaN/Inf；这些只是候选代码的静态保护/局部检查，不能替代 runtime S14，也不构成部署硬件 fail-safe。
- 文档 E 的 exporter 候选通过 `torch.where` 对有限 action 做 clamp、对 NaN/Inf 沿 `raw_actions` 原值传播，并在 export-time 检查中拒绝非有限样本；这是静态候选保护，不能替代部署 adapter 的 runtime finite reject、输出抑制、watchdog 或安全停机。
- push event 必须显式传 `asset_cfg=SceneEntityCfg(entity_name)`；mjlab 的默认实体名是 `robot`，而本任务实体名是 `go2`。候选代码已补齐该参数。
- `play=True` 必须同时关闭 startup DR、push、observation delay、actuator delay、action hold 和随机 reset 位姿；候选代码通过 fixed reset ranges、early return 与 `Go2Asset(cfg, play=play)` 共同保证，仍需 S17 运行验证。
- `randomize_restitution` 当前没有已确认的 mjlab 1.6.0 内置 `dr` term；不得把 friction、solref 或 solver 参数当作 restitution 的等价实现。

### 9.8 本轮证据边界

下面区分“文档候选的局部验证”和“当前项目的真实运行验证”，两者不能互换：

| 对象 | 已观察证据 | 不能推出的结论 |
|---|---|---|
| Markdown Python 代码块 | 11 个 Python fence 均可由 AST 解析 | 不等于项目源码已经替换或可运行 |
| 文档 B 的 Asset 候选 | 在内存中用当前 `go2.xml` 构造出 12 joints、12 actuators，三个 actuator group 均为 `ScaledIdealPdActuator` | 不等于完整 `ManagerBasedRlEnv` reset/step 已通过 |
| 文档环境候选 | synthetic cfg 的 `ManagerBasedRlEnvCfg` 构造成功；play 模式 events 只有固定 reset；17 个非零 reward term、四个 action key、hip scale `0.125`、clean critic 和五类 terrain 映射符合预期 | 不等于当前 registry、当前配置类和当前源码入口已通过 |
| 文档候选 reward/termination terms | 用纯 Tensor 假对象调用后，已覆盖的 term 均返回有限 `[N]` 输出 | 不等于真实接触传感器、history、manager 调度和奖励数值语义已通过 |
| 文档环境候选的 `auto_reset`/wrapper 终止帧 | 静态检查确认 `ManagerBasedRlEnvCfg.auto_reset=False`；wrapper 只断言/兼容真实 terminal frame，捕获后才 reset done env | 不等于底层 runtime terminal frame、partial reset 和 timeout bootstrap 已验证；不能依赖先 auto-reset，也不能替代 runtime S14/S17/S18 |
| 文档 B 的 custom actuator finite guard | 静态检查确认 PD torque、`force_limit`、`motor_strength`、缩放后 torque 和 clamp 后 torque 均有 finite guard；最小测试确认执行顺序为 `PD torque × motor_strength → effort clip` | 只是静态候选保护；不等于每个 physics substep 或 deployment adapter 的 finite reject、输出抑制和安全停机已验证，不能替代 runtime S14/S17/S18 |
| 文档 E 的 exporter finite/传播边界 | 静态检查确认有限 action 才走 clamp，NaN/Inf 经 `torch.where` 沿 `raw_actions` 原值传播，export-time check 会拒绝非有限样本 | 只是静态候选保护；不等于导出运行时或部署故障注入已验证，不能替代 runtime S14/S17/S18 |
| 1-env CPU 尝试 | 把 Warp kernel cache 指向 `/tmp` 后进入 CPU kernel 编译；当前环境报告无 CUDA driver | 没有得到 reset/step 的 `[N,45]`/`[N,235]` 运行断言，S4/S5 仍未通过 |
| 现有 CPU shape smoke test（第 6.3 节） | 只执行 `reset()` 与 `step(zeros)` 的 shape/返回值检查，未注入接触 | 不能证明 `nonfoot_ground_touch` 的 thigh/calf/base 终止路径；后续必须分别注入 thigh、calf、base 接触并断言 `terminated=True`、`truncated=False`，另行触发 timeout 并断言 `terminated=False`、`truncated=True` |
| 当前工作树 | `git diff --name-only -- '*.py'` 当前包含 `legged_mjlab/envs/him_go2/him_go2_env.py`；`him_go2_env.py:101` 仍是 `metrics = ,` | 不能 import 当前 `him_go2`，也不能触发 `[ALL_TESTS_PASSED]` |

`auto_reset=False`、custom actuator finite guard 和 exporter 的 `torch.where`/export-time finite check 目前都只有静态候选保护证据；它们不能替代 runtime S14、S17、S18，也不能把 clamp 或静态检查解释为硬件 fail-safe。

因此本轮新增的是文档候选的局部构造证据和更精确的阻断记录，不是项目运行绿灯。

## 9.9 通用参考项目迁移契约

本节只定义可复用边界，不把参考项目代码或版本当作当前实现证据；最终接口以本节契约为准。

| 真实路径 | 唯一可复用边界 |
|---|---|
| `/home/kk/github/AMP_mjlab` | mjlab 1.2.0、AMP group/外部 runner；仅借鉴组织方式，不直接搬 manager API。 |
| `/home/kk/github/unitree_rl_mjlab` | 该参考仓库明确 pinned `mjlab==1.2.0`、`mujoco-warp==3.5.0`；只借鉴 mjlab manager、Go2 terrain/contact/reward 语义，仍须按当前版本逐项核验。这两个 pin 属于参考仓库，不是当前 mjlab v1.6.0 的版本或 API 证据。 |
| `/home/kk/github/HIMLoco` | HIM estimator/history/action/deployment；不直接提供 mjlab manager。 |
| `/home/kk/github/2027_RC_legged_robot/legged_gym/envs/legged_gym_go1` | legacy config/reward dt/order；不直接搬 manager API。 |

**唯一目标契约：**

| 边界 | 固定契约 |
|---|---|
| 迁移形态 | `legacy cfg → builder`，字段映射必须可追溯。 |
| shape | action 12；actor 45；native critic 235；wrapper history 270；runner estimator 48。 |
| 数值 | `torch.float32`，`device=env.device`，单位为 `m/rad/m/s/N·m`。 |
| 顺序 | `_joint_names()` 返回 `FL,RL,FR,RR × hip/thigh/calf`，再由四个分腿 action term 固定策略动作顺序；禁止依赖 dict 或 XML 隐式顺序。 |
| 生命周期 | 明确区分 `reset`、`terminated`、`truncated`，并提供 terminal critic。 |

契约状态：当前源码的 action 顺序仍受按关节类型分组的顺序影响，且 `_build_actions` 仍是旧的 `actuator_name` 接口，尚未满足上述 canonical order 契约；当前状态为 `BLOCKED`。原因同时包括源码契约未满足，以及 S13 action round-trip 当前为 `NOT_RUN`。只有文档候选整体落地并跑完 S13 action round-trip，才允许将该项标记为 `PASS`。本次只修改本文档，不修改源码。

当前 `num_privileged_obs=235` 是本轮不变 scope 的 native/wrapper 目标，但当前源码尚未构造出该 shape。candidate `239` 只能作为另行启用 `foot_contact` 的扩展目标；235 与 239 不可混淆，任一数字都不能反写成当前运行已通过。

#### 参考 reward 语义差异

| reward | 候选选择 | 参考来源 | 状态 |
|---|---|---|---|
| `only_positive_rewards` | `super().step()` 后对汇总 reward 执行 `clamp_min(0)` | legacy `/home/kk/github/2027_RC_legged_robot/legged_gym/envs/base/legged_robot.py:1299-1305` 先 clamp、再追加 termination reward；该顺序需 runtime 决策 | `NOT_RUN` |
| `feet_air_time` | 沿候选/Unitree：threshold `0.4`，进入 two-contact mode（恰好两足接触）后取四足 mode time 的 `min` | Unitree `/home/kk/github/unitree_rl_mjlab/src/tasks/velocity/mdp/rewards.py:134-159`；legacy Go1 `/home/kk/github/2027_RC_legged_robot/legged_gym/envs/base/him_legged_robot.py:1555-1566` 为 `0.5`、first-contact 后按 per-foot 累加 | `NOT_RUN` |
| `collision` | candidate contact sensor `history_length=4`；`force_history` 的 H=4 是 physics-history 槽计数而非 contact total，先按接触项聚合，再按历史槽计数；`max_contact_force=100` 仅为候选策略值 | candidate 本文第4节唯一完整候选的 `HimGo2Env._reward_collision` 实现（见第4节）；legacy Go1 `/home/kk/github/2027_RC_legged_robot/legged_gym/envs/base/him_legged_robot.py:1532-1534` 使用当前接触力且阈值 `>0.1`；Unitree `/home/kk/github/unitree_rl_mjlab/src/tasks/velocity/mdp/rewards.py:87-106` 的默认 `self_collision` 阈值为 `10` | `NOT_RUN` |
| `stand_still` | 关节误差的平方 L2 范数/平方和，并含 `xy` 速度与 yaw 速度的 gate | legacy Go1 `/home/kk/github/2027_RC_legged_robot/legged_gym/envs/base/him_legged_robot.py:1573-1575` 为 L1，且只使用 `xy` gate | `NOT_RUN` |
| `hip_pos` | 相对 default pose 的平方 L2 范数/平方和，无 command gate | legacy Go1 `/home/kk/github/2027_RC_legged_robot/legged_gym/envs/base/him_legged_robot.py:1581-1583` 为 zero-target，并带 command gate | `NOT_RUN` |
| `foot_clearance` | candidate 使用 world/site height（`site_pos_w[..., 2]`）、world/site 水平速度（`site_lin_vel_w[..., :2]`）、高度绝对误差，并带 command gate | candidate 本文第4节唯一完整候选的 `HimGo2Env._reward_foot_clearance` 实现（见第4节）；Unitree `/home/kk/github/unitree_rl_mjlab/src/tasks/velocity/mdp/rewards.py:163-185`；legacy HIM Go1 `/home/kk/github/2027_RC_legged_robot/legged_gym/envs/base/him_legged_robot.py:1503-1514` 使用 body-frame、relative-to-root 的高度/速度、平方高度误差且无 gate | `NOT_RUN` |
| `dof_acc` | candidate 注册类内 `HimGo2Env._reward_dof_acc`，其实现读取 mjlab physics `joint_acc` | candidate 本文第4节唯一完整候选的 `HimGo2Env._reward_dof_acc` 实现（见第4节）；legacy HIM Go1 `/home/kk/github/2027_RC_legged_robot/legged_gym/envs/base/him_legged_robot.py:128,1490-1492` 用相邻 policy-step 的 velocity difference 除以 `self.dt`，时间离散不同 | `NOT_RUN` |
| `torque_limits` | candidate 对归一化超限量取平方后求和 | candidate 本文第4节唯一完整候选的 `HimGo2Env._reward_torque_limits` 实现（见第4节）；legacy HIM Go1 `/home/kk/github/2027_RC_legged_robot/legged_gym/envs/base/him_legged_robot.py:1551-1553` 对原始 torque 超限量作线性和 | `NOT_RUN` |

状态枚举仅允许 `PASS/FAIL/BLOCKED/NOT_RUN/UNSAFE`。

- `PASS` 仅表示该项证据已完成且满足契约。
- `FAIL`、`BLOCKED`、`NOT_RUN`、`UNSAFE` 都不是放行状态。
- 若任一上游依赖失败，所有下游依赖项禁止标记为 `PASS`，只能使用单值非 `PASS` 状态，具体值按依赖证据填写；修复根因后必须重新运行整条依赖链。

迁移 checklist（8 项）：

1. **版本与 commit**：记录四个参考路径的实际存在性、版本、commit，以及当前工作树 commit。
2. **字段映射**：逐项核对 legacy cfg → builder，含 dt、scale、命令、terrain、DR 和 timeout 语义。
3. **asset**：核对 XML、12 joints/actuators、sensor/site/geom 名称、effort limit，以及由 `_joint_names()` 与四个分腿 action term 固定的策略顺序。
4. **manager**：构造 scene、action、observation、critic、reward、event、termination manager，并记录字段来源。
5. **CPU reset/step**：在 CPU 上验证 reset、zero-action step、partial reset 及 45/235/270/48 shape 和 finite；若另行启用 foot-contact critic 扩展，再同步验证 239。
6. **reward/DR**：核对 reward dt 缩放、非零项、接触与 terrain 语义，以及 play 关闭 DR/push/delay。
7. **action/runner**：验证 12 维动作 round-trip、力矩/clip/顺序、HIM history、48 维 estimator 与 terminal critic。
8. **play/export/fail-safe**：验证 play、export、NaN/Inf、越界、通信超时、断连和安全停机；未完成不得部署。

P1 安装入口矛盾必须单独记录：根 `setup.py`、`docs/uv使用.md` 的 `uv sync`，但根目录无 `pyproject.toml`/lock；在入口统一前不得把安装成功当作环境证据。

每次验证必须记录 Python 版本、Torch 版本、CUDA 版本、driver 状态，以及 `mjlab.__file__`、`mujoco_warp.__file__`、`rsl_rl.__file__`；缺一项就不能把环境标为 `PASS`。

### 9.10 当前工作树启动/部署安全审计

以下审计以当前工作树为准；状态字段只表示当前证据，不把文档候选实现、静态检查或未运行步骤写成已实现/已放行。

| 审计项 | 状态 | 当前事实 | 证据（真实相对路径:行号） |
|---|---|---|---|
| 安装入口与依赖来源 | `FAIL` | 根 `setup.py` pin 了 `mjlab==1.6.0`、`mujoco-warp==3.11.0`，同时把仓库内 `rsl_rl` 纳入根包；独立 `rsl_rl/setup.py` 也分发同名 `rsl_rl`。`docs/setup.md` 连续安装 `./rsl_rl` 与 `.`。`docs/uv使用.md` 要求根 `pyproject.toml`/`uv sync`，但当前根目录没有 `pyproject.toml`、`uv.lock` 或 `requirements/`，因此安装入口不可复现。 | `setup.py:7-26`；`rsl_rl/setup.py:3-15`；`docs/setup.md:23-32`；`docs/uv使用.md:79-100`；`pyproject.toml`、`uv.lock`、`requirements/`：不存在（无文件行号） |
| task registry 与环境构造 | `FAIL` | registry 传 `device`，当前 `HimGo2Env` 构造仍接收 `sim_device`，且 `render_mode` 无默认值；按当前调用会在进入有效环境前失败。 | `legged_mjlab/utils/task_registry.py:201-212`；`legged_mjlab/envs/him_go2/him_go2_env.py:41-60` |
| 当前源码可解析性 | `FAIL` | `ManagerBasedRlEnvCfg` 构造中的 `:101 metrics = ,` 为语法错误；不能据此宣称 import、reset 或 step 已通过。 | `legged_mjlab/envs/him_go2/him_go2_env.py:71-102`，错误在 `:101 metrics = ,` |
| play/export 入口 | `FAIL` | `play.py` 为空文件（0 行、0 bytes）；`exporter.py` 仅含 1 个空行（1 byte），当前没有可执行回放或导出入口。 | `legged_mjlab/scripts/play.py:0（空文件）`；`legged_mjlab/utils/exporter.py:1（空行）` |
| 运行环境与 CPU gate | `BLOCKED` | `nvidia-smi` 当前报告无法与 NVIDIA driver 通信；CPU `reset()`/`step()` 本轮未运行，不能形成 shape/finite 证据。 | `nvidia-smi` 当前输出；审计命令 `nvidia-smi` 当前无法与 NVIDIA driver 通信（命令输出，无文件行号） |
| 配置声明与实际实现 | `FAIL` | 配置打开了 motor-strength、观测/动作 delay 等随机化，并声明了 termination/reward 项；源码对 payload 随机化仍为 `pass`，资产只创建普通 `IdealPdActuatorCfg`，终止管理器实际只返回 timeout，未落地声明的接触/故障终止。 | `legged_mjlab/envs/him_go2/him_go2_config.py:151-168,174-205`；payload `legged_mjlab/envs/him_go2/him_go2_env.py:283-284`；timeout-only termination `legged_mjlab/envs/him_go2/him_go2_env.py:454-459`；`legged_mjlab/envs/him_go2/go2_asset.py:127-164` |
| play 隔离 | `BLOCKED` | `play` 虽进入环境构造，但 manager builder 调用 `_build_events()`、`_build_observations()`、`_build_curriculum()` 时未传 `play`，与这些函数的签名不一致；不能证明 play 已关闭 DR、push、noise、delay、hold 和 curriculum。 | 调用点 `legged_mjlab/envs/him_go2/him_go2_env.py:69-95`；函数定义 `_build_events:228`、`_build_observations:410`、`_build_curriculum:461` |
| wrapper finite check 的安全边界 | `UNSAFE` | wrapper 只在收到动作后以 `FloatingPointError` 拒绝 NaN/Inf；这不是硬件 fail-safe，也没有证明输出抑制、通信监督或安全停机。 | `legged_mjlab/wrappers/him_wrapper.py:210-228,388-397` |
| 仿真/部署 effort 一致性 | `FAIL` | 仿真配置 calf effort 为 `45 N·m`，部署 YAML 的 calf torque limit 为 `35.55 N·m`，两者不一致。 | `legged_mjlab/envs/him_go2/him_go2_config.py:77-84`；`deploy/deploy_mujoco/him_go2/policy/config.yaml:31-34` |
| 部署 fail-safe 基础设施 | `UNSAFE` | 当前没有 communication watchdog、sensor-disconnect handling、output suppression、safe-stop 或可审计故障记录；没有部署放行证据。 | `deploy/deploy_mujoco/him_go2/src/library/fsm/fsm.py:1-164`（仅为通用 FSM，未提供上述链路）；`deploy/deploy_mujoco/him_go2/policy/config.yaml:1-54`；全树关键字核验无对应实现 |
| 文档候选代码边界 | `NOT_RUN` | 第 4 节候选环境、full exporter、custom actuator 和 contact termination 只是待落地方案；候选代码、静态 finite guard 或文档 smoke test 均不能改写为当前源码已实现。 | 本文第4节/第9.6节候选代码 |

阻断清单：

- `BLOCKED`：安装入口必须先统一根 `setup.py`、独立 `rsl_rl/setup.py` 与 `uv` 文档的唯一来源；在根项目元数据缺失且存在同名分发冲突时，不得启动训练或部署。
- `FAIL`：必须先修复当前 `HimGo2Env` 的参数契约与 `recorders` 语法错误；本节不修改源文件。
- `NOT_RUN`：CPU reset/step、S14 effort/finite safety、S16 termination latency、S17 play isolation、S18 deploy fail-safe 均没有当前运行证据；S14、S16、S17、S18 均未通过，具体单值状态以第10节表格为准；不得标为 `PASS`。
- `UNSAFE`：在补齐 finite reject、输出抑制、通信 watchdog、传感器断连处理、安全停机和可审计故障记录，并统一 calf effort 上限前，禁止把任何 exporter/actuator 候选或 clamp 当作硬件 fail-safe。

### 9.10.1 文档质量与待核验补充

9.1/9.2 是叙述性状态，严格的单值枚举仅适用于 9.3.1、9.9、9.10。`docs/setup.md:27-32,36-45` 的 shell fence 误标为 Python，属于文档质量 `FAIL`；第4节完整候选通过 AST；第12节无可执行 Python fence；将缩进片段复制到 class 外不能独立解析。collision candidate 的 `force_history` slot/contact-item 轴及 `any(dim=...)` 的物理时间聚合，必须在真实 `ContactSensor` 的 S6/S14 runtime 中核验，状态为 `NOT_RUN`；`torque_limits` candidate 读取的是已限幅的 `actuator_force`，因此 `soft_torque_limit=1.0` 的惩罚有效性也必须 runtime 核验，状态为 `NOT_RUN`。130/130 字段验证仅是文档覆盖率，不是 runtime 生效；最小只读核对沿用现有 9.2 矩阵，并逐项对照 `him_go2_env.py:8-226`。本补充不改变 `[ALL_TESTS_PASSED]` 缺失、S0 `FAIL` 及所有 runtime 阻断状态。

## 10. 最终验证矩阵与阻断规则

| 阶段 | 检查 | 预期证据 | 失败处理 | 当前状态 |
|---|---|---|---|---|
| S0 | `python -m py_compile` 目标 Python 文件 | 无 SyntaxError | 修复文档候选代码；当前源码已知失败 | FAIL |
| S1 | `import legged_mjlab.envs` | task 注册成功 | 检查 `recorders`、循环 import、rsl 来源 | BLOCKED |
| S2 | `Go2Asset(cfg)` + `MjSpec.from_file` | XML、joint、actuator、sensor 名称解析成功 | 对照 `go2.xml:41-159`；禁止猜名 | BLOCKED |
| S3 | manager config build | `integrator` 是字符串，gravity 是三元组，非空 terrain | 检查所有 builder 参数和 v1.6 dataclass 字段 | BLOCKED |
| S4 | 4 env CPU `reset()` | actor `[4,45]`、native critic `[4,235]` | 检查 term 顺序/传感器匹配 | BLOCKED |
| S5 | 4 env CPU `step(zeros)` | history `[4,270]`、critic `[4,235]`、reward `[4]`、terminated/truncated `[4]` | 检查 action dim、finite、dt | BLOCKED |
| S6 | reward audit | 非零 scale 全部出现在 manager，零 scale 不出现；每项输出 `[4]` | 逐项补 term 或明确禁用 | FAIL |
| S7 | event audit | startup/reset/interval 事件按 flag 生效，play 关闭 push/DR | 检查 model field 与随机化前后差异 | FAIL |
| S8 | partial reset | 只 reset done env，其他 env 的 command/history/timer 不变 | 这是 v1.6 partial-reset gate | NOT_RUN |
| S9 | HIM adapter | wrapper 返回 235，runner 显式消费 48，terminal obs 只替换 done 行；239 仅用于显式 foot-contact 扩展 | 任何 silent truncate 都是 blocker | NOT_RUN |
| S10 | one-iteration HIM PPO | estimator/optimizer/storage 无 shape 或 NaN 错误 | 先确认 `rsl_rl.__file__` 是项目 backend | NOT_RUN |
| S11 | play | `play=True` 无 curriculum/push/noise，checkpoint 可加载 | 当前 `play.py` 为空，未通过前不能宣称可回放 | FAIL |
| S12 | export/deploy | 导出输入维度、joint order、action scale/clip/delay 与部署一致 | 当前没有 Go2 HIM 的已验证导出 artifact | FAIL |
| S13 | action round-trip | policy/deploy canonical joint order 与 XML runtime actuator/joint 映射逐关节一致；零动作和单位脉冲均落到预期电机 | 任一腿/关节错位都是 Critical blocker | BLOCKED |
| S14 | effort/finite safety | exporter/actuator 的静态 finite guard 已存在；runtime 仍须证明每个 physics substep 的 torque/action 无 NaN/Inf，仿真与部署 effort clip 一致，calf 不超过 `35.55 N*m` | 静态 guard 或 clamp 不能写成硬件 fail-safe；部署 adapter 仍须 finite reject 并抑制非有限输出，任一越界或 runtime finite 检查缺失都不放行 | UNSAFE |
| S15 | reset/settle | 记录 `base z`、四足高度、接触力、第一步 impulse；零动作 settle 不穿透、不异常弹跳 | 失败则关闭 push/DR，回到初始姿态修正 | NOT_RUN |
| S16 | termination latency | 第 6.3 节 smoke test 不能证明该路径；后续必须分别注入 thigh、calf、base 接触，每次都断言一个 policy step 内 `terminated=True` 且 `truncated=False`，并确认不再执行危险动作；另行触发 timeout，断言 `terminated=False` 且 `truncated=True` | 未完成接触注入与 `terminated`/`truncated` 分别断言，或把 timeout 当作 failure termination，均是 blocker | FAIL |
| S17 | play isolation | `play=True` 明确关闭 push、DR、noise、随机延迟、action hold、curriculum，并固定 reset 位姿 | 任一随机化残留都不得用于部署回放结论 | BLOCKED |
| S18 | deploy fail-safe | exporter/actuator 的静态保护已存在，但仍须对 NaN/Inf、通信超时/断连、传感器断连/故障和 checkpoint 输出异常做故障注入；deployment adapter 必须 finite reject、抑制输出、运行 watchdog，并进入安全停机、记录原因 | 缺少 finite reject、输出抑制、watchdog、断连/传感器故障注入或可审计的安全停机日志，S18 失败；exporter/actuator 静态保护或 clamp 不得当作硬件 fail-safe | UNSAFE |

本轮没有执行上述运行验证，也没有触发 `[ALL_TESTS_PASSED]`。因此项目当前仍停在“文档化实现候选 + 静态审计”阶段，不能发布为可训练版本。

验证状态勘误：当前工作树存在未归属的 Python diff，集中在 `him_go2_env.py` 的 `_build_scene()`/sensor manager 调用附近；本轮文档补充不接管这些源码改动。`him_go2_env.py:101` 的 `metrics = ,` 仍在源码中，因此以当前工作树为准，S0/S1/S2 仍应从头执行，import 不能被描述为已通过。已有 `__pycache__` 变化是审查过程生成物，不是本次目标实现；最终提交前需单独确认其归属。

## 11. 落地顺序、风险与回滚

1. 先手工落地并静态编译 `him_go2_config.py`、`go2_asset.py`、`him_go2_env.py`，只解决 import/Asset/Manager 构造，不启动大规模训练。
2. 在任何训练前先完成 S13–S16：固定 XML runtime 与 policy/deploy canonical joint order 的逐关节映射，统一 `35.55 N*m` calf 上限，加入 finite/hard clip，验证初始姿态、零动作 settle、危险接触终止。
3. 关闭全部 domain randomization、push、noise、curriculum 和随机延迟，使用 plane 或单一 rough preset 做 4-env reset/step 和 shape gate。
4. 打开 actor noise、height scan，确认 45/235/270/48 分层契约，再打开 reward；若另行启用 critic foot-contact 扩展，则同步确认 239；S17 必须证明 play 隔离有效。
5. 按一类一类顺序打开 reset randomization：COM → joint friction/damping → friction → PD → mass/armature → push；每次保存参数快照和 episode metric，所有未验证项仍视为未放行。
6. 最后验证 partial reset、HIM PPO、checkpoint、play、ONNX/部署与 S18 故障注入，不得把训练曲线正常当作接口验证。

回滚策略是恢复三个目标 Python 文件到修改前版本，保留本文档作为手工迁移说明；不要用 `git reset --hard` 清理工作树，也不要删除未确认属于用户的文件。部署放行前必须同时满足：所有 S0–S18 有证据、`[ALL_TESTS_PASSED]`、动作范围/关节顺序/延迟单位与真实控制器签字确认。

## 12. 候选方案 C：奖励函数全部定义在 HimGo2Env 类内

本节是对前文候选代码的一个“类内实现”变体，服务于明确约束：所有会作为
RewardTermCfg.func 注册的 reward callable 都必须是 HimGo2Env 的静态方法。前文
的类外函数和本节的函数不能同时注册同一个 term；若采用本节，按本节的
_build_rewards 替换前文对应注册块即可。本节仍然只修改本文档，不修改任何
Python 文件，也没有声称 import、manager 构造或 runtime reward 已通过。

### 12.1 已核实事实与接口契约

当前工作树中的事实如下。

| 事实 | 证据 | 对候选实现的约束 |
|---|---|---|
| HimGo2RoughCfg.rewards.scales 有 17 个非零项，termination 是 -0.0 | legged_mjlab/envs/him_go2/him_go2_config.py:174-205 | 只把非零项放进 active terms；termination 方法可以存在但默认不注册 |
| HimGo2Env._build_rewards 当前构造空 terms 并返回 | legged_mjlab/envs/him_go2/him_go2_env.py:399-408 | 本节只设计替换后的 builder，不把文档候选描述成当前已接入 |
| 源码已有一组奖励函数，但函数体首参写成 env，未加 self 或 staticmethod | legged_mjlab/envs/him_go2/him_go2_env.py:488-697 | 类内函数必须加 @staticmethod；注册时传 HimGo2Env._reward_xxx，不能传会绑定 self 的实例方法 |
| 已安装 mjlab 为 1.6.0 | .venv/lib/python3.11/site-packages/mjlab | term callable 最终以 func(env, **params) 调用 |
| RewardManager 对每个 term 要求输出精确 shape 为 [num_envs] | mjlab/managers/manager_base.py 的 _check_term_shape 与 managers/reward_manager.py | 所有逐足、逐接触、逐关节量必须在 term 内归约 |
| Manager 在 policy action 后执行 decimation 个 physics step，再计算 termination 和 reward | mjlab/envs/manager_based_rl_env.py:386-506 | reward 看到的是一个 policy step 的末端状态；不要在 term 内重复乘 dt 或推进状态 |
| scale_rewards_by_dt 默认 True，当前源码却写成 False | ManagerBasedRlEnvCfg 与 him_go2_env.py:_build_mjlab_managercfg | 继续使用 legacy raw scale 时必须改为 True；这是迁移契约，不是本节已完成的 Python 改动 |
| ContactSensor 的 air time 每个 physics substep 更新，force_history 的 index 0 是最近 substep | mjlab/sensor/contact_sensor.py | feet_air_time 不能自己按 policy step 猜时间；collision 的 history 聚合必须明确物理含义 |

本节采用下面这组最小契约。

| 项目 | 契约 |
|---|---|
| 调用签名 | 每个 active term 是 @staticmethod，逻辑签名为 func(env, **resolved_params) |
| 环境批次 | env.num_envs = N；所有输出为与 env.device 相同设备上的 float tensor，shape 严格为 [N] |
| command | command_name=twist，返回 [N,3]，顺序为 vx、vy、wz；不从 actor observation 反切 command |
| entity | SceneEntityCfg 在 RewardManager 初始化阶段 resolve；reward 内使用 name、joint_ids、site_ids、actuator_ids，不自行按 XML 顺序猜索引 |
| 动作时序 | action_manager.action、prev_action、prev_prev_action 均是 policy action；action rate 用一阶差分，smoothness 用二阶差分 |
| 接触时序 | feet_ground_contact 开启 track_air_time=True；nonfoot_ground_touch 的 history_length=4 对应 decimation=4 的 physics history |
| dt | RewardTermCfg.weight 使用配置中的 raw scale；Manager 统一执行 value * weight * step_dt；reward 方法不得手工乘 step_dt |
| 状态副作用 | reward 方法只读 env/scene/manager；不在静态 term 内维护 feet_air_time、last_contact 等持久状态 |

### 12.2 推荐类内实现：可直接插入 HimGo2Env 的 reward 代码

下面代码块的缩进假定内容放在 class HimGo2Env(ManagerBasedRlEnv) 内。它复用当前
文件已有的 torch、Entity、ManagerBasedRlEnv、RewardTermCfg、SceneEntityCfg 和
ContactSensor 导入。代码以当前 builder 的无参签名 _build_rewards(self) 为准，
因为当前源码在 self.robot_cfg 中保存了 legacy 配置。

第4节是唯一规范完整实现；第12节不再复制代码。所有自定义 reward 必须在 HimGo2Env 类内使用 @staticmethod，并由 HimGo2Env._reward_* 注册。第12节其余表格和风险说明继续有效。

orientation 方法按 asset_cfg.name 取根实体，因而不会依赖
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")。当前 HimGo2RoughCfg.asset.name
是 go2；如果环境资产名变更，SceneEntityCfg 会随 builder 参数化，不应在 reward
函数中硬编码业务名称。最终 Python 文件只保留这一版 orientation 定义。

上面的代码是 callable 与注册契约，不是对当前源码的自动补丁。当前源码的
_build_rewards、_reward_scale、_add_reward 需要作为一个一致的组替换；不能只
复制静态方法而继续使用旧的空 builder。若项目后来把 builder 改成接收显式 cfg，
只需把 cfg = self.robot_cfg 换成该接口的 cfg，不要把 reward 函数移回模块级。

注意：本文第 4 节的完整替换候选曾使用
_add_reward(terms, cfg, name, func, params) 的文档级 helper 签名，而当前工作树
源码的 _add_reward 是 _add_reward(terms, name, func, params)。这两个候选不能
交叉复制；本节代码选择当前源码的无 cfg helper 签名，并把 cfg 只从
self.robot_cfg 读取。

### 12.3 当前 rewards 配置逐项注册表

下表以当前 him_go2_config.py 的 scale 为准。active 表示 scale 非零且本节
builder 会注册；zero 表示函数可以保留在类内，但当前配置不进入 RewardManager。

| 配置项 | scale | 类内 callable | mjlab 1.6 数据与归约 | 状态 |
|---|---:|---|---|---|
| tracking_lin_vel | 1.0 | _reward_tracking_lin_vel | root_link_lin_vel_b 的 xy 与 twist xy，exp(-平方误差 / tracking_sigma) | active |
| tracking_ang_vel | 0.5 | _reward_tracking_ang_vel | root_link_ang_vel_b 的 yaw 与 twist wz，exp(-平方误差 / tracking_sigma) | active |
| lin_vel_z | -1.5 | _reward_lin_vel_z | root_link_lin_vel_b[:,2] 平方 | active |
| ang_vel_xy | -0.05 | _reward_ang_vel_xy | root_link_ang_vel_b 的 xy 平方和 | active |
| orientation | -0.2 | _reward_orientation | projected_gravity_b 的 xy 平方和 | active |
| dof_acc | -2.5e-7 | _reward_dof_acc | joint_acc 选定 12 个 joint 后平方和 | active |
| joint_power | -2e-5 | _reward_joint_power | qfrc_actuator 与 joint_vel 的绝对乘积和 | active |
| base_height | -1.0 | _reward_base_height | height_scan 有效 hit 的 root z - hit z 均值与 target 的平方 | active |
| foot_clearance | -0.01 | _reward_foot_clearance | 四个 site 的 world z 绝对误差乘 world xy 速度，再按 xy 范数 + abs(wz) gate 归约 | active |
| action_rate | -0.01 | _reward_action_rate | action - prev_action 的平方和 | active |
| smoothness | -0.01 | _reward_smoothness | action - 2 prev_action + prev_prev_action 的平方和 | active |
| feet_air_time | 0.1 | _reward_feet_air_time | ContactSensor current air/contact time；当前 candidate/Unitree 语义：进入 two-contact mode（恰好两足接触）后取四足 mode time 最小值 | active；runtime `NOT_RUN` |
| collision | -0.5 | _reward_collision | `force_history` 的 H=4 是 physics-history 槽计数而非 contact total；无 history 时回退到 found 计数 | active |
| stand_still | -1.0 | _reward_stand_still | joint position 相对 default 的平方和，仅 xy 范数 + abs(wz) 不大于 0.1 时生效 | active |
| torques | -0.0 | _reward_torques | actuator_force 选定 actuator 后平方和 | zero |
| dof_vel | -0.0 | _reward_dof_vel | joint_vel 平方和 | zero |
| dof_pos_limits | -10.0 | _reward_dof_pos_limits | soft_joint_pos_limits 外的线性 violation 和 | active |
| dof_vel_limits | -0.0 | _reward_dof_vel_limits | 显式传入标量或每关节 `velocity_limits`，按 `soft_limit` 计算线性超限并逐关节截断到 1 后求和；无来源时不注册 | zero/contract-only |
| torque_limits | -1.0 | _reward_torque_limits | actuator_force 相对各 IdealPdActuator.force_limit 的归一化超限平方和 | active，需重点验证 |
| hip_pos | -1.0 | _reward_hip_pos | 四个 hip joint 相对 default 的平方和；本候选不加 legacy 的精确零侧向命令 gate | active |
| feet_stumble | -0.0 | _reward_feet_stumble | feet current force 的水平力大于 5 倍垂直力时按环境给事件 bit | zero |
| termination | -0.0 | _reward_termination | termination_manager.terminated，失败终止为 1，timeout 不计入 | zero，默认不注册 |

这里的 active 数量是 17。函数定义数量多于 active 数量是有意的：这样把
配置中可能重新打开的普通项也限制在 HimGo2Env 内，但没有把当前 zero scale
误报成运行时 active。dof_vel_limits 是例外：若没有明确的每关节速度上限来源，
不能仅因为配置里存在 soft_dof_vel_limit=1.0 就注册它。

### 12.4 四个参考项目的语义对齐

四个参考项目不能被当作同一个实现。下面先给出本候选的选择，再标出偏差和
迁移时必须保留的风险。

| 参考项目 | 事实锚点 | 本候选采用/不采用的语义 |
|---|---|---|
| AMP_mjlab | /home/kk/github/AMP_mjlab 的 mjlab 1.2.0 velocity 配置与 mdp/rewards.py | 采用 manager RewardTermCfg 组织和 projected-gravity、动作项的结构；不把 1.2.0 API 当成 1.6.0 证据，也不引入 AMP 专用 observation、runner 或额外 termination 权重 |
| unitree_rl_mjlab | /home/kk/github/unitree_rl_mjlab/src/tasks/velocity/mdp/rewards.py | 采用其 tracking 的 exp 形式、feet_air_time 的 current air/contact time + two-contact mode（恰好两足接触）+ min 以及 site world position/velocity 的 clearance 形式；其 1.2.0/任务本地实现只作语义参考 |
| HIMLoco | /home/kk/github/HIMLoco/legged_gym/.../go2w_legged_robot.py:51-52 与 rsl_rl/modules/him_estimator.py:76-84 | 保留 legacy 的 current-first history 解释和 actor/critic 分离；不把 legacy self.feet_air_time 的可变 buffer、reset_buf 或六帧观测直接当成 mjlab manager 字段 |
| 2027_RC_legged_robot | /home/kk/github/2027_RC_legged_robot/legged_gym/envs/base/him_legged_robot.py:1468-1583 | 作为当前 scale 与 legacy 数值基线；tracking 在本候选只跟踪 xy/yaw，z/xy angular 仍为独立 penalty；termination 的 post-clamp 顺序不由 manager 自动复现 |

逐项差异如下。

1. tracking。legacy Go1 只计算 xy 线速度和 yaw 角速度，且使用
   exp(-error / tracking_sigma)；Unitree mjlab 的线速度 tracking 会把 z
   误差加进 tracking，角速度 tracking 会把 xy 误差加进 tracking。当前配置
   同时存在 lin_vel_z 和 ang_vel_xy，因此本候选选择 legacy 的轴拆分，避免
   同一误差被 tracking 和独立 penalty 双重计算。

2. feet_air_time。本候选选择 Unitree 的 threshold=0.4、two-contact mode（恰好两足接触）和
   mode time 最小值，因为 mjlab ContactSensor 已提供按 physics substep 更新
   的 current_air_time/current_contact_time，静态 term 不需要复制状态。legacy
   HIMLoco/Go1 的等价物是 threshold=0.5、first contact 时读取各足累计 air
   time，并按足相加；两者不是数值等价。若训练复现目标是旧 checkpoint，应把
   本 callable 换成基于 ContactSensor.last_air_time 与
   compute_first_contact(dt=env.step_dt) 的 legacy 分支，不能只改 threshold。

3. collision。Unitree 示例的 force_history 分支通常把
   [N, primary, history, 3] 先取力范数、跨 primary 做 any，再跨 history
   求和；本候选保持这一语义。这里 H=4 表示 physics-history 槽计数而非
   contact total，不是四个独立接触；当前 Go2Asset helper 的 nonfoot sensor 实际为
   reduce=maxforce、num_slots=1、history_length=decimation；若改成 reduce=none，
   found 的 primary/slot 数量和 force_history 的 P 轴会变，reward 的绝对量也会
   变。legacy 当前接触判断用当前帧 force > 0.1，Unitree 常用 10，而当前配置
   max_contact_force=100；阈值必须作为明确迁移选择，不能宣称三者相同。

4. stand_still。Unitree/本候选是 default pose 的平方和并按 command gate；
   legacy 是 L1 joint error 且主要按线速度命令 gate。当前类内方案保持前文
   candidate 的 squared default-pose 选择，说明这是一项语义变更，不是机械的
   函数搬家。

5. hip_pos。legacy 是四个 hip 相对零角度、且只在侧向命令等于零时惩罚；
   本候选沿用前文 candidate 的相对 default pose 平方和，不加精确浮点等于零
   的 gate。若要复现 legacy，须显式增加目标为零和 command gate。

6. foot_clearance。Unitree/candidate 使用 site_pos_w 和 site_lin_vel_w 的
   world 高度、world xy 速度、绝对高度误差；legacy 使用相对 root 的 body-frame
   高度、平方误差，且没有同样的 gate。当前 clearance_height_target=-0.2
   不是 world foot target，builder 用 abs 后得到 0.2 是候选转换，必须在真实
   XML/地形上复核。

7. dof_acc 与 torque_limits。Unitree mjlab 可直接读 joint_acc；legacy 是
   (last_dof_vel-dof_vel)/legacy_dt。mjlab 的 joint_acc 来自 physics derived
   data，且 reward 计算点存在最后一个 mj_step 的一 substep freshness 约束。
   torque_limits 不能把 actuator_force 当作 legacy 未限幅 torque；mjlab
   IdealPdActuator 的 force_limit 会在 actuator 输出路径中做 clip，因此
   soft_torque_limit=1.0 时该 penalty 可能长期为零。

### 12.5 shape、时序与 contact history 风险清单

#### shape

- RewardManager 的硬检查是 [N]，不是 [N,1]、[N,4] 或 [N,4,3]。四个 feet
  的 contact/time、四元 site 高度、选定 joint/actuator 必须在 callable 内完成
  sum、min、any 或 count。
- root_link_lin_vel_b、root_link_ang_vel_b、projected_gravity_b 是 [N,3]；
  joint_pos、joint_vel、joint_acc 是 [N,J]；action 及其两级历史是 [N,A]，
  当前 A 应为 12。每个 term 的最终输出必须 batch 维仍为 N。
- ContactSensor 的 found 不是固定为 [N,4]：它可能是 [N,P] 或
  [N,P*num_slots]；force_history 是 [N,P_or_slots,H,3]。本节 collision
  明确把 H=4 的 physics-history 槽维 count 掉，不把它当作 contact total；
  fallback 明确把 found 的第二维 count 掉，但这两个结果的单位不同。
- ContactSensor 的 current_air_time/current_contact_time 是 [N,P]，P 必须在
  runtime 解析为四个 primary foot。不能仅根据 sensor 名称假定 FL、RL、FR、
  RR 顺序；应检查 sensor.primary_names 与 SceneEntityCfg.site_names 的解析表。
- SceneEntityCfg 在 manager 初始化时被 resolve；term 内不能把 joint_ids
  当作全局 MuJoCo joint id 再次映射，也不能把 actuator_ids 传给 joint_pos。

#### 时序

- Manager 每个 policy step 先 process_action 一次，再 apply_action 和
  sim.step 四次；action_manager.action 是当前 policy action，prev_action 和
  prev_prev_action 是 policy-level history。action rate/smoothness 不应在四个
  physics substep 中重复累加。
- Reward 在 termination.compute() 之后调用。由此 _reward_termination 读取的
  terminated 是当前 step 的失败终止 bit；time_out 属于 time_outs/truncated，
  不是 terminated。
- mjlab 文档明确提醒：reward/termination 读取的 derived data 可能比最后一次
  mj_step 少一个 physics forward 的新鲜度；不要把 joint_acc 与 legacy 的
  velocity finite difference 宣称为同一时刻同一量。
- RewardTermCfg.weight 是 raw scale 时，唯一的 dt 路径是
  RewardManager 的 value * weight * env.step_dt。当前 decimation=4、sim.dt=0.005，
  step_dt=0.02；手工把 scale 先乘 0.02 或把 scale_rewards_by_dt 设 False 都会
  破坏 legacy 数值标定。

#### contact history

- force_history 只有在 ContactSensorCfg.fields 包含 force 且 history_length>0
  时才有分配。只注册 _reward_collision 而没有相应 sensor field/history，
  不是“退化为同一语义”，而是会进入 found fallback 或直接失败。
- collision 的 H=4（history_length=4）表示四个 physics-history 槽、对应四个
  physics substep，不是 contact total，也不是四个 policy observation frame。
  index 0 是最近 substep，若要 count 应明确是否对 history 求和、取 any，不能
  把 index 0 当作当前 policy frame 后又乘 decimation。
- track_air_time=True 的 sensor 在 Sensor.reset(env_ids) 时按 env 部分重置。
  因而静态 reward 不应写 self.feet_air_time 累加器；若实现 legacy first-contact
  stateful 版本，必须使用 manager 可实例化的 class term 并验证 partial reset。
- 当前 go2_asset.py 的足端 sensor 也设置了 history_length=decimation，但
  _reward_feet_air_time 只读取 air/contact time；不要因此误把足端 history
  当作 collision history。
- found 是接触匹配数量/事件，不是力。reduce=maxforce/netforce/none 会改变
  found 或 force 的物理解释；force_threshold=100、10、0.1 之间不能只靠名称
  推断等价关系。

### 12.6 termination reward 与 only_positive_rewards 的边界

_reward_termination 只返回 termination_manager.terminated，符合 mjlab 1.6 的
failure-only 语义；timeout term 由 TerminationTermCfg(time_out=True) 进入
time_outs，不应再由 reward 读取 reset_buf 复制一份。

当前 scale 是 termination=-0.0，因此推荐默认不注册它。若后来把它改成非零，
RewardManager 会像其他项一样把它加入总和；而 legacy HIMLoco 的顺序是先对普通
reward 总和做 only_positive_rewards clamp，再把 termination reward 追加到 clamp
之后。mjlab 不会自动读取 legacy 的 only_positive_rewards，也不会自动执行这个
post-clamp injection。

因此存在两个必须显式选择的契约：

| 选择 | 行为 | 风险 |
|---|---|---|
| manager-native | termination 进入 RewardManager 总和，若外层统一 clamp 则和普通项一起 clamp | 不复现 legacy 的“clamp 后再扣 terminal penalty” |
| legacy-compatible | 在环境 step 的 reward 汇总边界保存普通总和、先 clamp，再追加 terminated penalty | 需要改环境生命周期/step 汇总代码，超出只搬 reward callable 的范围，必须另开审查 |

本节只提供 manager-native 的 callable，不把任一选择伪装成已运行。特别是
在 only_positive_rewards=True 时，不能因为看到 termination 方法存在就声称
legacy terminal penalty 已生效。

### 12.7 注册后验收契约

以下是落地 Python 修改后的最小检查清单；当前文档追加没有执行这些检查。

1. 静态检查：所有 RewardTermCfg.func 的 __qualname__ 应以
   HimGo2Env. 开头；active term 集合应为当前 scale 的 17 项，不能出现
   同名的模块级函数；所有 callable 都应能以 env 为第一个实参调用。
2. manager shape gate：在至少 4 个 env 的真实 manager 上执行 reset/step，
   对每个 active term 单独读取 get_active_iterable_terms，确认 value 是
   [N]、finite，且没有 [N,1]、per-foot 或 history 未归约结果。
3. dt gate：记录 scale_rewards_by_dt、env.step_dt 和 manager 输出，证明每项只
   经一次 0.02 缩放；用一个固定 synthetic state 对比 raw term、weight 后值、
   manager reward。
4. action timing gate：连续三次不同 action，验证 rate 使用一阶差分，
   smoothness 使用二阶差分；验证 decimation=4 不会把同一 action 项累计四次。
5. contact gate：构造单足、双足、四足接触和 partial reset，确认
   current_air_time/current_contact_time 在 physics substep 递增/重置，确认
   collision history 的输出范围为 0 到 H，并记录 reduce、P、H、threshold。
6. torque gate：在 actuator clip 前后分别记录原始 PD effort（若可得）和
   asset.data.actuator_force；若 force 已被限幅，则 soft_torque_limit=1.0 的
   active reward 应明确标记为 dead/无效，而不是把全零输出当作验证通过。
7. termination gate：分别注入非 timeout 失败和 timeout，确认
   terminated、time_outs、_reward_termination 三者只在失败 case 对应；再单独
   验证 only_positive_rewards 的 clamp 顺序。
8. full runtime gate：当前源码仍有 `metrics = ,`、builder 签名/
   调用不一致等已知阻断；在这些 Python 问题修复并实际运行前，本文状态保持
   NOT_RUN，不得输出 [ALL_TESTS_PASSED]。

本节的结论是：在不改变全局架构的前提下，“全部 reward callable 放进
HimGo2Env”可以通过静态方法 + 显式 SceneEntityCfg/ContactSensor 参数实现，
且能满足 mjlab 1.6 的 manager 调用协议；但它只解决 callable 的归属与 shape
契约，不自动解决当前 Python 的 import/manager 构造、only-positive 顺序、
actuator 未限幅 torque、四足 primary 映射或四个参考项目之间的数值语义差异。

## 13. 2026-08-28 追加审计：MJLab 1.6.0 API 与当前 scope 锁定

本节补充第 4 节候选实现的解释边界：本轮默认不扩展实现范围，保留当前
`HimGo2RoughCfg.env.num_privileged_obs = 235` 的 scope。`239` 只表示额外把
4 维 foot-contact 放入 critic 的可选扩展，不能再作为默认目标混用。

### 13.1 当前环境事实

| 项 | 事实 | 影响 |
|---|---|---|
| 依赖入口 | 根 `setup.py` pin `mjlab==1.6.0`、`mujoco-warp==3.11.0`，当前 `.venv` 中 `importlib.metadata.version()` 与之吻合 | 以本项目 `.venv/bin/python` 为准；系统 `python` 不存在，系统 `python3` 无 `mjlab` |
| Matplotlib cache | 导入 mjlab 时默认 `$HOME/.config/matplotlib` 不可写，会退到临时目录 | 后续 smoke/test 命令建议显式设置 `MPLCONFIGDIR=/tmp/mjlab-mpl` |
| 当前源码解析 | `legged_mjlab/envs/him_go2/him_go2_env.py:101` 仍为 `metrics = ,` | import、task registry、reset/step 全部不能被描述为已通过 |
| 当前 Python diff | 工作树中已有未归属的 `him_go2_env.py` diff，修改 `_build_scene()` 签名和 sensor manager 调用 | 文档补充只记录它，不合并、不回滚、不把它算成本轮实现 |

### 13.2 MJLab 1.6.0 接口契约

| 模块 | v1.6.0 契约 | 对 HIM-Go2 的要求 |
|---|---|---|
| `ManagerBasedRlEnvCfg` | `metrics`/`recorders` 已有默认空 dict；`auto_reset` 默认 `True`；`scale_rewards_by_dt` 默认 `True` | builder 里不要写空右值；若 wrapper 要保留真实 terminal critic，应显式 `auto_reset=False` |
| manager 加载顺序 | Event → Command → Action → Observation → Termination → Reward → Curriculum → Metrics/Recorder | 需要先定义 scene/sensor/action/obs，再注册依赖它们的 reward/termination |
| `ObservationTermCfg` | 执行顺序为 func → noise → clip → scale → delay → history；group history 会覆盖 term history | 原生 actor group 保持 45 单帧；6 帧历史只由 `HIMRslRlWrapper` 做 |
| `JointPositionActionCfg` | 字段是 `actuator_names`，支持 `scale`/`offset`/`clip` dict 和 `preserve_order` | 必须用四个分腿 term 固定策略顺序，不能继续使用旧的 `actuator_name` |
| `UniformVelocityCommandCfg` | yaw 命令字段是 `ang_vel_z`；`heading` 只有 `heading_command=True` 时才合法 | 当前 heading_command=False 时，actor 只拼 3 维 command，不拼 4 维 |
| `RewardTermCfg` | callable 形如 `func(env, **params) -> [N]`；`scale_rewards_by_dt=True` 时 manager 自动乘 `env.step_dt` | legacy scale 不应手工再乘 dt；若要复现 clamp 后 terminal penalty，需要另开生命周期修改 |
| Event/DR | event mode 包括 startup/reset/interval/step；DR 通过 `requires_model_fields` 扩展模型字段 | payload/mass/COM/friction/PD/armature/push 必须明确 mode 与参数快照 |
| DR 函数表 | v1.6.0 提供 `body_mass`、`pseudo_inertia`、`body_com_offset`、`geom_friction`、`pair_friction`、`joint_friction`、`joint_damping`、`joint_armature`、`pd_gains`、`effort_limits`、`encoder_bias` | 没有确认到内置 `geom_restitution`；`randomize_restitution` 只能标为未接入或写自定义 adapter |
| ContactSensor | per-contact force/history 与 per-primary air/contact time 是不同张量；partial reset 会按 env 重置 sensor state | reward/termination 必须显式选择 force/found/air_time 的 shape 归约方式 |
| RayCastSensor | grid ray 数为 frame 数 × pattern 点数；height scan 返回 `[N, num_rays]` | 17×11 height scan 是 187 维，和 actor/history 不混叠 |

官方文档入口：环境配置、observation、action、event、reward、termination、terrain、
RSL-RL 分别对应 `https://mujocolab.github.io/mjlab/main/source/environment_config.html`、
`observations.html`、`actions.html`、`events.html`、`rewards.html`、
`terminations.html`、`terrain.html`、`training/rsl_rl.html`。本节以本项目已安装
的 `mjlab==1.6.0` 源码为最终判据，文档 URL 只作为章节定位。

### 13.3 当前还没用上的配置

| 配置类 | 已声明但未闭环的项 | 推荐接入状态 |
|---|---|---|
| reward scales | `tracking_*`、orientation、torque/action rate/smoothness、collision、feet air/clearance/stumble、stand still、joint pos/vel/acc/limit、base height 等非零项 | 第 4 节候选已给出 manager-native callable；当前源码 `_build_rewards()` 仍没有返回 active reward dict |
| command | `num_commands=4`、heading range、yaw range | 保持 actor 3 维 command；`heading_command=False` 时不传 `heading` range |
| observation | actor noise、height measurement、privileged velocity/height | actor 45、critic 235；delay 用 ObservationTerm delay，不放进 wrapper history |
| terrain | rough/static/dynamic 选项、height scan grid | scene terrain 和 raycast sensor 尚未形成可运行闭环 |
| reset randomization | root pose/velocity、joint pos/vel | 用 reset event；必须和 partial reset gate 一起测 |
| domain randomization | payload、link mass、COM、joint friction/damping/armature、ground friction、PD gain、effort limit、encoder bias、push | 按 v1.6 DR 函数逐项接入；`restitution`、motor strength、action latency 需要自定义或明确不接入 |
| actuator/deploy | calf torque 上限、action clip、policy/deploy joint order、latency/fail-safe | 先统一 `FL,RL,FR,RR × hip/thigh/calf` 与 effort limit，再谈导出/部署 |
| play/export | play.py 为空，exporter.py 为空行 | 目前没有可执行回放和 ONNX/export gate |

### 13.4 参考项目可借鉴边界

| 参考项目 | 可借鉴 | 不可直接迁移 |
|---|---|---|
| `/home/kk/github/AMP_mjlab` | `ManagerBasedRlEnvCfg` 组装、task 注册、外部 runner/ONNX 结构、AMP 专用 observation 分组 | pin `mjlab==1.2.0`；AMP discriminator 390 维不是 HIM-Go2 critic |
| `/home/kk/github/unitree_rl_mjlab` | Go2 manager cfg 分层、Go2 常量表、官方 velocity reward/event/sensor 组织、deploy YAML/CPP 边界 | actor 使用 phase/height-scan 等 Unitree velocity 契约，不是当前 45/270 HIM history |
| `/home/kk/github/HIMLoco` | HIM estimator 的 velocity/latent/actor 拼接语义和 current-first history 习惯；Go2W actor 顺序为 command、base angular velocity、projected gravity、DOF error、DOF velocity、last action；DR/reward 名称能解释当前配置来源 | Go2W 是 16-action 轮足任务，单帧 actor 57、privileged 262；`terrain_crossing_reward`、wheel action 和 upstream critic 切片顺序不能直接变成当前 12-DOF Go2 的 45/270/235 契约 |
| `/home/kk/github/2027_RC_legged_robot/legged_gym/envs/legged_gym_go1` | 标准 12-DOF Go1 P 控制、`action_scale=0.25`、`decimation=4`、thigh/calf 惩罚接触、base 接触终止、base-height target 与 entropy 设置可作为 legged_gym 风格迁移参照 | Go1 参考不包含 HIM history/adaptation 结构，也不能绕过 mjlab manager 把 state buffer 和 explicit compute 直接搬进当前实现 |

### 13.5 当前默认代码候选的 shape 锁

第 4 节完整候选中，critic 默认应保持：

```python
critic_terms = dict(actor_terms)
critic_terms.update(
    {
        "base_lin_vel": ObservationTermCfg(
            func=envs_mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg(entity_name)},
        ),
        "height_scan": ObservationTermCfg(
            func=envs_mdp.height_scan,
            params={"sensor_name": "height_scan"},
        ),
    }
)
```

只有明确选择扩展 scope，并同步修改 `num_privileged_obs`、wrapper shape gate、
smoke test、runner/storage 说明之后，才允许加入：

```python
critic_terms["foot_contact"] = ObservationTermCfg(
    func=_foot_contact_observation,
    params={"sensor_name": "feet_ground_contact"},
)
```

因此本轮文档的结论是：当前 `him_go2` 仍处于“结构草稿 + 文档化完整候选”阶段；
缺口的主线不是再换架构，而是把已声明的 manager config、reward、event、sensor、
action、wrapper 和 deploy shape 按 235 默认契约逐项落地并通过 S0-S18 gate。
