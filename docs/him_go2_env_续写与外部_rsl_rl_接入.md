# `him_go2` 环境续写与外部 `rsl_rl` 接入说明

> 本文是 `/home/aira/docs/legged_mjlab` 的续写方案。
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
        │ native 观测：actor[45]、critic[239]
        ▼
HIMRslRlWrapper
        │ actor history[6, 45] -> [270]
        ▼
项目内 rsl_rl.runners.HIMOnPolicyRunner
```

这里的 `rsl_rl` 不是官方 mjlab runner。当前仓库已经提供了自己的 `HIMOnPolicyRunner`、`HIMActorCritic`、`HIMPPO` 和 wrapper；`task_registry.load_project_rsl()` 会在已安装包缺失或不兼容时加载仓库内的实现。

需要注意：当前 loader 的策略是“兼容的已安装 `rsl_rl` 优先，项目实现作为 fallback”。因此，若要求绝对只使用项目内实现，必须在运行时检查 `rsl_rl.__file__`，而不能只看 `setup.py`。

## 2. 当前文件的明确缺口

当前 `legged_mjlab/legged_mjlab/envs/him_go2/him_go2_env.py` 还不能被 Python 解析，主要缺口是：

| 位置 | 当前问题 | 处理方式 |
| --- | --- | --- |
| `_build_mjlab_managercfg` | `events =`、`viewer =` 等语法占位符 | 统一构造 manager 配置 |
| `MujocoCfg` | 把重力向量传给了 `integrator` | `integrator="implicitfast"`，另传 `gravity=tuple(cfg.sim.gravity)` |
| `_build_scene` | 依赖空的地形与传感器实现 | 使用合法的 generator、接触传感器和 raycast 传感器 |
| `_build_terrain` | generator 的 `sub_terrains={}` 无法生成地形，也没有非法类型报错 | 基于 `ROUGH_TERRAINS_CFG` 做 `dataclasses.replace` |
| `_joint_names` | 缺少 `self` | 改成静态方法或普通实例方法 |
| `_build_actions` | `actuator_name`、`hip_scale_reduction`、`action_clip` 均不符合当前项目接口 | 使用 `actuator_names`、`hip_reduction`，不把旧 clip 字段硬塞给 mjlab |
| `_reward_scale` | `BaseConfig` 子类不是 Mapping，不能调用 `.get()` | 使用 `getattr(cfg.rewards.scales, name, 0.0)` |
| `_build_observations` | 空实现 | 严格构造 actor 45 维和 critic 239 维 |
| `_build_events` | 空实现 | 加入 reset、push 和可用的 domain randomization |
| `_build_commands` | 空实现 | 注册名为 `twist` 的 `UniformVelocityCommandCfg` |
| `_build_rewards` | `_add_reward` 调用不完整且无返回值；上一版混用了官方 velocity task 函数与项目奖励 | 按 `unitree_rl_mjlab` 的本地 `mdp` 模式注册任务特有 reward，并覆盖所有非零旧 reward |
| `_build_terminations` / `_build_curriculum` | 空实现 | 增加 time-out、姿态、base 接触和 terrain curriculum |
| `__init__` | `render_mode` 没有默认值，但现有 `train.py` 不传它 | 将 `render_mode` 默认设为 `None` |

`Go2Asset.sensor` 里还保留了一套早期 API 风格的传感器 helper，例如 `prim_path`、`target_names_expr` 和 `ContactMatch.GEOM_NAME`。它与当前 mjlab 1.6.0 的 scene-level sensor API 不兼容。本方案不调用那套 helper，而是在 `SceneCfg.sensors` 中使用当前 API；后续如果要清理 stale helper，应另开一次代码变更。

## 3. HIM 的观测契约

### 3.1 actor 45 维

actor group 必须只产生一帧 45 维观测，顺序固定为：

| term | 维度 | 来源 |
| --- | ---: | --- |
| `base_ang_vel` | 3 | `go2/imu_ang_vel` |
| `projected_gravity` | 3 | mjlab 内置函数 |
| `command` | 3 | `twist` command 的 `vx, vy, wz` |
| `joint_pos` | 12 | 12 个 Go2 关节相对默认角 |
| `joint_vel` | 12 | 12 个 Go2 关节速度 |
| `actions` | 12 | 上一帧 policy action |
| **合计** | **45** | |

旧配置里的 `num_commands = 4` 包含 heading 这个命令配置字段，但 `heading_command=False` 时 heading 不进入 actor 观测；因此 actor 中必须使用 `generated_commands("twist")` 的 3 维输出。

### 3.2 critic 239 维

critic 以同样的 45 维 actor 一帧开头，再拼接：

| term | 维度 |
| --- | ---: |
| actor terms | 45 |
| `base_lin_vel` | 3 |
| `foot_contact` | 4 |
| `height_scan` | 17 × 11 = 187 |
| **合计** | **239** |

项目内 HIM runner 的 estimator 只消费 critic 的前 `45 + 3 = 48` 维，但 wrapper 仍然必须保留完整的 239 维 privileged observation；这样既满足 estimator 的窄输入约定，也保留完整 critic 信息供后续扩展。

原生 mjlab observation group 的 `history_length` 必须保持为 `1`。6 帧历史由已有的 `HIMRslRlWrapper` 维护，否则会把 45 维再次历史展开，导致 actor 维度重复。

## 4. `him_go2_env.py` 完整替换代码

下面的代码块是单文件替换版本。复制时覆盖目标文件的全部内容；本文本身不会自动覆盖它。

```python
from __future__ import annotations

import math
from dataclasses import replace

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
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
    # Current mjlab normally returns [N, feet].  Accept [N, feet, slots] as
    # well, because num_slots is a sensor configuration detail.
    if found.ndim == 3:
        found = found.any(dim=-1)
    if found.ndim != 2 or found.shape[1] != 4:
        raise RuntimeError(
            f"{sensor_name!r} must resolve to four feet, got {tuple(found.shape)}"
        )
    return (found > 0).float()


def track_linear_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_lin_vel_b
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    lin_vel_error = xy_error + (2 * z_error)
    return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_ang_vel_b
    z_error = torch.square(command[:, 2] - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    ang_vel_error = z_error + (0.05 * xy_error)
    return torch.exp(-ang_vel_error / std**2)


def body_orientation_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
    asset: Entity = env.scene[asset_cfg.name]

    if asset_cfg.body_ids:
        body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
        body_quat_w = body_quat_w.squeeze(1)
        gravity_w = asset.data.gravity_vec_w
        projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
        xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    else:
        xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return xy_squared


def lin_vel_z(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def ang_vel_xy(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def feet_air_time(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float = 0.4,
    command_name: str | None = None,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward feet air time."""
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    air_time = sensor_data.current_air_time
    contact_time = sensor_data.current_contact_time
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
    mode_time = torch.min(
        torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
    )[0]
    error = torch.abs(mode_time - threshold)
    reward = torch.clamp(threshold - error, min=0.0)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


def self_collision_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Penalize self-collisions."""
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        force_mag = torch.norm(data.force_history, dim=-1)
        hit = (force_mag > force_threshold).any(dim=1)
        return hit.sum(dim=-1).float()
    assert data.found is not None
    return data.found.squeeze(-1)


def illegal_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Terminate when a forbidden contact exceeds the force threshold."""
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        force_mag = torch.norm(data.force_history, dim=-1)
        return (force_mag > force_threshold).any(dim=-1).any(dim=-1)
    assert data.found is not None
    return torch.any(data.found, dim=-1)


def hip_pos(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize hip deviation from the robot's default joint pose."""

    asset: Entity = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    diff_angle = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.square(diff_angle), dim=1)


def _base_height_l2(env, target_height: float, asset_cfg: SceneEntityCfg):
    """Penalize root height relative to the terrain origin."""

    asset = env.scene[asset_cfg.name]
    height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.square(height - target_height)


def feet_clearance(
    env: ManagerBasedRlEnv,
    target_height: float,
    command_name: str | None = None,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize deviation from target clearance height, weighted by foot velocity."""
    asset: Entity = env.scene[asset_cfg.name]
    foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    vel_norm = torch.norm(foot_vel_xy, dim=-1)
    delta = torch.abs(foot_z - target_height)
    cost = torch.sum(delta * vel_norm, dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            active = (total_command > command_threshold).float()
            cost = cost * active
    return cost


def _joint_power_l1(env, asset_cfg: SceneEntityCfg):
    """Penalize absolute actuator mechanical power for the selected joints."""

    asset = env.scene[asset_cfg.name]
    torque = asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
    velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(torque * velocity), dim=1)


def stand_still(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command <= command_threshold).float()
            reward *= scale
    return reward


def _torque_limit_cost(
    env,
    soft_limit: float,
    asset_cfg: SceneEntityCfg,
):
    """Penalize actuator output above each configured Ideal-PD effort limit."""

    asset = env.scene[asset_cfg.name]
    force = asset.data.actuator_force
    limits = torch.full_like(force, float("inf"))

    # Go2Asset creates three IdealPdActuator groups.  Their force_limit tensors
    # are per-environment, so the term also remains valid after effort-limit DR.
    for actuator in asset.actuators:
        force_limit = getattr(actuator, "force_limit", None)
        if force_limit is None:
            continue
        limits[:, actuator.ctrl_ids] = force_limit

    force = force[:, asset_cfg.actuator_ids]
    limits = limits[:, asset_cfg.actuator_ids]
    denominator = torch.clamp(limits * max(float(soft_limit), 1.0e-6), min=1.0e-6)
    excess = torch.relu(torch.abs(force) / denominator - 1.0)
    return torch.sum(torch.square(excess), dim=1)


class HimGo2Env(ManagerBasedRlEnv):
    """Go2 rough-terrain manager environment for the repository HIM runner."""

    def __init__(
        self,
        cfg: HimGo2RoughCfg,
        sim_device=None,
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
        super().__init__(
            cfg=self.managercfg,
            device=sim_device or getattr(cfg.env, "device", "cpu"),
            render_mode=render_mode,
        )

    @staticmethod
    def _joint_names(cfg) -> tuple[str, ...]:
        """Keep the policy order declared by the legacy default-angle dict."""

        return tuple(cfg.init_state.default_joint_angles.keys())

    @staticmethod
    def _noise(cfg, field: str, enabled: bool):
        if not enabled or not cfg.noise.add_noise:
            return None
        scale = float(getattr(cfg.noise.noise_scales, field))
        level = float(cfg.noise.noise_level)
        return Unoise(n_min=-level * scale, n_max=level * scale)

    def _build_mjlab_managercfg(
        self,
        cfg,
        play: bool = False,
        debug_vis: bool = False,
    ) -> ManagerBasedRlEnvCfg:
        entity_name = cfg.asset.name
        asset = Go2Asset(cfg)

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
            auto_reset=True,
            # Keep the old config's reward magnitudes as per-policy-step
            # weights.  Enabling this would multiply them by step_dt again.
            scale_rewards_by_dt=False,
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
        nonfoot_pattern = r".*_collision\d*$"

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
                    pattern=nonfoot_pattern,
                    entity=entity_name,
                    exclude=foot_geoms,
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

        # ROUGH_TERRAINS_CFG is a valid mjlab preset.  An empty sub_terrains
        # mapping is not valid.  In curriculum mode mjlab uses one column per
        # configured subterrain type; num_cols is still retained for the
        # non-curriculum play grid.
        generator = replace(
            ROUGH_TERRAINS_CFG,
            curriculum=bool(cfg.terrain.curriculum and not play),
            size=(
                float(cfg.terrain.terrain_length),
                float(cfg.terrain.terrain_width),
            ),
            border_width=float(cfg.terrain.border_size),
            num_rows=5 if play else int(cfg.terrain.num_rows),
            num_cols=5 if play else int(cfg.terrain.num_cols),
        )
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
        joint_names = self._joint_names(cfg)
        scales = {
            name: float(
                cfg.control.action_scale
                * (cfg.control.hip_reduction if "_hip_joint" in name else 1.0)
            )
            for name in joint_names
        }
        return {
            "joint_position": JointPositionActionCfg(
                entity_name=cfg.asset.name,
                actuator_names=joint_names,
                scale=scales,
                use_default_offset=True,
                preserve_order=True,
                clip=None,
            )
        }

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

        events = {
            "reset_base": EventTermCfg(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (0.0, 0.0),
                        "yaw": (-math.pi, math.pi),
                    },
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

        if not play and cfg.domain_rand.push_robots:
            max_push = float(cfg.domain_rand.max_push_vel_xy)
            events["push_robot"] = EventTermCfg(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(
                    float(cfg.domain_rand.push_interval_s),
                    float(cfg.domain_rand.push_interval_s),
                ),
                params={
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

        if cfg.domain_rand.randomize_friction:
            events["foot_friction"] = EventTermCfg(
                mode="startup",
                func=dr.geom_friction,
                params={
                    "asset_cfg": foot_cfg,
                    "operation": "abs",
                    "ranges": (0.3, 1.6),
                    "shared_random": True,
                },
            )

        if cfg.domain_rand.randomize_motor_zero_offset:
            events["encoder_bias"] = EventTermCfg(
                mode="startup",
                func=dr.encoder_bias,
                params={
                    "asset_cfg": joint_cfg,
                    "bias_range": tuple(cfg.domain_rand.motor_zero_offset_range),
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
            events["effort_limits"] = EventTermCfg(
                mode="startup",
                func=dr.effort_limits,
                params={
                    "asset_cfg": actuator_cfg,
                    "operation": "scale",
                    "effort_limit_range": tuple(
                        cfg.domain_rand.motor_strength_range
                    ),
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

    def _reward_scale(self, cfg, name: str) -> float:
        return float(getattr(cfg.rewards.scales, name, 0.0))

    def _add_reward(self, terms, cfg, name: str, func, params=None):
        weight = self._reward_scale(cfg, name)
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
            track_linear_velocity,
            {"command_name": "twist", "std": std, "asset_cfg": SceneEntityCfg(entity_name)},
        )
        self._add_reward(
            terms,
            cfg,
            "tracking_ang_vel",
            track_angular_velocity,
            {"command_name": "twist", "std": std, "asset_cfg": SceneEntityCfg(entity_name)},
        )
        self._add_reward(
            terms,
            cfg,
            "lin_vel_z",
            lin_vel_z,
            {"asset_cfg": SceneEntityCfg(entity_name)},
        )
        self._add_reward(
            terms,
            cfg,
            "ang_vel_xy",
            ang_vel_xy,
            {"asset_cfg": SceneEntityCfg(entity_name)},
        )
        self._add_reward(
            terms,
            cfg,
            "orientation",
            body_orientation_l2,
            {"asset_cfg": SceneEntityCfg(entity_name)},
        )
        self._add_reward(
            terms,
            cfg,
            "dof_acc",
            envs_mdp.joint_acc_l2,
            {"asset_cfg": joint_cfg},
        )
        self._add_reward(
            terms,
            cfg,
            "joint_power",
            _joint_power_l1,
            {"asset_cfg": joint_cfg},
        )
        self._add_reward(
            terms,
            cfg,
            "base_height",
            _base_height_l2,
            {
                "target_height": float(cfg.rewards.base_height_target),
                "asset_cfg": SceneEntityCfg(entity_name),
            },
        )
        self._add_reward(
            terms,
            cfg,
            "foot_clearance",
            feet_clearance,
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
            envs_mdp.action_rate_l2,
        )
        self._add_reward(
            terms,
            cfg,
            "smoothness",
            envs_mdp.action_acc_l2,
        )
        self._add_reward(
            terms,
            cfg,
            "feet_air_time",
            feet_air_time,
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
            self_collision_cost,
            {
                "sensor_name": "nonfoot_ground_touch",
                "force_threshold": float(cfg.rewards.max_contact_force),
            },
        )
        self._add_reward(
            terms,
            cfg,
            "stand_still",
            stand_still,
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
            envs_mdp.joint_torques_l2,
            {"asset_cfg": actuator_cfg},
        )
        self._add_reward(
            terms,
            cfg,
            "dof_vel",
            envs_mdp.joint_vel_l2,
            {"asset_cfg": joint_cfg},
        )
        self._add_reward(
            terms,
            cfg,
            "dof_pos_limits",
            envs_mdp.joint_pos_limits,
            {"asset_cfg": joint_cfg},
        )
        self._add_reward(
            terms,
            cfg,
            "torque_limits",
            _torque_limit_cost,
            {
                "soft_limit": float(cfg.rewards.soft_torque_limit),
                "asset_cfg": actuator_cfg,
            },
        )
        self._add_reward(
            terms,
            cfg,
            "hip_pos",
            hip_pos,
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
            ),
            "projected_gravity": ObservationTermCfg(
                func=envs_mdp.projected_gravity,
                params={"asset_cfg": SceneEntityCfg(entity_name)},
                noise=self._noise(cfg, "gravity", noise_enabled),
            ),
            "command": ObservationTermCfg(
                func=envs_mdp.generated_commands,
                params={"command_name": "twist"},
            ),
            "joint_pos": ObservationTermCfg(
                func=envs_mdp.joint_pos_rel,
                params={"asset_cfg": joint_cfg},
                noise=self._noise(cfg, "dof_pos", noise_enabled),
                scale=float(cfg.normalization.obs_scales.dof_pos),
            ),
            "joint_vel": ObservationTermCfg(
                func=envs_mdp.joint_vel_rel,
                params={"asset_cfg": joint_cfg},
                noise=self._noise(cfg, "dof_vel", noise_enabled),
                scale=float(cfg.normalization.obs_scales.dof_vel),
            ),
            "actions": ObservationTermCfg(func=envs_mdp.last_action),
        }

        critic_terms = {
            **actor_terms,
            "base_lin_vel": ObservationTermCfg(
                func=envs_mdp.builtin_sensor,
                params={"sensor_name": f"{entity_name}/imu_lin_vel"},
                scale=float(cfg.normalization.obs_scales.lin_vel),
            ),
            "foot_contact": ObservationTermCfg(
                func=_foot_contact_observation,
                params={"sensor_name": "feet_ground_contact"},
            ),
            "height_scan": ObservationTermCfg(
                func=envs_mdp.height_scan,
                params={"sensor_name": "height_scan"},
                scale=float(cfg.normalization.obs_scales.height_measurements),
            ),
        }

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
                func=illegal_contact,
                params={
                    "sensor_name": "base_ground_touch",
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

本文件的完整替换代码为了保持“单文件可复制”，直接采用了 Unitree 的函数命名和函数签名，例如 `track_linear_velocity`、`feet_air_time`、`feet_clearance`、`self_collision_cost` 和 `stand_still`；这些实现按 `src/tasks/velocity/mdp/rewards.py` 对齐。若以后把奖励拆到独立的 `him_go2/mdp/rewards.py`，可以原样移动这些函数，然后在环境配置中以 `mdp.<function>` 注册，不要再引入 `velocity_mdp` 这个别名。

本次修订的关键映射如下：

| 旧 reward 名 | 修订后的函数 | 实现来源/语义 |
| --- | --- | --- |
| `tracking_lin_vel` | `track_linear_velocity` | 与 Unitree 相同的指数型线速度跟踪，额外抑制 base 的 z 速度 |
| `tracking_ang_vel` | `track_angular_velocity` | 与 Unitree 相同的 yaw 跟踪，同时弱惩罚 xy 角速度 |
| `lin_vel_z` / `ang_vel_xy` | `lin_vel_z` / `ang_vel_xy` | 补回旧配置中原本非零、但上一版漏注册的两项 |
| `orientation` | `body_orientation_l2` | 用 projected gravity 的 xy 分量衡量身体倾斜 |
| `feet_air_time` | `feet_air_time` | 使用 Unitree 的 `threshold`；不再传不存在的 `threshold_min/threshold_max` |
| `foot_clearance` | `feet_clearance` | 直接采用 Unitree 的 foot site 高度与水平速度实现，不把 `height_scan` 误传给官方同名函数 |
| `collision` | `self_collision_cost` | 使用 `nonfoot_ground_touch` 的 force history 和 `max_contact_force` |
| `stand_still` | `stand_still` | 低速命令时惩罚相对默认姿态的关节误差 |
| `hip_pos` | `hip_pos` | 修正为 hip 相对默认关节姿态的 L2，不再错误复用 `joint_pos_limits` |

`dof_acc`、`action_rate`、`smoothness`、`torques`、`dof_vel` 和 `dof_pos_limits` 仍显式使用 `envs_mdp` 中的通用 mjlab term；这与 Unitree 的 `mdp/__init__.py` 重新导出通用项的效果相同，但来源更清楚。

### 4.2 代码中的有意映射

有几个旧配置字段不能机械地一对一搬运：

- `clearance_height_target=-0.2` 是旧项目的高度约定；Unitree 的 `feet_clearance` 直接使用 `site_pos_w`，因此代码把旧值转换为正的 `0.2 m` 目标。这个值训练后仍应调参。
- 旧的 `torque_limits` 不是一个标准 mjlab reward term；代码使用每个 `IdealPdActuator` 的 `force_limit` 做超限惩罚。物理 actuator 限幅本身仍由 `Go2Asset` 完成。
- `tracking_sigma=0.25` 表示指数跟踪函数的 `std=sqrt(tracking_sigma)`；这与 Unitree 配置中显式写 `math.sqrt(0.25)` 的语义一致。
- `feet_air_time` 依赖 `feet_ground_contact` 开启 `track_air_time=True`，`collision` 与 `base_contact` 分别依赖接触传感器的 `force_history`；因此这两个 reward/termination 不能只注册函数而不配置对应 sensor field/history。
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
cd /home/aira/docs/legged_mjlab
source .venv/bin/activate
python -m pip install -e .
```

然后先验证 runner 来源，不要直接启动 4096 个环境：

```bash
cd /home/aira/docs/legged_mjlab
source .venv/bin/activate
PYTHONPATH=/home/aira/docs/legged_mjlab python - <<'PY'
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
cd /home/aira/docs/legged_mjlab
source .venv/bin/activate
PYTHONPATH=/home/aira/docs/legged_mjlab python -m legged_mjlab.scripts.train \
  --task him_go2 \
  --device cpu \
  --num-envs 4 \
  --max-iterations 1 \
  --log-dir /tmp/legged_mjlab_logs
```

确认 CPU smoke test 通过后，再切换到 GPU 和正式数量：

```bash
PYTHONPATH=/home/aira/docs/legged_mjlab python -m legged_mjlab.scripts.train \
  --task him_go2 \
  --device cuda:0 \
  --num-envs 4096 \
  --log-dir logs
```

`HIMRslRlWrapper` 的 `step()` 返回 7 项：

```python
(
    history_obs,              # [N, 270]
    privileged_obs,           # [N, 239]
    rewards,                  # [N]
    dones,                    # [N]
    infos,
    termination_ids,          # [K]
    termination_privileged,   # [K, 239] 或空 tensor
)
```

这正是项目内 `HIMOnPolicyRunner._unpack_step()` 支持的 terminal privileged observation 路径。wrapper 会把原生 `ManagerBasedRlEnvCfg.auto_reset` 改为 `False`，先保留真正的 terminal critic frame，再只 reset 已结束的环境；这一步不能删掉，否则 timeout bootstrap 会使用 reset 后的错误状态。

## 6. 训练前的 smoke test

### 6.1 静态检查

手工把代码复制到目标 `.py` 后执行：

```bash
cd /home/aira/docs/legged_mjlab
source .venv/bin/activate
python -m py_compile legged_mjlab/envs/him_go2/him_go2_env.py
```

### 6.2 manager 配置检查

```bash
cd /home/aira/docs/legged_mjlab
source .venv/bin/activate
PYTHONPATH=/home/aira/docs/legged_mjlab python - <<'PY'
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
assert manager_cfg.actions["joint_position"].actuator_names
print("manager config construction: OK")
PY
```

### 6.3 真正环境的 shape 检查

```bash
cd /home/aira/docs/legged_mjlab
source .venv/bin/activate
PYTHONPATH=/home/aira/docs/legged_mjlab python - <<'PY'
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
assert tuple(privileged.shape) == (4, 239), privileged.shape
assert env.num_actions == 12

action = torch.zeros((4, 12), device=history.device, dtype=history.dtype)
step_result = env.step(action)
next_history, next_privileged = step_result[:2]
assert tuple(next_history.shape) == (4, 270), next_history.shape
assert tuple(next_privileged.shape) == (4, 239), next_privileged.shape
print("HIM environment shape contract: OK")
PY
```

### 6.4 失败时按顺序定位

1. `SyntaxError`：目标文件没有完整替换，或把 Markdown fence 一并复制进了 Python 文件。
2. `TypeError: unexpected keyword actuator_name`：动作配置仍是旧代码，必须使用 `actuator_names`。
3. `ValueError: no matching names`：先打印 `Go2Asset` 的 `entity.joint_names`、`entity.actuator_names`、`entity.site_names`，核对 XML 名称；当前 XML 的足端 site 是 `FL/FR/RL/RR`，接触 geom 是 `*_foot_collision`。
4. critic 维度不是 239：检查 actor term 顺序、height scan 网格是否为 17×11，以及是否把 history 放进了原生 mjlab group。
5. wrapper 报 `auto_reset=False`：说明底层环境配置没有暴露 `ManagerBasedRlEnvCfg.auto_reset`，不能继续训练，必须先解决 terminal observation 保留问题。
6. `rsl_rl` 来源断言失败：运行时加载的是另一个兼容包；先处理包来源，再排查 PPO 参数。
7. GPU 启动失败但 CPU 通过：先降低 `num_envs`，确认 mujoco-warp、CUDA 和 `mujoco_warp` 版本，再恢复 4096。

## 7. 与参考实现的对应关系

- `unitree_rl_mjlab/src/tasks/velocity/velocity_env_cfg.py` 提供了当前 mjlab 1.6.0 的 manager 配置组织方式：scene、sensor、observation、action、command、event、reward、termination、curriculum 都在一个 flat `ManagerBasedRlEnvCfg` 中组装。
- `unitree_rl_mjlab/src/tasks/velocity/mdp/rewards.py` 才是 Unitree velocity task 的任务特有奖励实现；其中的 `track_linear_velocity`、`feet_clearance`、`feet_air_time` 和 `stand_still` 通过 `RewardTermCfg(func=mdp.<name>, ...)` 注册。
- `unitree_rl_mjlab/src/tasks/velocity/config/go2/env_cfgs.py` 提供了 Go2 的实际接触 geom、site、raycast frame、rough-terrain play override 和 `illegal_contact` 组织方式。
- `AMP_mjlab/src/tasks/amp_loco/amp_env_cfg.py` 与其 MDP term 展示了 `RewardTermCfg`、`TerminationTermCfg`、`SceneEntityCfg` 的可复制写法。
- `legged_gym` 的 legacy 配置仍然适合用来核对 observation/reward 的语义和维度，但不能直接把它的 manager、clip、reset 或旧 sensor 字段搬进 mjlab。
- 官方 mjlab 文档中的 [environment config](https://mujocolab.github.io/mjlab/main/source/environment_config.html)、[observations](https://mujocolab.github.io/mjlab/main/source/observations.html)、[actions](https://github.com/mujocolab/mjlab/blob/main/docs/source/actions.rst)、[events](https://mujocolab.github.io/mjlab/main/source/events.html)、[rewards](https://mujocolab.github.io/mjlab/main/source/rewards.html)、[terminations](https://mujocolab.github.io/mjlab/main/source/terminations.html)、[terrain](https://mujocolab.github.io/mjlab/main/source/terrain.html) 是当前 API 对照入口。

## 8. 完成判据

这部分手工代码落地后，`him_go2` 才算达到“可训练”而不是“文件能导入”：

- Python 文件可编译；
- `manager_cfg` 可以创建，且 `MujocoCfg.integrator`、`gravity` 类型正确；
- XML 的 actuator、joint、site、geom 全部匹配；
- actor 为 `[N, 45]`，wrapper history 为 `[N, 270]`；
- critic 为 `[N, 239]`，其中前 48 维符合 HIM estimator；
- done 环境的 terminal privileged observation 能被 runner 使用；
- runtime `rsl_rl.__file__` 指向项目内 `legged_mjlab/rsl_rl/rsl_rl`；
- 4 环境 CPU、1 iteration smoke test 通过后，才启动 GPU/4096 环境正式训练。
