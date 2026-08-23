"""mjlab 1.6.0 environment adapter for the HIM Go2 task.

The native environment deliberately exposes one actor frame.  The HIM wrapper
is responsible for stacking six frames, while the critic receives the actor
frame plus base linear velocity (45 + 3).
"""

import math

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as mjlab_mdp
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
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg

from legged_mjlab.envs.him_go2.go2_asset import Go2Asset
from legged_mjlab.envs.him_go2.him_go2_config import HimGo2RoughCfg


class HimGo2Env(ManagerBasedRlEnv):
    """Construct a Go2 manager-based environment with the project config."""

    def __init__(self, cfg=None, device=None, render_mode=None, play=False):
        self.task_cfg = cfg if cfg is not None else HimGo2RoughCfg()
        self.play = bool(play)
        manager_cfg = self._build_mjlab_cfg(self.task_cfg, play=self.play)
        selected_device = device if device is not None else self.task_cfg.env.device
        super().__init__(
            cfg=manager_cfg,
            device=selected_device,
            render_mode=render_mode,
        )

    @staticmethod
    def _entity_cfg(cfg):
        return SceneEntityCfg(
            name=cfg.asset.name,
            joint_names=Go2Asset.Names.joint_order,
            preserve_order=True,
        )

    @staticmethod
    def _noise_range(value, level):
        magnitude = abs(float(value)) * abs(float(level))
        if magnitude == 0.0:
            return None
        return UniformNoiseCfg(n_min=-magnitude, n_max=magnitude)

    @classmethod
    def _build_mjlab_cfg(cls, cfg, play=False):
        asset = Go2Asset(cfg, entity_name=cfg.asset.name)

        # Contact sensors are required by the termination/reward contract on
        # every terrain, including the default plane.  Only the optional terrain
        # ray scan is conditional on rough terrain.
        terrain_type = "plane" if cfg.terrain.mesh_type == "plane" else "generator"
        asset.add_foot_contact_sensor()
        asset.add_illegal_contact_sensor()
        if terrain_type != "plane":
            asset.add_terrain_scan_sensor(debug_vis=not play)
        asset.validate(require_effort_limits=True)

        entity_cfg = cls._entity_cfg(cfg)
        actor_terms = {
            # Keep this insertion order in sync with the deploy policy.
            "commands": ObservationTermCfg(
                func=mjlab_mdp.generated_commands,
                params={"command_name": "twist"},
                scale=tuple(cfg.commands.scale),
                noise=cls._noise_range(
                    getattr(cfg.noise, "command", 0.0),
                    getattr(cfg.noise, "noise_level", 1.0),
                ),
            ),
            "angular_velocity": ObservationTermCfg(
                func=mjlab_mdp.builtin_sensor,
                params={"sensor_name": f"{cfg.asset.name}/imu_ang_vel"},
                scale=0.25,
                noise=cls._noise_range(
                    cfg.noise.imu_ang_vel,
                    getattr(cfg.noise, "noise_level", 1.0),
                ),
            ),
            "projected_gravity": ObservationTermCfg(
                func=mjlab_mdp.projected_gravity,
                params={"asset_cfg": SceneEntityCfg(name=cfg.asset.name)},
                noise=cls._noise_range(
                    cfg.noise.projected_gravity,
                    getattr(cfg.noise, "noise_level", 1.0),
                ),
            ),
            "joint_pos": ObservationTermCfg(
                func=mjlab_mdp.joint_pos_rel,
                params={"asset_cfg": entity_cfg},
                scale=1.0,
                noise=cls._noise_range(
                    cfg.noise.joint_pos,
                    getattr(cfg.noise, "noise_level", 1.0),
                ),
            ),
            "joint_vel": ObservationTermCfg(
                func=mjlab_mdp.joint_vel_rel,
                params={"asset_cfg": entity_cfg},
                scale=0.05,
                noise=cls._noise_range(
                    cfg.noise.joint_vel,
                    getattr(cfg.noise, "noise_level", 1.0),
                ),
            ),
            "last_action": ObservationTermCfg(
                func=mjlab_mdp.last_action,
                params={"action_name": None},
            ),
        }
        critic_terms = dict(actor_terms)
        critic_terms["base_lin_vel"] = ObservationTermCfg(
            func=mjlab_mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg(name=cfg.asset.name)},
        )

        observations = {
            "actor": ObservationGroupCfg(
                terms=actor_terms,
                concatenate_terms=True,
                enable_corruption=bool(
                    getattr(cfg.noise, "add_noise", False) and not play
                ),
                # History belongs to HIMRslRlWrapper, not the native manager.
                history_length=None,
                nan_policy="error",
            ),
            "critic": ObservationGroupCfg(
                terms=critic_terms,
                concatenate_terms=True,
                enable_corruption=False,
                history_length=None,
                nan_policy="error",
            ),
        }

        terrain = TerrainEntityCfg(
            terrain_type=terrain_type,
            terrain_generator=ROUGH_TERRAINS_CFG if terrain_type != "plane" else None,
            max_init_terrain_level=(
                cfg.terrain.max_init_terrain_level if terrain_type != "plane" else None
            ),
            debug_vis=bool(getattr(cfg.terrain, "debug_vis", False) and not play),
        )
        scene = SceneCfg(
            num_envs=int(cfg.env.num_envs),
            env_spacing=float(cfg.env.env_spacing),
            entities={cfg.asset.name: asset.build_entity_cfg()},
            sensors=asset.build_scene_sensors(),
            terrain=terrain,
        )

        return ManagerBasedRlEnvCfg(
            decimation=int(cfg.control.decimation),
            episode_length_s=float(cfg.env.episode_length_s),
            scene=scene,
            observations=observations,
            actions={
                "joint_position": JointPositionActionCfg(
                    entity_name=cfg.asset.name,
                    actuator_names=(r".*",),
                    scale=float(cfg.control.action_scale),
                    clip=dict(cfg.control.action_clip),
                    use_default_offset=True,
                    preserve_order=True,
                )
            },
            commands=cls._build_commands(cfg, play=play),
            rewards=cls._build_rewards(cfg),
            terminations=cls._build_terminations(cfg),
            events=cls._build_events(cfg),
            curriculum=cls._build_curriculum(cfg),
            seed=int(cfg.env.seed),
            sim=SimulationCfg(
                mujoco=MujocoCfg(
                    timestep=float(cfg.sim.dt),
                    gravity=tuple(cfg.sim.gravity),
                )
            ),
            scale_rewards_by_dt=False,
        )

    @classmethod
    def _build_commands(cls, cfg, play=False):
        heading = tuple(cfg.commands.heading) if cfg.commands.heading_command else None
        return {
            "twist": UniformVelocityCommandCfg(
                resampling_time_range=tuple(cfg.commands.resampling_time_range),
                entity_name=cfg.asset.name,
                heading_command=bool(cfg.commands.heading_command),
                heading_control_stiffness=float(cfg.commands.heading_control_stiffness),
                rel_standing_envs=float(cfg.commands.rel_standing_envs),
                rel_heading_envs=float(cfg.commands.rel_heading_envs),
                rel_world_envs=float(getattr(cfg.commands, "rel_world_envs", 0.0)),
                rel_forward_envs=float(getattr(cfg.commands, "rel_forward_envs", 0.0)),
                init_velocity_prob=float(getattr(cfg.commands, "init_velocity_prob", 0.0)),
                debug_vis=bool(getattr(cfg.commands, "debug_vis", False) and not play),
                ranges=UniformVelocityCommandCfg.Ranges(
                    lin_vel_x=tuple(cfg.commands.lin_vel_x),
                    lin_vel_y=tuple(cfg.commands.lin_vel_y),
                    ang_vel_z=tuple(cfg.commands.ang_vel_yaw),
                    heading=heading,
                ),
            )
        }

    @staticmethod
    def _weight(rewards, name):
        return float(getattr(rewards, name, 0.0))

    @classmethod
    def _build_rewards(cls, cfg):
        rewards = cfg.rewards
        robot_cfg = SceneEntityCfg(name=cfg.asset.name)
        terms = {}

        if cls._weight(rewards, "tracking_lin_vel") != 0.0:
            terms["track_linear_velocity"] = RewardTermCfg(
                func=velocity_mdp.track_linear_velocity,
                weight=cls._weight(rewards, "tracking_lin_vel"),
                params={
                    "std": float(getattr(rewards, "tracking_std", 0.5)),
                    "command_name": "twist",
                    "asset_cfg": robot_cfg,
                },
            )
        if cls._weight(rewards, "tracking_ang_vel") != 0.0:
            terms["track_angular_velocity"] = RewardTermCfg(
                func=velocity_mdp.track_angular_velocity,
                weight=cls._weight(rewards, "tracking_ang_vel"),
                params={
                    "std": float(getattr(rewards, "tracking_std", 0.5)),
                    "command_name": "twist",
                    "asset_cfg": robot_cfg,
                },
            )
        if cls._weight(rewards, "lin_vel_z") != 0.0:
            terms["lin_vel_z"] = RewardTermCfg(
                func=cls._reward_lin_vel_z,
                weight=cls._weight(rewards, "lin_vel_z"),
            )
        if cls._weight(rewards, "orientation") != 0.0:
            terms["orientation"] = RewardTermCfg(
                func=mjlab_mdp.flat_orientation_l2,
                weight=cls._weight(rewards, "orientation"),
                params={"asset_cfg": robot_cfg},
            )
        if cls._weight(rewards, "action_rate") != 0.0:
            terms["action_rate"] = RewardTermCfg(
                func=mjlab_mdp.action_rate_l2,
                weight=cls._weight(rewards, "action_rate"),
            )
        if cls._weight(rewards, "hip_reduction") != 0.0:
            terms["hip_reduction"] = RewardTermCfg(
                func=cls._reward_hip_reduction,
                weight=cls._weight(rewards, "hip_reduction"),
            )
        if cls._weight(rewards, "termination") != 0.0:
            terms["termination"] = RewardTermCfg(
                func=mjlab_mdp.is_terminated,
                weight=cls._weight(rewards, "termination"),
            )
        if cls._weight(rewards, "dof_pos_limits") != 0.0:
            terms["dof_pos_limits"] = RewardTermCfg(
                func=mjlab_mdp.joint_pos_limits,
                weight=cls._weight(rewards, "dof_pos_limits"),
                params={"asset_cfg": robot_cfg},
            )

        # This term is enabled only when the corresponding sensor exists.  The
        # default plane mode intentionally has no contact sensor.
        if cls._weight(rewards, "feet_air_time") != 0.0:
            terms["feet_air_time"] = RewardTermCfg(
                func=velocity_mdp.feet_air_time,
                weight=cls._weight(rewards, "feet_air_time"),
                params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
            )
        return terms

    @classmethod
    def _build_terminations(cls, cfg):
        terms = {
            "time_out": TerminationTermCfg(
                func=mjlab_mdp.time_out,
                time_out=bool(cfg.terminations.time_out),
            ),
        }
        robot_cfg = SceneEntityCfg(name=cfg.asset.name)
        if float(cfg.terminations.base_height) > 0.0:
            terms["base_height"] = TerminationTermCfg(
                func=mjlab_mdp.root_height_below_minimum,
                params={
                    "minimum_height": float(cfg.terminations.base_height),
                    "asset_cfg": robot_cfg,
                },
                time_out=False,
            )
        if bool(cfg.terminations.bad_orientation):
            terms["bad_orientation"] = TerminationTermCfg(
                func=mjlab_mdp.bad_orientation,
                params={
                    "limit_angle": math.radians(
                        float(cfg.terminations.bad_orientation_limit_deg)
                    ),
                    "asset_cfg": robot_cfg,
                },
                time_out=False,
            )
        if bool(cfg.terminations.illegal_contact or cfg.terminations.base_contact):
            terms["illegal_contact"] = TerminationTermCfg(
                func=cls._termination_illegal_contact,
                params={
                    "force_threshold": float(cfg.terminations.illegal_contact_force)
                },
                time_out=False,
            )
        return terms

    @classmethod
    def _build_events(cls, cfg):
        events = {
            "reset_scene": EventTermCfg(
                func=mjlab_mdp.reset_scene_to_default,
                mode="reset",
            )
        }
        if bool(cfg.domain_rand.push_robots):
            max_velocity = abs(float(cfg.domain_rand.max_push_vel_xy))
            events["push_robot"] = EventTermCfg(
                func=mjlab_mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(
                    float(cfg.domain_rand.push_interval_s),
                    float(cfg.domain_rand.push_interval_s),
                ),
                params={
                    "velocity_range": {
                        "x": (-max_velocity, max_velocity),
                        "y": (-max_velocity, max_velocity),
                    },
                    "asset_cfg": SceneEntityCfg(name=cfg.asset.name),
                },
            )
        return events

    @classmethod
    def _build_curriculum(cls, cfg):
        if not bool(cfg.terrain.curriculum) or cfg.terrain.mesh_type == "plane":
            return {}
        return {
            "terrain_levels": CurriculumTermCfg(
                func=velocity_mdp.terrain_levels_vel,
                params={
                    "command_name": "twist",
                    "asset_cfg": SceneEntityCfg(name=cfg.asset.name),
                },
            )
        }

    @staticmethod
    def _finite(value, name):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        finite = torch.isfinite(value)
        if bool(finite.all()):
            return value

        if value.ndim == 0:
            env_ids = [0]
        else:
            invalid_envs = ~finite.reshape(value.shape[0], -1).all(dim=1)
            env_ids = torch.where(invalid_envs)[0].detach().cpu().tolist()
        raise FloatingPointError(
            f"NaN/Inf detected in observation '{name}' "
            f"for environments: {env_ids[:10]}"
        )

    @staticmethod
    def _robot(env, entity_name="robot"):
        return env.scene[entity_name]

    @staticmethod
    def _obs_imu_ang_vel(env, sensor_name):
        value = HimGo2Env._finite(mjlab_mdp.builtin_sensor(env, sensor_name), sensor_name)
        return value.reshape(value.shape[0], -1)

    @staticmethod
    def _obs_projected_gravity(env, asset_cfg=None):
        value = mjlab_mdp.projected_gravity(
            env, asset_cfg or SceneEntityCfg(name=Go2Asset.Names.entity)
        )
        return HimGo2Env._finite(value, "projected_gravity")

    @staticmethod
    def _obs_joint_pos_rel(env, asset_cfg):
        return HimGo2Env._finite(mjlab_mdp.joint_pos_rel(env, asset_cfg=asset_cfg), "joint_pos_rel")

    @staticmethod
    def _obs_joint_vel(env, asset_cfg):
        return HimGo2Env._finite(mjlab_mdp.joint_vel_rel(env, asset_cfg=asset_cfg), "joint_vel_rel")

    @staticmethod
    def _obs_commands(env):
        return HimGo2Env._finite(mjlab_mdp.generated_commands(env, command_name="twist"), "commands")

    @staticmethod
    def _obs_last_action(env):
        return HimGo2Env._finite(mjlab_mdp.last_action(env), "last_action")

    @staticmethod
    def _obs_base_lin_vel(env, asset_cfg=None):
        value = mjlab_mdp.base_lin_vel(
            env, asset_cfg or SceneEntityCfg(name=Go2Asset.Names.entity)
        )
        return HimGo2Env._finite(value, "base_lin_vel")

    @staticmethod
    def _obs_height_scan(env, sensor_name):
        # Kept as a compatibility helper for callers that explicitly request a
        # rough-terrain scan.  It is never part of the default 45-dim actor.
        sensor = env.scene[sensor_name]
        value = getattr(sensor.data, "distance", None)
        if value is None:
            value = getattr(sensor.data, "value", None)
        if value is None:
            raise AttributeError(f"height sensor {sensor_name!r} has no distance field")
        return HimGo2Env._finite(value.reshape(value.shape[0], -1), "height_scan")

    @staticmethod
    def _reward_lin_vel_z(env):
        value = HimGo2Env._robot(env).data.root_link_lin_vel_b[:, 2]
        return torch.nan_to_num(torch.square(value), nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _reward_hip_reduction(env):
        robot = HimGo2Env._robot(env)
        joint_ids, _ = robot.find_joints((r".*_hip_joint",), preserve_order=True)
        value = robot.data.joint_pos[:, joint_ids]
        return torch.nan_to_num(torch.sum(torch.square(value), dim=-1))

    @staticmethod
    def _termination_illegal_contact(env, force_threshold):
        sensor = env.scene.sensors.get("illegal_contact")
        if sensor is None:
            raise RuntimeError(
                "required contact sensor 'illegal_contact' is not registered; "
                "refusing to disable illegal-contact termination"
            )
        data = sensor.data
        force = getattr(data, "force", None)
        found = getattr(data, "found", None)
        if force is None and found is None:
            raise RuntimeError(
                "required contact sensor 'illegal_contact' exposes neither "
                "'force' nor 'found' data"
            )
        if force is not None:
            force = HimGo2Env._finite(force, "illegal_contact.force")
        if found is not None:
            found = HimGo2Env._finite(found, "illegal_contact.found")
        if force is not None:
            result = torch.any(
                torch.linalg.vector_norm(force, dim=-1) > float(force_threshold), dim=-1
            )
        else:
            result = torch.any(found, dim=-1)
        return result.to(dtype=torch.bool)


# A few old task entry points imported Go2Env.  Keep the alias local to this file
# instead of maintaining a second environment implementation.
Go2Env = HimGo2Env
