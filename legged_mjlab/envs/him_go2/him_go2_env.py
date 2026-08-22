import math

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as mjlab_mdp
from mjlab.managers.action_manager import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.tasks.velocity import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg

from legged_mjlab.envs.him_go2.go2_asset import Go2Asset
from legged_mjlab.envs.him_go2.him_go2_config import HimGo2RoughCfg, HimGo2RoughCfgPPO


class HimGo2Env(ManagerBasedRlEnv):
    def __init__(self, cfg=None, device=None, render_mode=None, play=False):
        self.task_cfg = cfg if cfg is not None else HimGo2RoughCfg()
        self.play = bool(play)
        manager_cfg = self._build_mjlab_cfg(self.task_cfg, self.play)
        selected_device = device or self.task_cfg.env.device
        super().__init__(
            cfg=manager_cfg,
            device=selected_device,
            render_mode=render_mode,
        )

    @classmethod
    def _build_mjlab_cfg(cls, cfg, play=False):
        asset = Go2Asset(cfg, entity_name=cfg.asset.name)
        if cfg.terrain.mesh_type != "plane":
            asset.add_terrain_scan_sensor(debug_vis=not play)
            asset.add_foot_contact_sensor()
        asset.validate()

        actor_terms = {
            "imu_ang_vel": ObservationTermCfg(
                func=cls._obs_imu_ang_vel,
                params={"sensor_name": "robot/imu_ang_vel"},
                noise=UniformNoiseCfg(
                    n_min=-cfg.noise.imu_ang_vel,
                    n_max=cfg.noise.imu_ang_vel,
                ),
            ),
            "projected_gravity": ObservationTermCfg(
                func=cls._obs_projected_gravity,
                noise=UniformNoiseCfg(
                    n_min=-cfg.noise.projected_gravity,
                    n_max=cfg.noise.projected_gravity,
                ),
            ),
            "joint_pos": ObservationTermCfg(
                func=cls._obs_joint_pos_rel,
                params={"asset_cfg": SceneEntityCfg(
                    name=cfg.asset.name,
                    joint_names=(".*",),
                )},
                noise=UniformNoiseCfg(
                    n_min=-cfg.noise.joint_pos,
                    n_max=cfg.noise.joint_pos,
                ),
            ),
            "joint_vel": ObservationTermCfg(
                func=cls._obs_joint_vel,
                params={"asset_cfg": SceneEntityCfg(
                    name=cfg.asset.name,
                    joint_names=(".*",),
                )},
                noise=UniformNoiseCfg(
                    n_min=-cfg.noise.joint_vel,
                    n_max=cfg.noise.joint_vel,
                ),
            ),
            "commands": ObservationTermCfg(func=cls._obs_commands),
            "last_action": ObservationTermCfg(func=cls._obs_last_action),
        }
        critic_terms = dict(actor_terms)
        critic_terms["base_lin_vel"] = ObservationTermCfg(
            func=cls._obs_base_lin_vel
        )
        if cfg.terrain.mesh_type != "plane":
            actor_terms["height_scan"] = ObservationTermCfg(
                func=cls._obs_height_scan,
                params={"sensor_name": "terrain_scan"},
                scale=0.2,
            )
            critic_terms["height_scan"] = actor_terms["height_scan"]

        observations = {
            "actor": ObservationGroupCfg(
                terms=actor_terms,
                concatenate_terms=True,
                enable_corruption=cfg.noise.add_noise and not play,
                history_length=cfg.him.history_length,
            ),
            "critic": ObservationGroupCfg(
                terms=critic_terms,
                concatenate_terms=True,
                enable_corruption=False,
                history_length=1,
            ),
        }

        terrain = TerrainEntityCfg(
            terrain_type=cfg.terrain.mesh_type,
            terrain_generator=ROUGH_TERRAINS_CFG
            if cfg.terrain.mesh_type != "plane"
            else None,
            max_init_terrain_level=cfg.terrain.max_init_terrain_level,
        )
        scene = SceneCfg(
            num_envs=cfg.env.num_envs,
            env_spacing=cfg.scene.env_spacing,
            entities={cfg.asset.name: asset.build_entity_cfg()},
            sensors=asset.build_scene_sensors(),
            terrain=terrain,
            replicate_physics=cfg.scene.replicate_physics,
        )

        return ManagerBasedRlEnvCfg(
            decimation=cfg.control.decimation,
            episode_length_s=cfg.env.episode_length_s,
            scene=scene,
            observations=observations,
            actions={
                "joint_position": JointPositionActionCfg(
                    entity_name=cfg.asset.name,
                    actuator_names=(".*",),
                    scale=cfg.control.action_scale,
                    clip=cfg.control.action_clip,
                    use_default_offset=True,
                )
            },
            commands=cls._build_commands(cfg),
            rewards=cls._build_rewards(cfg),
            terminations=cls._build_terminations(cfg),
            events=cls._build_events(cfg),
            curriculum=cls._build_curriculum(cfg),
        )

    @classmethod
    def _build_commands(cls, cfg):
        return {
            "twist": UniformVelocityCommandCfg(
                entity_name=cfg.asset.name,
                resampling_time=cfg.commands.resampling_time,
                heading_command=cfg.commands.heading_command,
                ranges=UniformVelocityCommandCfg.Ranges(
                    lin_vel_x=cfg.commands.lin_vel_x,
                    lin_vel_y=cfg.commands.lin_vel_y,
                    ang_vel_z=cfg.commands.ang_vel_yaw,
                    heading=cfg.commands.heading,
                ),
            )
        }

    @classmethod
    def _build_rewards(cls, cfg):
        return {
            "track_linear_velocity": RewardTermCfg(
                func=cls._reward_track_linear_velocity,
                weight=cfg.rewards.tracking_lin_vel,
                params={"std": cfg.rewards.tracking_std},
            ),
            "track_angular_velocity": RewardTermCfg(
                func=cls._reward_track_angular_velocity,
                weight=cfg.rewards.tracking_ang_vel,
                params={"std": cfg.rewards.tracking_std},
            ),
            "lin_vel_z": RewardTermCfg(
                func=cls._reward_base_lin_vel_z,
                weight=cfg.rewards.lin_vel_z,
            ),
            "orientation": RewardTermCfg(
                func=cls._reward_orientation,
                weight=cfg.rewards.orientation,
            ),
            "action_rate": RewardTermCfg(
                func=cls._reward_action_rate,
                weight=cfg.rewards.action_rate,
            ),
            "hip_reduction": RewardTermCfg(
                func=cls._reward_hip_reduction,
                weight=cfg.rewards.hip_reduction,
            ),
            "termination": RewardTermCfg(
                func=cls._reward_termination,
                weight=cfg.rewards.termination,
            ),
        }

    @classmethod
    def _build_terminations(cls, cfg):
        return {
            "time_out": TerminationTermCfg(
                func=cls._termination_time_out,
                time_out=cfg.terminations.time_out,
            ),
            "base_height": TerminationTermCfg(
                func=cls._termination_base_height,
                params={"minimum": cfg.terminations.base_height},
                time_out=False,
            ),
            "roll_pitch": TerminationTermCfg(
                func=cls._termination_roll_pitch,
                params={"maximum": cfg.terminations.roll_pitch},
                time_out=False,
            ),
            "illegal_contact": TerminationTermCfg(
                func=cls._termination_illegal_contact,
                params={"minimum_force": cfg.terminations.illegal_contact_force},
                time_out=False,
            ),
        }

    @classmethod
    def _build_events(cls, cfg):
        events = {
            "reset_robot": EventTermCfg(
                func=cls._event_reset_robot,
                mode="reset",
            ),
        }
        if cfg.domain_rand.randomize_friction:
            events["randomize_friction"] = EventTermCfg(
                func=cls._event_randomize_friction,
                mode="startup",
                params={"range": cfg.domain_rand.friction_range},
            )
        if cfg.domain_rand.push_robots:
            events["push_robot"] = EventTermCfg(
                func=cls._event_push_robot,
                mode="interval",
                interval_range_s=(
                    cfg.domain_rand.push_interval_s,
                    cfg.domain_rand.push_interval_s,
                ),
                params={"max_velocity": cfg.domain_rand.max_push_vel_xy},
            )
        return events

    @classmethod
    def _build_curriculum(cls, cfg):
        if not cfg.terrain.curriculum:
            return {}
        return {
            "terrain_level": cls._curriculum_terrain_level,
        }

    @staticmethod
    def _require_tensor(value, name):
        if not isinstance(value, torch.Tensor):
            raise TypeError(name + " must be a torch.Tensor")
        if not torch.isfinite(value).all():
            raise ValueError(name + " contains NaN or Inf")
        return value

    @staticmethod
    def _sensor(env, sensor_name):
        sensors = getattr(env.scene, "sensors", {})
        if sensor_name not in sensors:
            raise KeyError("sensor is not registered: " + sensor_name)
        sensor = sensors[sensor_name]
        data = getattr(sensor, "data", sensor)
        value = getattr(data, "value", data)
        return Go2Env._require_tensor(value, sensor_name)

    @staticmethod
    def _robot(env, entity_name="robot"):
        try:
            return env.scene[entity_name]
        except (KeyError, TypeError) as exc:
            raise KeyError("robot entity is not registered: " + entity_name) from exc

    @staticmethod
    def _obs_imu_ang_vel(env, sensor_name):
        value = Go2Env._sensor(env, sensor_name)
        if value.ndim != 2 or value.shape[-1] != 3:
            raise ValueError("imu angular velocity must have shape [N, 3]")
        return value

    @staticmethod
    def _obs_projected_gravity(env):
        robot = Go2Env._robot(env)
        value = robot.data.projected_gravity_b
        if value.shape[-1] != 3:
            raise ValueError("projected gravity must have shape [N, 3]")
        return Go2Env._require_tensor(value, "projected_gravity_b")

    @staticmethod
    def _obs_joint_pos_rel(env, asset_cfg):
        robot = Go2Env._robot(env, asset_cfg.name)
        joint_ids = robot.find_joints(asset_cfg.joint_names)[0]
        value = robot.data.joint_pos[:, joint_ids]
        return Go2Env._require_tensor(value, "joint_pos_rel")

    @staticmethod
    def _obs_joint_vel(env, asset_cfg):
        robot = Go2Env._robot(env, asset_cfg.name)
        joint_ids = robot.find_joints(asset_cfg.joint_names)[0]
        value = robot.data.joint_vel[:, joint_ids]
        return Go2Env._require_tensor(value, "joint_vel")

    @staticmethod
    def _obs_commands(env):
        value = env.command_manager.get_command("twist")
        return Go2Env._require_tensor(value, "commands")

    @staticmethod
    def _obs_last_action(env):
        value = env.action_manager.action
        return Go2Env._require_tensor(value, "last_action")

    @staticmethod
    def _obs_base_lin_vel(env):
        value = Go2Env._robot(env).data.root_lin_vel_b
        return Go2Env._require_tensor(value, "root_lin_vel_b")

    @staticmethod
    def _obs_height_scan(env, sensor_name):
        value = Go2Env._sensor(env, sensor_name)
        if value.ndim == 3:
            value = value.reshape(value.shape[0], -1)
        if value.ndim != 2:
            raise ValueError("height scan must have shape [N, D]")
        return value

    @staticmethod
    def _reward_track_linear_velocity(env, std):
        command = env.command_manager.get_command("twist")[:, :2]
        measured = Go2Env._robot(env).data.root_lin_vel_b[:, :2]
        error = torch.sum(torch.square(command - measured), dim=-1)
        return torch.exp(-error / (2.0 * std * std))

    @staticmethod
    def _reward_track_angular_velocity(env, std):
        command = env.command_manager.get_command("twist")[:, 2]
        measured = Go2Env._robot(env).data.root_ang_vel_b[:, 2]
        error = torch.square(command - measured)
        return torch.exp(-error / (2.0 * std * std))

    @staticmethod
    def _reward_base_lin_vel_z(env):
        value = Go2Env._robot(env).data.root_lin_vel_b[:, 2]
        return torch.square(value)

    @staticmethod
    def _reward_orientation(env):
        gravity = Go2Env._robot(env).data.projected_gravity_b
        return torch.sum(torch.square(gravity[:, :2]), dim=-1)

    @staticmethod
    def _reward_action_rate(env):
        action = env.action_manager.action
        previous = env.action_manager.prev_action
        return torch.sum(torch.square(action - previous), dim=-1)

    @staticmethod
    def _reward_hip_reduction(env):
        robot = Go2Env._robot(env)
        joint_ids = robot.find_joints((".*_hip_joint",))[0]
        hip_yaw = robot.data.joint_pos[:, joint_ids]
        return torch.sum(torch.square(hip_yaw), dim=-1)

    @staticmethod
    def _reward_termination(env):
        return env.termination_manager.terminated.float()

    @staticmethod
    def _termination_time_out(env):
        return env.episode_length_buf >= env.max_episode_length

    @staticmethod
    def _termination_base_height(env, minimum):
        height = Go2Env._robot(env).data.root_pos_w[:, 2]
        return height < minimum

    @staticmethod
    def _termination_roll_pitch(env, maximum):
        gravity = Go2Env._robot(env).data.projected_gravity_b[:, :2]
        return torch.linalg.vector_norm(gravity, dim=-1) > maximum

    @staticmethod
    def _termination_illegal_contact(env, minimum_force):
        sensor = Go2Env._sensor(env, "feet_ground_contact")
        force = torch.linalg.vector_norm(sensor, dim=-1)
        return torch.any(force > minimum_force, dim=-1)

    @staticmethod
    def _event_reset_robot(env, env_ids):
        robot = Go2Env._robot(env)
        count = len(env_ids)
        position = torch.as_tensor(
            env.task_cfg.init_state.pos,
            device=robot.data.root_pos_w.device,
        ).expand(count, 3)
        velocity = torch.zeros((count, 6), device=position.device)
        robot.write_root_pose_to_sim(
            torch.cat((position, torch.as_tensor(
                env.task_cfg.init_state.rot,
                device=position.device,
            ).expand(count, 4)), dim=-1),
            env_ids,
        )
        robot.write_root_velocity_to_sim(velocity, env_ids)
        joint_ids = robot.find_joints((".*",))[0]
        joint_position = torch.zeros(
            (count, len(joint_ids)),
            device=position.device,
        )
        for index, name in enumerate(robot.joint_names):
            if name in env.task_cfg.init_state.default_joint_angles:
                joint_position[:, index] = env.task_cfg.init_state.default_joint_angles[name]
        robot.write_joint_state_to_sim(
            joint_position,
            torch.zeros_like(joint_position),
            env_ids,
            joint_ids,
        )

    @staticmethod
    def _event_randomize_friction(env, range):
        lower, upper = range
        friction = torch.empty(
            (env.num_envs, 1),
            device=env.device,
        ).uniform_(lower, upper)
        env.scene["robot"].write_material_properties_to_sim(
            friction=friction
        )

    @staticmethod
    def _event_push_robot(env, max_velocity):
        robot = Go2Env._robot(env)
        velocity = torch.empty(
            (env.num_envs, 2),
            device=env.device,
        ).uniform_(-max_velocity, max_velocity)
        root_velocity = robot.data.root_vel_w.clone()
        root_velocity[:, :2] = root_velocity[:, :2] + velocity
        robot.write_root_velocity_to_sim(root_velocity)

    @staticmethod
    def _curriculum_terrain_level(env):
        command = env.command_manager.get_command("twist")
        progress = torch.linalg.vector_norm(command[:, :2], dim=-1)
        env.scene.terrain.update_env_origins(progress)

