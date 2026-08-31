import torch
import copy
import math

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg 
from mjlab.entity import Entity
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.envs import mdp as mdp
from mjlab.tasks.velocity import mdp
from mjlab.envs.mdp import dr
import mjlab.terrains as terrain_gen
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.viewer import ViewerConfig
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from mjlab.sensor import ContactSensor
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from legged_mjlab.envs.him_go2.go2_asset import Go2Asset
from legged_mjlab.envs.him_go2.him_go2_config import HimGo2RoughCfg

from legged_mjlab.utils.helpers import class_to_dict

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class HimGo2Env(ManagerBasedRlEnv):
    def __init__(self, cfg: HimGo2RoughCfg, sim_device, render_mode, play: bool = False, debug_vis: bool = False):
        """ 
        Args:
            cfg (HimGo2RoughCfg): 配置对象，包含环境、控制器和资产的参数
            sim_device: 模拟设备（如 CPU 或 GPU）
            render_mode: 渲染模式（如 "human" 或 "rgb_array"）
        """
        self.robot_cfg = cfg
        self.asset = Go2Asset(self.robot_cfg)
        self.play = bool(play)

        # 组装并生成 mjlab 规范的 ManagerBasedRlEnvCfg
        self.managercfg = self._build_mjlab_managercfg(play = self.play, debug_vis = debug_vis)

        # 记录是否只保留正向奖励（截断负奖励）
        self.only_positive_rewards = self.robot_cfg.rewards.only_positive_rewards

        # 完成底层 MuJoCo 仿真器与各 Manager 的实例化
        super().__init__(
            cfg = self.managercfg,
            device = sim_device,
            render_mode = render_mode,
        )

    def _build_mjlab_managercfg(self, play = False, debug_vis = False) -> ManagerBasedRlEnvCfg:
        """ 将 HimGo2RoughCfg 转换为 ManagerBasedRlEnvCfg 以便于使用 mjlab 的管理器进行环境管理。
        """
        entity_name = self.robot_cfg.asset.name

        return ManagerBasedRlEnvCfg(
            decimation = self.robot_cfg.control.decimation,                 # 控制步
            scene = self._build_scene(self.asset, play, debug_vis),         # 构建场景（包含实体、传感器、地形）
            observations = self._build_observations(play),                  # 挂载观测管理器配置
            actions = self._build_actions(),                                # 挂载动作管理器配置
            events = self._build_events(play),                              # 挂载事件管理器配置（reset, domain_rand）
            seed = self.robot_cfg.env.seed,                                 # 随机种子
            sim = SimulationCfg(                                            # 仿真引擎底层设置
                nconmax=35,
                njmax=1500,
                mujoco = MujocoCfg(
                    timestep = self.robot_cfg.sim.dt,          # 物理仿真步长
                    iterations=10,
                    ls_iterations=20,
                )
            ),                         
            viewer = ViewerConfig(                                          # 
                origin_type = ViewerConfig.OriginType.ASSET_BODY,
                entity_name = entity_name,
                body_name = "base",
                distance=3.0,
                elevation=-5.0,
                azimuth=90.0,
            ),                    
            episode_length_s = self.robot_cfg.env.episode_length_s,         # 单回合最长时间
            rewards = self._build_rewards(),                                # 挂载奖励管理器配置
            terminations = self._build_terminations(),                      # 挂载终止条件管理器配置
            commands = self._build_commands(debug_vis),                     # 构建指令管理器配置
            curriculum = self._build_curriculum(play),                      # 构建课程训练管理配置
        )

    def _build_scene(self, asset, play, debug_vis = False):
        """ 构建场景对象，包含地形和机器人实体
            官方文档：https://mujocolab.github.io/mjlab/v1.6.0/source/scene.html
        """
        entity_name = self.robot_cfg.asset.name
        return SceneCfg(
            num_envs = self.robot_cfg.env.num_envs,                                     # 环境数
            env_spacing = self.robot_cfg.env.env_spacing,                               # 并行环境间的网格间距
            terrain = self._build_terrain(),                                            # 挂载地形实体
            entities = {                                                                # 挂载机器人的 MJCF 实体配置
                entity_name: self.asset.entity.get_robot_cfg()
            },                                  
            sensors = tuple(                                                            # 构建并挂载传感器元组
                self._build_sensors(debug_vis = debug_vis)
            ),
            extent = self.robot_cfg.env.extent
        )

    def _build_sensors(self, entity_name, debug_vis):
        """ 构建
            官方文档：https://mujocolab.github.io/mjlab/v1.6.0/source/sensors/index.html
        """
        sensors = []

        # 足端触地浮空
        if self.robot_cfg.rewards.scales.feet_air_time:
            sensors.append(self.asset.sensor_mgr.get_foot_contact_sensor(entity_name))

        # 高程图
        if self.robot_cfg.terrain.measure_heights:
            sensors.append(self.asset.sensor_mgr.get_height_scan_sensor(entity_name, debug_vis))

        # 
        if self.robot_cfg.rewards.scales.collision and self.asset.penalized_contact_names:
            sensors.append(self.asset.sensor_mgr.get_illegal_contact_sensor(entity_name))

        return sensors

    def _build_terrain(self):
        """ 构建训练/测试地形（平面地形或基于课程进阶的程序化生成地形）
            官方文档：https://mujocolab.github.io/mjlab/v1.6.0/source/terrain.html
        """
        # plane
        if self.robot_cfg.terrain.mesh_type == "plane":
            return TerrainEntityCfg(
                terrain_type="plane",
                terrain_generator=None,
                debug_vis=False,
            )

        # terrain
        if self.robot_cfg.terrain.mesh_type == "generator":
            return TerrainEntityCfg(
                terrain_type = "generator",
                terrain_generator = TerrainGeneratorCfg(
                    curriculum = True,
                    size = (self.robot_cfg.terrain.terrain_length, self.robot_cfg.terrain.terrain_width),            # 子地形大小
                    num_rows = self.robot_cfg.terrain.num_rows,                                                # 地形行数（难度等级）
                    num_cols = self.robot_cfg.terrain.num_cols,                                                # 地形列数（地形类型）
                    border_width = self.robot_cfg.terrain.border_size,                                         # 边界宽度
                    sub_terrains = {
                        # 1. 平坦地面类型
                        "flat": terrain_gen.BoxFlatTerrainCfg(
                            proportion = 0.2        # 占比 20%
                        ),

                        # 2.金字塔台阶地形类型（楼梯）
                        "stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
                            proportion = 0.2,                             
                            step_height_range = (0.0, 0.20),              # 台阶高度范围（难度从 0.0m 逐渐加大到 0.20m）
                            step_width = 0.3,                             # 每个台阶的踏步宽度为 0.3 米
                            platform_width = 2.0,                         # 金字塔顶部的中央平坦平台宽度为 2.0 米
                        ),

                        # 3. 随机高度场崎岖地形类型（碎石路、粗糙砂石地面）
                        "rough": terrain_gen.HfRandomUniformTerrainCfg(
                            proportion = 0.2,                 
                            noise_range = (0.02, 0.10),       # 随机起伏高度范围（噪声幅度从 2cm 逐渐增加到 10cm）
                            noise_step = 0.02,                # 噪声高度的离散采样步长为 0.02 米
                        ),

                        # 4. 柏林噪声连续起伏地形（缓坡/土丘/旷野）
                        "perlin_noise": terrain_gen.HfPerlinNoiseTerrainCfg(
                            proportion=0.2,
                            noise_range = (0.05, 0.30),                   # 波峰/土丘最大起伏高度（从 5cm 递增至 30cm）
                            noise_step=0.02,  
                        ),

                        # 5. 离散凸起障碍高度场（随机柱状/方块障碍)
                        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
                            proportion = 0.2,
                            obstacle_height_range = (0.05, 0.20),           # 障碍高度范围（从 5cm 递增至 20cm）
                            obstacle_width_range = (0.4, 0.8),              # 障碍物宽度/边长范围（0.4m ~ 0.8m）
                            num_obstacles = 12,                             # 每个子地形块内生成的障碍物数量
                            platform_width = 1.5,                           # 中心预留平坦出生区域宽度（避免出生直接卡入障碍）
                        )
                    },
                ),
                max_init_terrain_level = 5
            )
        
    def _build_actions(self):
        """ 构建关节位置动作空间：输出值经 action_scale 缩放后，加到 default 关节偏置上，这里的 offset 为静态偏置
            官方文档：https://mujocolab.github.io/mjlab/v1.6.0/source/actions.html
        """
        action_scales = {}
        for name in self.asset.joint_names:
            if "hip" in name:
                action_scales[name] = float(self.robot_cfg.control.action_scale * self.robot_cfg.control.hip_scale_reduction)
            else:
                action_scales[name] = float(self.robot_cfg.control.action_scale)

        clip_val = self.robot_cfg.control.action_clip
        if isinstance(clip_val, (int, float)):
            action_clip = (-float(clip_val), float(clip_val))
        else:
            action_clip = clip_val

        return {
            "joint_position": JointPositionActionCfg(
                entity_name = self.robot_cfg.asset.name,
                actuator_name = self.asset.joint_names,
                scale = action_scales,    
                use_default_offset = True,                  # 使用位置增量
                preserve_order = True,                      # 严格保持关节顺序与 policy 输出对齐
                clip = action_clip,
            )
        }

    def _build_events(self, play):
        """ 构建重置事件与域随机化
            官方文档：https://mujocolab.github.io/mjlab/v1.6.0/source/events.html
        """
        entity_name = self.robot_cfg.asset.name

        joint_cfg = SceneEntityCfg(
            entity_name,
            joint_names = self.asset.joint_names,
            preserve_order = True,
        )

        actuator_cfg = SceneEntityCfg(
            entity_name,
            actuator_names = (".*",),
            preserve_order = True,
        )

        foot_cfg = SceneEntityCfg(
            entity_name,
            geom_names=tuple(
                f"{leg}_foot_collision" for leg in ("FL", "FR", "RL", "RR")
            ),
            preserve_order=True,
        )

        body_cfg = SceneEntityCfg(entity_name, body_names=("base_link",))

        non_base_bodies = tuple(b for b in self.asset.body_names if b != "base_link")
        link_cfg = SceneEntityCfg(
            entity_name,
            body_names = non_base_bodies,
            preserve_order = True,
        )

        base_pose = {}
        if self.robot_cfg.domain_rand.randomize_base_pose and not play:
            base_pose = {
                "z" : tuple(self.robot_cfg.domain_rand.base_pose_z_range),
                "roll": tuple(self.robot_cfg.domain_rand.base_pose_roll_range),
                "pitch": tuple(self.robot_cfg.domain_rand.base_pose_pitch_range),
            }

        # 初始化基础事件字典（默认包含每次环境 reset 时的状态重置项）
        events = {
            # 重置基座/机身状态
            "reset_base": EventTermCfg(
                func = mdp.reset_root_state_uniform,
                mode = "reset",                       # 触发时机：每次环境重置时
                params = {
                    "asset_cfg": body_cfg,
                    "pose_range": {
                        "x": (-0.0, 0.0),
                        "y": (-0.0, 0.0),
                        "z": base_pose.get("z", (0.0, 0.0)),
                        "roll": base_pose.get("roll", (0.0, 0.0)),
                        "pitch": base_pose.get("pitch", (0.0, 0.0)),
                        "yaw": (-0.0, 0.0),
                    },
                    "velocity_range": {},
                }
            ),
            # 重置各关节状态
            "reset_joints": EventTermCfg(
                func = mdp.reset_joints_by_offset,
                mode = "reset",
                params = {
                    "position_range" : (-0.0, 0.0),
                    "velocity_range" : (-0.0, 0.0),
                    "asset_cfg": joint_cfg,
                }
            ),
        }

        if not play:
            # 域随机化部分
            # --------- 动力学参数随机化 ----------
            # 基座负载
            if self.robot_cfg.domain_rand.randomize_payload_mass:
                events["base_mass"] = EventTermCfg(
                    mode = "startup",
                    func = dr.body_mass,
                    params = {
                        "asset_cfg": body_cfg,
                        "operation": "add",
                        "ranges": tuple(self.robot_cfg.domain_rand.payload_mass_range),
                    }
                )

            # 质心
            if self.robot_cfg.domain_rand.randomize_com_displacement:
                events["base_com"] = EventTermCfg(
                    mode = "startup",
                    func = dr.body_com_offset,
                    params = {
                        "asset_cfg": body_cfg,
                        "operation": "add",                     # 在默认质心坐标上累加偏移量
                        "ranges": {
                            0: tuple(self.robot_cfg.domain_rand.com_displacement_range), # X 轴偏移范围
                            1: tuple(self.robot_cfg.domain_rand.com_displacement_range), # Y 轴偏移范围
                            2: tuple(self.robot_cfg.domain_rand.com_displacement_range), # Z 轴偏移范围
                        },
                    },
                )

            # 除base外其他link质量：
            if self.robot_cfg.domain_rand.randomize_link_mass:
                events["link_mass"] = EventTermCfg(
                    mode = "startup",
                    func = dr.body_mass,
                    params = {
                        "asset_cfg": link_cfg,
                        "operation": "scale",
                        "ranges": tuple(self.robot_cfg.domain_rand.link_mass_range),
                    },
                )

            # 关节摩擦
            if self.robot_cfg.domain_rand.randomize_joint_friction:
                events["joint_friction"] = EventTermCfg(
                    mode="startup",
                    func=dr.joint_friction,
                    params={
                        "asset_cfg": joint_cfg,
                        "operation": "abs",
                        "ranges": tuple(self.robot_cfg.domain_rand.joint_friction_range),
                    },
                )

            # 关节阻尼
            if self.robot_cfg.domain_rand.randomize_joint_damping:
                events["joint_damping"] = EventTermCfg(
                    mode="startup",
                    func=dr.joint_damping,
                    params={
                        "asset_cfg": joint_cfg,
                        "operation": "abs",
                        "ranges": tuple(self.robot_cfg.domain_rand.joint_damping_range),
                    },
                )

            # 关节等效转动惯量
            if self.robot_cfg.domain_rand.randomize_joint_armature:
                events["joint_armature"] = EventTermCfg(
                    mode = "startup",
                    func = dr.joint_armature,
                    params = {
                        "asset_cfg": joint_cfg,
                        "operation": "abs",
                        "ranges": tuple(self.robot_cfg.domain_rand.joint_armature_range),
                    },
                )
            
            # --------- 接触与外力随机化 ----------
            # push
            if self.robot_cfg.domain_rand.push_robots:
                events["push_robot"] = EventTermCfg(
                    mode = "interval",
                    func = mdp.push_by_setting_velocity,
                    params = {
                        "velocity_range": {
                            "x": tuple(self.robot_cfg.domain_rand.push_vel_xy),
                            "y": tuple(self.robot_cfg.domain_rand.push_vel_xy),
                            "z": tuple(self.robot_cfg.domain_rand.push_vel_z),
                            "roll": tuple(self.robot_cfg.domain_rand.push_ang_rp),
                            "pitch": tuple(self.robot_cfg.domain_rand.push_ang_rp),
                            "yaw": tuple(self.robot_cfg.domain_rand.push_ang_y),
                        }
                    }
                )
                
            # 接触摩擦力
            if self.robot_cfg.domain_rand.frandomize_friction:
                events["friction"] = EventTermCfg(
                    mode = "startup",
                    func = dr.geom_friction,
                    params = {
                        "asset_cfg": foot_cfg,
                        "operation": "abs",
                        "ranges": tuple(self.robot_cfg.domain_rand.friction_range),
                        "shared_random": True,              # All foot geoms share the same friction.
                    }
                )

            # --------- 控制器与执行器随机化 ----------
            # motor offset
            if self.robot_cfg.domain_rand.randomize_motor_zero_offset:
                events["encoder_bias"] = EventTermCfg(
                    mode = "startup",
                    func = dr.encoder_bias,
                    params = {
                        "asset_cfg": joint_cfg,
                        "bias_range": tuple(self.robot_cfg.domain_rand.motor_zero_offset_range),
                    },
                )

            # pd
            if self.robot_cfg.domain_rand.randomize_pd_gains:
                events["pd_gains"] = EventTermCfg(
                    func = dr.pd_gains,
                    mode = "startup",
                    params = {
                        "asset_cfg": actuator_cfg,
                        "operation": "scale",          # 在标称增益上乘比例系数
                        "kp_range": tuple(self.robot_cfg.domain_rand.stiffness_multiplier_range),
                        "kd_range": tuple(self.robot_cfg.domain_rand.damping_multiplier_range),
                    }
                )

            # effort_limits
            if self.robot_cfg.domain_rand.randomize_motor_strength:
                events["effort_limits"]  = EventTermCfg(
                    mode = "startup",
                    func = dr.effort_limits,
                    params = {
                        "asset_cfg" : actuator_cfg,
                        "operation" : "scale",
                        "effort_limit_range" : tuple(self.robot_cfg.domain_rand.motor_strength_range)
                    }
                )
            
            return events

        return events

    def _build_commands(self, debug_vis):
        """
            官方文档：https://mujocolab.github.io/mjlab/v1.6.0/source/commands.html
        """
        ranges = self.robot_cfg.commands.ranges

        return {
            "twist": UniformVelocityCommandCfg(
                entity_name = self.robot_cfg.asset.name,
                resampling_time_range = tuple(self.robot_cfg.commands.resampling_time),
                rel_standing_envs = 0.05,               # 保持静止环境的比例
                rel_forward_envs = 0.25,                # 仅前向速度的比例
                debug_vis = debug_vis,                  # 是否在仿真视口中渲染指令的箭头/可视化标记
                heading_command = self.robot_cfg.commands.heading_command,
                ranges = UniformVelocityCommandCfg.Ranges(
                    lin_vel_x = tuple(ranges.lin_vel_x),
                    lin_vel_y = tuple(ranges.lin_vel_y),
                    ang_vel_z = tuple(ranges.ang_vel_yaw),
                    heading = tuple(ranges.heading),
                )
            )
        }

    def _build_observations(self, play: bool):
        """ 构建观测组
            官方文档：https://mujocolab.github.io/mjlab/v1.6.0/source/observations.html
        """
        entity_name = self.cfg.asset.name

        # clip
        clip_val = self.robot_cfg.normalization.clip_observations
        clip_obs = (-clip_val, clip_val)

        # delay
        imu_delay_min, imu_delay_max = 0, 0
        if self.robot_cfg.domain_rand.randomize_obs_imu_latency:
            imu_delay_min = self.robot_cfg.domain_rand.range_obs_imu_latency[0]
            imu_delay_max = self.robot_cfg.domain_rand.range_obs_imu_latency[1]

        motor_delay_min, motor_delay_max = 0, 0
        if self.robot_cfg.domain_rand.randomize_obs_motor_latency:
            motor_delay_min = self.robot_cfg.domain_rand.range_obs_motor_latency[0]
            motor_delay_max = self.robot_cfg.domain_rand.range_obs_motor_latency[1]

        # noise
        noise = {}
        if not play and self.robot_cfg.noise.add_noise:
            noise_level = self.robot_cfg.noise.noise_level
            scales = self.robot_cfg.noise.noise_scales
            noise = {
                "ang_vel": UniformNoiseCfg(n_min=-scales.ang_vel * noise_level, n_max=scales.ang_vel * noise_level),
                "gravity": UniformNoiseCfg(n_min=-scales.gravity * noise_level, n_max=scales.gravity * noise_level),
                "dof_pos": UniformNoiseCfg(n_min=-scales.dof_pos * noise_level, n_max=scales.dof_pos * noise_level),
                "dof_vel": UniformNoiseCfg(n_min=-scales.dof_vel * noise_level, n_max=scales.dof_vel * noise_level),
            }

        # actor obs
        actor_terms = {
            # cmd + ang + gra + pos + vel + last_action
            "command": ObservationTermCfg(
                func = mdp.generated_commands,
                params = {"command_name": "twist"},
                clip = clip_obs,
            ),
            "base_ang_vel": ObservationTermCfg(
                func = mdp.builtin_sensor,
                params = {"sensor_name": f"{entity_name}/imu_ang_vel"},
                noise = noise.get("ang_vel"),
                scale = self.robot_cfg.normalization.obs_scales.ang_vel,
                delay_min_lag = imu_delay_min,
                delay_max_lag = imu_delay_max,
                clip = clip_obs,
            ),
            "projected_gravity": ObservationTermCfg(
                func = mdp.projected_gravity,
                noise = noise.get("gravity"),
                delay_min_lag = imu_delay_min,
                delay_max_lag = imu_delay_max,
                clip = clip_obs,
            ),
            "joint_pos": ObservationTermCfg(
                func = mdp.joint_pos_rel,
                scale = self.robot_cfg.normalization.obs_scales.dof_pos,
                noise = noise.get("dof_pos"),
                delay_min_lag = motor_delay_min,
                delay_max_lag = motor_delay_max,
                clip = clip_obs,
            ),
            "joint_vel": ObservationTermCfg(
                func = mdp.joint_vel_rel,
                scale = self.robot_cfg.normalization.obs_scales.dof_vel,
                noise = noise.get("dof_vel"),
                delay_min_lag = motor_delay_min,
                delay_max_lag = motor_delay_max,
                clip = clip_obs,
            ),
            "last_action": ObservationTermCfg(
                func = mdp.last_action
            ),
        }

        # critic obs
        critic_terms = {}
        for name, term in actor_terms.items():
            critic_term = copy.deepcopy(term)
            critic_term.delay_min_lag = 0  # 强制归零延迟
            critic_term.delay_max_lag = 0
            critic_term.noise = None       # 强制移除噪声
            critic_terms[name] = critic_term
        critic_terms.update({
            "base_lin_vel": ObservationTermCfg(
                func = mdp.builtin_sensor,
                params={"sensor_name": "robot/imu_lin_vel"},
            ),
            "height_scan": ObservationTermCfg(
                func = mdp.height_scan,
                params={"sensor_name": "terrain_scan"},
                scale = self.robot_cfg.normalization.obs_scales.height_measurements,
            ),
        })

        obs = {
            "actor": ObservationGroupCfg(
                terms=actor_terms,
                concatenate_terms = True,
                enable_corruption = True,
                history_length=1,
            ),
            "critic": ObservationGroupCfg(
                terms = critic_terms,
                concatenate_terms = True,
                enable_corruption = False,
                history_length = 1,
            ),
        }

        return obs

    def _build_terminations(self):
        """ 构建回合提前终止条件
            官方文档：https://mujocolab.github.io/mjlab/v1.6.0/source/terminations.html
        """
        return {
            "time_out": TerminationTermCfg(
                func = mdp.time_out,
                time_out = True,
            ),
        }
    
    def _build_curriculum(self, play):
        """ 构建课程学习机制
            官方文档：https://mujocolab.github.io/mjlab/v1.6.0/source/curriculum.html
        """
        curriculums = {}

        if play:
            return curriculums

        if self.robot_cfg.terrain.curriculum:
            curriculums["terrain_levels"] = CurriculumTermCfg(
                func = mdp.terrain_levels_vel,
                params={"command_name": "twist"},
            )

        if self.robot_cfg.commands.curriculum:
            curriculums["commands_levels"] = CurriculumTermCfg(
                func = mdp.commands_vel,
                params = {
                    "command_name": "twist",
                    "velocity_stages": [
                        {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
                        {"step": 5000 * 24, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0)},
                    ],
                },
            )

        return curriculums

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        self.reward_scales = class_to_dict(self.robot_cfg.rewards.scales)

        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key, None) 

        reward_functions = {}
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            reward_functions[name] = RewardTermCfg(
                weight = scale,
                func = getattr(self, '_reward_' + name),
            )

        return reward_functions

# ----------------- rewards -----------------

    def _reward_tracking_lin_vel(
        self,
        env: ManagerBasedRlEnv,
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
        return torch.exp(-lin_vel_error / self.robot_cfg.rewards.tracking_sigma**2)
    
    def _reward_tracking_ang_vel(
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

    def _reward_lin_vel_z(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        return torch.square(asset.data.root_link_lin_vel_b[:, 2])

    def _reward_ang_vel_xy(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)

    def _reward_body_orientation_l2(
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

    def _reward_feet_air_time(
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

    def _reward__collision_cost(
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

    def _reward_hip_pos(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        default_joint_pos = asset.data.default_joint_pos
        assert default_joint_pos is not None
        diff_angle = (
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - default_joint_pos[:, asset_cfg.joint_ids]
        )
        return torch.sum(torch.square(diff_angle), dim=1)

    def _reward_base_height_l2(env, target_height: float, asset_cfg: SceneEntityCfg):
        asset = env.scene[asset_cfg.name]
        height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
        return torch.square(height - target_height)

    def _reward_feet_clearance(
        env: ManagerBasedRlEnv,
        target_height: float,
        command_name: str | None = None,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
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

    def _reward_joint_power_l1(env, asset_cfg: SceneEntityCfg):
        asset = env.scene[asset_cfg.name]
        torque = asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
        velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
        return torch.sum(torch.abs(torque * velocity), dim=1)

    def _reward_stand_still(
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

    def _reward_torque_limit_cost(
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

    def _reward_action_rate(env: ManagerBasedRlEnv):
        # Penalize changes in actions
        action_manager = env.action_manager
        action_rate = action_manager.action - action_manager.prev_action
        return torch.sum(torch.square(action_rate), dim=1)

    def _reward_smoothness(env: ManagerBasedRlEnv):
        # second order smoothness
        action_manager = env.action_manager
        smoothness = (action_manager.action - 2.0 * action_manager.prev_action + action_manager.prev_prev_action)
        return torch.sum(torch.square(smoothness), dim=1)
    
    def _reward_dof_pos_limits(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg):
        """惩罚：关节角度超出软限位（防止撞击机械限位）。"""
        asset: Entity = env.scene[asset_cfg.name]
        joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        soft_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids] # 获取软限位
        lower, upper = soft_limits[..., 0], soft_limits[..., 1]
        # 计算低于下限或高于上限的部分
        violation = torch.relu(lower - joint_pos) + torch.relu(joint_pos - upper)
        return torch.sum(violation, dim=1)

    