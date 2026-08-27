import torch


from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg 
from mjlab.entity import Entity
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.tasks.velocity import mdp
from mjlab.envs.mdp import dr
import mjlab.terrains as terrain_gen
from mjlab.utils.lab_api.math import quat_apply_inverse

from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from mjlab.sensor import (
    ContactMatch,          # 接触匹配规则（用于定义 geom/body 碰撞过滤）
    ContactSensor,
    ContactSensorCfg,      # 接触力/碰撞传感器配置
    GridPatternCfg,        # 射线扫描网格模式配置
    ObjRef,                # 实体/刚体引用对象
    RayCastSensorCfg,      # 射线投射传感器（测高）配置
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from legged_mjlab.envs.him_go2.go2_asset import Go2Asset
from legged_mjlab.envs.him_go2.him_go2_config import HimGo2RoughCfg

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

        self.managercfg = self._build_mjlab_managercfg(play = self.play, debug_vis = debug_vis)

        # 完成底层 MuJoCo 仿真器与各 Manager 的实例化
        super().__init__(
            cfg = self.managercfg,
            device = sim_device,
            render_mode = render_mode,
        )

    def _build_mjlab_managercfg(self, play=False, debug_vis = False) -> ManagerBasedRlEnvCfg:
        """ 将 HimGo2RoughCfg 转换为 ManagerBasedRlEnvCfg 以便于使用 mjlab 的管理器进行环境管理。
        """
        return ManagerBasedRlEnvCfg(
            decimation = self.cfg.control.decimation,                       # 控制步
            scene = self._build_scene(self.asset, play, debug_vis),         # 构建场景（包含实体、传感器、地形）
            observations = self._build_observations(),                      # 挂载观测管理器配置
            actions = self._build_actions(),                                # 挂载动作管理器配置
            events = self._build_events(),                                  # 挂载事件管理器配置（reset, domain_rand）
            seed = self.robot_cfg.env.seed,                                 # 随机种子
            sim = SimulationCfg(                                            # 仿真引擎底层设置
                mujoco = MujocoCfg(
                    timestep = self.robot_cfg.sim.dt,          # 物理仿真步长
                    integrator = self.robot_cfg.sim.gravity,   # 重力向量
                )
            ),                                             
            episode_length_s = self.robot_cfg.env.episode_length_s,         # 单回合最长时间
            rewards = self._build_rewards(),                                # 挂载奖励管理器配置
            terminations = self._build_terminations(),                      # 挂载终止条件管理器配置
            commands = self._build_commands(),
            curriculum = self._build_curriculum(),
            # recorders = ,
            scale_rewards_by_dt = False,
        )

    def _build_scene(self, asset, play, debug_vis = False):
        """ 构建场景对象，包含地形和机器人实体
            官方文档：https://mujocolab.github.io/mjlab/v1.6.0/source/scene.html
        """
        entity_name = self.robot_cfg.asset.name

        return SceneCfg(
            num_envs = self.robot_cfg.env.num_envs,                                           # 环境数
            env_spacing = self.robot_cfg.env.env_spacing,                                     # 并行环境间的网格间距
            terrain = self._build_terrain(),                                            # 挂载地形实体
            entities = {                                                                # 挂载机器人的 MJCF 实体配置
                entity_name: asset.entity.get_robot_cfg()
            },                                  
            sensors = tuple(                                                            # 构建并挂载传感器元组
                self._build_sensors(entity_name, debug_vis = debug_vis)
            ),
        )

    def _build_sensors(self, entity_name, debug_vis):
        """ 构建
            官方文档：https://mujocolab.github.io/mjlab/v1.6.0/source/sensors/index.html
        """
        # 定义四条腿足端碰撞体的名字元组

        contact_geom_patterns = {
            "base": r"^base[123]_collision$",
            "thigh": r"^(FR|FL|RR|RL)_thigh_collision$",
            "calf": r"^(FR|FL|RR|RL)_calf[12]_collision$",
        }
        penalize_contact_patterns = tuple(
            contact_geom_patterns[part_name]
            for part_name in self.robot_cfg.asset.penalize_contacts_on
            if part_name in contact_geom_patterns
        )

        sensors = []

        # 足端触地浮空
        if self.robot_cfg.rewards.scales.feet_air_time:
            sensors.append(self.asset.sensor.get_foot_contact_sensor(entity_name))

        if self.robot_cfg.terrain.measure_heights:
            x_points = tuple(self.robot_cfg.terrain.measured_points_x)
            y_points = tuple(self.robot_cfg.terrain.measured_points_y)

            sensors.append(
                RayCastSensorCfg(
                    name = "height_scan",
                    frame = ObjRef(                                     # 附加射线的实体
                        type = "body",
                        name = "base_link",          # 以 base_link 为基准坐标系向下发射射线
                        entity = entity_name,
                    ),
                    pattern=GridPatternCfg(
                        size = (
                            max(x_points) - min(x_points),  # 网格长
                            max(y_points) - min(y_points),  # 网格宽
                        ),
                        resolution = self.robot_cfg.terrain.horizontal_scale,  # 采样分辨率
                    ),
                    ray_alignment = "yaw",
                    max_distance = 2.0,              
                    debug_vis = bool(debug_vis),     # 是否在 GUI 中绘制扫描射线
                )
            )

        if self.robot_cfg.rewards.scales.collision and penalize_contact_patterns:
            sensors.append(
                ContactSensorCfg(
                    name = "nonfoot_ground_touch", 
                    primary = ContactMatch(   
                        mode = "geom",
                        pattern = penalize_contact_patterns,
                        entity = entity_name,
                    ),
                    secondary = ContactMatch(
                        mode = "body",
                        pattern = "terrain",       
                    ),

                )
            )

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
        
    def _joint_names(self):
        """ 从默认关节角字典中提取全部关节名元组，保持严格的顺序
        """
        return tuple(self.robot_cfg.init_state.default_joint_angles.keys())

    def _build_actions(self):
        """ 构建关节位置动作空间：输出值经 action_scale 缩放后，加到 default 关节偏置上，这里的 offset 为静态偏置
            官方文档：https://mujocolab.github.io/mjlab/v1.6.0/source/actions.html
        """
        joint_names = self._joint_names()

        action_scales = {}
        for name in joint_names:
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
                actuator_name = joint_names,
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
            joint_names=self._joint_names(),
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

        base_pose = {}

        # 初始化基础事件字典（默认包含每次环境 reset 时的状态重置项）
        events = {
            # 重置基座/机身状态
            "reset_base": EventTermCfg(
                func = mdp.reset_root_state_uniform,
                mode = "reset",                       # 触发时机：每次环境重置时
                params = {

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

        # 域随机化部分
        if self.robot_cfg.domain_rand.randomize_payload_mass:
            pass

        if self.robot_cfg.domain_rand.randomize_motor_zero_offset:
            events["encoder_bias"] = EventTermCfg(
                mode="startup",
                func=dr.encoder_bias,
                params={
                    "asset_cfg": joint_cfg,
                    "bias_range": tuple(self.robot_cfg.domain_rand.motor_zero_offset_range),
                },
            )

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

        if self.robot_cfg.domain_rand.randomize_com_displacement:
            events["base_com"] = EventTermCfg(
                mode="startup",
                func=dr.body_com_offset,
                params={
                    "asset_cfg": body_cfg,
                    "operation": "add",                     # 在默认质心坐标上累加偏移量
                    "ranges": {
                        0: tuple(self.robot_cfg.domain_rand.com_displacement_range), # X 轴偏移范围
                        1: tuple(self.robot_cfg.domain_rand.com_displacement_range), # Y 轴偏移范围
                        2: tuple(self.robot_cfg.domain_rand.com_displacement_range), # Z 轴偏移范围
                    },
                },
            )

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
                heading_command = self.robot_cfg.commands.heading_command,
                rel_standing_envs = 0.05,               # 保持静止环境的比例
                rel_forward_envs = 0.25,                # 仅前向速度的比例
                debug_vis = debug_vis,                  # 是否在仿真视口中渲染指令的箭头/可视化标记
                ranges = UniformVelocityCommandCfg.Ranges(
                    lin_vel_x = tuple(ranges.lin_vel_x),
                    lin_vel_y = tuple(ranges.lin_vel_y),
                    ang_vel_z = tuple(ranges.heading),
                )
            )
        }

    def _reward_scale(self, name):
        """ 返回指定奖励项的缩放系数，若不存在则返回 0
        """
        return self.robot_cfg.rewards.scales.get(name, 0.0)

    def _add_reward(self, terms, name, func, params=None):
        """ 通用奖励项注册辅助函数：仅当权重非零时才加入计算图
        """
        weight = self._reward_scale(name)

        if weight == 0.0:
            return

        terms[name] = RewardTermCfg(
            func = func,
            weight = weight,
            params = dict(params or {}),
        )

    def _entity_term_cfg(self):
        """ 
        """
        return SceneEntityCfg(
            name=self.cfg.asset.name,
            joint_names=self._joint_names(),
            preserve_order=True,
        )

    def _build_rewards(self):
        """ 
            官方文档：https://mujocolab.github.io/mjlab/v1.6.0/source/rewards.html
        """

        robot_cfg = SceneEntityCfg(name = self.robot_cfg.asset.name)
        joint_cfg = self._entity_term_cfg()
        terms = {}

        return terms

    def _build_observations(self, play: bool):
        """ 构建观测组
            官方文档：https://mujocolab.github.io/mjlab/v1.6.0/source/observations.html
        """
        entity_name = self.cfg.asset.name

        joint_cfg = SceneEntityCfg(
            entity_name,
            joint_names = self._joint_names(),
            preserve_order = True,
        )

        noise = {}

        if play:
            noise = {}
        elif self.cfg.noise.add.add_noise:
            self._get_noise()
        else:
            noise = {}

        # actor_obs
        actor_terms = {
            # cmd + ang + gra + pos + vel + last_action
            "command": ObservationTermCfg(
                func = envs_mdp.generated_commands,
                params = {"command_name": "twist"},
            ),
            "base_ang_vel": ObservationTermCfg(
                func = envs_mdp.builtin_sensor,
                params = {"sensor_name": f"{entity_name}/imu_ang_vel"},
                noise = noise.get("ang_vel"),
                scale = float(self.robot_cfg.normalization.obs_scales.ang_vel),

            ),


        }

    def _build_terminations(self):
        """ 构建回合提前终止条件
            官方文档：https://mujocolab.github.io/mjlab/v1.6.0/source/terminations.html
        """

        return {
            "time_out": TerminationTermCfg(
                func=envs_mdp.time_out,
                time_out=True,
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
                func=mdp.terrain_levels_vel,
                params={"command_name": "twist"},
            )

        if self.robot_cfg.commands.curriculum:
            curriculums["commands_levels"] = CurriculumTermCfg(
                func = mdp.commands_vel,
                params={
                    "command_name": "twist",
                    "max_lin_vel_x": self.robot_cfg.commands.max_curriculum,
                    "max_ang_vel_yaw": self.robot_cfg.commands.max_curriculum,
                },
            )

        return curriculums
        
# ----------------- rewards -----------------

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

    def hip_pos(
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

    def _base_height_l2(env, target_height: float, asset_cfg: SceneEntityCfg):
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


    
