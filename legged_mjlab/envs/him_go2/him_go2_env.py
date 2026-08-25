import torch


from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg 
from mjlab.sim import MujocoCfg, SimulationCfg

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
    ContactSensorCfg,      # 接触力/碰撞传感器配置
    GridPatternCfg,        # 射线扫描网格模式配置
    ObjRef,                # 实体/刚体引用对象
    RayCastSensorCfg,      # 射线投射传感器（测高）配置
)

from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg



from legged_mjlab.envs.him_go2.go2_asset import Go2Asset
from legged_mjlab.envs.him_go2.him_go2_config import HimGo2RoughCfg



class HimGo2Env(ManagerBasedRlEnv):
    def __init__(self, cfg: HimGo2RoughCfg, sim_device, render_mode, play: bool = False, debug_vis: bool = False):
        """ 
        Args:
            cfg (HimGo2RoughCfg): 配置对象，包含环境、控制器和资产的参数
            sim_device: 模拟设备（如 CPU 或 GPU）
            render_mode: 渲染模式（如 "human" 或 "rgb_array"）
        """
        self.cfg = cfg
        self.play = bool(play)

        self.managercfg = self._build_mjlab_managercfg(self.cfg, self.play, debug_vis)

        # 完成底层 MuJoCo 仿真器与各 Manager 的实例化
        super().__init__(
            cfg = self.managercfg,
            device = sim_device,
            render_mode = render_mode,
        )


    def _build_mjlab_managercfg(self, cfg, play=False, debug_vis = False) -> ManagerBasedRlEnvCfg:
        """ 将 HimGo2RoughCfg 转换为 ManagerBasedRlEnvCfg 以便于使用 mjlab 的管理器进行环境管理。
        """
        asset = Go2Asset(cfg)

        return ManagerBasedRlEnvCfg(
            decimation = cfg.control.decimation,                            # 控制步
            scene = self._build_scene(cfg, asset, play, debug_vis),         # 构建场景（包含实体、传感器、地形）
            observations = self._build_observations(),                      # 挂载观测管理器配置
            actions = self._build_actions(cfg),                             # 挂载动作管理器配置
            events = ,
            seed = cfg.env.seed,                                            # 随机种子
            sim = SimulationCfg(                                            # 仿真引擎底层设置
                mujoco = MujocoCfg(
                    timestep = cfg.sim.dt,          # 物理仿真步长
                    integrator = cfg.sim.gravity,   # 重力向量
                )
            ),                                             
            viewer = ,
            episode_length_s = cfg.env.episode_length_s,                    # 单回合最长时间
            rewards = ,
            terminations = ,
            commands = ,
            curriculum = ,
            metrics = , 
            recorders = ,
            is_finite_horizon = ,
            scale_rewards_by_dt = False,
        )

    def _build_scene(self, cfg, asset, play, debug_vis = False):
        """ 构建场景对象，包含地形和机器人实体
            官方文档：https://mujocolab.github.io/mjlab/v1.6.0/source/scene.html
        """
        entity_name = cfg.asset.name

        return SceneCfg(
            num_envs = cfg.env.num_envs,                                                # 环境数
            env_spacing = cfg.env.env_spacing,                                          # 并行环境间的网格间距
            terrain = self._build_terrain(cfg),                            # 挂载地形实体
            entities = {asset.entity.get_robot_cfg()},                                  # 挂载机器人的 MJCF 实体配置
            sensors = tuple(                                                            # 构建并挂载传感器元组
                self._build_sensors(cfg, entity_name, debug_vis = debug_vis)
            ),
        )

    def _build_sensors(self, cfg, entity_name, debug_vis):
        """ 构建
            官方文档：https://mujocolab.github.io/mjlab/v1.6.0/source/sensors/index.html
        """
        # 定义四条腿足端碰撞体的名字元组
        foot_geoms = tuple(
            f"{leg}_foot_collision"
            for leg in ("FR", "FL", "RR", "RL")
        )

        sensors = []

        if cfg.rewards.scales.feet_air_time:
            sensors.append(
                ContactSensorCfg(
                    name="feet_ground_contact",                     # 传感器名称
                    primary = ContactMatch(                         
                        mode = "geom",                  # 匹配类型
                        pattern = foot_geoms,           # 匹配四个足端 Geom
                        entity = entity_name,
                    ),
                    secondary = ContactMatch(
                        mode = "body",
                        pattern = "terrain",            # 碰撞目标必须是地形
                    ),
                    fields = ("found", "force"),                    # 提取是否发生碰撞及碰撞力大小
                    reduce = "netforce",                            # 对接触力进行合力计算
                    num_slots = 1,                                  # 槽位数
                    track_air_time = True,                          # 开启滞空时间跟踪
                )
            )

        if cfg.terrain.measure_heights:
            x_points = tuple(cfg.terrain.measured_points_x)
            y_points = tuple(cfg.terrain.measured_points_y)

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
                        resolution = cfg.terrain.horizontal_scale,  # 采样分辨率
                    ),
                    ray_alignment = "yaw",
                    max_distance = 2.0,              
                    debug_vis = bool(debug_vis),     # 是否在 GUI 中绘制扫描射线
                )
            )

        if :
            pass
        

        return sensors

    def _build_terrain(self, cfg):
        """ 构建训练/测试地形（平面地形或基于课程进阶的程序化生成地形）
            官方文档：https://mujocolab.github.io/mjlab/v1.6.0/source/terrain.html
        """
        # plane
        if cfg.terrain.mesh_type == "plane":
            return TerrainEntityCfg(
                terrain_type="plane",
                terrain_generator=None,
                debug_vis=False,
            )

        # terrain
        if cfg.terrain.mesh_type == "generator":
            return TerrainEntityCfg(
                terrain_type = "generator",
                terrain_generator = TerrainGeneratorCfg(
                    curriculum = True,
                    size = (cfg.terrain.terrain_length, cfg.terrain.terrain_width),                 # 子地形大小
                    num_rows = cfg.terrain.num_rows,                                                # 地形行数（难度等级）
                    num_cols = cfg.terrain.num_cols,                                                # 地形列数（地形类型）
                    border_width = cfg.terrain.border_size,                                         # 边界宽度
                    sub_terrains = {

                    },
                ),

            )
        

    def _joint_names(cfg):
        """ 从默认关节角字典中提取全部关节名元组，保持严格的顺序
        """
        return tuple(cfg.init_state.default_joint_angles.keys())

    def _build_actions(self, cfg):
        """ 构建关节位置动作空间：输出值经 action_scale 缩放后，加到 default 关节偏置上
            官方文档：https://mujocolab.github.io/mjlab/v1.6.0/source/actions.html
        """
        joint_names = self._joint_names(cfg)

        action_scales = {}
        for name in joint_names:
            if "hip" in name:
                action_scales[name] = float(cfg.control.action_scale * cfg.control.hip_scale_reduction)
            else:
                action_scales[name] = float(cfg.control.action_scale)

        clip_val = cfg.control.action_clip
        if isinstance(clip_val, (int, float)):
            action_clip = (-float(clip_val), float(clip_val))
        else:
            action_clip = clip_val

        return {
            "joint_position": JointPositionActionCfg(
                entity_name = cfg.asset.name,
                actuator_name = joint_names,
                scale = action_scales,    
                use_default_offset = True,                  # 使用位置增量
                preserve_order = True,                      # 严格保持关节顺序与 policy 输出对齐
                clip = action_clip,
            )
        }

    def _build_events(self, cfg, play):
        """ 构建重置事件与域随机化
            官方文档：https://mujocolab.github.io/mjlab/v1.6.0/source/events.html
        """

    def _build_commands(self):
        """
        """

    def _reward_scale(cfg, name):
        """ 返回指定奖励项的缩放系数，若不存在则返回 0
        """
        return cfg.rewards.scales.get(name, 0.0)

    def _add_reward(self, terms, cfg, name, func, params=None):
        """ 通用奖励项注册辅助函数：仅当权重非零时才加入计算图
        """
        weight = self._reward_scale(cfg, name)

        if weight == 0.0:
            return

        terms[name] = RewardTermCfg(
            func=func,
            weight=weight,
            params=dict(params or {}),
        )

    def _entity_term_cfg(self, cfg):
        """ 
        """
        return SceneEntityCfg(
            name=cfg.asset.name,
            joint_names=self._joint_names(cfg),
            preserve_order=True,
        )

    def _build_rewards(self, cfg):
        """ 
            官方文档：https://mujocolab.github.io/mjlab/v1.6.0/source/rewards.html
        """

        robot_cfg = SceneEntityCfg(name=cfg.asset.name)
        joint_cfg = self._entity_term_cfg(cfg)
        terms = {}

        self._add_reward(terms, cfg, "tracking_lin_vel",
                         )


    def _build_observations(self):
        """ 构建观测组
        """

    def _build_terminations(self):
        """  构建回合提前终止条件
        """

    def _build_curriculum(self):
        """ 构建课程学习机制
        """

    

        
