import torch


from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg 
from mjlab.sim import MujocoCfg, SimulationCfg

from mjlab.sensor import (
    ContactMatch,          # 接触匹配规则（用于定义 geom/body 碰撞过滤）
    ContactSensorCfg,      # 接触力/碰撞传感器配置
    GridPatternCfg,        # 射线扫描网格模式配置
    ObjRef,                # 实体/刚体引用对象
    RayCastSensorCfg,      # 射线投射传感器（测高）配置
)

from mjlab.terrains import TerrainEntityCfg




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
            actions = self._build_actions(),                                # 挂载动作管理器配置
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
            auto_reset = ,
            scale_rewards_by_dt = ,
        )

    def _build_scene(self, cfg, asset, play, debug_vis = False):
        """ 构建场景对象，包含地形和机器人实体
            官方文档：https://mujocolab.github.io/mjlab/v1.6.0/source/scene.html
        """
        entity_name = cfg.asset.name

        return SceneCfg(
            num_envs = cfg.env.num_envs,                                                # 环境数
            env_spacing = cfg.env.env_spacing,                                          # 并行环境间的网格间距
            terrain = self._build_terrain(cfg, play = play),                            # 挂载地形实体
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

    def _build_terrain(self, cfg, play=False):
        """
        """
        # plane
        if cfg.terrain.mesh_type == "plane":
            return TerrainEntityCfg(
                terrain_type="plane",
                terrain_generator=None,
                debug_vis=False,
            )

        # terrain




    def _build_actions(self):
        """
        """

    def _build_commands(self):
        """
        """

    def _build_rewards(self):
        """
        """

    def _build_observations(self):
        """
        """

    def _build_terminations(self):
        """
        """

    def _build_events(self):
        """
        """

    def _build_curriculum(self):
        """
        """

    

        
