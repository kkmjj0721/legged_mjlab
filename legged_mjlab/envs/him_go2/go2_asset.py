"""Canonical Go2 robot entity and scene-sensor configuration."""
from legged_mjlab import LEGGED_MJLAB_ROOT_DIR

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict
from pathlib import Path

import mujoco

from legged_mjlab.utils.paths import PROJECT_ROOT, resource_path
from mjlab.actuator import  IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
)
from mjlab.utils.spec_config import CollisionCfg


from legged_mjlab.envs.him_go2.him_go2_config import HimGo2RoughCfg


class Go2Asset:
    def __init__(self, cfg: HimGo2RoughCfg):
        self.cfg = cfg
        self._parse_cfg(self.cfg)

        self.entity = self.entitycfg(self)
        self.sensor_mgr = self.sensor(self)

    def _resolve_xml_path(self, raw_path: str) -> Path:
        """解析 MJCF 路径：替换占位符、解析绝对路径并校验存在性"""
        if not raw_path:
            raise ValueError("Asset xml 路径未在配置中指定 (cfg.asset.file 为空)")

        # 1. 兼容多种占位符格式与相对路径
        formatted_path = raw_path.format(
            LEGGED_MJLAB_ROOT_DIR=str(PROJECT_ROOT)
        ).replace("{LEGGED_MJLAB_ROOT_DIR}", str(PROJECT_ROOT))

        # 2. 转换为 Path 并获取规范绝对路径
        xml_path = Path(formatted_path).expanduser().resolve()

        # 3. 校验目标文件是否存在
        if not xml_path.is_file():
            raise FileNotFoundError(
                f"Go2 MJCF 文件不存在，解析路径为: {xml_path}\n"
                f"原始配置路径: {raw_path}"
            )

        return xml_path

    def _parse_cfg(self, cfg):
        # 解析并保存为 Path 对象
        raw_file = getattr(self.cfg.asset, "file", "")
        self.xml_path: Path = self._resolve_xml_path(raw_file)

        self.effort_limit = self.cfg.control.effort_limit
        self.stiffness = self.cfg.control.stiffness
        self.damping = self.cfg.control.damping
        self.armature = self.cfg.asset.armature

        self.pos = self.cfg.init_state.pos
        self.rot = self.cfg.init_state.rot
        self.default_joint_angles = self.cfg.init_state.default_joint_angles

        self.vel_limit = self.cfg.rewards.soft_dof_vel_limit
        self.pos_limit = self.cfg.rewards.soft_dof_pos_limit

        if self.cfg.domain_rand.randomize_cmd_action_latency:
            self.action_delay = self.cfg.domain_rand.range_cmd_action_latency
            self.action_delay_min = self.action_delay[0]
            self.action_delay_max = self.action_delay[1]
            self.action_hold_prob = self.cfg.domain_rand.action_hold_prob
        else:
            self.action_delay = 0
            self.action_delay_min = 0
            self.action_delay_max = 0
            self.action_hold_prob = 0

        # 从 MJCF 中读取真实 body 名称，排除 world body
        spec = self.asset.entity.get_spec()
        self.body_names = tuple(
            body.name for body in spec.worldbody.find_all("body")
        )

        # 根据配置中的字符串筛选 body
        self.penalized_contact_names = []
        for name in self.asset.cfg.asset.penalize_contacts_on:
            self.penalized_contact_names.extend(
                [s for s in self.body_names if name in s]
            )

        self.termination_contact_names = []
        for name in self.asset.cfg.asset.terminate_after_contacts_on:
            self.termination_contact_names.extend(
                [s for s in self.body_names if name in s]
            )

        self.penalized_contact_names = tuple(
            dict.fromkeys(self.penalized_contact_names)
        )
        self.termination_contact_names = tuple(
            dict.fromkeys(self.termination_contact_names)
        )

    class entitycfg():
        def __init__(self, asset: "Go2Asset"):
            self.asset = asset
        
        def get_spec(self) -> mujoco.MjSpec:
            """ 解析 MJCF 生成 MjSpec，并完成 assets 资源表绑定。
            """
            return mujoco.MjSpec.from_file(str(self.asset.xml_path))

        def _get_val(self, param: Any, key: str) -> float:
            if isinstance(param, Mapping):
                return float(param[key])
            return float(param)
            
        def add_actuator_cfg(self):
            ACTUATOR_HIP = IdealPdActuatorCfg(
                target_names_expr = (".*hip_.*",),
                stiffness = self._get_val(self.asset.stiffness, "hip"),
                damping = self._get_val(self.asset.damping, "hip"),
                effort_limit = self._get_val(self.asset.effort_limit, "hip"),
                delay_min_lag = self.asset.action_delay_min,
                delay_max_lag = self.asset.action_delay_max,
                delay_hold_prob = self.asset.action_hold_prob,      
                delay_update_period = 10,       
                armature = self.asset.armature,
            )

            ACTUATOR_THIGH = IdealPdActuatorCfg(
                target_names_expr = (".*thigh_.*",),
                stiffness = self._get_val(self.asset.stiffness, "thigh"),
                damping = self._get_val(self.asset.damping, "thigh"),
                effort_limit = self._get_val(self.asset.effort_limit, "thigh"),
                delay_min_lag = self.asset.action_delay_min,
                delay_max_lag = self.asset.action_delay_max,
                delay_hold_prob = self.asset.action_hold_prob,      
                delay_update_period = 10,  
                armature = self.asset.armature,
            )

            ACTUATOR_CALF = IdealPdActuatorCfg(
                target_names_expr = (".*calf_.*",),
                stiffness = self._get_val(self.asset.stiffness, "calf"),
                damping = self._get_val(self.asset.damping, "calf"),
                effort_limit = self._get_val(self.asset.effort_limit, "calf"),
                delay_min_lag = self.asset.action_delay_min,
                delay_max_lag = self.asset.action_delay_max,
                delay_hold_prob = self.asset.action_hold_prob,      
                delay_update_period = 10,   
                armature = self.asset.armature,
            )

            return (ACTUATOR_HIP, ACTUATOR_THIGH, ACTUATOR_CALF)

        def get_robot_cfg(self) -> EntityCfg:
            """ 组装并返回 Go2 实体的完整 EntityCfg
            """
            foot_regex = "^(FL|FR|RL|RR)_foot_collision$"
            static_friction = getattr(self.asset.cfg.terrain, "static_friction", 0.6)

            INIT_STATE = EntityCfg.InitialStateCfg(
                pos = self.asset.pos,
                rot = self.asset.rot,
                joint_pos=self.asset.default_joint_angles,
                joint_vel={".*": 0.0},
            )

            FULL_COLLISION = CollisionCfg(
                geom_names_expr = (".*_collision",),
                condim = {foot_regex: 3, ".*_collision": 1},
                priority = {foot_regex: 1, ".*": 0},
                friction = {foot_regex: (static_friction,), ".*": (0.6,)},
                solimp = {foot_regex: (0.9, 0.95, 0.023), ".*": (0.9, 0.95, 0.001)},
                contype = 1,
                conaffinity = 0,
            )

            ARTICULATION = EntityArticulationInfoCfg(
                actuators = self.add_actuator_cfg(),
                soft_joint_pos_limit_factor = self.asset.pos_limit
            )

            return EntityCfg(
                init_state = INIT_STATE,
                collisions = (FULL_COLLISION,),
                spec_fn = self.get_spec,
                articulation = ARTICULATION,
              )

    class sensor:
        def __init__(self, asset: "Go2Asset"):
            self.asset = asset

        def get_foot_contact_sensor(self, entity_name: str) -> ContactSensorCfg:
            """ 足端触地力与接触判定传感器。"""
            return ContactSensorCfg(
                name = "feet_ground_contact",                     # 传感器名称
                primary = ContactMatch(
                    mode = "geom",
                    pattern = r"^(FL|FR|RL|RR)_foot_collision$",
                    entity = entity_name,
                ),
                secondary = ContactMatch(
                    mode = "body",
                    pattern = "terrain",
                ),
                fields = ("found", "force"),
                reduce = "netforce",
                num_slots = 1,
                track_air_time = True,
                history_length = self.asset.cfg.control.decimation
            )

        def get_illegal_contact_sensor(self, entity_name: str) -> ContactSensorCfg:
            """ 非期望碰撞检测传感器
            """
            penalized_body_names = tuple(
                self.asset.penalized_contact_names
            )

            return ContactSensorCfg(
                name = "nonfoot_ground_touch",
                primary = ContactMatch(
                    mode = "body",
                    pattern = penalized_body_names,
                    entity = entity_name,
                ),
                secondary = ContactMatch(
                    mode = "body",
                    pattern = "terrain",
                ),
                fields = ("found", "force"),
                reduce = "maxforce",
                num_slots = 1,
                history_length = self.asset.cfg.control.decimation,
            )

        def get_height_scan_sensor(self, entity_name: str, debug_vis: bool = False) -> RayCastSensorCfg:
            """ 基座底部高程图扫描传感器（17x11=187 网格点）
            """
            x_points = tuple(self.asset.cfg.terrain.measured_points_x)
            y_points = tuple(self.asset.cfg.terrain.measured_points_y)

            return RayCastSensorCfg(
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
                    resolution = self.asset.cfg.terrain.horizontal_scale,  # 采样分辨率
                ),
                ray_alignment = "yaw",
                max_distance = 2.0,              
                debug_vis = bool(debug_vis),     # 是否在 GUI 中绘制扫描射线
                history_length = self.asset.cfg.control.decimation
            )

        def get_all_sensors(self, debug_vis: bool = False) -> Dict[str, Any]:
            sensors = {
                "feet_ground_contact": self.get_foot_contact_sensor(),
                "illegal_contact": self.get_illegal_contact_sensor(),
            }
            if getattr(self.asset.cfg.terrain, "measure_heights", False):
                sensors["height_scan"] = self.get_height_scan_sensor(debug_vis=debug_vis)
            return sensors


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    Go2_asset = Go2Asset(HimGo2RoughCfg())

    robot = Entity(Go2_asset.entity.get_robot_cfg())

    viewer.launch(robot.spec.compile())
