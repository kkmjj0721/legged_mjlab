"""Canonical Go2 robot entity and scene-sensor configuration."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
import os
import re
from typing import Any, Dict, Tuple

import mujoco

from legged_mjlab.utils.paths import PROJECT_ROOT, resource_path
from mjlab.actuator import  IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
)
from mjlab.utils.spec_config import CollisionCfg


from .him_go2_config import HimGo2RoughCfg


class Go2Asset:
    def __init__(self, cfg: HimGo2RoughCfg):
        self.cfg = cfg
        self._parse_cfg(self.cfg)

        self.entity = self.entitycfg(self)
        self.sensor_mgr = self.sensor(self)


    def _parse_cfg(self, cfg):
        # 解析文件路径
        self.xml_path = self.cfg.asset.file

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

    class entitycfg():
        def __init__(self, asset: "Go2Asset"):
            self.asset = asset

        def get_assets(self, meshdir: str) -> dict[str, bytes]:
            """ 扫描并注入 MJCF 引用的 mesh/texture 外部二进制资源
            """
            assets: dict[str, bytes] = {}
            assets_dir = self.asset.xml_path.parent / "assets"
            if assets_dir.exists():
                update_assets(assets, assets_dir, meshdir)
            return assets
        
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
                geom_names_expr=(".*_collision",),
                condim={foot_regex: 3, ".*_collision": 1},
                priority={foot_regex: 1},
                friction={foot_regex: (static_friction,)},
                solimp={foot_regex: (0.9, 0.95, 0.023)},
                contype=1,
                conaffinity=0,
            )

            ARTICULATION = EntityArticulationInfoCfg(
                actuators=self.add_actuator_cfg(),
                soft_joint_pos_limit_factor = self.asset.pos_limit
            )

            return EntityCfg(
                init_state=INIT_STATE,
                collisions=(FULL_COLLISION,),
                spec_fn=self.get_spec,
                articulation=ARTICULATION,
              )

    class sensor:
        def __init__(self, asset: "Go2Asset"):
            self.asset = asset

        def get_foot_contact_sensor(self) -> ContactSensorCfg:
            """ 足端触地力与接触判定传感器。"""
            return ContactSensorCfg(
                prim_path=".*",
                target_names_expr=("^(FL|FR|RL|RR)_foot_collision$",),
                match=ContactMatch.GEOM_NAME,
                history_length=1,
                track_air_time=True,
                force_threshold=1.0,
            )

        def get_illegal_contact_sensor(self) -> ContactSensorCfg:
            """ 机身/大腿/小腿非期望碰撞检测传感器。
            """
            return ContactSensorCfg(
                prim_path=".*",
                target_names_expr=(r"^(base[123]_collision|(FL|FR|RL|RR)_(thigh_collision|calf[12]_collision))$",),
                match=ContactMatch.GEOM_NAME,
                history_length=1,
                track_air_time=False,
                force_threshold=getattr(
                    getattr(self.asset.cfg, "terminations", None),
                    "illegal_contact_force",
                    1.0,
                ),
            )

        def get_height_scan_sensor(self, debug_vis: bool = False) -> RayCastSensorCfg:
            """基座底部高程图扫描传感器（17x11=187 网格点）。
            """
            return RayCastSensorCfg(
                prim_path="base_link",
                attach_to_frame=True,
                pattern_cfg=GridPatternCfg(
                    size=(1.6, 1.0),
                    resolution=getattr(self.asset.cfg.terrain, "horizontal_scale", 0.1),
                ),
                ray_alignment="z_negative",
                ray_length=2.0,
                offset_pos=(0.0, 0.0, 0.5),
                debug_vis=debug_vis,
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