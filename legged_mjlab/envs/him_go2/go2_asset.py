"""Canonical Go2 robot entity and scene-sensor configuration."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
import os
import re
from typing import Any, Dict, Tuple

import mujoco

from legged_mjlab.utils.paths import PROJECT_ROOT, resource_path
from mjlab.actuator import BuiltinPositionActuatorCfg, IdealPdActuatorCfg
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

        self.entity = self.EntityCfg(self)
        self.sensor_mgr = self.sensor(self)


    def _parse_cfg(self, cfg):
        self.xml_path = self.cfg.asset.file

        self.effort_limit = self.cfg.control.effort_limit
        self.stiffness = self.cfg.control.stiffness
        self.damping = self.cfg.control.damping
        self.armature = self.cfg.asset.armature

        self.pos = self.cfg.init_state.pos
        self.rot = self.cfg.init_state.rot
        self.default_joint_angles = self.cfg.init_state.default_joint_angles

        if self.cfg.domain_rand.randomize_cmd_action_latency:

            self.action_delay = self.cfg.domain_rand.range_cmd_action_latency
            self.action_delay_min = self.action_delay[0]
            self.action_delay_max = self.action_delay[1]
            self.action_hold_prob = self.cfg.domain_rand.action_hold_prob

    class EntityCfg():
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
            spec = mujoco.MjSpec.from_file(str(self.asset.xml_path))
            spec.assets = self.get_assets(spec.meshdir)
            return spec

        def _get_val(self, param: Any, key: str, default: float) -> float:
            if isinstance(param, Mapping):
                return float(param.get(key, param.get(".*", default)))
            elif isinstance(param, (int, float)):
                return float(param)
            return default
            
        def add_actuator_cfg(self):
            ACTUATOR_HIP = IdealPdActuatorCfg(
                target_names_expr = (".*hip_.*",),
                stiffness = self._get_val(self.asset.stiffness, "hip"),
                damping = self._get_val(self.asset.damping, "hip"),
                effort_limit = 
                delay_min_lag =
                delay_max_lag =
                delay_hold_prob =          
                delay_update_period =      
            )

            ACTUATOR_THIGH = IdealPdActuatorCfg(
                target_names_expr = (".*thigh_.*",),
                stiffness =
                damping = 
                effort_limit = 
                delay_min_lag =
                delay_max_lag =
                delay_hold_prob =          
                delay_update_period =      
            )

            ACTUATOR_CALF = IdealPdActuatorCfg(
                target_names_expr = (".*calf_.*",),
                stiffness =
                damping = 
                effort_limit = 
                delay_min_lag =
                delay_max_lag =
                delay_hold_prob =          
                delay_update_period =      
            )

        def get_robot_cfg():

        



    class sensor:
        pass

