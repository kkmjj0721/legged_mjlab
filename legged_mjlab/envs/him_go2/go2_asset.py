""" 官方文档：Entity 与 Actuators 章节，各个版本的 mjlab 中实现会有一些差异 """

from pathlib import Path

import mujoco

from legged_mjlab.utils.paths import resource_path
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab.utils.spec_config import CollisionCfg


from .him_go2_config import HimGo2RounghCfg


