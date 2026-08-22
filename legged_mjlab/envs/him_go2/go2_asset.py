import os
from legged_mjlab.utils.paths import PROJECT_ROOT, resource_path
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.spec_config import CollisionCfg

from .him_go2_config import HimGo2RounghCfg

class Go2Asset:
    pass