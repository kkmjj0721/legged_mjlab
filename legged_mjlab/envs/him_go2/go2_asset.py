from copy import deepcopy
from functools import partial
from pathlib import Path

import mujoco

from legged_mjlab.utils.paths import resource_path
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from him_go2_config import HimGo2Config