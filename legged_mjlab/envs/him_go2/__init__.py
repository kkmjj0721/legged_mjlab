from legged_mjlab.utils.task_registry import task_registry

from .him_go2_env import Go2Env
from .him_go2_config import HimGo2RoughCfg, HimGo2CfgPPO


task_registry.register(
    "him_go2",
    Go2Env,
    HimGo2RoughCfg,
    HimGo2CfgPPO,
)