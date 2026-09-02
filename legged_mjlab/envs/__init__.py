from legged_mjlab.envs.him_go2.him_go2_config import HimGo2CfgPPO, HimGo2RoughCfg
from legged_mjlab.envs.him_go2.him_go2_env import HimGo2Env
from legged_mjlab.utils.task_registry import task_registry


task_registry.register("him_go2", HimGo2Env, HimGo2RoughCfg(), HimGo2CfgPPO())

__all__ = ["HimGo2Env"]
