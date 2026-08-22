from legged_mjlab.envs.him_go2.him_go2_config import HimGo2RoughCfg, HimGo2CfgPPO
from legged_mjlab.envs.him_go2.him_go2_env import HimGo2Env
from legged_mjlab.utils.task_registry import task_registry


task_registry.register(
    task_id="him_go2",
    env_cls=HimGo2Env,
    env_cfg_cls=HimGo2RoughCfg,
    train_cfg_cls=HimGo2CfgPPO,
    wrapper_name="him",
)
