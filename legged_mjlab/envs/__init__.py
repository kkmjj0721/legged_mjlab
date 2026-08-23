"""Environment package and deterministic task-registration entry point."""

from legged_mjlab.utils.task_registry import task_registry


def _register_him_go2():
    try:
        from legged_mjlab.envs.him_go2 import HimGo2Env
    except (ImportError, ModuleNotFoundError) as exc:
        # Keep package/config discovery and ``train --help`` usable when the
        # optional simulator stack is absent.  Construction still raises the
        # original dependency/interface error at the point it is requested.
        from legged_mjlab.envs.him_go2.him_go2_config import (
            HimGo2CfgPPO,
            HimGo2RoughCfg,
        )

        class HimGo2Env:
            __name__ = "HimGo2Env"
            import_error = exc

            def __new__(cls, *args, **kwargs):
                raise RuntimeError(
                    "him_go2 environment could not be imported; install the "
                    "mjlab runtime and resolve the core environment import "
                    "error before constructing it"
                ) from cls.import_error

        task_registry.register(
            task_id="him_go2",
            env_cls=HimGo2Env,
            env_cfg_cls=HimGo2RoughCfg,
            train_cfg_cls=HimGo2CfgPPO,
            wrapper_name="him",
        )
        return HimGo2Env

    # him_go2/__init__.py normally performs this registration itself.  The
    # guard also makes this package robust to a registration-only task module.
    if "him_go2" not in task_registry.list_tasks():
        from legged_mjlab.envs.him_go2.him_go2_config import (
            HimGo2CfgPPO,
            HimGo2RoughCfg,
        )

        task_registry.register(
            task_id="him_go2",
            env_cls=HimGo2Env,
            env_cfg_cls=HimGo2RoughCfg,
            train_cfg_cls=HimGo2CfgPPO,
            wrapper_name="him",
        )
    return HimGo2Env


HimGo2Env = _register_him_go2()

__all__ = ["HimGo2Env"]
