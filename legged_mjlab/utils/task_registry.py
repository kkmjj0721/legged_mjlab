"""Task registration and the narrow environment/runner construction boundary."""

import importlib
import importlib.util
import sys
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from legged_mjlab.wrappers import HIMRslRlWrapper, RslRlVecEnvWrapper


@dataclass
class TaskSpec:
    task_id: str
    env_cls: type
    env_cfg_cls: type
    train_cfg_cls: type
    wrapper_name: str


def _positive_int(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _train_config_dict(train_cfg):
    if isinstance(train_cfg, Mapping):
        return dict(train_cfg)
    to_dict = getattr(train_cfg, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("train_cfg must be a mapping or a BaseConfig instance")
    config = to_dict()
    if not isinstance(config, Mapping):
        raise TypeError("BaseConfig.to_dict() must return a mapping")
    return dict(config)


def _rsl_module_source(module):
    """Return a diagnostic source string for an imported ``rsl_rl`` module."""

    origin = getattr(module, "__file__", None)
    if origin:
        return str(origin)
    locations = getattr(module, "__path__", None)
    if locations:
        return "namespace package (" + ", ".join(map(str, locations)) + ")"
    return "unknown source"


def _project_rsl_package():
    """Return the source package directory and initializer for this checkout."""

    package_dir = Path(__file__).resolve().parents[2] / "rsl_rl" / "rsl_rl"
    return package_dir, package_dir / "__init__.py"


def _rsl_modules_snapshot():
    return {
        name: module
        for name, module in sys.modules.items()
        if name == "rsl_rl" or name.startswith("rsl_rl.")
    }


def _remove_rsl_modules():
    for name in list(sys.modules):
        if name == "rsl_rl" or name.startswith("rsl_rl."):
            sys.modules.pop(name, None)


def _load_project_rsl_source(reason):
    """Load the repository backend as one complete ``rsl_rl`` namespace.

    The source tree has an outer ``rsl_rl/`` directory and the importable
    package below it.  Loading the package with its canonical name means its
    absolute imports (for example ``rsl_rl.modules``) remain self-consistent.
    Existing modules in that namespace are removed first so a failed or
    incomplete site-packages backend cannot be mixed with the source tree.
    """

    package_dir, init_file = _project_rsl_package()
    original_modules = _rsl_modules_snapshot()
    _remove_rsl_modules()

    try:
        if not init_file.is_file():
            raise FileNotFoundError(
                f"project rsl_rl initializer does not exist: {init_file}"
            )

        spec = importlib.util.spec_from_file_location(
            "rsl_rl",
            init_file,
            submodule_search_locations=[str(package_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"could not create import spec for {init_file}")

        project_module = importlib.util.module_from_spec(spec)
        sys.modules["rsl_rl"] = project_module
        spec.loader.exec_module(project_module)
        project_runners = importlib.import_module("rsl_rl.runners")
        if getattr(project_runners, "HIMOnPolicyRunner", None) is None:
            raise AttributeError(
                f"project rsl_rl runners at {project_runners.__file__} does not "
                "export HIMOnPolicyRunner"
            )
    except Exception as exc:
        _remove_rsl_modules()
        sys.modules.update(original_modules)
        raise RuntimeError(
            "unable to load the repository rsl_rl backend after rejecting the "
            f"currently imported backend ({reason}); expected package at "
            f"{package_dir}; {type(exc).__name__}: {exc}"
        ) from exc

    warnings.warn(
        "the imported rsl_rl backend is incompatible with this project "
        f"({reason}); using the repository backend at {package_dir}",
        RuntimeWarning,
        stacklevel=3,
    )
    return project_runners


def load_project_rsl():
    """Load and validate the RSL-RL backend exposed by this project.

    A compatible installed backend remains the default.  When the interpreter
    resolves a same-named but incompatible package, the source checkout is
    loaded as a complete namespace so no installed submodule can be reused.
    """

    imported = None
    imported_source = "not imported"
    try:
        imported = importlib.import_module("rsl_rl")
    except Exception as exc:
        reason = (
            f"rsl_rl could not be imported ({type(exc).__name__}: {exc}); "
            f"source={imported_source}"
        )
        return _load_project_rsl_source(reason)

    imported_source = _rsl_module_source(imported)
    try:
        runners = importlib.import_module("rsl_rl.runners")
    except Exception as exc:
        reason = (
            f"imported from {imported_source}, but rsl_rl.runners failed "
            f"with {type(exc).__name__}: {exc}"
        )
        return _load_project_rsl_source(reason)

    if getattr(runners, "HIMOnPolicyRunner", None) is not None:
        return runners

    reason = (
        f"imported from {imported_source}, but rsl_rl.runners has no "
        "HIMOnPolicyRunner"
    )
    return _load_project_rsl_source(reason)


class TaskRegistry:
    def __init__(self):
        self.task_specs = {}

    def register(self, task_id, env_cls, env_cfg_cls, train_cfg_cls, wrapper_name):
        if task_id in self.task_specs:
            raise KeyError("task already registered: " + task_id)
        if wrapper_name not in {"ppo", "him", "amp"}:
            raise ValueError("unsupported wrapper: " + wrapper_name)
        self.task_specs[task_id] = TaskSpec(
            task_id=task_id,
            env_cls=env_cls,
            env_cfg_cls=env_cfg_cls,
            train_cfg_cls=train_cfg_cls,
            wrapper_name=wrapper_name,
        )

    def list_tasks(self):
        return tuple(sorted(self.task_specs))

    def get(self, task_id):
        if task_id not in self.task_specs:
            raise KeyError(
                "unknown task " + task_id + "; choices=" + str(self.list_tasks())
            )
        return self.task_specs[task_id]

    def make_env(self, task_id, device=None, play=False, num_envs=None):
        spec = self.get(task_id)
        env_cfg = spec.env_cfg_cls()
        if num_envs is not None:
            env_cfg.env.num_envs = _positive_int(num_envs, "num_envs")

        configured_device = getattr(env_cfg.env, "device", "cuda:0")
        env = spec.env_cls(
            cfg=env_cfg,
            device=device if device is not None else configured_device,
            play=play,
        )
        if spec.wrapper_name == "him":
            history_length = getattr(
                env_cfg.env,
                "history_length",
                getattr(getattr(env_cfg, "him", None), "history_length", 6),
            )
            one_step_obs_dim = getattr(
                env_cfg.env, "num_one_step_observations", 45
            )
            expected_privileged_obs_dim = getattr(
                env_cfg.env, "num_privileged_obs", None
            )
            action_dim = getattr(env_cfg.env, "num_actions", None)
            return HIMRslRlWrapper(
                env,
                history_length=history_length,
                one_step_obs_dim=one_step_obs_dim,
                expected_privileged_obs_dim=expected_privileged_obs_dim,
                action_dim=action_dim,
            ), env_cfg
        if spec.wrapper_name == "ppo":
            return RslRlVecEnvWrapper(env), env_cfg
        if spec.wrapper_name == "amp":
            try:
                from legged_mjlab.wrappers.amp_wrapper import AMPRslRlWrapper
            except (ImportError, ModuleNotFoundError) as exc:
                raise RuntimeError(
                    "AMP task selected but legged_mjlab.wrappers.amp_wrapper "
                    "is not available"
                ) from exc
            return AMPRslRlWrapper(env), env_cfg
        raise ValueError("unsupported wrapper: " + spec.wrapper_name)

    def make_alg_runner(self, task_id, env, train_cfg, log_dir):
        self.get(task_id)
        config = _train_config_dict(train_cfg)
        try:
            runner_cfg = config["runner"]
            runner_name = runner_cfg["runner_class_name"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "train_cfg must contain runner.runner_class_name"
            ) from exc

        runners = load_project_rsl()
        runner_cls = getattr(runners, runner_name, None)
        if runner_cls is None:
            raise ValueError("unsupported runner: " + str(runner_name))
        return runner_cls(
            env,
            config,
            log_dir,
            device=getattr(env, "device", "cpu"),
        )


task_registry = TaskRegistry()


__all__ = ["TaskRegistry", "TaskSpec", "load_project_rsl", "task_registry"]
