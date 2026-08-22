import os

from rsl_rl

from legged_mjlab.wrappers import HIMRslRlWrapper
from legged_mjlab.utils.helpers import class_to_dict


class TaskRegistry:
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(self, name, task_class, env_cfg, train_cfg):
        if name in self.task_classes:
            raise ValueError(f"task already registered: {name}")
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_cfgs(self, name):
        if name not in self.task_classes:
            raise KeyError(f"unknown task: {name}")
        return self.env_cfgs[name](), self.train_cfgs[name]()

    def make_env(self, name, args=None, env_cfg=None):
        if env_cfg is None:
            env_cfg, train_cfg = self.get_cfgs(name)
        else:
            _, train_cfg = self.get_cfgs(name)

        if args is not None and args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs

        env = self.task_classes[name](
            cfg=env_cfg,
            sim_device=args.sim_device if args else "cuda:0",
            headless=args.headless if args else False,
        )

        runner_name = train_cfg.runner.runner_class_name
        if runner_name == "HIMOnPolicyRunner":
            env = HIMRslRlWrapper(
                env,
                history_length=env_cfg.env.history_length,
                one_step_obs_dim=env_cfg.env.num_one_step_observations,
            )
        else:
            env = RslRlVecEnvWrapper(env)

        return env, env_cfg

    def make_alg_runner(self, env, name, train_cfg=None, log_root="logs"):
        if train_cfg is None:
            _, train_cfg = self.get_cfgs(name)

        log_dir = None
        if log_root is not None:
            log_dir = os.path.join(
                log_root,
                train_cfg.runner.experiment_name,
                train_cfg.runner.run_name or "default",
            )
            os.makedirs(log_dir, exist_ok=True)

        cfg = class_to_dict(train_cfg)
        runner_name = cfg["runner"]["runner_class_name"]
        if runner_name == "HIMOnPolicyRunner":
            from rsl_rl.runners import HIMOnPolicyRunner
            runner_cls = HIMOnPolicyRunner
        elif runner_name == "OnPolicyRunner":
            runner_cls = OnPolicyRunner
        else:
            raise ValueError(f"unsupported runner: {runner_name}")

        return runner_cls(
            env,
            cfg,
            log_dir=log_dir,
            device=env.device,
        ), train_cfg


task_registry = TaskRegistry()