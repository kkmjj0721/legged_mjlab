import os
from typing import Tuple
import torch
from rsl_rl.runners import OnPolicyRunner, HIMOnPolicyRunner
from rsl_rl.env import VecEnv
from legged_mjlab.utils.helpers import class_to_dict

class TaskRegistry:
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(self, name: str, task_class: type, env_cfg, train_cfg):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_cfgs(self, name: str):
        return self.env_cfgs[name](), self.train_cfgs[name]()

    def make_env(self, name: str, args=None, env_cfg=None) -> Tuple[VecEnv, object]:
        if env_cfg is None:
            env_cfg, _ = self.get_cfgs(name)
        if args is not None and getattr(args, "num_envs", None) is not None:
            env_cfg.env.num_envs = args.num_envs
        task_class = self.task_classes[name]
        env = task_class(
            cfg=env_cfg,
            sim_device=args.sim_device if args else "cuda:0",
            headless=args.headless if args else False,
        )
        return env, env_cfg

    def make_alg_runner(self, env: VecEnv, name: str, args=None, train_cfg=None, log_root="logs"):
        if train_cfg is None:
            _, train_cfg = self.get_cfgs(name)
        
        log_dir = None
        if log_root is not None:
            exp_name = train_cfg.runner.experiment_name
            run_name = train_cfg.runner.run_name if train_cfg.runner.run_name else "default"
            log_dir = os.path.join(log_root, exp_name, run_name)
            os.makedirs(log_dir, exist_ok=True)

        train_cfg_dict = class_to_dict(train_cfg)
        runner_cls_name = getattr(train_cfg.runner, "runner_class_name", "HIMOnPolicyRunner")
        runner_cls = HIMOnPolicyRunner if runner_cls_name == "HIMOnPolicyRunner" else OnPolicyRunner
        runner = runner_cls(env, train_cfg_dict, log_dir=log_dir, device=env.device)
        return runner, train_cfg

task_registry = TaskRegistry()