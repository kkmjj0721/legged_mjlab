import os
import sys
from datetime import datetime
import torch

from legged_mjlab.utils.paths import PROJECT_ROOT

local_rsl_rl_path = os.path.join(PROJECT_ROOT.parent, "rsl_rl") 
if os.path.exists(local_rsl_rl_path):
    sys.path.insert(0, local_rsl_rl_path)

from rsl_rl.runners import OnPolicyRunner
try:
    from rsl_rl.runners import HIMOnPolicyRunner
except ImportError:
    print("Warning: HIMOnPolicyRunner not found in local rsl_rl. Fallback to OnPolicyRunner.")
    HIMOnPolicyRunner = OnPolicyRunner

from legged_mjlab.wrappers import HIMRslRlWrapper, RslRlVecEnvWrapper
from legged_mjlab.utils.helpers import get_args, class_to_dict, get_load_path, set_seed


class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(self, name: str, task_class, env_cfg, train_cfg):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str):
        return self.task_classes[name]

    def get_cfgs(self, name):
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg

    def make_env(self, name, args=None, env_cfg=None):
        if args is None:
            args = get_args()
            
        if name not in self.task_classes:
            raise ValueError(f"Task with name: {name} was not registered")
            
        if env_cfg is None:
            env_cfg, _ = self.get_cfgs(name)
            
        set_seed(env_cfg.seed)

        render_mode = None if getattr(args, "headless", False) else "human"
        
        env = self.task_classes[name](
            cfg=env_cfg,
            sim_device=args.sim_device,
            render_mode=render_mode,
            play=getattr(args, "play", False)
        )
        
        runner_name = getattr(self.train_cfgs[name].runner, "runner_class_name", "OnPolicyRunner")
        if runner_name == "HIMOnPolicyRunner":
            env = HIMRslRlWrapper(
                env,
                history_length=env_cfg.env.history_length,
                one_step_obs_dim=env_cfg.env.num_one_step_observations,
                expected_privileged_obs_dim=env_cfg.env.num_privileged_obs,
                action_dim=env_cfg.env.num_actions
            )
        else:
            env = RslRlVecEnvWrapper(env)

        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default", train_path=None):
        if args is None:
            args = get_args()
            
        if train_cfg is None:
            _, train_cfg = self.get_cfgs(name)

        if log_root == "default":
            log_root = os.path.join(PROJECT_ROOT, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + getattr(train_cfg.runner, 'run_name', ''))
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + getattr(train_cfg.runner, 'run_name', ''))
        
        train_cfg_dict = class_to_dict(train_cfg)

        runner_class_name = getattr(train_cfg.runner, "runner_class_name", "OnPolicyRunner")
        if runner_class_name == "HIMOnPolicyRunner":
            runner = HIMOnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
        else:
            runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)

        if train_cfg.runner.resume:
            resume_path = train_path if train_path is not None else get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
            
        return runner, train_cfg

task_registry = TaskRegistry()