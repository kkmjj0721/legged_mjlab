import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.runners import HIMOnPolicyRunner 

from legged_mjlab.wrappers import HIMRslRlWrapper, RslRlVecEnvWrapper

from legged_mjlab.utils.helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed
from legged_mjlab.utils.paths import PROJECT_ROOT

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
        """基于 mjlab 参数创建环境，并自动套上 rsl_rl wrapper"""
        if args is None:
            args = get_args()
            
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
            
        if env_cfg is None:
            env_cfg, _ = self.get_cfgs(name)
            
        # env_cfg, _ = update_cfg_from_args(env_cfg, None, args) # 根据你的 helpers 接口调整
        set_seed(env_cfg.seed)

        render_mode = None if args.headless else "human"

        env = task_class(
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
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
                
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

        resume = train_cfg.runner.resume
        if resume:
            resume_path = train_path if train_path is not None else get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
            
        return runner, train_cfg

task_registry = TaskRegistry()