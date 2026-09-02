import os
import copy
import torch
import numpy as np
import random
import torch.nn.functional as F
import argparse

def class_to_dict(obj) -> dict:
    """ 将类实例转为字典
    """
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    """ 用字典来更新类实例属性
    """
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    """ 定义固定随机种子的函数
    """
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_load_path(root, load_run=-1, checkpoint=-1):
    """ 定义获取预训练模型路径的函数
    """
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    """ 用命令行参数覆盖默认配置的方法
    """
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        if args.seed is not None:
            env_cfg.seed = args.seed
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

def get_args():
    """ 解析终端命令行输入参数的函数[
    """
    parser = argparse.ArgumentParser(description="RL Policy")

    parser.add_argument("--sim_device", type=str, default="cuda:0", help="Device for MuJoCo simulation (cpu or cuda)")

    parser.add_argument("--task", type=str, default="aliengo", help="Resume training or start testing from a checkpoint. Overrides config file if provided.")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from a checkpoint")
    parser.add_argument("--experiment_name", type=str, help="Name of the experiment to run or load. Overrides config file if provided.")
    parser.add_argument("--run_name", type=str, help="Name of the run. Overrides config file if provided.")
    parser.add_argument("--load_run", type=str, help="Name of the run to load when resume=True. If -1: will load the last run.")
    parser.add_argument("--checkpoint", type=int, help="Saved model checkpoint number. If -1: will load the last checkpoint.")
    
    parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times")
    parser.add_argument("--horovod", action="store_true", default=False, help="Use horovod for multi-gpu training")
    parser.add_argument("--rl_device", type=str, default="cuda:0", help="Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)")
    parser.add_argument("--num_envs", type=int, help="Number of environments to create. Overrides config file if provided.")
    parser.add_argument("--seed", type=int, help="Random seed. Overrides config file if provided.")
    parser.add_argument("--max_iterations", type=int, help="Maximum number of training iterations.")

    args = parser.parse_args()

    args.sim_device = args.rl_device
    
    return args

