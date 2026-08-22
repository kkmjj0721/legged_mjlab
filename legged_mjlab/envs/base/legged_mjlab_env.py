import torch
import numpy as np
from typing import Tuple, Union, Dict
from rsl_rl.env import VecEnv
from mjlab.envs import ManagerBasedRlEnv



class LeggedMjlabEnv(VecEnv):
    """
    桥接 mjlab (MuJoCo Warp GPU 仿真) 与 rsl_rl (VecEnv) 的核心环境包装器
    """
    def __init__(self, cfg, sim_device="cuda:0", headless=True):
        self.cfg = cfg
        self.device = torch.device(sim_device)

        
    
    