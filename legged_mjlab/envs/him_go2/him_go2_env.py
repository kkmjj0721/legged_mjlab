import torch

from mjlab.envs import ManagerBasedRlEnv
from .him_go2_config import HimGo2RounghCfg


class Go2Env(ManagerBasedRlEnv):
    def __init__(self, cfg: HimGo2RounghCfg, sim_device, headless):
        self.cfg = cfg
        super().__init__(
            cfg=self._build_mjlab_cfg(cfg),
            device=sim_device,
            render_mode=None if headless else "human",
        )

    def _build_mjlab_cfg(self, cfg):
        return compile_go2_env_cfg(cfg)


    def _reward_hip_reduction(self):
        hip_yaw = self.data.joint_pos[:, self.hip_yaw_ids]
        reduction = self.cfg.control.hip_reduction
        return -reduction * torch.sum(torch.square(hip_yaw), dim=-1)