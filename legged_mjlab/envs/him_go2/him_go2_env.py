import torch


from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg


from legged_mjlab.envs.him_go2.go2_asset import Go2Asset
from legged_mjlab.envs.him_go2.him_go2_config import HimGo2RoughCfg



class HimGo2Env(ManagerBasedRlEnv):
    def __init__(self, cfg: HimGo2RoughCfg, sim_device):
        self.cfg = cfg

        self.managercfg = self._build_mjlab_managercfg(self.cfg)

        super().__init__(
            cfg = self.managercfg,
            device = sim_device,
            render_mode = ,
        )


    def _build_mjlab_managercfg(self, cfg):
        pass
