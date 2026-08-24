import torch


from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg


from legged_mjlab.envs.him_go2.go2_asset import Go2Asset
from legged_mjlab.envs.him_go2.him_go2_config import HimGo2RoughCfg



class HimGo2Env(ManagerBasedRlEnv):
    def __init__(self, cfg: HimGo2RoughCfg, sim_device, render_mode):
        """ 
        Args:
            cfg (HimGo2RoughCfg): 配置对象，包含环境、控制器和资产的参数
            sim_device: 模拟设备（如 CPU 或 GPU）
            render_mode: 渲染模式（如 "human" 或 "rgb_array"）
        """
        self.cfg = cfg

        self.managercfg = self._build_mjlab_managercfg(self.cfg)

        super().__init__(
            cfg = self.managercfg,
            device = sim_device,
            render_mode = ,
        )


    def _build_mjlab_managercfg(self, cfg) -> ManagerBasedRlEnvCfg:
        """ 将 HimGo2RoughCfg 转换为 ManagerBasedRlEnvCfg 以便于使用 mjlab 的管理器进行环境管理。
        """
        asset = Go2Asset(cfg)


    def _build_scene(cls, cfg, asset):
        """ 构建场景对象，包含地形和机器人实体。
        """

    def _build_actions(self):
        """
        """

    def _build_commands(self):
        """
        """

    def _build_rewards(self):
        """
        """

        
