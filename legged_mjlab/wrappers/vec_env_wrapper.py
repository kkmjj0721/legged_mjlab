import torch
from rsl_rl.env import VecEnv


class VecEnvWrapper(VecEnv):
    """
    向量化环境抽象基类包装器。
    严格对齐 rsl_rl.env.VecEnv 声明的全部属性与抽象方法。
    """
    def __init__(self, env):
        self.env = env
        self.device = env.device if hasattr(env, "device") else torch.device("cuda:0")
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_length

        # 提取观测维度
        if hasattr(env, "num_obs"):
            self.num_obs = env.num_obs
            self.num_privileged_obs = getattr(env, "num_privileged_obs", None)
        else:
            obs_dict = env.observation_manager.compute()
            self.num_obs = obs_dict["policy"].shape[-1]
            self.num_privileged_obs = obs_dict["critic"].shape[-1] if "critic" in obs_dict else None

        # 初始化 VecEnv 所需的核心张量缓冲区
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros((self.num_envs, self.num_privileged_obs), device=self.device, dtype=torch.float)
        else:
            self.privileged_obs_buf = None

        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.uint8)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

    def step(self, actions):
        return self.env.step(actions)

    def reset(self, env_ids=None):
        return self.env.reset(env_ids=env_ids)

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def __getattr__(self, name):
        return getattr(self.env, name)
