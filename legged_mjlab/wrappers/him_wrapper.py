import torch
from .vec_env_wrapper import VecEnvWrapper


class HIMRslRlWrapper(VecEnvWrapper):
    """
    HIM-Loco 适配器 (对接 HIMOnPolicyRunner)。
    提供 num_one_step_obs、历史时序滑窗、拦截重置前末帧特权态并输出 7 元组。
    """

    def __init__(self, env, history_length=6):
        super().__init__(env)
        self.history_length = history_length

        obs_dict = self.env.observation_manager.compute()
        
        # HIMActorCritic 必需的单步观测维度
        self.num_one_step_obs = obs_dict["policy"].shape[-1]
        self.num_obs = self.num_one_step_obs * self.history_length

        if "critic" in obs_dict:
            self.num_privileged_obs = obs_dict["critic"].shape[-1]
            self.privileged_obs_buf = obs_dict["critic"]
            self._prev_privileged_obs = torch.zeros(
                (self.num_envs, self.num_privileged_obs),
                device=self.device,
                dtype=torch.float,
            )
        else:
            self.num_privileged_obs = None
            self.privileged_obs_buf = None
            self._prev_privileged_obs = None

        # 历史滑窗队列: [num_envs, history_length, num_one_step_obs]
        self.obs_history_buf = torch.zeros(
            (self.num_envs, self.history_length, self.num_one_step_obs),
            device=self.device,
            dtype=torch.float,
        )

        # 填充初始帧
        self.obs_history_buf[:, 0] = obs_dict["policy"]
        self.obs_buf = self.obs_history_buf.view(self.num_envs, -1)

    def step(self, actions):
        # 1. 在 mjlab 执行 auto-reset 覆盖前，缓存当前步真实特权态
        if self.privileged_obs_buf is not None:
            self._prev_privileged_obs.copy_(self.privileged_obs_buf)

        # 2. 底层步进
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        # 3. 提取终止信号与 ID
        dones = (terminated | truncated).to(torch.uint8)
        self.reset_buf = dones
        termination_ids = dones.nonzero(as_tuple=False).flatten()

        # 4. 捕获重置前的真实末帧 (用于 HIMEstimator 的自监督对比学习和速度预测)
        if self._prev_privileged_obs is not None and len(termination_ids) > 0:
            termination_privileged_obs = self._prev_privileged_obs[termination_ids].clone()
        else:
            termination_privileged_obs = torch.empty((0, self.num_privileged_obs), device=self.device) if self.num_privileged_obs else None

        # 5. 更新特权观测与历史滑窗
        if "critic" in obs_dict:
            self.privileged_obs_buf = obs_dict["critic"]

        raw_one_step_obs = obs_dict["policy"]
        if len(termination_ids) > 0:
            self.obs_history_buf[termination_ids] = 0.0

        self.obs_history_buf = torch.roll(self.obs_history_buf, shifts=1, dims=1)
        self.obs_history_buf[:, 0] = raw_one_step_obs
        self.obs_buf = self.obs_history_buf.view(self.num_envs, -1)

        # 6. 维护回合步数与 extras
        self.rew_buf = rew.squeeze(-1) if rew.ndim > 1 else rew
        
        if hasattr(self.env, "episode_length_buf"):
            self.episode_length_buf = self.env.episode_length_buf
        else:
            self.episode_length_buf += 1
            if len(termination_ids) > 0:
                self.episode_length_buf[termination_ids] = 0

        self.extras = extras if extras is not None else {}
        self.extras["time_outs"] = truncated

        # 7. 返回 HIMOnPolicyRunner 必需的 7 元组
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            termination_ids,
            termination_privileged_obs,
        )

    def reset(self, env_ids=None):
        obs_dict, extras = self.env.reset(env_ids=env_ids)
        raw_one_step_obs = obs_dict["policy"]
        if "critic" in obs_dict:
            self.privileged_obs_buf = obs_dict["critic"]

        if env_ids is None:
            self.obs_history_buf.zero_()
            self.obs_history_buf[:, 0] = raw_one_step_obs
            self.episode_length_buf.zero_()
        else:
            self.obs_history_buf[env_ids] = 0.0
            self.obs_history_buf[env_ids, 0] = raw_one_step_obs[env_ids] if raw_one_step_obs.ndim > 1 else raw_one_step_obs
            self.episode_length_buf[env_ids] = 0

        self.obs_buf = self.obs_history_buf.view(self.num_envs, -1)
        self.extras = extras if extras is not None else {}
        return self.obs_buf, self.privileged_obs_buf