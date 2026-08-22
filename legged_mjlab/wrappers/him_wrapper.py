import torch

from legged_mjlab.wrappers.vec_env_wrapper import VecEnvWrapper


class HIMRslRlWrapper(VecEnvWrapper):
    def __init__(self, env, history_length):
        super().__init__(env)
        if history_length < 1:
            raise ValueError("history_length must be positive")
        self.history_length = int(history_length)
        self.obs_history_buf = None
        self.termination_privileged_obs = None

    @staticmethod
    def _flatten_group(value, name):
        if value is None:
            return None
        if not isinstance(value, torch.Tensor):
            raise TypeError(name + " must be a tensor")
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim > 2:
            value = value.reshape(value.shape[0], -1)
        if value.ndim != 2:
            raise ValueError(name + " must be [N, D]")
        return value

    def _split(self, obs_dict):
        if not isinstance(obs_dict, dict):
            raise TypeError("native observation must be a dictionary")
        actor = obs_dict.get("actor", obs_dict.get("policy"))
        critic = obs_dict.get("critic", obs_dict.get("privileged"))
        if actor is None:
            raise KeyError("observation dictionary has no actor or policy group")
        return (
            self._flatten_group(actor, "actor"),
            self._flatten_group(critic, "critic"),
        )

    def _ensure_history(self, obs):
        if self.obs_history_buf is None:
            self.obs_history_buf = torch.zeros(
                (
                    obs.shape[0],
                    self.history_length,
                    obs.shape[1],
                ),
                device=obs.device,
                dtype=obs.dtype,
            )

    def _append(self, obs):
        self._ensure_history(obs)
        self.obs_history_buf[:, 1:] = self.obs_history_buf[:, :-1].clone()
        self.obs_history_buf[:, 0] = obs
        return self.obs_history_buf.reshape(obs.shape[0], -1)

    def reset(self):
        obs, privileged = super().reset()
        self.obs_history_buf = None
        history = self._append(obs)
        self._last_obs = history
        self._last_privileged_obs = privileged
        return history, privileged

    def step(self, actions):
        result = self.env.step(actions)
        obs_dict, rewards, terminated, truncated, infos = result
        obs, privileged = self._split(obs_dict)
        self._ensure_history(obs)
        done = torch.as_tensor(
            terminated, device=obs.device, dtype=torch.bool
        ).reshape(-1)
        timeout = torch.as_tensor(
            truncated, device=obs.device, dtype=torch.bool
        ).reshape(-1)
        terminal_privileged = privileged.clone() if privileged is not None else None
        history = self._append(obs)
        infos = dict(infos or {})
        infos["terminated"] = done
        infos["truncated"] = timeout
        infos["time_outs"] = timeout & ~done
        infos["termination_privileged_obs"] = terminal_privileged
        self.termination_privileged_obs = terminal_privileged
        self._last_obs = history
        self._last_privileged_obs = privileged
        rewards = torch.as_tensor(
            rewards, device=obs.device, dtype=torch.float32
        ).reshape(-1)
        return (
            history,
            privileged,
            rewards,
            done | timeout,
            infos,
            torch.nonzero(done | timeout, as_tuple=False).flatten(),
            terminal_privileged,
        )