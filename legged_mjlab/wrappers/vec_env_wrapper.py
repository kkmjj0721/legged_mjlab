from typing import Any


class VecEnvWrapper:
    def __init__(self, env):
        self.env = env
        self._last_obs = None
        self._last_privileged_obs = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def device(self):
        return self.env.device

    @property
    def num_envs(self):
        return self.env.num_envs

    def reset(self):
        result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs_dict, infos = result
        else:
            obs_dict, infos = result, {}
        self._last_obs = obs_dict
        self._last_privileged_obs = None
        return obs_dict, infos

    def get_observations(self):
        return self._last_obs

    def get_privileged_observations(self):
        return self._last_privileged_obs

    def step(self, actions):
        result = self.env.step(actions)
        if len(result) != 5:
            raise ValueError(
                "native env must return obs, reward, terminated, truncated, infos"
            )
        self._last_obs = result[0]
        return result

    def close(self):
        close = getattr(self.env, "close", None)
        if callable(close):
            close()