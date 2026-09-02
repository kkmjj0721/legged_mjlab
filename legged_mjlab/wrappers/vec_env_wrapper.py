from typing import Any


def _optional_positive_int(value, name):
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


class VecEnvWrapper:
    """Small forwarding adapter shared by the RSL-RL wrappers.

    The native mjlab environment owns the simulator state.  This class only
    stores the most recent RSL-RL observations and exposes the metadata that a
    runner reads before the first reset.
    """

    def __init__(self, env):
        self.env = env
        self._last_obs = None
        self._last_privileged_obs = None
        self._num_envs = _optional_positive_int(
            getattr(env, "num_envs", None), "num_envs"
        )
        self._num_obs = _optional_positive_int(
            getattr(env, "num_obs", None), "num_obs"
        )
        self._num_one_step_obs = _optional_positive_int(
            getattr(env, "num_one_step_obs", None)
            or getattr(env, "num_one_step_observations", None),
            "num_one_step_obs",
        )
        self._num_privileged_obs = _optional_positive_int(
            getattr(env, "num_privileged_obs", None), "num_privileged_obs"
        )
        self._num_actions = _optional_positive_int(
            getattr(env, "num_actions", None), "num_actions"
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def num_obs(self):
        return self._num_obs

    @property
    def num_one_step_obs(self):
        return self._num_one_step_obs

    @property
    def num_privileged_obs(self):
        return self._num_privileged_obs

    @property
    def num_actions(self):
        return self._num_actions

    def _native_reset(self, env_ids=None):
        if env_ids is None:
            return self.env.reset()
        try:
            return self.env.reset(env_ids=env_ids)
        except TypeError:
            # A few vectorized environments expose reset(env_ids) positionally.
            return self.env.reset(env_ids)

    @staticmethod
    def _unpack_reset(result):
        if isinstance(result, tuple):
            if len(result) != 2:
                raise ValueError("env.reset must return observations and infos")
            return result[0], dict(result[1] or {})
        return result, {}

    def reset(self, env_ids=None):
        obs_dict, infos = self._unpack_reset(self._native_reset(env_ids))
        self._last_obs = obs_dict
        self._last_privileged_obs = None
        return obs_dict, infos

    def get_observations(self):
        return self._last_obs

    def get_privileged_observations(self):
        return self._last_privileged_obs

    def step(self, actions):
        result = self.env.step(actions)
        if not isinstance(result, tuple) or len(result) != 5:
            raise ValueError(
                "native env must return obs, reward, terminated, truncated, infos"
            )
        self._last_obs = result[0]
        return result

    def close(self):
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
