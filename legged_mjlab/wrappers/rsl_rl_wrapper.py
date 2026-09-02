"""Adapters from native manager envs to legacy RSL-RL VecEnv APIs."""

from collections.abc import Mapping
import torch

from legged_mjlab.wrappers.vec_env_wrapper import VecEnvWrapper


def _cfg_get(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _check_finite(value, name):
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{name} contains NaN or Inf")


def flatten_batch(value, name):
    """Validate a tensor and flatten feature dimensions to ``[N, F]``."""

    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or None")
    if value.ndim == 0:
        raise ValueError(f"{name} must have a batch dimension")
    if value.ndim == 1:
        value = value.unsqueeze(0)
    elif value.ndim > 2:
        value = value.reshape(value.shape[0], -1)
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [N, F], got {tuple(value.shape)}")
    _check_finite(value, name)
    return value


class RslRlVecEnvWrapper(VecEnvWrapper):
    """Standard PPO adapter for the repository-local legacy RSL-RL runner.

    Native mjlab-style envs return ``(obs_dict, reward, terminated, truncated,
    infos)``.  Old ``OnPolicyRunner`` consumes ``(actor_obs, critic_obs,
    rewards, dones, infos)`` and reads ``infos["time_outs"]`` directly for
    bootstrap, so finite-horizon handling belongs here.
    """

    def __init__(self, env, is_finite_horizon=None):
        super().__init__(env)
        self._is_finite_horizon = self._resolve_finite_horizon(is_finite_horizon)

    def _resolve_finite_horizon(self, explicit_value):
        if explicit_value is not None:
            return bool(explicit_value)
        cfg = getattr(self.env, "cfg", None)
        for source in (cfg, _cfg_get(cfg, "env")):
            value = _cfg_get(source, "is_finite_horizon", None)
            if value is not None:
                return bool(value)
        return False

    def _split_obs(self, obs_dict):
        if not isinstance(obs_dict, Mapping):
            raise TypeError("native observation must be a dictionary")

        actor = obs_dict.get("actor", obs_dict.get("policy"))
        if actor is None:
            raise KeyError("observation dictionary has no actor or policy group")
        privileged = obs_dict.get("critic", obs_dict.get("privileged"))
        return flatten_batch(actor, "actor"), flatten_batch(privileged, "critic")

    def _update_metadata(self, actor, privileged):
        if self.num_envs is None:
            self._num_envs = int(actor.shape[0])
        if actor.shape[0] != self.num_envs:
            raise ValueError(
                f"actor batch mismatch: expected {self.num_envs}, got {actor.shape[0]}"
            )

        actor_width = int(actor.shape[-1])
        if self.num_obs is None:
            self._num_obs = actor_width
        elif self.num_obs != actor_width:
            raise ValueError(
                f"actor width mismatch: expected {self.num_obs}, got {actor_width}"
            )
        if self.num_one_step_obs is None:
            self._num_one_step_obs = actor_width

        if privileged is None:
            if self.num_privileged_obs is not None:
                raise ValueError(
                    "native environment declares privileged observations but did not "
                    "return a critic group"
                )
            return

        if privileged.shape[0] != self.num_envs:
            raise ValueError("critic observation batch dimension does not match num_envs")
        privileged_width = int(privileged.shape[-1])
        if self.num_privileged_obs is None:
            self._num_privileged_obs = privileged_width
        elif self.num_privileged_obs != privileged_width:
            raise ValueError(
                "critic width mismatch: "
                f"expected {self.num_privileged_obs}, got {privileged_width}"
            )
        if privileged.device != actor.device:
            raise ValueError("actor and critic observations must share a device")

    def reset(self, env_ids=None):
        obs_dict, _ = self._unpack_reset(self._native_reset(env_ids))
        actor, privileged = self._split_obs(obs_dict)
        self._update_metadata(actor, privileged)
        self._last_obs = actor
        self._last_privileged_obs = privileged
        return actor, privileged

    def _validate_actions(self, actions):
        if not isinstance(actions, torch.Tensor):
            raise TypeError("actions must be a torch.Tensor")
        if actions.ndim != 2:
            raise ValueError(f"actions must have shape [N, A], got {tuple(actions.shape)}")
        if self.num_envs is not None and actions.shape[0] != self.num_envs:
            raise ValueError("action batch dimension does not match num_envs")
        if self.num_actions is None:
            self._num_actions = int(actions.shape[-1])
        elif actions.shape[-1] != self.num_actions:
            raise ValueError(
                f"action width mismatch: expected {self.num_actions}, got {actions.shape[-1]}"
            )
        _check_finite(actions, "actions")

    def _as_batch_bool(self, value, name, device):
        tensor = torch.as_tensor(value, device=device)
        if tensor.numel() != self.num_envs:
            raise ValueError(f"{name} must contain {self.num_envs} values, got {tensor.numel()}")
        if tensor.is_floating_point():
            _check_finite(tensor, name)
        return tensor.reshape(-1).to(dtype=torch.bool)

    def _as_rewards(self, value, device):
        rewards = torch.as_tensor(value, device=device, dtype=torch.float32)
        if rewards.numel() != self.num_envs:
            raise ValueError(f"rewards must contain {self.num_envs} values, got {rewards.numel()}")
        _check_finite(rewards, "rewards")
        return rewards.reshape(-1)

    def _time_outs(self, terminated, truncated):
        # 有限时域任务到达时间上限就是任务终点，不能 bootstrap。
        if self._is_finite_horizon:
            return torch.zeros_like(truncated)
        return truncated & ~terminated

    def step(self, actions):
        self._validate_actions(actions)
        result = self.env.step(actions)
        if not isinstance(result, tuple) or len(result) != 5:
            raise ValueError(
                "env.step must return (obs_dict, rewards, terminated, truncated, infos)"
            )

        obs_dict, rewards, terminated, truncated, infos = result
        actor, privileged = self._split_obs(obs_dict)
        self._update_metadata(actor, privileged)
        terminated = self._as_batch_bool(terminated, "terminated", actor.device)
        truncated = self._as_batch_bool(truncated, "truncated", actor.device)
        rewards = self._as_rewards(rewards, actor.device)
        if infos is not None and not isinstance(infos, Mapping):
            raise TypeError("infos must be a dictionary or None")

        infos = dict(infos or {})
        infos["terminated"] = terminated
        infos["truncated"] = truncated
        infos["time_outs"] = self._time_outs(terminated, truncated)
        infos["is_finite_horizon"] = self._is_finite_horizon

        self._last_obs = actor
        self._last_privileged_obs = privileged
        return actor, privileged, rewards, terminated | truncated, infos


__all__ = ["RslRlVecEnvWrapper", "flatten_batch"]