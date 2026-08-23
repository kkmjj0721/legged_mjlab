"""History and terminal-observation adapter for the project's HIM runner."""

from collections.abc import Mapping

try:
    import torch
except (ImportError, ModuleNotFoundError):  # pragma: no cover - CLI help path
    torch = None

from legged_mjlab.wrappers.rsl_rl_wrapper import require_torch
from legged_mjlab.wrappers.vec_env_wrapper import VecEnvWrapper


class HIMRslRlWrapper(VecEnvWrapper):
    """Expose a fixed ``[N, 6 * 45]`` actor history to ``HIMOnPolicyRunner``."""

    def __init__(
        self,
        env,
        history_length=6,
        one_step_obs_dim=45,
        expected_privileged_obs_dim=None,
        action_dim=None,
    ):
        super().__init__(env)
        if int(history_length) != 6:
            raise ValueError("HIMRslRlWrapper requires exactly 6 history frames")
        if int(one_step_obs_dim) != 45:
            raise ValueError("HIMRslRlWrapper requires single-frame observations [N, 45]")

        self.history_length = 6
        self.one_step_obs_dim = 45
        self._num_one_step_obs = 45
        self._num_obs = 270
        if action_dim is not None:
            self._num_actions = int(action_dim)
            if self._num_actions < 1:
                raise ValueError("action_dim must be positive")
        if expected_privileged_obs_dim is not None:
            self._num_privileged_obs = int(expected_privileged_obs_dim)
            if self._num_privileged_obs < 1:
                raise ValueError("expected_privileged_obs_dim must be positive")
        self.obs_history_buf = None
        self.termination_privileged_obs = None
        self.termination_ids = None
        self._privileged_obs = None
        self._manual_reset = self._configure_manual_reset()

    def _configure_manual_reset(self):
        """Disable native auto-reset so a step exposes the true terminal frame."""

        cfg = getattr(self.env, "cfg", None)
        if cfg is None:
            # Keep lightweight non-mjlab test doubles usable.  They must provide
            # an explicit terminal observation in ``infos``; this path never
            # falls back to the pre-step critic frame.
            return False
        try:
            if isinstance(cfg, Mapping):
                if "auto_reset" not in cfg:
                    raise AttributeError("env.cfg has no auto_reset field")
                cfg["auto_reset"] = False
                configured = cfg["auto_reset"]
            else:
                if not hasattr(cfg, "auto_reset"):
                    raise AttributeError("env.cfg has no auto_reset field")
                cfg.auto_reset = False
                configured = cfg.auto_reset
        except Exception as exc:
            raise RuntimeError(
                "HIMRslRlWrapper could not set env.cfg.auto_reset=False; "
                "a terminal-observation-preserving runtime is required"
            ) from exc
        if configured is not False:
            raise RuntimeError(
                "HIMRslRlWrapper set env.cfg.auto_reset=False but the value "
                f"read back as {configured!r}"
            )
        return True

    @staticmethod
    def _split_obs(obs_dict):
        if not isinstance(obs_dict, Mapping):
            raise TypeError("native observation must be a dictionary")
        actor = obs_dict.get("actor", obs_dict.get("policy"))
        if actor is None:
            raise KeyError("observation dictionary has no actor or policy group")
        privileged = obs_dict.get("critic", obs_dict.get("privileged"))
        return actor, privileged

    def _validate_frame(self, value, name, required=True, validate_batch=True):
        require_torch()
        if value is None:
            if required:
                raise ValueError(f"{name} observation is required")
            return None
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 2:
            raise ValueError(
                f"{name} must have shape [N, F], got {tuple(value.shape)}"
            )
        if validate_batch and value.shape[0] != self.num_envs:
            raise ValueError(
                f"{name} batch mismatch: expected {self.num_envs}, got {value.shape[0]}"
            )
        if not torch.is_floating_point(value):
            raise TypeError(f"{name} must be a floating-point tensor")
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"{name} contains NaN or Inf")
        return value

    def _split_and_validate(self, obs_dict):
        actor, privileged = self._split_obs(obs_dict)
        actor = self._validate_frame(actor, "actor")
        if actor.shape[-1] != self.one_step_obs_dim:
            raise ValueError(
                "actor observation shape mismatch: expected "
                f"({self.num_envs}, {self.one_step_obs_dim}), got {tuple(actor.shape)}"
            )

        # HIM's estimator always consumes a critic frame.  Keeping this strict
        # prevents a later runner assignment from turning into a cryptic shape
        # error when a native environment silently drops its critic group.
        privileged = self._validate_frame(privileged, "critic")
        if self.num_privileged_obs is None:
            self._num_privileged_obs = int(privileged.shape[-1])
        elif privileged.shape[-1] != self.num_privileged_obs:
            raise ValueError(
                "critic observation width mismatch: "
                f"expected {self.num_privileged_obs}, got {privileged.shape[-1]}"
            )
        if privileged.device != actor.device:
            raise ValueError("actor and critic observations must share a device")
        return actor, privileged

    def _ensure_history(self, actor):
        require_torch()
        expected_shape = (
            self.num_envs,
            self.history_length,
            self.one_step_obs_dim,
        )
        if self.obs_history_buf is None:
            self.obs_history_buf = torch.zeros(
                expected_shape,
                device=actor.device,
                dtype=actor.dtype,
            )
        elif tuple(self.obs_history_buf.shape) != expected_shape:
            raise ValueError(
                "history buffer shape mismatch: "
                f"expected {expected_shape}, got {tuple(self.obs_history_buf.shape)}"
            )
        elif (
            self.obs_history_buf.device != actor.device
            or self.obs_history_buf.dtype != actor.dtype
        ):
            raise ValueError("history buffer and actor observation must share device/dtype")

    def _append_history(self, actor):
        self._ensure_history(actor)
        if self.history_length > 1:
            self.obs_history_buf[:, 1:].copy_(self.obs_history_buf[:, :-1].clone())
        self.obs_history_buf[:, 0].copy_(actor)
        history = self.obs_history_buf.reshape(self.num_envs, self.num_obs)
        if not bool(torch.isfinite(history).all().item()):
            raise FloatingPointError("actor history contains NaN or Inf")
        return history

    def _reset_history(self, actor, env_ids=None):
        self._ensure_history(actor)
        if env_ids is None:
            self.obs_history_buf.zero_()
            self.obs_history_buf[:, 0].copy_(actor)
        else:
            ids = self._normalize_env_ids(env_ids, actor.device)
            self.obs_history_buf[ids] = 0.0
            self.obs_history_buf[ids, 0] = actor[ids]
        return self.obs_history_buf.reshape(self.num_envs, self.num_obs)

    def _normalize_env_ids(self, env_ids, device):
        ids = torch.as_tensor(env_ids, device=device)
        if ids.dtype == torch.bool:
            if ids.numel() != self.num_envs:
                raise ValueError("boolean env_ids must have one value per environment")
            ids = torch.nonzero(ids.reshape(-1), as_tuple=False).flatten()
        else:
            ids = ids.reshape(-1).to(dtype=torch.long)
        if ids.numel() and (
            int(ids.min().item()) < 0 or int(ids.max().item()) >= self.num_envs
        ):
            raise IndexError("env_ids contains an out-of-range environment index")
        return ids

    def reset(self, env_ids=None):
        result = self._native_reset(env_ids)
        obs_dict, _ = self._unpack_reset(result)
        actor, privileged = self._split_and_validate(obs_dict)
        history = self._reset_history(actor, env_ids=env_ids)
        self._last_obs = history
        self._last_privileged_obs = privileged
        self._privileged_obs = privileged
        self.termination_privileged_obs = None
        self.termination_ids = torch.empty(
            (0,), dtype=torch.long, device=actor.device
        )
        return history, privileged

    def _validate_actions(self, actions):
        require_torch()
        if not isinstance(actions, torch.Tensor):
            raise TypeError("actions must be a torch.Tensor")
        if actions.ndim != 2:
            raise ValueError(
                f"actions must have shape [N, A], got {tuple(actions.shape)}"
            )
        if actions.shape[0] != self.num_envs:
            raise ValueError("action batch dimension does not match num_envs")
        if self.num_actions is None:
            self._num_actions = int(actions.shape[-1])
        elif actions.shape[-1] != self.num_actions:
            raise ValueError(
                f"action width mismatch: expected {self.num_actions}, "
                f"got {actions.shape[-1]}"
            )
        if not bool(torch.isfinite(actions).all().item()):
            raise FloatingPointError("actions contains NaN or Inf")

    def _as_batch_bool(self, value, name, device):
        tensor = torch.as_tensor(value, device=device)
        if tensor.numel() != self.num_envs:
            raise ValueError(
                f"{name} must contain {self.num_envs} values, got {tensor.numel()}"
            )
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            raise FloatingPointError(f"{name} contains NaN or Inf")
        return tensor.reshape(-1).to(dtype=torch.bool)

    def _as_rewards(self, value, device):
        rewards = torch.as_tensor(value, device=device, dtype=torch.float32)
        if rewards.numel() != self.num_envs:
            raise ValueError(
                f"rewards must contain {self.num_envs} values, got {rewards.numel()}"
            )
        if not bool(torch.isfinite(rewards).all().item()):
            raise FloatingPointError("rewards contains NaN or Inf")
        return rewards.reshape(-1)

    @staticmethod
    def _privileged_from_observation(observation):
        """Extract a critic group from a final/terminal observation payload."""

        if isinstance(observation, Mapping):
            for key in (
                "critic",
                "privileged",
                "critic_obs",
                "privileged_obs",
                "final_privileged_obs",
                "terminal_privileged_obs",
            ):
                candidate = observation.get(key)
                if candidate is not None:
                    return candidate
            # Some vector wrappers nest the actual observation one level below
            # ``final_observation``/``terminal_observation``.
            for key in ("observation", "obs", "final_observation", "terminal_observation"):
                nested = observation.get(key)
                if nested is not None:
                    candidate = HIMRslRlWrapper._privileged_from_observation(nested)
                    if candidate is not None:
                        return candidate
            return None

        # A few adapters represent (actor, critic) as a two-item tuple.  Do not
        # treat an arbitrary per-environment list as this representation.
        if isinstance(observation, tuple) and len(observation) == 2:
            return observation[1]
        return observation

    def _terminal_candidate(self, infos):
        """Find a real terminal critic observation, never a pre-step snapshot."""

        for key in (
            "termination_privileged_obs",
            "terminal_privileged_obs",
            "final_privileged_obs",
            "final_privileged_observation",
            "terminal_critic_obs",
            "final_critic_obs",
        ):
            candidate = infos.get(key)
            if candidate is not None:
                return candidate, key

        final_keys = (
            "final_observation",
            "final_observations",
            "final_obs",
            "terminal_observation",
            "terminal_observations",
            "terminal_obs",
        )
        containers = (("infos", infos),)
        for container_name, container in containers:
            if not isinstance(container, Mapping):
                continue
            for key in final_keys:
                observation = container.get(key)
                if observation is None:
                    continue
                candidate = self._privileged_from_observation(observation)
                if candidate is not None:
                    return candidate, f"{container_name}.{key}"

        # Gymnasium-style wrappers may put the final observation in final_info.
        for container_name in ("final_info", "terminal_info", "info"):
            container = infos.get(container_name)
            if not isinstance(container, Mapping):
                continue
            for key in final_keys:
                observation = container.get(key)
                if observation is None:
                    continue
                candidate = self._privileged_from_observation(observation)
                if candidate is not None:
                    return candidate, f"{container_name}.{key}"
        return None, "unavailable_no_final_observation"

    def _select_terminal_privileged(self, candidate, done_ids):
        # A native vector environment may preserve terminal critic observations
        # for every environment ([N, P]) or compact them to the environments
        # that ended on this step ([K, P]).  Validate the frame independently of
        # the regular observation batch before choosing the representation.
        candidate = self._validate_frame(
            candidate,
            "termination privileged",
            validate_batch=False,
        )
        if (
            self.num_privileged_obs is not None
            and candidate.shape[-1] != self.num_privileged_obs
        ):
            raise ValueError(
                "termination privileged width mismatch: "
                f"expected {self.num_privileged_obs}, got {candidate.shape[-1]}"
            )
        done_count = int(done_ids.numel())
        if candidate.shape[0] == self.num_envs:
            return candidate[done_ids]
        if candidate.shape[0] == done_count:
            return candidate
        raise ValueError(
            "termination privileged obs must have shape [N, P] or [K, P], "
            f"got {tuple(candidate.shape)} for N={self.num_envs}, K={done_count}"
        )

    def _reset_done_envs(self, done_ids):
        """Reset done native environments and return their next observations."""

        try:
            result = self.env.reset(env_ids=done_ids)
        except Exception as exc:
            raise RuntimeError(
                "HIMRslRlWrapper requires native reset(env_ids=...) after "
                "switching env.cfg.auto_reset=False"
            ) from exc
        obs_dict, _ = self._unpack_reset(result)
        return self._split_and_validate(obs_dict)

    def _merge_reset_observations(
        self, actor, privileged, done_ids, reset_actor, reset_privileged
    ):
        """Replace only done rows with the first frame of their reset episodes."""

        if reset_actor.shape[0] != self.num_envs or reset_privileged.shape[0] != self.num_envs:
            raise ValueError(
                "native reset(env_ids=...) must return full [N, F] actor and critic "
                "batches for HIM history alignment"
            )
        next_actor = actor.clone()
        next_privileged = privileged.clone()
        next_actor[done_ids] = reset_actor[done_ids]
        next_privileged[done_ids] = reset_privileged[done_ids]
        return next_actor, next_privileged

    def step(self, actions):
        require_torch()
        self._validate_actions(actions)
        result = self.env.step(actions)
        if not isinstance(result, tuple) or len(result) != 5:
            raise ValueError(
                "env.step must return "
                "(obs_dict, rewards, terminated, truncated, infos)"
            )
        obs_dict, rewards, terminated, truncated, infos = result
        actor, privileged = self._split_and_validate(obs_dict)
        terminated = self._as_batch_bool(terminated, "terminated", actor.device)
        truncated = self._as_batch_bool(truncated, "truncated", actor.device)
        rewards = self._as_rewards(rewards, actor.device)
        if infos is not None and not isinstance(infos, Mapping):
            raise TypeError("infos must be a dictionary or None")
        infos = dict(infos or {})
        dones = terminated | truncated
        done_ids = torch.nonzero(dones, as_tuple=False).flatten()

        # With auto_reset disabled, ``privileged`` is the true post-action
        # terminal critic frame.  Save it before resetting done environments.
        terminal_step_privileged = privileged
        candidate, source = self._terminal_candidate(infos)
        if candidate is None and self._manual_reset and done_ids.numel() > 0:
            candidate, source = terminal_step_privileged, "step_terminal_observation"

        if self._manual_reset and done_ids.numel() > 0:
            reset_actor, reset_privileged = self._reset_done_envs(done_ids)
            actor, privileged = self._merge_reset_observations(
                actor,
                privileged,
                done_ids,
                reset_actor,
                reset_privileged,
            )

        self._ensure_history(actor)
        if done_ids.numel() > 0:
            # Native environments commonly return the first observation of the
            # reset episode here.  Clear only those rows before appending it.
            self.obs_history_buf[done_ids] = 0.0
        history = self._append_history(actor)

        if candidate is None:
            # No terminal observation was supplied by the native step/info
            # contract.  Never manufacture one from s_t or a reset observation.
            terminal_privileged = (
                privileged.new_empty((0, privileged.shape[-1]))
                if done_ids.numel() == 0
                else None
            )
            terminal_available = torch.zeros_like(dones)
        else:
            terminal_privileged = self._select_terminal_privileged(
                candidate, done_ids
            )
            terminal_available = torch.zeros_like(dones)
            if done_ids.numel() > 0:
                terminal_available[done_ids] = True
        infos["terminated"] = terminated
        infos["truncated"] = truncated
        time_outs = truncated & ~terminated
        infos["time_outs"] = time_outs
        infos["timeout_bootstrap"] = time_outs & terminal_available
        infos["termination_privileged_obs"] = terminal_privileged
        infos["termination_privileged_obs_source"] = source

        self._last_obs = history
        self._last_privileged_obs = privileged
        self._privileged_obs = privileged
        self.termination_ids = done_ids
        self.termination_privileged_obs = terminal_privileged
        return (
            history,
            privileged,
            rewards,
            dones,
            infos,
            done_ids,
            terminal_privileged,
        )


__all__ = ["HIMRslRlWrapper"]
