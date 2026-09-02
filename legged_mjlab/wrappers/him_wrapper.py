"""History and terminal-observation adapter for legacy HIM RSL-RL."""

from collections.abc import Mapping
import math
import torch

from legged_mjlab.wrappers.vec_env_wrapper import VecEnvWrapper


class HIMRslRlWrapper(VecEnvWrapper):
    """Expose current-first ``[N, 6 * 45]`` actor history to HIMOnPolicyRunner."""

    def __init__(
        self,
        env,
        history_length=6,
        one_step_obs_dim=45,
        expected_privileged_obs_dim=None,
        action_dim=12,
        action_clip=None,
        max_action_rate=None,
        is_finite_horizon=None,
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

        self._action_clip = self._resolve_action_clip(action_clip)
        self._max_action_rate = self._resolve_max_action_rate(max_action_rate)
        self._is_finite_horizon = self._resolve_finite_horizon(is_finite_horizon)
        self._previous_accepted_actions = None
        self.obs_history_buf = None
        self.termination_privileged_obs = None
        self.termination_ids = None
        self._privileged_obs = None
        self._manual_reset = self._configure_manual_reset()

        

    @staticmethod
    def _cfg_get(obj, name, default=None):
        if obj is None:
            return default
        if isinstance(obj, Mapping):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def _resolve_finite_horizon(self, explicit_value):
        if explicit_value is not None:
            return bool(explicit_value)

        cfg = getattr(self.env, "cfg", None)
        for source in (cfg, self._cfg_get(cfg, "env")):
            value = self._cfg_get(source, "is_finite_horizon", None)
            if value is not None:
                return bool(value)
        return False

    def _resolve_action_clip(self, explicit_value):
        # 不复用 normalization.clip_actions；它属于 native action manager，
        # 在当前配置里可能是 100.0，不适合当 policy raw action 边界。
        value = explicit_value
        if value is None:
            cfg = getattr(self.env, "cfg", None)
            env_cfg = self._cfg_get(cfg, "env")
            sources = (
                self._cfg_get(cfg, "wrapper"),
                self._cfg_get(cfg, "wrappers"),
                self._cfg_get(cfg, "him"),
                self._cfg_get(env_cfg, "wrapper"),
                self._cfg_get(env_cfg, "him"),
            )
            for source in sources:
                for name in ("action_clip", "clip_actions"):
                    candidate = self._cfg_get(source, name, None)
                    if candidate is not None:
                        value = candidate
                        break
                if value is not None:
                    break
        if value is None:
            return None

        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("action_clip must be a finite positive float") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("action_clip must be a finite positive float")
        return value

    @staticmethod
    def _resolve_max_action_rate(value):
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_action_rate must be a finite positive float or None") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("max_action_rate must be a finite positive float or None")
        return value

    def _configure_manual_reset(self):
        """Disable native auto-reset so step can expose the real terminal frame."""

        cfg = getattr(self.env, "cfg", None)
        if cfg is None:
            # 测试 double 可不带 cfg；这类 env 必须在 infos 中显式给 terminal critic。
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

    def _validate_frame(self, value, name, required=True, validate_batch=True, width=None):
        if value is None:
            if required:
                raise ValueError(f"{name} observation is required")
            return None
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape [N, F], got {tuple(value.shape)}")
        if validate_batch and value.shape[0] != self.num_envs:
            raise ValueError(f"{name} batch mismatch: expected {self.num_envs}, got {value.shape[0]}")
        if width is not None and value.shape[-1] != width:
            raise ValueError(f"{name} width mismatch: expected {width}, got {value.shape[-1]}")
        if not torch.is_floating_point(value):
            raise TypeError(f"{name} must be a floating-point tensor")
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"{name} contains NaN or Inf")
        return value

    def _split_and_validate(self, obs_dict):
        actor, privileged = self._split_obs(obs_dict)
        actor = self._validate_frame(actor, "actor", width=self.one_step_obs_dim)
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
        expected_shape = (self.num_envs, self.history_length, self.one_step_obs_dim)
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
        elif self.obs_history_buf.device != actor.device or self.obs_history_buf.dtype != actor.dtype:
            raise ValueError("history buffer and actor observation must share device/dtype")

    def _append_history(self, actor):
        self._ensure_history(actor)
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
        if ids.numel() and (int(ids.min().item()) < 0 or int(ids.max().item()) >= self.num_envs):
            raise IndexError("env_ids contains an out-of-range environment index")
        return ids

    def _ensure_action_state(self, reference):
        if self.num_actions is None:
            raise ValueError("num_actions is required before action sanitation")
        expected_shape = (self.num_envs, self.num_actions)
        if self._previous_accepted_actions is None:
            self._previous_accepted_actions = torch.zeros(
                expected_shape,
                device=reference.device,
                dtype=reference.dtype,
            )
        elif tuple(self._previous_accepted_actions.shape) != expected_shape:
            raise ValueError(
                "previous action buffer shape mismatch: "
                f"expected {expected_shape}, got {tuple(self._previous_accepted_actions.shape)}"
            )
        elif (
            self._previous_accepted_actions.device != reference.device
            or self._previous_accepted_actions.dtype != reference.dtype
        ):
            self._previous_accepted_actions = torch.zeros(
                expected_shape,
                device=reference.device,
                dtype=reference.dtype,
            )

    def _reset_action_state(self, reference, env_ids=None):
        self._ensure_action_state(reference)
        if env_ids is None:
            self._previous_accepted_actions.zero_()
            return
        ids = self._normalize_env_ids(env_ids, reference.device)
        self._previous_accepted_actions[ids] = 0.0

    def reset(self, env_ids=None):
        result = self._native_reset(env_ids)
        obs_dict, _ = self._unpack_reset(result)
        actor, privileged = self._split_and_validate(obs_dict)
        history = self._reset_history(actor, env_ids=env_ids)
        self._last_obs = history
        self._last_privileged_obs = privileged
        self._privileged_obs = privileged
        self.termination_privileged_obs = privileged.new_empty((0, privileged.shape[-1]))
        self.termination_ids = torch.empty((0,), dtype=torch.long, device=actor.device)
        self._reset_action_state(actor, env_ids=env_ids)
        return history, privileged

    def _validate_actions(self, actions):
        if not isinstance(actions, torch.Tensor):
            raise TypeError("actions must be a torch.Tensor")
        if actions.ndim != 2:
            raise ValueError(f"actions must have shape [N, A], got {tuple(actions.shape)}")
        if actions.shape[0] != self.num_envs:
            raise ValueError("action batch dimension does not match num_envs")
        if self.num_actions is None:
            self._num_actions = int(actions.shape[-1])
        elif actions.shape[-1] != self.num_actions:
            raise ValueError(
                f"action width mismatch: expected {self.num_actions}, got {actions.shape[-1]}"
            )
        if not torch.is_floating_point(actions):
            raise TypeError("actions must be floating point")
        if not bool(torch.isfinite(actions).all().item()):
            raise FloatingPointError("actions contains NaN or Inf")
        return actions

    def _sanitize_actions(self, actions):
        raw_actions = self._validate_actions(actions)
        if self._action_clip is None:
            clipped_actions = raw_actions
            clip_mask = torch.zeros(raw_actions.shape[0], dtype=torch.bool, device=raw_actions.device)
        else:
            clipped_actions = raw_actions.clamp(-self._action_clip, self._action_clip)
            clip_mask = (clipped_actions != raw_actions).any(dim=-1)
        accepted_actions = clipped_actions
        rate_mask = torch.zeros(raw_actions.shape[0], dtype=torch.bool, device=raw_actions.device)

        if self._max_action_rate is not None:
            # PPO 的 log-prob 对应 raw action；启用 rate limit 前要保证 previous
            # accepted action 已进入 observation，否则训练语义会被改写。
            self._ensure_action_state(raw_actions)
            delta = (clipped_actions - self._previous_accepted_actions).clamp(
                -self._max_action_rate,
                self._max_action_rate,
            )
            accepted_actions = self._previous_accepted_actions + delta
            rate_mask = (accepted_actions != clipped_actions).any(dim=-1)

        self._previous_accepted_actions = accepted_actions.detach().clone()
        return (
            raw_actions.detach().clone(),
            clipped_actions.detach().clone(),
            accepted_actions,
            clip_mask,
            rate_mask,
        )

    def _as_batch_bool(self, value, name, device):
        tensor = torch.as_tensor(value, device=device)
        if tensor.numel() != self.num_envs:
            raise ValueError(f"{name} must contain {self.num_envs} values, got {tensor.numel()}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            raise FloatingPointError(f"{name} contains NaN or Inf")
        return tensor.reshape(-1).to(dtype=torch.bool)

    def _as_rewards(self, value, device):
        rewards = torch.as_tensor(value, device=device, dtype=torch.float32)
        if rewards.numel() != self.num_envs:
            raise ValueError(f"rewards must contain {self.num_envs} values, got {rewards.numel()}")
        if not bool(torch.isfinite(rewards).all().item()):
            raise FloatingPointError("rewards contains NaN or Inf")
        return rewards.reshape(-1)

    @staticmethod
    def _privileged_from_observation(observation):
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
            for key in ("observation", "obs", "final_observation", "terminal_observation"):
                nested = observation.get(key)
                if nested is not None:
                    candidate = HIMRslRlWrapper._privileged_from_observation(nested)
                    if candidate is not None:
                        return candidate
            return None
        if isinstance(observation, tuple) and len(observation) == 2:
            return observation[1]
        return observation

    def _terminal_candidate(self, infos):
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
        for container_name in ("infos", "final_info", "terminal_info", "info"):
            container = infos if container_name == "infos" else infos.get(container_name)
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
        candidate = self._validate_frame(
            candidate,
            "termination privileged",
            validate_batch=False,
        )
        if self.num_privileged_obs is not None and candidate.shape[-1] != self.num_privileged_obs:
            raise ValueError(
                "termination privileged width mismatch: "
                f"expected {self.num_privileged_obs}, got {candidate.shape[-1]}"
            )
        done_count = int(done_ids.numel())
        if candidate.shape[0] == self.num_envs:
            return candidate[done_ids]
        if candidate.shape[0] == done_count:
            # compact [K, P] 必须已经按 done_ids 顺序排列；wrapper 无法从
            # tensor 本身验证乱序。除非 native infos 同时给出 compact_env_ids
            # 并严格等于 done_ids，否则不要接受乱序 compact terminal。
            return candidate
        raise ValueError(
            "termination privileged obs must have shape [N, P] or [K, P] ordered by done_ids, "
            f"got {tuple(candidate.shape)} for N={self.num_envs}, K={done_count}"
        )

    def _reset_done_envs(self, done_ids):
        try:
            result = self.env.reset(env_ids=done_ids)
        except Exception as exc:
            raise RuntimeError(
                "HIMRslRlWrapper requires native reset(env_ids=...) after "
                "switching env.cfg.auto_reset=False"
            ) from exc
        obs_dict, _ = self._unpack_reset(result)
        return self._split_and_validate(obs_dict)

    def _merge_reset_observations(self, actor, privileged, done_ids, reset_actor, reset_privileged):
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

    def _time_outs(self, terminated, truncated):
        # 有限时域任务到时间上限就是终止，不允许旧 HIM timeout bootstrap。
        if self._is_finite_horizon:
            return torch.zeros_like(truncated)
        return truncated & ~terminated

    def step(self, actions):
        raw_actions, clipped_actions, accepted_actions, clip_mask, rate_mask = self._sanitize_actions(actions)
        result = self.env.step(accepted_actions)
        if not isinstance(result, tuple) or len(result) != 5:
            raise ValueError(
                "env.step must return (obs_dict, rewards, terminated, truncated, infos)"
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

        terminal_step_privileged = privileged
        candidate, source = self._terminal_candidate(infos)
        if candidate is None and self._manual_reset and done_ids.numel() > 0:
            candidate, source = terminal_step_privileged, "step_terminal_observation"
        if candidate is None and done_ids.numel() > 0:
            raise RuntimeError(
                "done environments require reset-before terminal privileged obs; "
                "disable native auto_reset or provide termination_privileged_obs in infos"
            )

        if hasattr(self.env, "reward_manager") and hasattr(self.env.reward_manager, "_episode_sums"):
            if done_ids.numel() > 0:
                episode_dict = {}
                for name, ep_sum in self.env.reward_manager._episode_sums.items():
                    # 直接保存这批已结束环境的累计 Tensor，无 CPU 同步开销
                    episode_dict[f"rew_{name}"] = ep_sum[done_ids].clone()
                infos["episode"] = episode_dict

        if self._manual_reset and done_ids.numel() > 0:
            reset_actor, reset_privileged = self._reset_done_envs(done_ids)
            actor, privileged = self._merge_reset_observations(
                actor,
                privileged,
                done_ids,
                reset_actor,
                reset_privileged,
            )
            self._reset_action_state(actor, env_ids=done_ids)

        self._ensure_history(actor)
        if done_ids.numel() > 0:
            self.obs_history_buf[done_ids] = 0.0
        history = self._append_history(actor)

        terminal_available = torch.zeros_like(dones)
        if candidate is None:
            terminal_privileged = privileged.new_empty((0, privileged.shape[-1]))
        else:
            terminal_privileged = self._select_terminal_privileged(candidate, done_ids)
            if done_ids.numel() > 0:
                terminal_available[done_ids] = True

        time_outs = self._time_outs(terminated, truncated)
        infos["terminated"] = terminated
        infos["truncated"] = truncated
        infos["time_outs"] = time_outs
        infos["timeout_bootstrap"] = time_outs & terminal_available
        infos["termination_privileged_obs"] = terminal_privileged
        infos["termination_privileged_obs_source"] = source
        infos["policy_raw_actions"] = raw_actions
        infos["clipped_actions"] = clipped_actions
        infos["accepted_actions"] = accepted_actions.detach().clone()
        infos["action_clip_mask"] = clip_mask
        infos["action_rate_limit_mask"] = rate_mask
        infos["is_finite_horizon"] = self._is_finite_horizon

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