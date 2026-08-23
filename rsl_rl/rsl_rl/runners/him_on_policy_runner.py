# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
from collections.abc import Mapping
import inspect
import statistics

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard is optional for shape and CPU smoke tests
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

from rsl_rl.algorithms import HIMPPO
from rsl_rl.modules import HIMActorCritic
from rsl_rl.env import VecEnv


def _config_value(config, key, default=None):
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _config_section(config, name):
    section = _config_value(config, name, None)
    if section is None:
        raise KeyError("training configuration has no '{}' section".format(name))
    return section


def _config_to_dict(config):
    """Convert either a mapping or nested config object for ``**kwargs``."""
    if isinstance(config, Mapping):
        return {key: _config_to_dict(value) for key, value in config.items()}
    if isinstance(config, (list, tuple)):
        converted = [_config_to_dict(value) for value in config]
        return type(config)(converted)
    if isinstance(config, (str, bytes, int, float, bool, type(None))):
        return config
    if not hasattr(config, "__dict__"):
        return config

    result = {}
    for key in dir(config):
        if key.startswith("_"):
            continue
        value = getattr(config, key)
        if inspect.isroutine(value) or inspect.isclass(value):
            continue
        result[key] = _config_to_dict(value)
    return result


class HIMOnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = _config_to_dict(_config_section(train_cfg, "runner"))
        self.alg_cfg = _config_to_dict(_config_section(train_cfg, "algorithm"))
        self.policy_cfg = _config_to_dict(_config_section(train_cfg, "policy"))
        # Some config serializers flatten class/runner metadata into every
        # section.  Resolve the names first, then keep metadata out of the
        # constructor kwargs consumed by HIMPPO/HIMActorCritic.
        self.policy_class_name = self.cfg.get(
            "policy_class_name",
            self.policy_cfg.get("policy_class_name", "HIMActorCritic"),
        )
        self.algorithm_class_name = self.cfg.get(
            "algorithm_class_name",
            self.alg_cfg.get("algorithm_class_name", "HIMPPO"),
        )
        metadata_keys = {
            "policy_class_name",
            "algorithm_class_name",
            "runner_class_name",
            "metadata",
        }
        self.alg_cfg = {
            key: value
            for key, value in self.alg_cfg.items()
            if key not in metadata_keys
        }
        self.policy_cfg = {
            key: value
            for key, value in self.policy_cfg.items()
            if key not in metadata_keys
        }
        self.device = torch.device(device)
        self.env = env

        # Reset once up front and use the returned tensors as the source of
        # truth.  The wrapper exposes actual group widths, while an underlying
        # environment/config may still advertise stale legacy dimensions.
        reset_result = self.env.reset()
        reset_obs, reset_critic_obs, reset_infos = self._unpack_reset(reset_result)
        if reset_obs is None:
            reset_obs, embedded_critic = self._read_current_observations()
            if reset_critic_obs is None:
                reset_critic_obs = embedded_critic
        reset_obs = self._flatten_batch(reset_obs, "actor observation")
        if reset_obs is None:
            raise ValueError("HIM environment reset did not return actor observations")

        self.metadata = self._read_metadata(train_cfg, reset_infos)
        self.num_actor_obs = reset_obs.shape[-1]
        configured_actor_obs = self._metadata_number(
            ("num_actor_obs", "actor_obs_dim", "num_observations", "obs_dim")
        )
        if configured_actor_obs is not None and int(configured_actor_obs) != self.num_actor_obs:
            raise ValueError(
                "actor observation metadata disagrees with reset shape: "
                f"{configured_actor_obs} vs {self.num_actor_obs}"
            )

        self.num_one_step_obs = self._resolve_one_step_obs()
        if self.num_actor_obs % self.num_one_step_obs != 0:
            raise ValueError(
                "actor observation width must be a multiple of one-step width: "
                f"{self.num_actor_obs} vs {self.num_one_step_obs}"
            )
        # HIM's critic layout is always the current one-step observation plus
        # the three privileged base-velocity values.  Extra privileged terms
        # (for example terrain scans) are not part of this contract.
        self.num_critic_obs = self.num_one_step_obs + 3
        self.num_actions = self._resolve_num_actions()

        reset_critic_obs = self._prepare_critic_obs(reset_critic_obs, reset_obs)
        actor_critic_class = self._resolve_class(self.policy_class_name)
        actor_critic: HIMActorCritic = actor_critic_class(
            self.num_actor_obs,
            self.num_critic_obs,
            self.num_one_step_obs,
            self.num_actions,
            **self.policy_cfg,
        ).to(self.device)
        alg_class = self._resolve_class(self.algorithm_class_name)
        self.alg: HIMPPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = int(self.cfg["num_steps_per_env"])
        self.save_interval = int(self.cfg.get("save_interval", 1))

        # init storage and model
        num_envs = int(getattr(self.env, "num_envs", reset_obs.shape[0]))
        self.alg.init_storage(
            num_envs,
            self.num_steps_per_env,
            [self.num_actor_obs],
            [self.num_critic_obs],
            [self.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self._initial_obs = reset_obs
        self._initial_critic_obs = reset_critic_obs

    @staticmethod
    def _split_observation(observation):
        if isinstance(observation, Mapping):
            actor = observation.get("actor", observation.get("policy"))
            critic = observation.get("critic", observation.get("privileged"))
            return actor, critic
        return observation, None

    @classmethod
    def _privileged_from_observation(cls, observation):
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
                    candidate = cls._privileged_from_observation(nested)
                    if candidate is not None:
                        return candidate
            return None
        if isinstance(observation, tuple) and len(observation) == 2:
            return observation[1]
        return observation

    @classmethod
    def _terminal_privileged_from_infos(cls, infos):
        if not isinstance(infos, Mapping):
            return None
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
                return candidate
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
                candidate = cls._privileged_from_observation(observation)
                if candidate is not None:
                    return candidate
        return None

    @staticmethod
    def _normalize_timeout_bootstrap(infos, terminal_privileged, dones):
        # Preserve the wrapper's explicit mask, but never synthesize one from
        # ``time_outs`` or terminal-observation availability.
        return dict(infos or {})

    @classmethod
    def _unpack_reset(cls, result):
        if isinstance(result, (tuple, list)):
            if len(result) == 0:
                return None, None, {}
            actor, embedded_critic = cls._split_observation(result[0])
            second = result[1] if len(result) > 1 else None
            if isinstance(second, Mapping):
                return actor, embedded_critic, second
            return actor, second if second is not None else embedded_critic, {}
        actor, embedded_critic = cls._split_observation(result)
        return actor, embedded_critic, {}

    @classmethod
    def _unpack_step(cls, result):
        if not isinstance(result, (tuple, list)):
            raise ValueError("HIM environment step must return a tuple/list")
        if len(result) == 7:
            (
                actor,
                critic,
                rewards,
                dones,
                infos,
                termination_ids,
                terminal_privileged,
            ) = result
            infos = dict(infos or {})
            if terminal_privileged is None:
                terminal_privileged = cls._terminal_privileged_from_infos(infos)
            dones = torch.as_tensor(dones).reshape(-1).bool()
            if termination_ids is None:
                termination_ids = torch.nonzero(
                    dones, as_tuple=False
                ).flatten()
            infos = cls._normalize_timeout_bootstrap(
                infos, terminal_privileged, dones
            )
            return (
                actor,
                critic,
                rewards,
                dones,
                infos,
                termination_ids,
                terminal_privileged,
            )
        if len(result) == 5:
            obs, rewards, terminated, truncated, infos = result
            actor, embedded_critic = cls._split_observation(obs)
            infos = dict(infos or {})
            terminal = cls._terminal_privileged_from_infos(infos)
            dones = torch.as_tensor(terminated).reshape(-1).bool() | torch.as_tensor(
                truncated
            ).reshape(-1).bool()
            ids = torch.nonzero(
                dones, as_tuple=False
            ).flatten()
            infos = cls._normalize_timeout_bootstrap(infos, terminal, dones)
            return actor, embedded_critic, rewards, dones, infos, ids, terminal
        raise ValueError(
            "HIM environment step must return either 5 or 7 values, "
            f"got {len(result)}"
        )

    @staticmethod
    def _flatten_batch(value, name):
        if value is None:
            return None
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim > 2:
            value = value.reshape(value.shape[0], -1)
        if value.ndim != 2:
            raise ValueError(f"{name} must be a batched tensor")
        return value

    @staticmethod
    def _safe_attr(obj, name, default=None):
        try:
            return getattr(obj, name, default)
        except (AttributeError, TypeError):
            return default

    def _read_current_observations(self):
        getter = self._safe_attr(self.env, "get_observations")
        observations = getter() if callable(getter) else None
        actor, embedded_critic = self._split_observation(observations)
        privileged_getter = self._safe_attr(self.env, "get_privileged_observations")
        privileged = privileged_getter() if callable(privileged_getter) else None
        return actor, privileged if privileged is not None else embedded_critic

    def _read_metadata(self, train_cfg, reset_infos):
        metadata = {}
        sources = [
            self._safe_attr(self.env, "metadata"),
            self._safe_attr(self.env, "observation_metadata"),
            _config_value(train_cfg, "metadata"),
        ]
        if isinstance(reset_infos, Mapping):
            sources.extend(
                [reset_infos.get("metadata"), reset_infos.get("observation_metadata")]
            )
        for source in sources:
            if isinstance(source, Mapping):
                metadata.update(source)
        return metadata

    def _metadata_number(self, names):
        def lookup(source, depth=0):
            if source is None or depth > 3:
                return None
            if isinstance(source, Mapping):
                for name in names:
                    if name in source and source[name] is not None:
                        return source[name]
                for child_name in (
                    "observations",
                    "observation",
                    "actor",
                    "policy",
                    "critic",
                    "privileged",
                    "actions",
                    "action",
                ):
                    value = lookup(source.get(child_name), depth + 1)
                    if value is not None:
                        return value
            else:
                for name in names:
                    value = self._safe_attr(source, name)
                    if value is not None:
                        return value
            return None

        for source in (self.metadata, self.env):
            value = lookup(source)
            if value is not None:
                return int(value)
        return None

    def _resolve_one_step_obs(self):
        value = self._metadata_number(
            ("num_one_step_obs", "one_step_obs_dim", "num_one_step_observations")
        )
        if value is not None:
            return value

        history_length = self._metadata_number(
            ("history_length", "num_history_steps", "temporal_steps")
        )
        if history_length is None:
            history_length = 1
        if history_length <= 0:
            raise ValueError("HIM history length must be positive")
        if self.num_actor_obs % history_length != 0:
            raise ValueError(
                "actor observation width is not divisible by history length: "
                f"{self.num_actor_obs} vs {history_length}"
            )
        return self.num_actor_obs // history_length

    def _resolve_num_actions(self):
        value = self._metadata_number(("num_actions", "action_dim", "num_action"))
        if value is not None:
            return value

        action_space = self._safe_attr(self.env, "action_space")
        shape = self._safe_attr(action_space, "shape")
        if shape:
            return int(shape[-1])

        action_manager = self._safe_attr(self.env, "action_manager")
        for name in ("total_action_dim", "action_dim", "num_actions"):
            value = self._safe_attr(action_manager, name)
            if value is not None:
                return int(value)
        raise ValueError("HIM environment does not expose num_actions metadata")

    def _resolve_class(self, class_name):
        if inspect.isclass(class_name):
            return class_name
        if isinstance(class_name, str) and class_name in globals():
            return globals()[class_name]
        raise ValueError("unknown HIM class: {}".format(class_name))

    def _prepare_critic_obs(self, critic_obs, actor_obs):
        critic_obs = self._flatten_batch(critic_obs, "critic observation")
        if critic_obs is None:
            if actor_obs is None or actor_obs.shape[-1] < self.num_one_step_obs:
                raise ValueError("critic observation is missing from the HIM environment")
            # This fallback preserves the tensor contract for environments
            # without privileged observations; its zero velocity target is
            # intentionally visible as a runtime limitation rather than a
            # shape mismatch later in PPO.
            velocity = torch.zeros(
                actor_obs.shape[0],
                3,
                dtype=actor_obs.dtype,
                device=actor_obs.device,
            )
            return torch.cat((actor_obs[..., :self.num_one_step_obs], velocity), dim=-1)
        if critic_obs.shape[-1] < self.num_critic_obs:
            raise ValueError(
                "critic observation is narrower than the HIM contract: "
                f"{critic_obs.shape[-1]} vs {self.num_critic_obs}"
            )
        # The wrapper's critic group starts with the one-step actor group and
        # appends base velocity.  Ignore optional privileged extras.
        return critic_obs[..., :self.num_critic_obs]

    @staticmethod
    def _termination_indices(termination_ids, num_envs, device):
        if termination_ids is None:
            return torch.empty(0, dtype=torch.long, device=device)
        ids = torch.as_tensor(termination_ids, device=device)
        if ids.dtype == torch.bool:
            if ids.numel() != num_envs:
                raise ValueError("boolean termination mask has wrong batch size")
            ids = torch.nonzero(ids.reshape(-1), as_tuple=False).flatten()
        else:
            ids = ids.reshape(-1).long()
        if ids.numel() and (ids.min() < 0 or ids.max() >= num_envs):
            raise IndexError("termination index is outside the environment batch")
        return ids

    def _apply_terminal_critic_obs(
        self, next_critic_obs, termination_ids, terminal_privileged_obs
    ):
        ids = self._termination_indices(
            termination_ids,
            next_critic_obs.shape[0],
            next_critic_obs.device,
        )
        if ids.numel() == 0 or terminal_privileged_obs is None:
            return next_critic_obs

        terminal = self._flatten_batch(
            terminal_privileged_obs, "terminal privileged observation"
        )
        if terminal.shape[0] == next_critic_obs.shape[0]:
            terminal = terminal.index_select(0, ids)
        elif terminal.shape[0] != ids.numel():
            raise ValueError(
                "terminal privileged observation batch does not match done subset: "
                f"{terminal.shape[0]} vs {ids.numel()}"
            )
        terminal = self._prepare_critic_obs(terminal, None).to(
            device=next_critic_obs.device,
            dtype=next_critic_obs.dtype,
        )
        next_critic_obs = next_critic_obs.clone()
        next_critic_obs.index_copy_(0, ids, terminal)
        return next_critic_obs
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            episode_length_buf = self._safe_attr(self.env, "episode_length_buf")
            max_episode_length = self._safe_attr(self.env, "max_episode_length")
            if episode_length_buf is not None and max_episode_length is not None:
                self.env.episode_length_buf = torch.randint_like(
                    episode_length_buf, high=int(max_episode_length)
                )
        raw_obs, raw_critic_obs = self._read_current_observations()
        obs = self._flatten_batch(raw_obs, "actor observation")
        if obs is None:
            obs = self._initial_obs
        obs = obs.to(self.device)
        critic_obs = self._prepare_critic_obs(raw_critic_obs, obs).to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        num_envs = int(getattr(self.env, "num_envs", obs.shape[0]))
        cur_reward_sum = torch.zeros(num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    (
                        raw_obs,
                        raw_critic_obs,
                        rewards,
                        dones,
                        infos,
                        termination_ids,
                        termination_privileged_obs,
                    ) = self._unpack_step(self.env.step(actions))
                    embedded_critic = None
                    raw_obs, embedded_critic = self._split_observation(raw_obs)
                    if raw_critic_obs is None:
                        raw_critic_obs = embedded_critic
                    obs = self._flatten_batch(raw_obs, "actor observation")
                    if obs is None:
                        raise ValueError("HIM environment step did not return actor observations")
                    obs = obs.to(self.device)
                    critic_obs = self._prepare_critic_obs(raw_critic_obs, obs).to(self.device)
                    rewards = torch.as_tensor(
                        rewards, device=self.device, dtype=torch.float32
                    ).reshape(-1)
                    dones = torch.as_tensor(
                        dones, device=self.device, dtype=torch.bool
                    ).reshape(-1)
                    infos = dict(infos or {})
                    if termination_privileged_obs is None:
                        termination_privileged_obs = self._terminal_privileged_from_infos(
                            infos
                        )
                    infos = self._normalize_timeout_bootstrap(
                        infos, termination_privileged_obs, dones
                    )

                    next_critic_obs = self._apply_terminal_critic_obs(
                        critic_obs.detach(),
                        termination_ids,
                        termination_privileged_obs,
                    )

                    self.alg.process_env_step(rewards, dones, infos, next_critic_obs)
                
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = torch.nonzero(dones, as_tuple=False).flatten()
                        rewbuffer.extend(cur_reward_sum[new_ids].cpu().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids].cpu().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
                
            mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if self.log_dir is not None and it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/Estimation Loss', locs['mean_estimation_loss'], locs['it'])
        self.writer.add_scalar('Loss/Swap Loss', locs['mean_swap_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"""
                          f"""{'Swap loss:':>{pad}} {locs['mean_swap_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"""
                          f"""{'Swap loss:':>{pad}} {locs['mean_swap_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'estimator_optimizer_state_dict': self.alg.actor_critic.estimator.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            self.alg.actor_critic.estimator.optimizer.load_state_dict(loaded_dict['estimator_optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
