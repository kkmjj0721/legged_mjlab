from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv


class LeggedMjlabEnv(ManagerBasedRlEnv):
  robot_entity_name = "robot"

    def get_robot(self):
        """Return the task's robot entity from the official Scene."""

        return self.scene[self.robot_entity_name]

    def get_env_origins(self) -> torch.Tensor:
        """Return per-environment origins owned by the Scene."""

        return self.scene.env_origins

    def reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        """Compatibility name for legged-gym-style partial reset calls."""

        self._reset_idx(env_ids)
