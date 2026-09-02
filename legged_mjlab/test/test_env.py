import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from mjlab.envs import ManagerBasedRlEnv

from legged_mjlab.envs.him_go2.him_go2_env import HimGo2Env


def _make_env(rewards_cfg):
    env = HimGo2Env.__new__(HimGo2Env)
    env.robot_cfg = SimpleNamespace(rewards = rewards_cfg)
    return env


def _fake_parent_step(self, action):
    rewards = torch.tensor([-1.5, 0.0, 2.25], device = action.device)
    self.reward_buf = rewards

    obs = {"policy": torch.zeros((3, 1), device = action.device)}
    terminated = torch.tensor([False, False, True], device = action.device)
    truncated = torch.tensor([False, True, False], device = action.device)
    infos = {"source": "fake_parent_step"}

    return obs, rewards, terminated, truncated, infos


class HimGo2EnvTests(unittest.TestCase):
    def test_only_positive_rewards_clips_negative_total_rewards(self):
        env = _make_env(SimpleNamespace(only_positive_rewards = True))
        action = torch.zeros(3)

        with patch.object(ManagerBasedRlEnv, "step", _fake_parent_step):
            _, rewards, terminated, truncated, _ = HimGo2Env.step(env, action)

        torch.testing.assert_close(rewards, torch.tensor([0.0, 0.0, 2.25]))
        self.assertIs(env.reward_buf, rewards)
        self.assertTrue(torch.equal(terminated, torch.tensor([False, False, True])))
        self.assertTrue(torch.equal(truncated, torch.tensor([False, True, False])))

    def test_only_positive_rewards_false_preserves_total_rewards(self):
        env = _make_env(SimpleNamespace(only_positive_rewards = False))
        action = torch.zeros(3)

        with patch.object(ManagerBasedRlEnv, "step", _fake_parent_step):
            _, rewards, _, _, _ = HimGo2Env.step(env, action)

        torch.testing.assert_close(rewards, torch.tensor([-1.5, 0.0, 2.25]))
        self.assertIs(env.reward_buf, rewards)

    def test_missing_only_positive_rewards_preserves_total_rewards(self):
        env = _make_env(SimpleNamespace())
        action = torch.zeros(3)

        with patch.object(ManagerBasedRlEnv, "step", _fake_parent_step):
            _, rewards, _, _, _ = HimGo2Env.step(env, action)

        torch.testing.assert_close(rewards, torch.tensor([-1.5, 0.0, 2.25]))
        self.assertIs(env.reward_buf, rewards)


if __name__ == "__main__":
    unittest.main()
