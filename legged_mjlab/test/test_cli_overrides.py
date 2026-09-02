import argparse
import sys
import unittest
from unittest.mock import patch

from legged_mjlab.envs.him_go2.him_go2_config import HimGo2CfgPPO, HimGo2RoughCfg
from legged_mjlab.utils.helpers import get_args, update_cfg_from_args


def _args(**overrides):
    values = {
        "num_envs": None,
        "seed": None,
        "num_steps_per_env": None,
        "max_iterations": None,
        "resume": False,
        "experiment_name": None,
        "run_name": None,
        "load_run": None,
        "checkpoint": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class CliOverrideTests(unittest.TestCase):
    def test_update_cfg_applies_rollout_steps_and_preserves_max_iterations(self):
        env_cfg = HimGo2RoughCfg()
        train_cfg = HimGo2CfgPPO()

        update_cfg_from_args(
            env_cfg,
            train_cfg,
            _args(num_envs=128, num_steps_per_env=8),
        )

        self.assertEqual(env_cfg.env.num_envs, 128)
        self.assertEqual(train_cfg.runner.num_steps_per_env, 8)
        self.assertEqual(train_cfg.runner.max_iterations, 10000)

    def test_update_cfg_keeps_rollout_steps_when_only_max_iterations_changes(self):
        env_cfg = HimGo2RoughCfg()
        train_cfg = HimGo2CfgPPO()

        update_cfg_from_args(env_cfg, train_cfg, _args(max_iterations=3))

        self.assertEqual(env_cfg.env.num_envs, 4096)
        self.assertEqual(train_cfg.runner.num_steps_per_env, 100)
        self.assertEqual(train_cfg.runner.max_iterations, 3)

    def test_get_args_parses_rollout_steps_as_int(self):
        argv = [
            "train.py",
            "--task=him_go2",
            "--rl_device=cpu",
            "--num_envs=16",
            "--num_steps_per_env=7",
            "--max_iterations=2",
        ]

        with patch.object(sys, "argv", argv):
            args = get_args()

        self.assertEqual(args.task, "him_go2")
        self.assertEqual(args.rl_device, "cpu")
        self.assertEqual(args.sim_device, "cpu")
        self.assertEqual(args.num_envs, 16)
        self.assertEqual(args.num_steps_per_env, 7)
        self.assertEqual(args.max_iterations, 2)

    def test_get_args_rejects_invalid_rollout_steps_with_argparse(self):
        argv = ["train.py", "--num_steps_per_env=not-an-int"]

        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            get_args()

    def test_train_entrypoint_applies_overrides_before_runner_construction(self):
        from legged_mjlab.scripts import train as train_module

        class DummyRunner:
            def __init__(self):
                self.learn_iterations = None

            def learn(self, num_learning_iterations, init_at_random_ep_len):
                self.learn_iterations = num_learning_iterations
                self.init_at_random_ep_len = init_at_random_ep_len

        class DummyRegistry:
            def __init__(self):
                self.runner = DummyRunner()
                self.seen_env_cfg = None
                self.seen_train_cfg = None

            def get_cfgs(self, name):
                self.task_name = name
                return HimGo2RoughCfg(), HimGo2CfgPPO()

            def make_env(self, name, args, env_cfg):
                self.seen_env_cfg = env_cfg
                return object(), env_cfg

            def make_alg_runner(self, env, name, args, train_cfg):
                self.seen_train_cfg = train_cfg
                return self.runner, train_cfg

        registry = DummyRegistry()
        args = _args(
            task="him_go2",
            num_envs=64,
            num_steps_per_env=5,
            max_iterations=9,
        )

        with patch.object(train_module, "task_registry", registry):
            train_module.train(args)

        self.assertFalse(args.play)
        self.assertEqual(registry.task_name, "him_go2")
        self.assertEqual(registry.seen_env_cfg.env.num_envs, 64)
        self.assertEqual(registry.seen_train_cfg.runner.num_steps_per_env, 5)
        self.assertEqual(registry.seen_train_cfg.runner.max_iterations, 9)
        self.assertEqual(registry.runner.learn_iterations, 9)
        self.assertTrue(registry.runner.init_at_random_ep_len)


if __name__ == "__main__":
    unittest.main()
