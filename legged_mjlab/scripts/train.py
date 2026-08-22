import argparse
from pathlib import Path

import legged_mjlab.envs
from legged_mjlab.envs.him_go2.him_go2_config import HimGo2CfgPPO
from legged_mjlab.utils.task_registry import task_registry


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="him_go2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--max-iterations", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    env, env_cfg = task_registry.make_env(
        args.task,
        device=args.device,
        play=False,
    )
    train_cfg = HimGo2CfgPPO()
    if args.max_iterations is not None:
        train_cfg.runner.max_iterations = args.max_iterations
    log_dir = Path(args.log_dir) / train_cfg.runner.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = task_registry.make_alg_runner(
        args.task,
        env,
        train_cfg,
        str(log_dir),
    )
    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    main()