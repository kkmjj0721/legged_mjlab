"""Command-line entry point for configured RSL-RL training."""

import argparse
from pathlib import Path


def _positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="him_go2")
    parser.add_argument("--device", "--sim-device", dest="device", default=None)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument(
        "--num-envs",
        "--num_envs",
        dest="num_envs",
        type=_positive_int,
        default=None,
        help="override the configured vectorized environment count",
    )
    parser.add_argument(
        "--max-iterations",
        type=_positive_int,
        default=None,
        help="positive smoke/training override for runner.max_iterations",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Keep --help/import smoke independent from optional mjlab and torch
    # imports.  Runtime dependencies are loaded only after argparse succeeds.
    import legged_mjlab.envs  # noqa: F401  # triggers task registration
    from legged_mjlab.utils.task_registry import load_project_rsl, task_registry

    load_project_rsl()
    spec = task_registry.get(args.task)
    train_cfg = spec.train_cfg_cls()
    if args.max_iterations is not None:
        train_cfg.runner.max_iterations = args.max_iterations

    env, _ = task_registry.make_env(
        args.task,
        device=args.device,
        play=False,
        num_envs=args.num_envs,
    )
    train_cfg_dict = train_cfg.to_dict()
    runner_cfg = train_cfg_dict["runner"]
    experiment_name = runner_cfg.get("experiment_name", args.task)
    log_dir = Path(args.log_dir) / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = task_registry.make_alg_runner(
        args.task,
        env,
        train_cfg_dict,
        str(log_dir),
    )
    runner.learn(
        num_learning_iterations=runner_cfg["max_iterations"],
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    main()
