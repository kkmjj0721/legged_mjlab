"""Command-line entry point for configured RSL-RL training."""
import legged_mjlab.envs 
from legged_mjlab.utils.task_registry import task_registry
from legged_mjlab.utils.helpers import get_args, update_cfg_from_args

def train(args):
    args.play = False
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    update_cfg_from_args(env_cfg, train_cfg, args)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True
    )

if __name__ == '__main__':
    args = get_args()
    train(args)