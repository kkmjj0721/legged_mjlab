"""Command-line entry point for configured RSL-RL training."""

import legged_mjlab.envs
from legged_mjlab.utils.task_registry import task_registry
from legged_mjlab.utils.helpers import get_args, update_cfg_from_args
from legged_mjlab.utils.training_viewer import TrainingViewerHook

def train(args):
    args.play = False
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    update_cfg_from_args(env_cfg, train_cfg, args)
    
    if not args.headless and getattr(args, "num_envs", None) is None:
        env_cfg.env.num_envs = 1
        
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    
    viewer_hook = None
    if not args.headless:
        viewer_hook = TrainingViewerHook(
            env=env,
            backend=args.viewer,
            interval=getattr(args, "viewer_interval", 1),
            exit_action=getattr(args, "viewer_exit", "continue"),
        )
        viewer_hook.setup()

        original_step = env.step

        def step_with_viewer(actions):
            result = original_step(actions)
            viewer_hook.render_if_needed()
            return result

        env.step = step_with_viewer

    try:
        runner.learn(
            num_learning_iterations=train_cfg.runner.max_iterations,
            init_at_random_ep_len=True
        )
    finally:
        if viewer_hook is not None:
            viewer_hook.close()

if __name__ == '__main__':
    args = get_args()
    train(args)