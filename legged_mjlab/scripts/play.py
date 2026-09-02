import os
import torch
import legged_mjlab.envs
from legged_mjlab.utils.task_registry import task_registry
from legged_mjlab.utils.helpers import get_args, update_cfg_from_args

def play(args):
    args.play = True
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    train_cfg.runner.resume = True

    if args.num_envs is None:
        env_cfg.env.num_envs = 1

    update_cfg_from_args(env_cfg, train_cfg, args)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    if args.agent == "trained":
        runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
        policy = runner.get_inference_policy(device=args.rl_device)
    elif args.agent == "zero":
        policy = lambda obs: torch.zeros((env.num_envs, env.num_actions), device=args.rl_device)
    elif args.agent == "random":
        policy = lambda obs: 2.0 * torch.rand((env.num_envs, env.num_actions), device=args.rl_device) - 1.0
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")

    if args.viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        resolved_viewer = "native" if has_display else "viser"
    else:
        resolved_viewer = args.viewer

    env.reset()

    print(f"[INFO] Launching viewer backend: {resolved_viewer}")
    if resolved_viewer == "native":
        from mjlab.viewer import NativeMujocoViewer
        viewer = NativeMujocoViewer(env, policy)
        viewer.run()
    elif resolved_viewer == "viser":
        from mjlab.viewer import ViserPlayViewer
        viewer = ViserPlayViewer(env, policy)
        viewer.run()
    else:
        raise ValueError(f"Unsupported viewer backend: {resolved_viewer}")

if __name__ == '__main__':
    args = get_args()
    play(args)