import torch
import legged_mjlab.envs  
from legged_mjlab.utils.task_registry import task_registry
from legged_mjlab.utils.helpers import get_args, update_cfg_from_args

def play(args):
    args.play = True
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    train_cfg.runner.resume = True     
    if args.num_envs is None:
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
        
    update_cfg_from_args(env_cfg, train_cfg, args)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

    policy = runner.get_inference_policy(device=args.rl_device)
    
    # 可选：如果需要导出 ONNX 或 JIT 部署，可以在这里调用 utils.helpers 中的导出函数
    # from legged_mjlab.utils.helpers import export_policy_as_jit
    # export_policy_as_jit(runner.alg.actor_critic, "policy_1.pt")

    obs, _ = env.reset()
    
    with torch.no_grad():
        while True:
            actions = policy(obs)
            step_results = env.step(actions)
            obs = step_results[0]

if __name__ == '__main__':
    args = get_args()

    if not args.headless:
        args.headless = False
        
    play(args)