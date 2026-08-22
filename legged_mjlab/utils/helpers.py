import argparse
import inspect

def get_args():
    parser = argparse.ArgumentParser(description="legged_mjlab Training Framework")
    parser.add_argument("--task", type=str, default="him_go2", help="Task name")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to create")
    parser.add_argument("--headless", action="store_true", default=False, help="Force display off")
    parser.add_argument("--sim_device", type=str, default="cuda:0", help="Simulation device")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from a checkpoint")
    parser.add_argument("--load_run", type=str, default=None, help="Name of the run to load when resume")
    parser.add_argument("--checkpoint", type=int, default=-1, help="Saved model checkpoint number")
    parser.add_argument("--max_iterations", type=int, default=None, help="Override maximum training iterations")
    return parser.parse_args()

def class_to_dict(obj) -> dict:
    """递归将嵌套配置类转换为原生 Python 字典"""
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        val = getattr(obj, key)
        if inspect.isclass(val) or isinstance(val, object):
            if hasattr(val, "__dict__"):
                result[key] = class_to_dict(val)
            else:
                result[key] = val
        else:
            result[key] = val
    return result