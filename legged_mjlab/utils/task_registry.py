from dataclasses import dataclass

from legged_mjlab.wrappers.him_wrapper import HIMRslRlWrapper


@dataclass
class TaskSpec:
    task_id: str
    env_cls: type
    env_cfg_cls: type
    train_cfg_cls: type
    wrapper_name: str


class TaskRegistry:
    def __init__(self):
        self.task_specs = {}

    def register(self, task_id, env_cls, env_cfg_cls, train_cfg_cls, wrapper_name):
        if task_id in self.task_specs:
            raise KeyError("task already registered: " + task_id)
        if wrapper_name not in {"ppo", "him", "amp"}:
            raise ValueError("unsupported wrapper: " + wrapper_name)
        self.task_specs[task_id] = TaskSpec(
            task_id=task_id,
            env_cls=env_cls,
            env_cfg_cls=env_cfg_cls,
            train_cfg_cls=train_cfg_cls,
            wrapper_name=wrapper_name,
        )

    def list_tasks(self):
        return tuple(sorted(self.task_specs))

    def get(self, task_id):
        if task_id not in self.task_specs:
            raise KeyError(
                "unknown task " + task_id + "; choices=" + str(self.list_tasks())
            )
        return self.task_specs[task_id]

    def make_env(self, task_id, device=None, play=False):
        spec = self.get(task_id)
        env_cfg = spec.env_cfg_cls()
        env = spec.env_cls(
            cfg=env_cfg,
            device=device or env_cfg.env.device,
            play=play,
        )
        if spec.wrapper_name == "him":
            return HIMRslRlWrapper(
                env,
                history_length=env_cfg.him.history_length,
            ), env_cfg
        return RslRlVecEnvWrapper(env), env_cfg

    def make_alg_runner(self, task_id, env, train_cfg, log_dir):
        spec = self.get(task_id)
        runner_name = train_cfg.runner.runner_class_name
        if runner_name == "OnPolicyRunner":
            from rsl_rl.runners import OnPolicyRunner
            return OnPolicyRunner(env, train_cfg, log_dir)
        if runner_name == "HIMOnPolicyRunner":
            from rsl_rl.runners import HIMOnPolicyRunner
            return HIMOnPolicyRunner(env, train_cfg, log_dir)
        if runner_name == "AMPOnPolicyRunner":
            from rsl_rl.runners import AMPOnPolicyRunner
            return AMPOnPolicyRunner(env, train_cfg, log_dir)
        raise ValueError("unsupported runner: " + runner_name)


task_registry = TaskRegistry()