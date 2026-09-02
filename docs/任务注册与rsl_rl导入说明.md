# 任务注册与 rsl_rl 导入说明

这篇文档解决两个问题：第一，为什么在创建任务或训练入口里写 `from rsl_rl.runners import HIMOnPolicyRunner` 会导入失败；第二，当前 `legged_mjlab` 的任务注册到底是不是自己写的，以及如果想参考 `register_mjlab_task(...)` 的写法，应该怎么改得更像 `legged_gym`、也更简便。

## 1. 先说结论

当前项目里确实有自己的 `HIMOnPolicyRunner` 源码，它在：

```text
rsl_rl/rsl_rl/runners/him_on_policy_runner.py
```

并且本地源码的导出口也已经写了：

```python
# rsl_rl/rsl_rl/runners/__init__.py
from .on_policy_runner import OnPolicyRunner
from .him_on_policy_runner import HIMOnPolicyRunner
```

所以问题不是 `HIMOnPolicyRunner` 类不存在，而是 Python 实际导入到的 `rsl_rl` 不是这个本地源码包，或者仓库根目录下的外层 `rsl_rl/` 目录把内层包路径挡住了。

当前项目的“创建任务/注册任务”部分是自己写的，入口主要在：

```text
legged_mjlab/utils/task_registry.py
legged_mjlab/envs/__init__.py
```

它不是参考项目里的：

```python
from mjlab.tasks.registry import register_mjlab_task
```

那套注册机制。

## 2. 为什么 `from rsl_rl.runners import HIMOnPolicyRunner` 会失败

仓库里有两层 `rsl_rl`：

```text
legged_mjlab/
  rsl_rl/
    setup.py
    rsl_rl/
      __init__.py
      runners/
        __init__.py
        him_on_policy_runner.py
        on_policy_runner.py
```

真正的 Python 包在：

```text
rsl_rl/rsl_rl/
```

但是如果你在项目根目录直接运行 Python，`sys.path` 里通常会先有当前目录，也就是：

```text
/home/kk/legged_mjlab
```

这时 Python 看到的是外层目录：

```text
/home/kk/legged_mjlab/rsl_rl
```

外层目录本身没有 `runners/`，真正的 `runners/` 在下一层 `rsl_rl/rsl_rl/runners/`。所以就可能出现：

```text
ModuleNotFoundError: No module named 'rsl_rl.runners'
```

另一种情况是虚拟环境里安装了别的同名包。例如有些版本的 `rsl_rl` 或 `rsl_rl_lib` 只导出：

```python
from .on_policy_runner import OnPolicyRunner
from .distillation_runner import DistillationRunner
```

它没有 `HIMOnPolicyRunner`，所以会出现：

```text
ImportError: cannot import name 'HIMOnPolicyRunner' from 'rsl_rl.runners'
```

可以用下面这个命令确认当前到底导入了哪个包：

```bash
python3 - <<'PY'
import rsl_rl
print(rsl_rl)
print(getattr(rsl_rl, "__file__", None))

import rsl_rl.runners
print(rsl_rl.runners.__file__)
print(dir(rsl_rl.runners))
PY
```

如果路径不是项目里的：

```text
/home/kk/legged_mjlab/rsl_rl/rsl_rl/...
```

那就是导入源不对。

## 3. 推荐的导入修复方式

优先推荐修虚拟环境的安装来源，而不是在代码里到处手动改 `sys.path`。先确认当前导入到哪里：

```bash
.venv/bin/python - <<'PY'
import rsl_rl
import rsl_rl.runners

print(rsl_rl.__file__)
print(rsl_rl.runners.__file__)
print(hasattr(rsl_rl.runners, "HIMOnPolicyRunner"))
PY
```

如果看到的是：

```text
.venv/lib/python3.11/site-packages/rsl_rl/...
```

并且 `hasattr(..., "HIMOnPolicyRunner")` 是 `False`，说明当前环境里的同名 `rsl_rl` 包遮住了仓库自带的 HIM 版本。

更稳的修复顺序是：

```bash
source .venv/bin/activate

# 先安装本地 HIM 版本 rsl_rl
uv pip install -e ./rsl_rl --force-reinstall
uv pip install -e . --force-reinstall
```

如果验证后仍然导入到 `site-packages/rsl_rl` 且没有 `HIMOnPolicyRunner`，再检查是否存在同名顶层包冲突：

```bash
uv pip list | grep -E "rsl|RSL"
```

当前仓库的 `.venv` 里曾经同时出现 `rsl_rl-1.0.2` 和 `rsl_rl_lib-5.4.2`，而它们都声明顶层包名 `rsl_rl`。这种情况下，需要保留一个明确的导入来源。对于本项目的 HIM 训练，目标应该是让导入解析到：

```text
/home/kk/legged_mjlab/rsl_rl/rsl_rl
```

然后验证：

```bash
python3 - <<'PY'
from rsl_rl.runners import HIMOnPolicyRunner, OnPolicyRunner
print(HIMOnPolicyRunner)
print(OnPolicyRunner)
PY
```

如果你确实需要使用 `mjlab` 官方 registry 或 `mjlab.rl` 里的 runner，再决定是否保留 `rsl_rl_lib`。不要在还没确认依赖关系时无差别卸载所有 `rsl_rl` 相关包。

如果你只是临时验证，也可以这样跑：

```bash
PYTHONPATH=/home/kk/legged_mjlab/rsl_rl:$PYTHONPATH python3 - <<'PY'
from rsl_rl.runners import HIMOnPolicyRunner
print(HIMOnPolicyRunner)
PY
```

这个临时方式能说明问题，但不建议作为长期方案。长期方案还是让本地 `rsl_rl` 被 editable install 正确接管。

## 4. 当前项目的任务注册方式

当前项目更接近 `legged_gym` 的传统写法：任务配置、环境类、训练配置通过一个 registry 绑定起来。

注册点在：

```python
# legged_mjlab/envs/__init__.py
from legged_mjlab.envs.him_go2.him_go2_config import HimGo2CfgPPO, HimGo2RoughCfg
from legged_mjlab.envs.him_go2.him_go2_env import HimGo2Env
from legged_mjlab.utils.task_registry import task_registry

task_registry.register("him_go2", HimGo2Env, HimGo2RoughCfg(), HimGo2CfgPPO())
```

registry 在：

```python
# legged_mjlab/utils/task_registry.py
class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(self, name: str, task_class, env_cfg, train_cfg):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
```

这个思路和 `legged_gym` 是一致的：

```python
task_registry.register(name, EnvClass, EnvCfg(), TrainCfg())
```

区别是，当前项目又引入了 `mjlab`、`MuJoCo`、本地 `rsl_rl` 和 wrapper，所以 `make_env()` 和 `make_alg_runner()` 需要额外负责：

1. 创建 mjlab 环境；
2. 根据 runner 类型选择 wrapper；
3. 把 config 转成 `rsl_rl` runner 需要的 dict；
4. 实例化 `OnPolicyRunner` 或 `HIMOnPolicyRunner`。

## 5. 参考项目 `register_mjlab_task(...)` 是什么风格

你贴的参考项目是这种：

```python
from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
    unitree_go2_flat_env_cfg,
    unitree_go2_rough_env_cfg,
)
from .rl_cfg import unitree_go2_ppo_runner_cfg

register_mjlab_task(
    task_id="Unitree-Go2-Rough",
    env_cfg=unitree_go2_rough_env_cfg(),
    play_env_cfg=unitree_go2_rough_env_cfg(play=True),
    rl_cfg=unitree_go2_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
```

这套写法依赖的是 `mjlab` 自己的 task registry。它通常要求：

- `env_cfg` 是 mjlab 期望的环境配置对象；
- `play_env_cfg` 是 play 模式下的环境配置对象；
- `rl_cfg` 是 mjlab/RSL-RL 适配后的 runner 配置对象；
- `runner_cls` 是一个 runner 类对象，不是字符串。

它的优点是简洁：注册时直接告诉系统“这个 task id 对应什么环境配置、play 配置、RL 配置和 runner 类”。

但是当前 `legged_mjlab` 还不是完全这套结构。当前配置是 `legged_gym` 式 class config，例如：

```python
class HimGo2RoughCfg(LeggedMjlabCfg):
    class env(LeggedMjlabCfg.env):
        num_envs = 4096
        num_one_step_observations = 45
        history_length = 6
        num_observations = num_one_step_observations * history_length
        num_privileged_obs = num_one_step_observations + 3 + 187
        num_actions = 12
```

所以它不能直接无脑替换成参考项目那种写法。能用，但要先决定谁做主 registry。

## 6. 推荐方案：保留 legged_gym 风格，但把 runner 改成类对象

我建议先不要马上切到 `mjlab.tasks.registry.register_mjlab_task`。更稳的做法是保留当前 `legged_gym` 风格，然后把 `runner_class_name = "HIMOnPolicyRunner"` 这种字符串判断，改成注册时传 `runner_cls`。

也就是从：

```python
class HimGo2CfgPPO(LeggedMjlabCfgPPO):
    class runner(LeggedMjlabCfgPPO.runner):
        policy_class_name = "HIMActorCritic"
        algorithm_class_name = "HIMPPO"
        experiment_name = "him_go2"
```

加上或保留：

```python
runner_class_name = "HIMOnPolicyRunner"
```

逐步改成注册时直接传类：

```python
from rsl_rl.runners import HIMOnPolicyRunner

task_registry.register(
    name="him_go2",
    task_class=HimGo2Env,
    env_cfg=HimGo2RoughCfg(),
    train_cfg=HimGo2CfgPPO(),
    runner_cls=HIMOnPolicyRunner,
)
```

对应 registry 可以设计成：

```python
class TaskSpec:
    def __init__(self, name, task_class, env_cfg, train_cfg, runner_cls):
        self.name = name
        self.task_class = task_class
        self.env_cfg = env_cfg
        self.train_cfg = train_cfg
        self.runner_cls = runner_cls


class TaskRegistry:
    def __init__(self):
        self.tasks = {}

    def register(self, name, task_class, env_cfg, train_cfg, runner_cls=None):
        self.tasks[name] = TaskSpec(
            name=name,
            task_class=task_class,
            env_cfg=env_cfg,
            train_cfg=train_cfg,
            runner_cls=runner_cls,
        )

    def get(self, name):
        if name not in self.tasks:
            raise ValueError(f"Task with name {name} was not registered")
        return self.tasks[name]
```

这样写的好处是：

- 任务注册仍然像 `legged_gym`；
- runner 选择不靠字符串硬判断；
- 以后接 `VelocityOnPolicyRunner`、`HIMOnPolicyRunner`、普通 `OnPolicyRunner` 都是一套入口；
- `train.py` 里也可以通过 `task_registry.get(args.task)` 拿到完整 task spec。

## 7. 更接近参考项目的写法

如果想写得像参考项目，可以在本项目里包一层自己的函数，例如：

```python
def register_legged_mjlab_task(
    task_id,
    env_cls,
    env_cfg_cls,
    train_cfg_cls,
    play_env_cfg_cls=None,
    runner_cls=None,
):
    task_registry.register(
        name=task_id,
        task_class=env_cls,
        env_cfg_cls=env_cfg_cls,
        train_cfg_cls=train_cfg_cls,
        play_env_cfg_cls=play_env_cfg_cls,
        runner_cls=runner_cls,
    )
```

然后任务文件里可以写成：

```python
from rsl_rl.runners import HIMOnPolicyRunner
from legged_mjlab.utils.task_registry import register_legged_mjlab_task

from .him_go2_env import HimGo2Env
from .him_go2_config import HimGo2RoughCfg, HimGo2CfgPPO

register_legged_mjlab_task(
    task_id="him_go2",
    env_cls=HimGo2Env,
    env_cfg_cls=HimGo2RoughCfg,
    play_env_cfg_cls=HimGo2RoughCfg,
    train_cfg_cls=HimGo2CfgPPO,
    runner_cls=HIMOnPolicyRunner,
)
```

这个写法和参考项目很像，但还是保留了本项目自己的 config 和 env 创建逻辑。也就是说，我们不用强行迁移到 `mjlab.tasks.registry`，只需要吸收它“注册时显式传 runner 类”的优点。

## 8. 如果一定要用 `mjlab.tasks.registry.register_mjlab_task`

可以，但要满足一个前提：你的 `env_cfg`、`play_env_cfg`、`rl_cfg` 要符合 `mjlab` 那边的配置协议。

参考项目之所以能这样写：

```python
register_mjlab_task(
    task_id="Unitree-Go2-Rough",
    env_cfg=unitree_go2_rough_env_cfg(),
    play_env_cfg=unitree_go2_rough_env_cfg(play=True),
    rl_cfg=unitree_go2_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
```

是因为它的配置函数本来就是按 `mjlab` 的 task registry 设计的。

当前项目的配置是 class-based legged_gym 风格：

```python
HimGo2RoughCfg()
HimGo2CfgPPO()
```

如果直接塞给 `register_mjlab_task`，可能会遇到这些问题：

- `mjlab` 期望的配置字段不存在；
- `rl_cfg` 的类型不匹配；
- `runner_cls` 初始化参数和当前 `rsl_rl` runner 不一致；
- wrapper 没有接上，尤其是 HIM 需要历史观测和 privileged obs；
- 当前环境类构造参数和 `mjlab` 标准任务创建器不一致。

所以“能不能用”不是取决于你是不是自己的 `rsl_rl`，而是取决于你是否愿意把 env cfg、rl cfg、runner 和 wrapper 全部适配到 `mjlab` 的 registry 协议。

## 9. 当前代码还有一个需要统一的地方

当前仓库里 `train.py` 和 `task_registry.py` 的接口看起来不完全一致。

`train.py` 里像是在用新式 registry：

```python
from legged_mjlab.utils.task_registry import load_project_rsl, task_registry

load_project_rsl()
spec = task_registry.get(args.task)
train_cfg = spec.train_cfg_cls()
```

但是当前 `task_registry.py` 里主要是旧式接口：

```python
def register(self, name: str, task_class, env_cfg, train_cfg):
    ...

def make_env(self, name, args=None, env_cfg=None):
    ...

def make_alg_runner(self, env, name=None, args=None, train_cfg=None, ...):
    ...
```

也就是说，后续修复时不要只盯着 `HIMOnPolicyRunner` 的 import，还要把下面三件事统一：

1. `task_registry.register()` 到底存实例，还是存 cfg class；
2. `task_registry.get()` 是否存在，返回什么结构；
3. `make_env()` 和 `make_alg_runner()` 的参数顺序是否和 `train.py` 一致。

还有一个容易忽略的点：当前 `HimGo2CfgPPO` 里 `runner_class_name = "HIMOnPolicyRunner"` 是放在外层 cfg 上的，但 `task_registry.py` 里读取的是：

```python
getattr(train_cfg.runner, "runner_class_name", "OnPolicyRunner")
```

也就是说，如果不把这个字段移动到 `class runner(...)` 里面，或者不改 registry 的读取逻辑，HIM 任务可能会被默认当成普通 `OnPolicyRunner` 处理。

如果不统一，即使 `HIMOnPolicyRunner` 导入成功，训练入口也可能在下一步报 `AttributeError` 或参数不匹配。

## 10. 建议落地路线

建议按这个顺序改，不要一上来把所有东西迁到 `mjlab.tasks.registry`：

### 第一步：先修 import 源

目标是让下面这句在项目虚拟环境里稳定成功：

```python
from rsl_rl.runners import HIMOnPolicyRunner
```

优先通过 editable install 修：

```bash
uv pip uninstall rsl-rl rsl_rl rsl_rl_lib
uv pip install -e ./rsl_rl
uv pip install -e .
```

### 第二步：统一项目自己的 registry

建议让 registry 保存一个 `TaskSpec`：

```python
TaskSpec(
    name="him_go2",
    env_cls=HimGo2Env,
    env_cfg_cls=HimGo2RoughCfg,
    train_cfg_cls=HimGo2CfgPPO,
    runner_cls=HIMOnPolicyRunner,
)
```

这样 `train.py` 可以直接：

```python
spec = task_registry.get(args.task)
env_cfg = spec.env_cfg_cls()
train_cfg = spec.train_cfg_cls()
runner = spec.runner_cls(env, train_cfg.to_dict(), log_dir, device=args.rl_device)
```

### 第三步：保留 HIM wrapper 特殊处理

HIM 不只是换 runner。它的 actor 输入是历史观测，critic 还可能用 privileged obs，所以 wrapper 仍然要根据 runner 或 policy 类型切换：

```python
if spec.runner_cls is HIMOnPolicyRunner:
    env = HIMRslRlWrapper(...)
else:
    env = RslRlVecEnvWrapper(env)
```

这一点不能省。否则 runner 能创建，但 observation 形状可能不对。

### 第四步：再考虑是否封装成参考项目写法

如果前三步稳定了，再加一个薄封装：

```python
register_legged_mjlab_task(...)
```

让任务注册看起来像：

```python
register_legged_mjlab_task(
    task_id="him_go2",
    env_cls=HimGo2Env,
    env_cfg_cls=HimGo2RoughCfg,
    play_env_cfg_cls=HimGo2RoughCfg,
    train_cfg_cls=HimGo2CfgPPO,
    runner_cls=HIMOnPolicyRunner,
)
```

这个是最适合当前项目的折中方案：外观接近参考项目，内部仍然保持 `legged_gym` 的配置组织方式。

## 11. 最简版模板

如果只是想新建一个任务，建议先按这种结构写。

目录：

```text
legged_mjlab/envs/my_robot/
  __init__.py
  my_robot_env.py
  my_robot_config.py
```

`my_robot_config.py`：

```python
from legged_mjlab.envs.base.legged_mjlab_config import LeggedMjlabCfg, LeggedMjlabCfgPPO


class MyRobotRoughCfg(LeggedMjlabCfg):
    class env(LeggedMjlabCfg.env):
        num_envs = 4096
        num_one_step_observations = 45
        history_length = 6
        num_observations = num_one_step_observations * history_length
        num_privileged_obs = num_one_step_observations + 3 + 187
        num_actions = 12


class MyRobotCfgPPO(LeggedMjlabCfgPPO):
    class policy(LeggedMjlabCfgPPO.policy):
        policy_class_name = "HIMActorCritic"

    class algorithm(LeggedMjlabCfgPPO.algorithm):
        algorithm_class_name = "HIMPPO"

    class runner(LeggedMjlabCfgPPO.runner):
        policy_class_name = "HIMActorCritic"
        algorithm_class_name = "HIMPPO"
        experiment_name = "my_robot_him"
        max_iterations = 10000
```

注册：

```python
from rsl_rl.runners import HIMOnPolicyRunner

from legged_mjlab.envs.my_robot.my_robot_config import MyRobotCfgPPO, MyRobotRoughCfg
from legged_mjlab.envs.my_robot.my_robot_env import MyRobotEnv
from legged_mjlab.utils.task_registry import task_registry


task_registry.register(
    "my_robot_him",
    MyRobotEnv,
    MyRobotRoughCfg(),
    MyRobotCfgPPO(),
    runner_cls=HIMOnPolicyRunner,
)
```

如果暂时还没有把 `runner_cls` 加进 registry，就只能先用当前项目已有的字符串方式：

```python
class MyRobotCfgPPO(LeggedMjlabCfgPPO):
    runner_class_name = "HIMOnPolicyRunner"

    class runner(LeggedMjlabCfgPPO.runner):
        policy_class_name = "HIMActorCritic"
        algorithm_class_name = "HIMPPO"
```

但长期建议换成 `runner_cls`，因为类对象比字符串更稳，也更接近参考项目。

## 12. 最后的判断

你不是因为“使用自己的 `rsl_rl`”所以不能用参考项目那种写法。真正的区别是：

- 参考项目使用 `mjlab` 官方/项目级 registry，注册时显式传 `runner_cls`；
- 当前项目使用自己写的 `TaskRegistry`，更像 `legged_gym`；
- 当前导入失败来自包路径或同名依赖冲突，不是 HIM runner 源码缺失；
- 当前最稳路线是先修本地 `rsl_rl` 导入，再把自己的 registry 改成支持 `runner_cls`；
- 暂时不建议直接把全部任务切到 `mjlab.tasks.registry.register_mjlab_task`，除非同时把 env cfg、rl cfg、runner 初始化和 wrapper 协议都迁过去。

所以推荐目标是：

```text
legged_gym 的任务组织方式
+ mjlab 的环境执行
+ 本地 rsl_rl 的 HIM runner
+ register(..., runner_cls=HIMOnPolicyRunner) 的简洁注册接口
```

这样最贴近当前仓库，也最不容易把后续训练入口改乱。

## 13. 对 Gemini 建议的校正

你后面贴的 Gemini 说法里，有几处方向是对的，但不能完全照抄成当前项目的方案。

第一，`mjlab` 的环境确实是 manager-based，这一点是对的。当前 `HimGo2Env` 继承自 `ManagerBasedRlEnv`，并且在 `__init__()` 里先保存 `robot_cfg`，再调用 `_build_mjlab_managercfg()` 组装 `ManagerBasedRlEnvCfg`，最后通过 `super().__init__(cfg=self.managercfg, device=sim_device, render_mode=render_mode)` 交给 mjlab 底层环境初始化。

```python
class HimGo2Env(ManagerBasedRlEnv):
    def __init__(self, cfg, sim_device, render_mode, play=False, debug_vis=False):
        self.robot_cfg = cfg
        self.asset = Go2Asset(self.robot_cfg)
        self.play = bool(play)
        self.managercfg = self._build_mjlab_managercfg(play=self.play, debug_vis=debug_vis)

        super().__init__(
            cfg=self.managercfg,
            device=sim_device,
            render_mode=render_mode,
        )
```

这说明一个关键点：外层 task registry 不需要知道 scene、action manager、observation manager、reward manager 怎么挂载。它只需要创建 `HimGo2Env(cfg=..., sim_device=..., render_mode=..., play=...)`。真正的 mjlab manager 配置转换应该继续封装在 env 内部。

第二，Gemini 说当前 `task_registry.py` “非常抽象复杂”，这和当前仓库状态不一致。现在的 `task_registry.py` 已经是三字典结构：

```python
self.task_classes = {}
self.env_cfgs = {}
self.train_cfgs = {}
```

也就是说，它本身已经很接近 `legged_gym` 的简单 registry。当前问题不是 registry 太复杂，而是下面几件事没有统一好：

- `rsl_rl` 导入源不稳定；
- `train.py` 期待 `load_project_rsl()`、`task_registry.get()` 和 `TaskSpec`，但当前 registry 没有这些接口；
- `train.py` 调用 `make_env(task, device=..., play=..., num_envs=...)`，但当前 `make_env()` 只接受 `name, args=None, env_cfg=None`；
- `train.py` 调用 `make_alg_runner(args.task, env, train_cfg_dict, str(log_dir))` 的参数顺序和当前 `make_alg_runner(env, name=None, args=None, train_cfg=None, log_root=...)` 不匹配；
- `HimGo2CfgPPO.runner_class_name = "HIMOnPolicyRunner"` 目前放在外层 cfg，而 registry 读取的是 `train_cfg.runner.runner_class_name`。

第三，Gemini 建议在 `task_registry.py` 里写：

```python
local_rsl_rl_path = os.path.join(PROJECT_ROOT.parent, "rsl_rl")
sys.path.insert(0, local_rsl_rl_path)
```

这个不建议作为正式方案，而且在当前仓库里路径还可能是错的。当前 `PROJECT_ROOT` 是仓库根目录：

```text
/home/kk/legged_mjlab
```

所以 `PROJECT_ROOT.parent / "rsl_rl"` 会变成：

```text
/home/kk/rsl_rl
```

但当前本地源码实际在：

```text
/home/kk/legged_mjlab/rsl_rl/rsl_rl
```

如果一定要临时验证，应该让 `PYTHONPATH` 指到外层包工程目录：

```bash
PYTHONPATH=/home/kk/legged_mjlab/rsl_rl:$PYTHONPATH python3 - <<'PY'
from rsl_rl.runners import HIMOnPolicyRunner
print(HIMOnPolicyRunner)
PY
```

但这只能算临时诊断。正式工程里不建议在业务代码顶部改 `sys.path`，因为它会让脚本、测试、安装包、IDE 和训练命令看到的导入顺序不一致。更稳的是修虚拟环境安装来源，让 `rsl_rl` 解析到本仓库 vendored 的 `rsl_rl/rsl_rl`。

第四，Gemini 的 fallback 方案：

```python
try:
    from rsl_rl.runners import HIMOnPolicyRunner
except ImportError:
    HIMOnPolicyRunner = OnPolicyRunner
```

不适合当前 HIM 任务作为默认行为。因为 HIM 不只是换一个 runner 名字，它还对应：

- `HIMActorCritic`；
- `HIMPPO`；
- 6 帧历史 actor observation；
- privileged critic observation；
- `HIMRslRlWrapper` 对 terminal observation、history buffer 和 action shape 的特殊处理。

如果静默 fallback 到 `OnPolicyRunner`，代码可能不在导入阶段报错，但后面会在 observation shape、policy class、algorithm class 或 rollout storage 上报更隐蔽的错误。更好的策略是：如果任务声明需要 `HIMOnPolicyRunner`，但导入不到，就直接抛出清晰错误，提示修本地 `rsl_rl` 安装。

## 14. 更准确的目标架构

如果你的目标是“做一个 `legged_gym` 风格的 `mjlab` 实现”，推荐分层应该是这样：

```text
legged_gym 风格 cfg
    ↓
HimGo2Env / MyRobotEnv 内部转换成 ManagerBasedRlEnvCfg
    ↓
mjlab ManagerBasedRlEnv 负责 scene/action/obs/reward/termination 等 manager
    ↓
RslRlVecEnvWrapper 或 HIMRslRlWrapper 把 mjlab obs_dict 适配成 rsl_rl 接口
    ↓
OnPolicyRunner 或 HIMOnPolicyRunner 训练
```

这套结构里，每一层的职责应该很窄：

- `env_cfg.py`：保留 `legged_gym` 式嵌套 class config，方便迁移原来的机器人参数、reward scale、domain randomization；
- `env.py`：负责把 `legged_gym` 式 cfg 翻译成 `mjlab` 的 `ManagerBasedRlEnvCfg`，也就是 scene、actions、observations、events、rewards、terminations、commands、curriculum；
- `wrapper.py`：负责把 mjlab 的 `(obs_dict, reward, terminated, truncated, infos)` 转成 rsl_rl runner 期望的 `(actor_obs, critic_obs, rewards, dones, infos)`；
- `task_registry.py`：只负责注册、取 cfg、创建 env、套 wrapper、创建 runner，不要把 manager 细节写进去；
- `train.py/play.py`：只通过 `task_registry` 拿任务，不直接关心 mjlab manager 和 wrapper 细节。

所以，`env 全是挂载在管理器下` 不和 `legged_gym` 风格冲突。冲突只会出现在你让 registry 直接处理 manager 配置时。当前更好的做法是：registry 保持 legged_gym 风格，manager-based 细节留在 env 内部。

## 15. registry 应该怎么基于 mjlab 但保持 legged_gym 风格

我建议不要直接使用 `mjlab.tasks.registry.register_mjlab_task` 作为本项目主入口。原因不是“自己的 `rsl_rl` 不能用”，而是当前项目已经选择了另一种封装边界：

```text
外部：legged_gym 风格 task id + EnvClass + EnvCfg + TrainCfg
内部：EnvClass 把 EnvCfg 转成 mjlab ManagerBasedRlEnvCfg
中间：Wrapper 把 mjlab 环境接口转成 rsl_rl 接口
```

如果改用 `register_mjlab_task`，通常意味着你要让外部直接暴露 mjlab 风格的 `env_cfg`、`play_env_cfg`、`rl_cfg` 和 `runner_cls`。这更接近参考项目，但会削弱你现在想保留的 `legged_gym` 迁移体验。

更合理的 registry 设计是保留自己的 `TaskRegistry`，但把注册信息补全。最少应该显式记录：

```text
task_id
env_cls
env_cfg_cls 或 env_cfg_factory
play_env_cfg_cls 或 play_env_cfg_factory
train_cfg_cls 或 train_cfg_factory
runner_cls
wrapper_cls 或 wrapper_factory
```

如果想最像原版 `legged_gym`，可以继续使用多字典：

```text
self.task_classes
self.env_cfgs
self.train_cfgs
self.runner_classes
self.wrapper_factories
```

如果想让 `train.py` 更清楚，建议用一个很薄的 `TaskSpec`。这不算背离 `legged_gym`，因为它只是把几张字典里的内容集中到一个对象里，核心仍然是显式注册、显式实例化。

推荐的注册外观可以是：

```python
task_registry.register(
    name="him_go2",
    task_class=HimGo2Env,
    env_cfg_cls=HimGo2RoughCfg,
    train_cfg_cls=HimGo2CfgPPO,
    runner_cls=HIMOnPolicyRunner,
    wrapper_cls=HIMRslRlWrapper,
)
```

或者如果想贴近参考项目命名，可以在自己 registry 上包一层薄函数：

```python
register_legged_mjlab_task(
    task_id="him_go2",
    env_cls=HimGo2Env,
    env_cfg_cls=HimGo2RoughCfg,
    play_env_cfg_cls=HimGo2RoughCfg,
    train_cfg_cls=HimGo2CfgPPO,
    runner_cls=HIMOnPolicyRunner,
    wrapper_cls=HIMRslRlWrapper,
)
```

这个接口外观看起来像参考项目的 `register_mjlab_task(...)`，但内部仍然是你自己的 `legged_gym` 风格注册器。

## 16. 当前代码最应该先统一的接口

下一步如果要真的改代码，我建议不是直接替换成 Gemini 给的整份 `task_registry.py`，而是先统一下面这些接口。

### 16.1 统一 cfg 是 class 还是 instance

原版 `legged_gym` 常见写法是注册 cfg 实例：

```python
task_registry.register("anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO())
```

当前 `envs/__init__.py` 也是实例写法：

```python
task_registry.register("him_go2", HimGo2Env, HimGo2RoughCfg(), HimGo2CfgPPO())
```

但当前 `train.py` 又像是在期待 class/factory：

```python
spec = task_registry.get(args.task)
train_cfg = spec.train_cfg_cls()
```

这两种都可以，但必须选一种。为了避免多个训练 run 共享同一个 cfg 实例，我更建议 registry 存 class/factory，`make_env()` 和 `make_alg_runner()` 每次创建新实例。如果你想完全贴近旧 `legged_gym`，也可以存实例，但 `get_cfgs()` 里最好做 `copy.deepcopy()`。

### 16.2 统一 runner 选择方式

不建议长期用字符串：

```python
runner_class_name = "HIMOnPolicyRunner"
```

更推荐注册时传类对象：

```python
runner_cls=HIMOnPolicyRunner
```

这样 runner 不需要靠字符串判断，也不会出现 cfg 外层和 `train_cfg.runner` 内层字段位置不一致的问题。

### 16.3 统一 wrapper 选择方式

HIM 任务应该明确使用 `HIMRslRlWrapper`，普通 PPO 任务使用 `RslRlVecEnvWrapper`。不要只靠 runner 名字推断，至少要允许注册时显式覆盖：

```text
HIMOnPolicyRunner + HIMRslRlWrapper
OnPolicyRunner + RslRlVecEnvWrapper
```

这样以后接 `VelocityOnPolicyRunner` 或其他 runner 时，不会被 `if runner_name == "HIMOnPolicyRunner"` 卡死。

### 16.4 统一 train.py 调用 registry 的方式

建议让 `train.py` 只保留一种调用形态：

```python
env, env_cfg = task_registry.make_env(
    name=args.task,
    args=args,
)

runner, train_cfg = task_registry.make_alg_runner(
    env=env,
    name=args.task,
    args=args,
    log_root=args.log_dir,
)
```

也就是说，命令行参数解析可以在 `train.py`，但环境创建、wrapper、runner 创建都交给 registry。这样才像 `legged_gym`：训练脚本很薄，任务细节都在 registry 和 config 里。

## 17. 最终建议

所以对 Gemini 那份建议，建议这样取舍：

- 接受：保留简单字典 registry 的方向；
- 接受：mjlab 环境创建只需要 `cfg/sim_device/render_mode/play` 这类参数，不需要 Isaac Gym 的 `sim_params/physics_engine`；
- 接受：mjlab 到 rsl_rl 中间必须有 wrapper；
- 修正：当前 `task_registry.py` 已经不是复杂动态导入版本，真正问题是 API 不一致；
- 修正：不要把 `sys.path.insert` 写进正式 registry，最多用于临时诊断；
- 修正：不要把 `HIMOnPolicyRunner` 静默 fallback 成 `OnPolicyRunner`；
- 修正：`env 全是 manager` 不代表 registry 要用 `mjlab.tasks.registry`，因为当前 env 已经内部封装了 manager cfg 构建。

最终推荐路线还是：

```text
自己的 TaskRegistry
+ legged_gym 风格 cfg
+ env 内部构建 ManagerBasedRlEnvCfg
+ wrapper 适配 mjlab obs_dict 和 rsl_rl Tensor 接口
+ 注册时显式传 runner_cls / wrapper_cls
```

这比直接照搬 `register_mjlab_task(...)` 更符合你这个仓库的目标：保留 `legged_gym` 的任务组织体验，同时使用 `mjlab` 的 manager-based 仿真执行。
