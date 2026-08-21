# 安装配置

## 支持矩阵与前置条件

当前项目元数据锁定的目标是 Ubuntu 22.04、x86_64 和 Python 3.11。`cu128` 与
`cpu` 是互斥的 accelerator extra，安装时只能选择其中一个。

GPU 运行涉及三个不同层次，请分别确认：

- **Python 依赖层**：由 uv 根据 `pyproject.toml` 安装 `mjlab`、
  `mujoco-warp`、`warp-lang`、PyTorch 等 Python 包；
- **用户态 CUDA wheel 层**：`cu128` extra 从 PyTorch 的 CUDA 12.8 wheel
  索引选择 PyTorch wheel，安装在项目 Python 环境中；它不会安装或升级系统
  NVIDIA driver；
- **系统 driver 层**：NVIDIA kernel/user driver 和 GPU 由操作系统或机器维护者
  提供。请用 `nvidia-smi` 确认 driver 正常工作，并确认它与 CUDA wheel 兼容。

`mujoco-warp` 和 GPU 训练还需要可用的 NVIDIA GPU/driver。当前审查环境中的
`nvidia-smi` 无法连接 NVIDIA driver，因此这里只能把 CPU、导入和打包检查作为
通过依据，不能宣称 GPU smoke test 已通过。CPU extra 只选择 CPU PyTorch wheel，
不等于把 GPU-oriented 的上游运行时变成 CPU 支持。

## uv 与锁文件

本项目安装流程以已经安装的 uv 为前提：

```bash
uv --version
```

如果本机尚未安装 uv，请使用操作系统官方包管理器或 uv 官方发布包；执行前应
审阅包来源、版本和官方安装说明。本项目文档不通过未经审阅的远程脚本管道安装 uv。

当前检出中**没有 `uv.lock`**。联网维护者必须先在目标平台和 Python 版本上执行
`uv lock`，审阅生成的解析结果，并将 `uv.lock` 提交到仓库；本分支不生成或伪造
该文件。锁文件一致性检查在该文件提交前保持阻塞。

## 安装

从仓库根目录选择一个 accelerator 流程执行；当前分支不预先生成或伪造 `uv.lock`。

### NVIDIA CUDA 12.8（`cu128`）

```bash
uv sync --python 3.11 --extra cu128
```

### CPU（`cpu`）

```bash
uv sync --python 3.11 --extra cpu
```

上述同步命令会把 `pyproject.toml` 中对应的可选依赖组纳入项目环境；选择的 extra
必须和 accelerator 一致。它只影响 Python 依赖解析
和安装，不负责安装系统 NVIDIA driver。不要在同一环境中混用 `cu128` 和 `cpu`。

## 验证安装与资源

以下 `uv run` 命令都明确写出与安装相同的 extra。`uv run --extra <name>` 会在
本次命令的项目环境中包含指定可选依赖组；它不是 CUDA driver 检查，也不会替代
前面的项目同步命令。

### `cu128` 环境

```bash
uv run --extra cu128 python -c "import importlib.metadata as m; import legged_mjlab, mjlab, torch; print('legged_mjlab', m.version('legged_mjlab')); print('mjlab', m.version('mjlab')); print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"
uv run --extra cu128 python -c "from importlib.resources import files; root=files('resources'); checks=[root.joinpath('robots/unitree_go2/xmls/go2.xml'), root.joinpath('robots/unitree_go2/xmls/scene_go2.xml'), root.joinpath('robots/unitree_go2/xmls/assets/foot.obj')]; missing=[str(path) for path in checks if not path.is_file()]; assert not missing, missing; print('resource_root=', root)"
```

### `cpu` 环境

```bash
uv run --extra cpu python -c "import importlib.metadata as m; import legged_mjlab, mjlab, torch; print('legged_mjlab', m.version('legged_mjlab')); print('mjlab', m.version('mjlab')); print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"
uv run --extra cpu python -c "from importlib.resources import files; root=files('resources'); checks=[root.joinpath('robots/unitree_go2/xmls/go2.xml'), root.joinpath('robots/unitree_go2/xmls/scene_go2.xml'), root.joinpath('robots/unitree_go2/xmls/assets/foot.obj')]; missing=[str(path) for path in checks if not path.is_file()]; assert not missing, missing; print('resource_root=', root)"
```

资源命令只是当前仓库发行包的 smoke test，不把顶层 `resources` namespace 宣传为
通用业务 API。它必须使用目标环境中 `importlib.resources.files('resources')`
返回的实际结果进行检查和打印；不要根据某个机器上的路径臆造“已安装路径”。由于
当前包布局仍使用顶层 namespace，发行 wheel 后必须再次运行该 smoke test。

锁文件提交后可单独检查其与项目元数据是否一致：

```bash
uv lock --check
```

本仓库没有自己的 `demo`、训练或回放 console script。上游 `mjlab` 的命令只有在
对应任务和依赖确实安装后才可运行，并且必须使用与同步相同的形式：
`uv run --extra cu128 ...` 或 `uv run --extra cpu ...`。

## setup.py 兼容入口

依赖和包发现只维护在 `pyproject.toml`。需要兼容旧工具时，可在已选 accelerator
环境中执行：

```bash
uv run --extra cu128 python setup.py --version
```

CPU 环境对应使用 `uv run --extra cpu python setup.py --version`。`uv build`
属于独立的 wheel 构建检查，不会替代上述依赖同步；本分支不启动构建网络任务。

不要从仓库内的 `rsl_rl/setup.py` 安装旧版 `rsl_rl==1.0.2`。根项目使用
`mjlab==1.6.0` 带来的 `rsl-rl-lib==5.4.2`，本地旧包不纳入根 wheel，以避免同名
`rsl_rl` 导入冲突。
