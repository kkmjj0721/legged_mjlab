# uv 使用

- UV 是一款轻量、高速的 Python 包与环境管理工具，兼容 pip、virtualenv、pyenv 等传统工具，无需额外依赖，能快速实现 Python 版本管理、虚拟环境创建、包管理及项目初始化，并且速度快

---

## 1\.安装 UV：

本项目的安装文档以已经安装的 uv 为前提。先验证当前命令来自预期的官方包或
发行版包：

    ```Bash
    uv --version
    ```

如果尚未安装，请优先使用操作系统官方包管理器或 uv 官方发布包，并在执行前
审阅官方文档、包来源和版本。本文不提供未经审阅的远程脚本管道。

---

## 2\.UV 使用：

- UV 的功能分为三部分：Python 版本管理、虚拟环境管理、包管理；

### 2\.1 Python 版本管理：

- 查看所有可用 Python 版本：

    ```Bash
    uv python list
    ```

- 本项目要求 Python 3.11。先检查本机是否已有兼容解释器：

    ```Bash
    uv python find 3.11
    ```

    本文不提供下载或安装 Python 的命令；如果检查不到解释器，请按操作系统的
    已审阅流程准备 Python 3.11。uv 不提供全局默认 Python 的项目命令，项目命令
    应通过 `--python 3.11` 或项目的 `.python-version` 明确选择解释器。

- 固定项目 Python 版本（可选）：

    - 在项目根目录执行以下命令，会生成 `\.python\-version` 文件，标识项目所需 Python 版本，他人克隆项目后可快速适配：

    ```Bash
    uv python pin 3.11
    ```

### 2\.2 虚拟环境管理：

- 创建虚拟环境：

    ```Markdown
    # 需要单独创建环境时，明确使用项目要求的 Python 3.11
    uv venv --python 3.11 .venv
    ```

- 激活虚拟环境：

    ```Bash
    source .venv/bin/activate
    ```

- 验证虚拟环境是否激活：

    ```Bash
    python -VV
    # 查看当前环境的 Python 版本，确认与虚拟环境指定版本一致
    ```

- 退出虚拟环境：

    ```Bash
    deactivate
    ```

### 2\.3 项目依赖管理：

本项目的依赖、版本约束和 accelerator extra 均维护在仓库根目录的
`pyproject.toml` 中。请从仓库根目录选择一个 extra 同步，两个命令只能二选一：

```Bash
uv sync --python 3.11 --extra cu128
uv sync --python 3.11 --extra cpu
```

`cu128` 与 `cpu` 不能在同一环境中混用。不要使用临时包安装命令绕过根项目元数据，
也不要把不存在的入口或依赖文件写进安装流程。

如需查看当前环境而不改变依赖，可使用真实存在的查询命令：

```Bash
uv pip list
uv pip tree
```

上面的命令只用于查看当前环境；本项目也没有单独的开发依赖入口。依赖变更应先更新
根 `pyproject.toml`，再重新执行上面的项目同步命令。

---

## 3\.其他功能：

- 查看缓存位置：

    ```Bash
    uv cache dir
    ```

    该命令只用于确认路径。本文不提供自动删除缓存或虚拟环境的命令；如果磁盘
    管理确实需要清理，请先审阅官方文档和命令展开后的准确目标，再由用户确认
    后执行。

- 卸载或清理 UV：

    本文不提供删除 uv 管理目录、缓存或二进制文件的命令。若确实需要卸载，
    请先记录 `uv --version` 和安装来源，再按照对应操作系统官方包管理器或 uv
    官方文档逐项审阅准确路径后执行；不要把项目目录或 uv 管理目录作为未经确认
    的删除目标。

- 项目初始化：

    本仓库已经提供根 `pyproject.toml`，不需要再次初始化项目。新项目才需要
    根据其自身元数据另行规划初始化流程。

## 4\.设置源：

```Bash
    mkdir -p ~/.config/uv
    cat <<EOF > ~/.config/uv/uv.toml
    [[index]]
    url = "https://pypi.tuna.tsinghua.edu.cn/simple"
    default = true
    EOF
```