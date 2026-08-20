# uv 使用

- UV 是一款轻量、高速的 Python 包与环境管理工具，兼容 pip、virtualenv、pyenv 等传统工具，无需额外依赖，能快速实现 Python 版本管理、虚拟环境创建、包管理及项目初始化，并且速度快

---

## 1\.安装 UV：

- 安装：

    ```Bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

- 验证：

    ```Bash
    uv --version
    ```

---

## 2\.UV 使用：

- UV 的功能分为三部分：Python 版本管理、虚拟环境管理、包管理；

### 2\.1 Python 版本管理：

- 查看所有可用 Python 版本：

    ```Bash
    uv python list
    ```

- 安装特定版本 python：

    ```Bash
    # 安装指定具体版本（如 3.11.6，避免版本兼容问题）
    uv python install 3.11.6
    
    # 安装 PyPy 版本（轻量高效，适合生产环境）
    uv python install pypy3.10
    ```

- 设置全局默认 python 版本：

    ```Bash
    uv python default 3.12
    python --version *# 输出 Python 3.12.x 即成功*
    ```

- 固定项目 python 版本：

    - 在项目根目录执行以下命令，会生成 `\.python\-version` 文件，标识项目所需 Python 版本，他人克隆项目后可快速适配：

    ```Bash
    uv python pin 3.11
    ```

### 2\.2 虚拟环境管理：

- 创建虚拟环境：

    ```Markdown
    # 方式1：创建默认名称（.venv）的虚拟环境（推荐，符合行业规范）
    uv venv
    
    # 方式2：指定 Python 版本创建虚拟环境（如 3.11）
    uv venv --python 3.11 .venv
    uv venv --python=3.11 .venv # 两种写法均可
    
    # 方式3：创建自定义名称的虚拟环境（区分不同项目）
    uv venv --python 3.12 .venv312 # 适配 Python 3.12 项目
    ```

- 激活虚拟环境：

    ```Bash
    source .venv/bin/activate
    ```

- 验证虚拟环境是否激活：

    ```Bash
    python -VV *# 查看当前环境的 Python 版本，确认与虚拟环境指定版本一致*
    ```

- 退出虚拟环境：

    ```Bash
    deactivate
    ```

### 2\.3 包管理：

- 安装包：

    ```Bash
    # 安装最新版本的包（如 requests）
    uv pip install requests
    
    # 安装指定版本的包（避免版本兼容问题）
    uv pip install requests==2.31.0
    
    # 从 requirements.txt 文件批量安装依赖（迁移项目必备）
    uv pip install -r requirements.txt
    
    # 安装包到开发环境（仅开发时使用，如测试工具 pytest）
    uv pip install --dev pytest
    ```

- 升级或卸载包：

    ```Bash
    *# 升级指定包到最新版本*
    uv pip upgrade requests
    
    *# 升级所有已安装的包（谨慎使用，避免版本冲突）*
    uv pip upgrade --all
    
    *# 卸载指定包（彻底删除，无残留）*
    uv pip uninstall requests
    ```

- 导出依赖：

    - 将当前环境的依赖包导出为 `requirements.txt` 文件

    ```Bash
    *# 导出当前环境所有依赖（包括开发依赖）*
    uv pip freeze > requirements.txt
    
    *# 导出生产环境依赖（排除开发依赖，上线必备）*
    uv pip freeze --production > requirements.txt
    ```

---

## 3\.其他功能：

- 清理 UV 缓存：

    ```Bash
    uv cache clean
    ```

- 卸载 UV：

    ```Bash
    # 1. 清理缓存和相关文件
    uv cache clean
    rm -r "$(uv python dir)"
    rm -r "$(uv tool dir)"
    
    # 2. 删除二进制文件（macOS/Linux）
    rm ~/.local/bin/uv ~/.local/bin/uvx
    
    # 2. 删除二进制文件（Windows）
    rm $HOME.local\bin\uv.exe
    rm $HOME.local\bin\uvx.exe
    ```

- 项目初始化：

    ```Bash
    uv init my_project *# 创建名为 my_project 的项目*
    cd my_project *# 进入项目目录*
    ```


