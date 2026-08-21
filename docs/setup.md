# 安装配置文档

## 系统要求

- **操作系统**：推荐使用 Ubuntu 22\.04

- **显卡**：Nvidia 显卡  

- **驱动版本**：建议使用 550 或更高版本  

---

## 1\.安装：

### 1\.1 下载：

- 通过 Git 克隆仓库：

  ```Bash
  git clone https://github.com/kkmjj0721/legged_mjlab.git
  ```

### 1\.2 创建虚拟环境并且安装依赖：

```Python
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ./rsl_rl 
uv pip install -e .
```

### 1\.3 运行和验证：

```Python
# 验证 mjlab 仿真环境 
uv run demo  

# 运行训练脚本 
uv run python legged_mjlab/scripts/train.py  

# 运行可视化回放 
uv run python legged_mjlab/scripts/play.py
```



