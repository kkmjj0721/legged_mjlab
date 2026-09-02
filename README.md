# legged_mjlab

面向 MuJoCo/mjlab 的腿式机器人强化学习框架设计。目标是保留 `legged_gym` 的“配置优先、
最小父类、一个目录一个训练任务、task id”体验，同时使用 mjlab 的 Entity、Scene、manager、
MuJoCo Warp 和独立 RSL-RL 算法包。

## 文档导航：
- [前言](docs/前言.md)：项目目标、参考边界和阅读顺序。
- [uv 使用](docs/uv使用.md)：uv 项目、extras、lock 和日常工作流。
- [安装与启动](docs/setup.md)：uv、CPU/GPU、Conda、打包状态和分层验证。
- [训练实时可视化说明](docs/训练可视化说明.md)：说明 `him_go2` 如何在训练启动时选择 viewer、当前缺口和参考项目方案。

## 快速使用：


## TODO：

* [ ] 进一步完善训练框架代码编写（2026\.9\.18）

* [ ] 加入 HIMLoco\_Amp（2026\.9\.30）

* [ ] 足臂任务训练搭建（待定）



## 参考项目：

- [Unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)
- [mjlab](https://github.com/mujocolab/mjlab)
- [himloco](https://github.com/InternRobotics/HIMLoco)
