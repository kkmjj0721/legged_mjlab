# HIM Go2 recovery reward 迁移审计

日期：2026-09-01

本文件只记录当前 `him_go2` env/config/asset 迁移状态和下一步修复顺序。本文档不表示代码已写入。

审计范围：

- 当前项目：`/home/kk/legged_mjlab`
- recovery 参考：`/home/kk/github/uw-himloco-hop`
- mjlab 参考：`/home/kk/github/unitree_rl_mjlab`、`/home/kk/github/AMP_mjlab`
- mjlab 源码：`/home/kk/legged_mjlab/.venv/lib/python3.11/site-packages/mjlab`
- 当前约束：只写本文档，不改源码、测试、构建、wrapper、pycache 或其他 docs。

## 1. 当前结论

`him_go2` 还不能认为已经到可训练闭环状态。config 和 asset 已有雏形，env 也已按 mjlab manager 风格搭起来；已修好的 obs/curriculum `asset_cfg=go2` 问题不再作为 blocker 记录。

当前文档不再保留已确认修好的 P0 项。下一步以 smoke 为准，重新确认 `make_env -> reset -> step` 是否还有运行 blocker。

已删除的过期项：

- actor obs 缺 `asset_cfg=go2`：当前 `projected_gravity`、`joint_pos_rel`、`joint_vel_rel` 已传 `base_cfg/joint_cfg`。
- terrain curriculum 缺 `asset_cfg=go2`：当前 `terrain_levels_vel` 已传 `SceneEntityCfg(entity_name)`。

## 2. 不是确定错，但必须记录的风险

这些项不要再写成“当前必然构造失败”，但后续训练/部署前必须收敛：

| 项目 | 重新定性 | 处理建议 |
|---|---|---|
| `orientation = g_x^2 + g_y^2` | 对 locomotion flatness shaping 可用，但不能作为 recovery 主指标。当前实现见 `him_go2_env.py:707-727`。 | recovery 主项改用能区分正立/倒立的 `upright_linear = 0.5 * (1 - g_z)`；原 orientation 可作为辅助项。 |
| `reset_root_state_uniform` | 函数本身没错；当前参数不是贴地倒姿。配置 z offset `[0.35, 0.50]`，默认 base z `0.42`，实际约 `[0.77, 0.92] m`。 | 正常行走 reset 可继续用；recovery reset 应单独写低高度倒姿 reset 或用 AMP 式 motion frame reset。 |
| 关节/action 顺序 | 不必然错，但必须冻结 ABI。当前 dict 顺序、XML leg-major、部署 YAML/参考项目顺序不完全一致。 | 训练、wrapper、导出、部署统一一份 name-based joint order，不靠隐式 dict 顺序。 |
| `randomize_restitution` | 配置存在但 env 未接线。 | 先标为 DR 覆盖缺口；不要假设它已生效。 |

## 3. 参考项目带来的修正

`unitree_rl_mjlab`、`AMP_mjlab` 和 mjlab 1.6.0 支持上面的降级判断：

- entity name：Unitree/AMP/mjlab velocity 模板普遍把主机器人注册为 `"robot"`，所以省略 `asset_cfg` 对模板不是错；当前项目注册成 `"go2"` 后，obs/curriculum 需要显式传 `SceneEntityCfg("go2")`。
- curriculum：mjlab `terrain_levels_vel` 本身支持 `asset_cfg`，不需要重写一个 go2 版本 curriculum。
- reset：Unitree velocity reset 可用 `reset_root_state_uniform`；AMP motion reset 直接写 root pose/velocity/joint state。当前问题是 recovery 参数语义，不是 mjlab reset API。

## 4. 当前代码状态

| 模块 | 状态 |
|---|---|
| config | 45D 单帧 actor obs、6 帧历史、12D action、HIM runner/policy/algorithm 已声明；reward scales 仍偏 locomotion。 |
| asset | `go2.xml`、mesh、floating base、12 joints、foot sites、IMU sensor 已存在；termination contact sensor 仍是占位接口。 |
| observation | actor 维度意图是 `3 command + 3 ang vel + 3 gravity + 12 q + 12 qd + 12 last action = 45`；gravity/q/qd 已显式走 `go2` asset cfg。 |
| termination | 目前只有 timeout；对 recovery 是合理方向，成功不应直接终止。 |
| wrapper | 本轮不审。env native reset/step 跑通后再冻结 HIM history、critic obs、terminal obs 和旧 `rsl_rl` ABI。 |

## 5. 单策略联合训练原则

目标是同一个 policy 同时学 locomotion 和倒地恢复，不建议拆成 recovery-only env。

首版边界：

- actor 仍保持 45D，action 仍保持 12D。
- fallen/upright reset 使用同一套 velocity command 分布；debug smoke 可以临时零命令，但正式训练不要长期零命令。
- reward gate 由当前物理状态推导，不用 actor 看不到的 hidden reset mode。
- `tracking_lin_vel`、`tracking_ang_vel`、`feet_air_time`、`foot_clearance` 保留给 locomotion，但倒地阶段乘低 `w_loco`。
- `upright_linear`、`stand_height`、`joint_to_default`、`upside_down_penalty` 用于 recovery，但站稳行走时不能把步态拖成僵硬站姿。
- recovery 成功只做 reward/metric/curriculum latch，不写成 episode termination。

建议 gate：正立时 `projected_gravity_b[:, 2] ~= -1`，倒立时接近 `+1`。用 `g_z` 做连续 `w_loco/w_recovery`，避免硬切。

## 6. 下一步顺序

按这个顺序修，不要同时大改 wrapper：

1. 跑小规模 smoke：`num_envs=2/4`，验证 reset、zero-action step、random finite action step、actor obs shape `[N,45]`、critic obs shape、reward tensor shape。
2. 再做 recovery reset/reward：低高度倒姿 reset、`upright_linear`、`w_loco/w_recovery`、成功 latch 和 fallen bucket curriculum。
3. 最后再审 wrapper/runner/deploy：HIM history、critic dim、terminal obs、`rsl_rl.__file__`、policy export 和部署顺序。

## 7. 当前验证状态

| 检查 | 状态 | 备注 |
|---|---|---|
| 文档一致性 | UPDATED | 已删除 obs/curriculum `asset_cfg` 两个过期 P0。 |
| 源码修改 | NOT_DONE | 本轮只写文档。 |
| env smoke | NOT_RERUN | obs/curriculum 已修，下一步直接跑 smoke。 |
| 测试 | NOT_RUN | 文档-only 修改，不运行会写缓存/日志的测试。 |

最小完成标准：必须实际跑通 `make_env -> reset -> step`，并确认 obs/reward shape 正确；否则不要把 env 标成“已写好”。
