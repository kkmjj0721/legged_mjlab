# HIM Go2 Env 剩余修改清单

日期：2026-09-01

范围：只看 `legged_mjlab/envs/`，先不管 runner / wrapper / deploy。

## 1. Gemini 结论核验

| Gemini 说法 | 结论 | 证据 | 文档动作 |
|---|---|---|---|
| `_reward_dof_pos_limits()` 被未闭合 docstring 吞掉，返回 `None`。 | 错 | `him_go2_env.py:892-904` 现在有闭合 docstring 和 `return`；`ast.parse` 通过。 | 删除 |
| 两个 `mdp` import 覆盖后会丢 `reset_root_state_uniform` 等基础函数。 | 错 | `him_go2_env.py:11-12` 确实覆盖；但 `.venv/.../mjlab/tasks/velocity/mdp/__init__.py:1` re-export 了 `mjlab.envs.mdp.*`。 | 删除 |
| `_reward_collision()` 把 batch 归约成标量。 | 错 | `ContactData.force_history` 是 `[B,N,H,3]`；`him_go2_env.py:853-856` 等价于 mjlab `self_collision_cost` 的 `[B,N,H] -> [B,H] -> [B]`。 | 删除 |
| `_reward_hip_pos()` 惩罚 12 个关节。 | 错 | `him_go2_env.py:860-871` 已写 `hip_ids = [0, 3, 6, 9]`，不是默认全关节。 | 删除 |
| `_reward_foot_clearance()` 目标高度是负数且默认选所有 site。 | 部分对 | `him_go2_config.py:210` 目标已是 `0.20`，不是负数；但 `him_go2_env.py:788-789` 默认 `site_ids=slice(None)`，Go2 XML 有 `imu` 和四个脚 site。 | 写入待修 |
| `_reward_base_height()` 在 rough terrain 上只减 env origin，会和台阶/坡面冲突。 | 部分对 | `him_go2_env.py:771-779` 用 `root_z - env_origins_z`；这不是运行崩溃，但 rough terrain 上不是脚下真实地表高度。 | 写入待修 |
| roll/pitch reset 全范围且没有 fall termination，会污染轨迹。 | 部分对 | `him_go2_config.py:176-177` 是 `[-3.14, 3.14]`，`him_go2_env.py:621-626` 只有 timeout；但起身任务需要倒地姿态，不应简单加 `bad_orientation` 终止。 | 改成待修：拆 reset / 加 gate |
| viewer `body_name="base"` 应改成 `base_link`。 | 对 | `him_go2_env.py:84-88` 用 `base`，Go2 XML 主体是 `base_link`。 | 写入待修 |
| actor obs 45、critic obs 235 是对的。 | 对 | `him_go2_env.py:526-599`：actor 为 `3+3+3+12+12+12=45`；critic 再加 `3+187=235`，配置在 `him_go2_config.py:10-13`。 | 删除 |
| `go2_asset.py` actuator 分 hip/thigh/calf、delay/limit 设计合理。 | 对 | `go2_asset.py:127-164` 三组 `IdealPdActuatorCfg` 分别接 stiffness/damping/effort/delay。 | 删除 |

## 2. 现在还要改什么

| 哪里错了 | 改哪里 | 怎么改 |
|---|---|---|
| `_reward_orientation()` 默认 `asset_cfg.body_ids=slice(None)`，`if asset_cfg.body_ids:` 会走 all bodies 分支，返回形状可能不是 `[num_envs]`。 | `legged_mjlab/envs/him_go2/him_go2_env.py:727-747` | 默认直接用 `asset.data.projected_gravity_b`；只有显式传了单个 body id/name 时才读 `body_link_quat_w`。 |
| `_reward_foot_clearance()` 默认选了全部 site，会把 `imu` 混进脚高奖励。 | `legged_mjlab/envs/him_go2/him_go2_env.py:781-802` | 在函数内固定四个脚 site：`FL/FR/RL/RR`；更稳的是用 foot height sensor 算相对地形高度。 |
| `_reward_base_height()` 用 env origin 近似地面高度，rough terrain 上不准。 | `legged_mjlab/envs/him_go2/him_go2_env.py:771-779`、`him_go2_config.py:189` | rough locomotion 先把 `base_height` scale 置 `0`，或改成 height scan / terrain sensor 下的相对地表高度；recovery 再单独做站高 reward。 |
| reset 只有一条路径：所有 env 都可能全 roll/pitch 倒地，但没有 upright/fallen 区分。 | `legged_mjlab/envs/him_go2/him_go2_env.py:271-307`、`him_go2_config.py:172-177` | 保留 full roll/pitch 给 fallen reset；新增 upright reset 小姿态扰动，并按概率混合；fallen reset 再加关节扰动。 |
| 倒地状态还在吃完整 locomotion reward / collision penalty。 | `legged_mjlab/envs/him_go2/him_go2_env.py:685-858` | 加 `w_loco / w_recovery` gate；倒地早期降低 tracking、feet_air_time、foot_clearance、collision 权重，起身后恢复 locomotion。 |
| 起身成功状态没有记录。 | `legged_mjlab/envs/him_go2/him_go2_env.py:617-659` 附近新增 buffer / metric | 定义 `recovered = upright & height_ok & joint_ok & stable_ok`；只做 reward/metric latch，不要直接 done。 |
| viewer 跟踪 body 名错。 | `legged_mjlab/envs/him_go2/him_go2_env.py:84-88` | 把 `body_name="base"` 改成 `body_name="base_link"`。 |

## 3. 需要 smoke 确认

- 构造 env 后跑 native `reset -> step`，确认 actor `[N,45]`、critic `[N,235]`、每个 reward term `[N]` 且 finite。
- `foot_clearance` 的 `0.20m` 目标不是静态 bug，但是否过高要通过 reward 分布 / smoke 再看。
