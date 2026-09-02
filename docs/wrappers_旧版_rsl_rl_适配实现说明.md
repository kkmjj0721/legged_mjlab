# `wrappers` 旧版 `rsl_rl` 适配待办

本文只列仍需处理项。

## 1. 仍需完成

| 优先级 | 剩余项 | 验收标准 |
| --- | --- | --- |
| P0 | 清理 worktree 里的无关状态 | `git status --short` 不再显示无关 `.pyc` 删除/修改，且不再显示 `legged_mjlab/utils/exporter.py`、`legged_mjlab/utils/terrain.py` 删除。 |
| P0 | 确认本阶段未改 `legged_mjlab/utils` | `git diff --exit-code -- legged_mjlab/utils` 和 `git diff --cached --exit-code -- legged_mjlab/utils` 通过。 |
| P0 | 跑已有 terminal critic 单测 | `.venv/bin/python -m pytest legged_mjlab/test/test_him_wrapper_terminal_privileged.py -q` 通过。 |
| P0 | 新增 `RslRlVecEnvWrapper` fake-env 单测 | 覆盖 reset 两元组、getter 缓存、step exactly 5 values、finite / non-finite timeout。 |
| P0 | 新增 `HIMRslRlWrapper` timeout/action fake-env 单测 | 覆盖 finite-horizon、`timeout_bootstrap`、NaN/Inf、显式 `action_clip`、`max_action_rate`、terminal critic 缺失 fail-closed。 |
| P1 | 最小 `him_go2` reset/step smoke | 小 `num_envs` 下 reset/step shape 符合 `[N,270]`、`[N,235]`、reward/done `[N]`、HIM 七元组。 |

## 2. 建议新增测试文件

如果后续允许写测试，新增：

```text
legged_mjlab/test/test_rsl_rl_wrapper_timeout.py
legged_mjlab/test/test_him_wrapper_timeout_and_actions.py
```

`test_rsl_rl_wrapper_timeout.py` 覆盖：

| 场景 | 期望 |
| --- | --- |
| `RslRlVecEnvWrapper.reset()` | 返回 `(actor, privileged)`，getter 读取同一批缓存。 |
| `RslRlVecEnvWrapper.step()` | 返回 exactly 5 values：`actor, privileged, rewards, dones, infos`。 |
| `is_finite_horizon=True` 且 `truncated=True` | `infos["time_outs"]` 全 False。 |
| `is_finite_horizon=False` 且 `truncated=True`、`terminated=False` | 对应行 `infos["time_outs"] = True`。 |
| `terminated=True` 且 `truncated=True` | `infos["time_outs"] = False`。 |

`test_him_wrapper_timeout_and_actions.py` 覆盖：

| 场景 | 期望 |
| --- | --- |
| finite-horizon timeout | `time_outs` 和 `timeout_bootstrap` 全 False。 |
| non-finite timeout 且 terminal critic 可用 | 对应 done 行 `timeout_bootstrap=True`。 |
| terminal failure | `terminated=True` 行不 bootstrap。 |
| action 含 NaN/Inf | 抛 `FloatingPointError`，native env 不收到动作。 |
| 显式 `action_clip=1.0` 且 action 超界 | native env 收到的 `accepted_actions` 不超过 clip。 |
| 启用 `max_action_rate` | 相邻 accepted action 差值不超过 rate。 |
| done 但无 terminal critic 且不能 manual reset | 抛 `RuntimeError`。 |
| compact `termination_privileged` | `[K, P]` 行顺序必须等于 `termination_ids`；不能证明顺序时改用 full `[N, P]`。 |

## 3. 验证命令

```bash
# 无关删除状态应先清掉
git status --short

# 不允许误碰 utils
git diff --exit-code -- legged_mjlab/utils
git diff --cached --exit-code -- legged_mjlab/utils

# 已有单测；当前 `.venv` 缺 `pytest` 时先补测试环境
.venv/bin/python -m pytest legged_mjlab/test/test_him_wrapper_terminal_privileged.py -q

# 新增单测后再跑
.venv/bin/python -m pytest \
  legged_mjlab/test/test_rsl_rl_wrapper_timeout.py \
  legged_mjlab/test/test_him_wrapper_timeout_and_actions.py \
  -q

```

## 4. 保留边界

| 边界 | 说明 |
| --- | --- |
| 不改 `utils` | task 创建、source gate、显式 wrapper 参数传递都留到后续阶段。 |
| 不改 `rsl_rl` | runner / PPO / storage / module 当前不在本文范围。 |
| 不改训练入口 | train / play / export 不在本文范围。 |
| 不声明训练完成 | 只有单测、backend smoke、reset/step smoke 通过后，才能进入短训练验证。 |
| 不声明实机安全 | wrapper action guard 不是硬件安全层；实机仍需独立控制器 safety gate。 |
