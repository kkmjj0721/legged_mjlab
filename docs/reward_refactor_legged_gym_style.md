# HIM-Go2 奖励重构：legged_gym 风格的 config-first 注册

目标是把 rewards.scales 作为唯一注册来源：每个非零 scale 自动绑定
HimGo2Env._reward_<name>(env)，并继续使用 mjlab RewardManager 执行计算。
这只重构 reward builder 和 reward callable，不重写 compute_reward，也不绕开 manager。

代码观察：HimGo2RoughCfg 的 rewards.scales 当前有 17 个非零项。BaseConfig 会把
嵌套 class 变为属性对象，所以 cfg.rewards.scales 不是 Mapping，不能调用 .get()。
当前 builder 仍有手工 _add_reward 路径、漏传参数和 helper 命名不统一的问题。

未执行/未验证：本文只提供可复制的迁移代码，没有运行环境、reset、step 或训练。
TaskRegistry/HimGo2Env 的构造 ABI、Go2Asset._parse_cfg 的 self.robot_cfg/self.asset
引用，以及 _build_scene 调用 _build_sensors 时漏传 entity_name，仍是 reward runtime
之前的外部 blocker。
## 为什么适合改成 config-first

legged_gym 的 _prepare_reward_function 核心是：遍历 scale，跳过零项，再按名称寻找
_reward_<name>。HIM-Go2 可以复用这个组织方式，但最终仍返回 RewardTermCfg：
~~~text
cfg.rewards.scales
  -> 过滤非数值和零权重
  -> 找到 HimGo2Env._reward_<name>
  -> RewardTermCfg(func=..., weight=...)
  -> mjlab RewardManager.compute(step_dt)
~~~
这样做的收益是：

- 新增、关闭或调权重只改 config，不需要再维护一份手工注册清单。
- 非零项缺 helper 会在创建 manager 前报错，不会静默漏掉奖励。
- manager 继续负责 [N] shape 检查、nan_to_num、episode_sums、日志和 dt 缩放。
- callable 从 env 读取 robot、command、scene，不把 task 语义散落在 params 中。

当前 scale_rewards_by_dt=False 只能验证注册结构和当前行为不变，不是 legged_gym
数值等价。当前 step_dt=0.02 时，raw legacy scale 加 False 会比 legged_gym 的
dt-scaled reward 大约 1 / 0.02 = 50 倍；raw scale 加 True 才是接近其数值语义的路径。
因此先用 False 隔离注册改动，随后在独立实验中切到 True；不能同时手工乘 dt 和打开
manager dt 缩放。
## 代码块 A：自动注册 helper

把下面代码放进 HimGo2Env 类，替换现有 scale 读取和手工注册 helper。
_iter_reward_scales 是唯一的 scale 读取和验证路径；文件顶部如没有
对应 import，再添加 Mapping 和 Real 的 import。
~~~python
from collections.abc import Mapping
from numbers import Real
import math

class HimGo2Env(ManagerBasedRlEnv):
    @staticmethod
    def _iter_reward_scales(scales):
        """Yield public numeric scale pairs from a dict or BaseConfig object."""
        if isinstance(scales, Mapping):
            items = scales.items()
        else:
            # dir() retains inherited public settings on nested BaseConfig objects.
            items = (
                (name, value)
                for name in dir(scales)
                if not name.startswith("_")
                for value in (getattr(scales, name),)
                if not callable(value)
            )
        for name, value in items:
            if isinstance(value, bool) or not isinstance(value, Real):
                continue
            weight = float(value)
            if not math.isfinite(weight):
                raise ValueError(f"Reward scale '{name}' must be finite, got {weight!r}.")
            yield name, weight
    def _build_rewards(self) -> dict[str, RewardTermCfg]:
        """Config-first registration, not numerical legged_gym equivalence."""
        terms: dict[str, RewardTermCfg] = {}
        missing: list[str] = []
        for name, weight in self._iter_reward_scales(self.robot_cfg.rewards.scales):
            if weight == 0.0:
                continue
            helper_name = f"_reward_{name}"
            helper = getattr(type(self), helper_name, None)
            if helper is None or not callable(helper):
                missing.append(helper_name)
                continue
            # Task parameters are read inside helper(env); params remains {}.
            terms[name] = RewardTermCfg(func=helper, weight=weight)
        if missing:
            raise AttributeError(
                "Nonzero reward scales require helpers: "
                + ", ".join(sorted(missing))
            )
        return terms
~~~
这里故意不保留第二条取值路径或通用 _add_reward。无论 scales 是 Mapping 还是
BaseConfig，_iter_reward_scales 都产出同一份已验证的 (name, weight)；自动注册后，
漏项、拼写错误都会在启动期显式失败。正常情况下 type(self) 会解析到
HimGo2Env._reward_<name>；它也允许任务子类覆盖某一项。
## 代码块 B：统一 reward 函数模板

下面片段替换 class 内现有 reward helper。每个 active term 都是
@staticmethod _reward_<name>(env) 并返回精确的 [N]。现有模块已经 import 了
torch、Entity 和 ManagerBasedRlEnv。
~~~python
class HimGo2Env(ManagerBasedRlEnv):
    @staticmethod
    def _asset(env: ManagerBasedRlEnv) -> Entity:
        return env.scene[env.robot_cfg.asset.name]
    @staticmethod
    def _twist(env: ManagerBasedRlEnv) -> torch.Tensor:
        command = env.command_manager.get_command("twist")
        if command is None:
            raise KeyError("Reward requires command term 'twist'.")
        return command
    @staticmethod
    def _moving(env: ManagerBasedRlEnv) -> torch.Tensor:
        command = HimGo2Env._twist(env)
        threshold = float(getattr(env.robot_cfg.rewards, "command_threshold", 0.1))
        return (
            torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        ) > threshold
    @staticmethod
    def _reward_tracking_lin_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
        asset = HimGo2Env._asset(env)
        command = HimGo2Env._twist(env)
        sigma = float(env.robot_cfg.rewards.tracking_sigma)
        if sigma <= 0.0:
            raise ValueError("rewards.tracking_sigma must be positive.")
        xy_error = torch.sum(
            torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]),
            dim=1,
        )
        return torch.exp(
            -(xy_error + 2.0 * torch.square(asset.data.root_link_lin_vel_b[:, 2]))
            / sigma
        )
    @staticmethod
    def _reward_tracking_ang_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
        asset = HimGo2Env._asset(env)
        command = HimGo2Env._twist(env)
        sigma = float(env.robot_cfg.rewards.tracking_sigma)
        if sigma <= 0.0:
            raise ValueError("rewards.tracking_sigma must be positive.")
        yaw_error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
        xy_error = torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)
        return torch.exp(-(yaw_error + 0.05 * xy_error) / sigma)
    @staticmethod
    def _reward_lin_vel_z(env: ManagerBasedRlEnv) -> torch.Tensor:
        return torch.square(HimGo2Env._asset(env).data.root_link_lin_vel_b[:, 2])
    @staticmethod
    def _reward_ang_vel_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
        value = HimGo2Env._asset(env).data.root_link_ang_vel_b[:, :2]
        return torch.sum(torch.square(value), dim=1)
    @staticmethod
    def _reward_orientation(env: ManagerBasedRlEnv) -> torch.Tensor:
        gravity = HimGo2Env._asset(env).data.projected_gravity_b[:, :2]
        return torch.sum(torch.square(gravity), dim=1)
    @staticmethod
    def _reward_dof_acc(env: ManagerBasedRlEnv) -> torch.Tensor:
        # Conservative: mjlab 1.6.0 EntityData.joint_acc is [N, J].
        return torch.sum(torch.square(HimGo2Env._asset(env).data.joint_acc), dim=1)
    @staticmethod
    def _reward_joint_power(env: ManagerBasedRlEnv) -> torch.Tensor:
        asset = HimGo2Env._asset(env)
        return torch.sum(
            torch.abs(asset.data.qfrc_actuator * asset.data.joint_vel), dim=1
        )
    @staticmethod
    def _reward_base_height(env: ManagerBasedRlEnv) -> torch.Tensor:
        asset = HimGo2Env._asset(env)
        height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
        return torch.square(height - float(env.robot_cfg.rewards.base_height_target))
    @staticmethod
    def _reward_foot_clearance(env: ManagerBasedRlEnv) -> torch.Tensor:
        asset = HimGo2Env._asset(env)
        site_ids, _ = asset.find_sites(("FL", "FR", "RL", "RR"), preserve_order=True)
        foot_z = asset.data.site_pose_w[:, site_ids, 2]
        foot_vel_xy = asset.data.site_vel_w[:, site_ids, :2]
        target = float(env.robot_cfg.rewards.clearance_height_target)
        cost = torch.sum(
            torch.abs(foot_z - target) * torch.norm(foot_vel_xy, dim=-1), dim=1
        )
        return cost * HimGo2Env._moving(env).to(dtype=cost.dtype)
    @staticmethod
    def _reward_action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
        actions = env.action_manager
        return torch.sum(torch.square(actions.action - actions.prev_action), dim=1)
    @staticmethod
    def _reward_smoothness(env: ManagerBasedRlEnv) -> torch.Tensor:
        actions = env.action_manager
        second_order = actions.action - 2.0 * actions.prev_action + actions.prev_prev_action
        return torch.sum(torch.square(second_order), dim=1)
    @staticmethod
    def _reward_feet_air_time(env: ManagerBasedRlEnv) -> torch.Tensor:
        data = env.scene["feet_ground_contact"].data
        if data.current_air_time is None or data.current_contact_time is None:
            raise RuntimeError("feet_ground_contact requires track_air_time=True.")
        target = float(getattr(env.robot_cfg.rewards, "feet_air_time_target", 0.4))
        in_contact = data.current_contact_time > 0.0
        in_mode = torch.where(in_contact, data.current_contact_time, data.current_air_time)
        single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
        mode_time = torch.min(
            torch.where(single_stance.unsqueeze(-1), in_mode, torch.zeros_like(in_mode)),
            dim=1,
        ).values
        reward = torch.clamp(target - torch.abs(mode_time - target), min=0.0)
        return reward * HimGo2Env._moving(env).to(dtype=reward.dtype)
    @staticmethod
    def _reward_collision(env: ManagerBasedRlEnv) -> torch.Tensor:
        data = env.scene["nonfoot_ground_touch"].data
        forces = data.force_history if data.force_history is not None else data.force
        if forces is None:
            raise RuntimeError("nonfoot_ground_touch requires contact force data.")
        if not bool(torch.isfinite(forces).all()):
            raise RuntimeError("nonfoot_ground_touch contains NaN/Inf contact force.")
        if forces.ndim == 4 and forces.shape[-1] == 3:
            # Verified contract must be [N, C, H, 3]; H is history, not contacts.
            per_contact = torch.norm(forces, dim=-1).amax(dim=-1)
        elif forces.ndim == 3 and forces.shape[-1] == 3:
            per_contact = torch.norm(forces, dim=-1)
        else:
            raise RuntimeError(f"Unexpected contact-force shape: {tuple(forces.shape)}")
        # If force_history layout is not verified, keep collision scale=0.0.
        # Never flatten/reshape C and H together and count history slots as contacts.
        threshold = float(env.robot_cfg.rewards.max_contact_force)
        return (per_contact > threshold).sum(dim=1).to(dtype=forces.dtype)
    @staticmethod
    def _reward_stand_still(env: ManagerBasedRlEnv) -> torch.Tensor:
        asset = HimGo2Env._asset(env)
        error = torch.sum(
            torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1
        )
        return error * (~HimGo2Env._moving(env)).to(dtype=error.dtype)
    @staticmethod
    def _reward_dof_pos_limits(env: ManagerBasedRlEnv) -> torch.Tensor:
        asset = HimGo2Env._asset(env)
        lower = asset.data.soft_joint_pos_limits[..., 0]
        upper = asset.data.soft_joint_pos_limits[..., 1]
        violation = torch.relu(lower - asset.data.joint_pos)
        violation += torch.relu(asset.data.joint_pos - upper)
        return torch.sum(violation, dim=1)
    @staticmethod
    def _reward_torque_limits(env: ManagerBasedRlEnv) -> torch.Tensor:
        asset = HimGo2Env._asset(env)
        force = asset.data.actuator_force
        if not bool(torch.isfinite(force).all()):
            raise RuntimeError("actuator_force contains NaN/Inf.")
        actuators = asset.actuators
        if not actuators:
            raise RuntimeError("torque_limits requires at least one configured actuator.")
        limits = torch.empty_like(force)
        covered = torch.zeros(force.shape[1], dtype=torch.bool, device=force.device)
        for actuator in asset.actuators:
            force_limit = getattr(actuator, "force_limit", None)
            if force_limit is None:
                raise RuntimeError("Every actuator must expose force_limit.")
            if not bool(torch.isfinite(force_limit).all()):
                raise RuntimeError("actuator.force_limit contains NaN/Inf.")
            limits[:, actuator.ctrl_ids] = force_limit
            covered[actuator.ctrl_ids] = True
        if not bool(covered.all()) or not bool(torch.isfinite(limits).all()):
            raise RuntimeError("Actuator force limits do not fully and finitely cover force.")
        soft_limit = max(float(env.robot_cfg.rewards.soft_torque_limit), 1.0e-6)
        excess = torch.relu(torch.abs(force) / torch.clamp(limits * soft_limit, min=1.0e-6) - 1.0)
        return torch.sum(torch.square(excess), dim=1)
    @staticmethod
    def _reward_hip_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
        asset = HimGo2Env._asset(env)
        hip_names = tuple(
            name for name in env.robot_cfg.init_state.default_joint_angles
            if "_hip_" in name
        )
        hip_ids, _ = asset.find_joints(hip_names, preserve_order=True)
        if not hip_ids:
            raise RuntimeError("No hip joints resolved.")
        difference = asset.data.joint_pos[:, hip_ids] - asset.data.default_joint_pos[:, hip_ids]
        return torch.sum(torch.square(difference), dim=1)
    # PENDING while their scales remain 0.0:
    # _reward_feet_stumble needs a verified horizontal/vertical force frame.
    # _reward_dof_vel_limits needs verified per-joint velocity limits.
    # torques and dof_vel are zero-weight and intentionally have no helper.
    # A nonzero scale must fail fast here until its helper is implemented.
~~~
语义风险需要保留在评审记录里：

- dof_acc 使用已存在的 joint_acc，属于保守实现；它不保证等价于 legacy 的速度差分。
- foot_clearance 使用 site world z。当前 target=-0.2 是否应是 world、origin 或地形相对
  坐标尚未验证，先观察分布再决定是否改 frame。
- feet_air_time 和 collision 依赖实际 sensor 名称、fields 和 scene wiring。
- feet_stumble、dof_vel_limits、torques、dof_vel 当前均为零权重且有意未实现 helper；
  scale 改为非零时应先补数据契约，或让代码块 A fail-fast，而不是猜公式。
### 安全边界

action_rate、smoothness、collision 和 torque reward 只是策略优化的软 shaping，不是
动作限幅、扭矩限幅、急停、安全联锁或 termination。本文代码只能作为仿真候选；
不得直接上实机，也不得在未完成 fault/termination 验证前启动大规模训练。
## 代码块 C：可选 only_positive_rewards

mjlab RewardManager 默认不读取 env.only_positive_rewards。当前把 config 值保存到
HimGo2Env 属性，不会自动裁剪 reward。若需要 total reward 的 max(total, 0)，可选用
下面的薄子类；它保留 manager 的 shape、nan_to_num、episode_sums 和 dt 行为。
~~~python
from mjlab.managers import RewardManager

class OnlyPositiveRewardManager(RewardManager):
    def compute(self, dt: float) -> torch.Tensor:
        reward = super().compute(dt)
        return reward.clamp_min_(0.0)

class HimGo2Env(ManagerBasedRlEnv):
    def load_managers(self) -> None:
        super().load_managers()
        if self.robot_cfg.rewards.only_positive_rewards:
            self.reward_manager = OnlyPositiveRewardManager(
                self.cfg.rewards,
                self,
                scale_by_dt=self.cfg.scale_rewards_by_dt,
            )

class PositiveRewardStepWrapper:
    """Use this only when policy-facing clipping is enough."""
    def __init__(self, env):
        self.env = env
    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        return obs, rewards.clamp_min(0.0), terminated, truncated, infos
    def __getattr__(self, name):
        return getattr(self.env, name)
~~~
子类方案会让 native env 返回裁剪后的 reward；但 super().compute 已把各 term 的
episode_sums 按裁剪前值累计，故 Episode_Reward 日志和 policy-facing total 不完全一致。
wrapper 同样只裁剪返回值，reward_buf/manager 日志未裁剪，也有相同日志偏差。两者都
不是包含 termination reward 时的逐字节 legged_gym 复现：mjlab 是 termination ->
所有 reward term -> auto-reset，而常见 legged_gym 是普通项求和 -> only_positive
clamp -> 再追加 termination reward。未来若加入 termination penalty，需要单独定义
两阶段汇总顺序，不能把它偷偷塞进 helper。
## 代码块 D：静态和运行 smoke 验证

先运行静态部分，它不会构造 MuJoCo。运行部分必须在一次性小规模环境中执行；
RewardManager.compute 会累计 episode sums，smoke 后应丢弃该 env。
~~~python
import torch
import math
from legged_mjlab.envs.him_go2.him_go2_config import HimGo2RoughCfg
from legged_mjlab.envs.him_go2.him_go2_env import HimGo2Env

def active_scale_names(cfg):
    scales = list(HimGo2Env._iter_reward_scales(cfg.rewards.scales))
    assert all(math.isfinite(weight) for _, weight in scales)
    return {
        name for name, weight in scales if weight != 0.0
    }

# Static: validates scale iteration, automatic naming and no params.
cfg = HimGo2RoughCfg()
assert dict(HimGo2Env._iter_reward_scales({"mapping_probe": 1.0})) == {
    "mapping_probe": 1.0
}
probe = object.__new__(HimGo2Env)
probe.robot_cfg = cfg
terms = HimGo2Env._build_rewards(probe)
expected = active_scale_names(cfg)
assert len(expected) == 17, expected
assert set(terms) == expected
for name, term_cfg in terms.items():
    assert term_cfg.func is getattr(HimGo2Env, f"_reward_{name}")
    assert term_cfg.params == {}

def assert_high_risk_inputs(env):
    contact = env.scene["nonfoot_ground_touch"].data
    forces = contact.force_history if contact.force_history is not None else contact.force
    asset = env.scene[env.robot_cfg.asset.name]
    limits = [getattr(actuator, "force_limit", None) for actuator in asset.actuators]
    assert forces is not None and bool(torch.isfinite(forces).all())
    assert bool(torch.isfinite(asset.data.actuator_force).all())
    assert limits and all(
        limit is not None and bool(torch.isfinite(limit).all()) for limit in limits
    )

def smoke_reward_terms(env):
    expected = active_scale_names(env.robot_cfg)
    assert set(env.reward_manager.active_terms) == expected
    manual = torch.zeros(env.num_envs, device=env.device)
    dt_scale = env.step_dt if env.cfg.scale_rewards_by_dt else 1.0
    for name in env.reward_manager.active_terms:
        term_cfg = env.reward_manager.get_term_cfg(name)
        value = term_cfg.func(env, **term_cfg.params)
        assert value.shape == (env.num_envs,), (name, value.shape)
        assert torch.isfinite(value).all(), name
        manual += torch.nan_to_num(value * term_cfg.weight * dt_scale)
    manager_value = env.reward_manager.compute(dt=env.step_dt)
    if type(env.reward_manager).__name__ == "OnlyPositiveRewardManager":
        manual.clamp_min_(0.0)
    torch.testing.assert_close(manager_value, manual)
    assert manager_value.shape == (env.num_envs,)
    assert torch.isfinite(manager_value).all()
    print(
        {
            "active_terms": env.reward_manager.active_terms,
            "scale_rewards_by_dt": env.cfg.scale_rewards_by_dt,
            "step_dt": env.step_dt,
        }
    )

# Runtime prerequisite: constructor, asset and sensor blockers must be fixed first.
runtime_cfg = HimGo2RoughCfg()
runtime_cfg.env.num_envs = 2
env = None
try:
    env = HimGo2Env(runtime_cfg, sim_device="cpu", render_mode=None)
    env.reset()
    actions = torch.zeros((env.num_envs, runtime_cfg.env.num_actions), device=env.device)
    _, rewards, terminated, truncated, infos = env.step(actions)
    assert rewards.shape == (env.num_envs,)
    assert terminated.shape == (env.num_envs,)
    assert truncated.shape == (env.num_envs,)
    assert_high_risk_inputs(env)
    smoke_reward_terms(env)
finally:
    if env is not None:
        env.close()
# Create a fresh env and rerun after deliberately switching scale_rewards_by_dt.
# Never flip the flag after the manager has already been constructed.
~~~
验证顺序是：先比 active_terms 与非零 scale 的集合，再查每项 [N] 和 finite，
最后比较 manager 输出与手工加权值。分别用 False 和 True 的全新环境跑一次，记录
step_dt 与 reward 量级；不要在 helper 或 builder 中额外乘 dt。
本 smoke 只覆盖健康输入；NaN/Inf fault injection、termination 标志和终止顺序必须由
独立测试覆盖，不能用此脚本替代。
## 迁移与回滚

建议顺序：

1. 先解除构造、asset、sensor blocker，确认 scene 中的 go2、twist、
   feet_ground_contact、nonfoot_ground_touch 可解析。
2. 合入代码块 A 和 B，先保持 scale_rewards_by_dt=False，只做注册结构/当前行为回归。
3. 跑静态 gate；缺 helper 必须修复或将其 scale 明确设为 0.0。
4. 用两个环境跑 runtime gate，再观察 foot_clearance、contact 和 torque 的分布。
5. 明确选择是否启用代码块 C；不要让 config 写 True 而运行时实际上未裁剪。
6. 独立实验中再切换 dt 语义，最后才启动长训练。

回滚必须恢复同一已验证提交中的完整已知安全组合：rewards.scales、builder/helper、
scale_rewards_by_dt、only-positive manager 或 wrapper、termination/sensor 接线和对应
smoke 基线。若某个公式尚不可信，优先将该项 scale 设为 0.0 并标为 PENDING；不要
静默跳过非零项。若 return 突变，恢复这套完整组合后再排查重复 dt 缩放。

未执行/未验证：本文创建时没有运行上述验证脚本。环境初始化前置 blocker 未解决前，
不能把静态设计或代码片段描述为运行通过。
