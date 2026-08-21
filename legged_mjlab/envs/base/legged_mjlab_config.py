from __future__ import annotations

from .base_config import BaseConfig


class LeggedMjlabCfg(BaseConfig):
    """环境与任务的基础配置类
    """
    class env:
        num_envs = 4096
        num_observations = 45
        num_privileged_obs = num_observations + 3 + 187        # num_obs + vel + height
        num_actions = 12
        env_spacing = 2.0
        episode_length_s = 20.0
        seed = 42

    class init_state:
        pos = (0.0, 0.0, 1.0)
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {}

    class control:
        control_type = "P"
        stiffness = {}
        damping = {}
        action_scale = 0.25
        decimation = 4

    class asset:
        xml = ""
        name = "robot"
        armature = 0.0
        effort_limit = None

    class domain_rand:
        enabled = False
        randomize_friction = False
        friction_range = (0.5, 1.25)
        push_robots = False
        randomize_armature = False
        armature_range = (0.8, 1.2)

    class rewards:
        class scales:
            termination = 0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            torques = -1.0e-4
            action_rate = -0.05

    class terminations:
        time_out = True
        bad_orientation = True
        bad_orientation_limit_deg = 70.0

    class play:
        episode_length_s = 1.0e9
        disable_observation_corruption = True
        disable_push = True
        disable_curriculum = True


class LeggedMjlabCfgPPO(BaseConfig):
    seed = 42

    class policy:
        hidden_dims = (256, 128, 128)
        activation = "elu"
        obs_normalization = True

    class algorithm:
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        entropy_coef = 0.005
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        num_steps_per_env = 24
        max_iterations = 1500
        save_interval = 50
        experiment_name = "legged_mjlab"
        run_name = ""
