from .base_config import BaseConfig


class LeggedMjlabCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_one_step_observations = 45
        history_length = 1
        num_observations = num_one_step_observations * history_length
        num_privileged_obs = num_one_step_observations + 3 + 187        # num_obs + vel + height
        num_actions = 12
        env_spacing = 2.0
        episode_length_s = 20.0
        seed = 42

    class terrain:
        mesh_type = "plane"
        curriculum = False

    class commands:
        resampling_time_range = (5.0, 15.0)
        rel_standing_envs = 0.05
        heading_command = True
        heading_control_stiffness = 0.5

        class ranges:
            lin_vel_x = (-1.0, 1.0)
            lin_vel_y = (-1.0, 1.0)
            ang_vel_z = (-1.0, 1.0)
            heading = (-3.14, 3.14)

    class init_state:
        pos = (0.0, 0.0, 1.0)
        default_joint_angles = {}

        reset_pose_range = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        reset_velocity_range = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

    class control:
        control_type = "P"
        stiffness = {}
        damping = {}
        action_scale = 0.25
        decimation = 4
        hip_reduction = 0.5

    class asset:
        xml = ""
        name = "robot"
        body_names = ()
        armature = 0.0
        effort_limit = None

    class domain_rand:
        # ---------------------------------- 动力学参数随机化 ---------------------------------- #
        # 基座负载质量
        randomize_payload_mass = True
        payload_mass_range = [-2.5, 2.5]

        # 连杆质量
        randomize_link_mass = True
        link_mass_range = [0.9, 1.1]

        # 质心偏移
        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]

        # 关节摩擦
        randomize_joint_friction = True
        joint_friction_range = [0.01, 1.15]
        
        # 关节阻尼
        randomize_joint_damping = True
        joint_damping_range = [0.3, 1.5]

        # 关节等效转动惯量
        randomize_joint_armature = True
        joint_armature_range = [0.0001, 0.05]

        # ---------------------------------- 接触与外力随机化 ---------------------------------- #
        # 地面摩擦力
        randomize_friction = True
        friction_range = [0.2, 1.3]
        
        # 恢复系数
        randomize_restitution = True
        restitution_range = [0., 0.4]

        # ---------------------------------- 控制器与执行器随机化 ------------------------------- #
        # 比例增益
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.8, 1.2]  
        damping_multiplier_range = [0.8, 1.2] 

        # 电机零位误差
        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035]

        # 电机输出强度
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]

        # ---------------------------------- 观测延迟随机化 ------------------------------------ #
        # 电机观测延迟
        randomize_obs_motor_latency = False
        range_obs_motor_latency = [1, 4]

        # IMU 观测延迟
        randomize_obs_imu_latency = False
        range_obs_imu_latency = [1, 3]

        # ---------------------------------- 动作指令延迟随机化 --------------------------------- #
        # 动作指令延迟
        randomize_cmd_action_latency = False
        range_cmd_action_latency = [1, 4]

    class rewards:
        class scales:
            termination = 0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            torques = -1.0e-4
            action_rate = -0.05

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

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
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
        obs_normalization = True

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.005
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 1500
        save_interval = 50

        experiment_name = "legged_mjlab"
        run_name = ""
        resume = False