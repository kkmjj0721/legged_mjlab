import math

from .base_config import BaseConfig


class LeggedMjlabCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_one_step_observations = 45
        history_length = 1
        num_observations = num_one_step_observations * history_length
        num_privileged_obs = None  # if not None, critic uses privileged_obs
        num_actions = 12
        env_spacing = 3.0  # [m] spacing between sub-environments
        extent = 2.0
        send_timeouts = True  # send time out information to algorithm
        episode_length_s = 20.0  # [s] duration of one episode
        seed = 42

    class terrain:
        mesh_type = "plane"  # "plane" | "heightfield" | "trimesh" | "generator"
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25.0  # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        # rough terrain options
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this are deemed unwalkable
        # height measurement
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 17
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5] # 11
        selected = False  # select a unique terrain type and get rid of curriculum
        terrain_kwargs = None  # dict of arguments passed to the specific terrain generator
        max_init_terrain_level = 5  # starting curriculum level

    class commands:
        curriculum = False
        max_curriculum = 1.0
        num_commands = 4  # [lin_vel_x, lin_vel_y, ang_vel_yaw, heading]
        resampling_time = [4.0, 8.0]  # [s] time before new command is given
        heading_command = False  # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        # mjlab quaternions use (w, x, y, z), not the xyzw convention used by some
        # deployment runtimes.
        pos = (0.0, 0.0, 1.0)
        rot = (1.0, 0.0, 0.0, 0.0)
        lin_vel = (0.0, 0.0, 0.0)
        ang_vel = (0.0, 0.0, 0.0)
        default_joint_angles = {}

    class control:
        control_type = "P"  # "P" | "V" | "T"
        stiffness = {}  # [N*m/rad]
        damping = {}  # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4  # control frequency = sim frequency / decimation
        # optional actuator dynamics
        armature = 0.0
        hip_reduction = 0.5

    class asset:
        file = ""
        name = "legged_robot"
        foot_name = "foot"  # name of the foot body/link/geom
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        fix_base_link = False  # fix robot base to world
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0: none, 1: pos, 2: vel, 3: effort)
        self_collisions = 0  # 1 to disable, 0 to enable
        replace_cylinder_with_capsule = True
        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

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

        # 随机推机器人
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 1.0

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
        add_obs_latency = False

        # 电机观测延迟
        randomize_obs_motor_latency = False
        range_obs_motor_latency = [1, 4]

        # IMU 观测延迟
        randomize_obs_imu_latency = False
        range_obs_imu_latency = [1, 3]

        # ---------------------------------- 动作指令延迟随机化 --------------------------------- #
        add_cmd_action_latency = False

        # 动作指令延迟
        randomize_cmd_action_latency = False
        range_cmd_action_latency = [1, 4]

    class rewards:
        class scales:
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            base_height = 0.0
            torques = -0.0001
            dof_vel = -0.0
            dof_acc = -2.5e-7
            action_rate = -0.01
            feet_air_time = 1.0
            collision = -1.0
            termination = -0.0
            dof_pos_limits = -10.0
            hip_reduction = -0.0

        only_positive_rewards = False  # if true: negative total rewards clipped to zero
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9  # percentage of limits where penalty starts
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.35
        max_contact_force = 100.0  # forces above this trigger collision penalties

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.0
        clip_actions = 100.0

    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class viewer:
        ref_env = 0
        pos = [10, 0, 6] 
        lookat = [11., 5, 3.]  

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]
        up_axis = 1  # 0 is y, 1 is z


class LeggedMjlabCfgPPO(BaseConfig):
    seed = 42

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
        obs_normalization = False

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.005
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 3.0e-4
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        class_name = "OnPolicyRunner"
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 1500
        save_interval = 50
        experiment_name = "legged_mjlab"
        run_name = ""
        resume = False
        load_run = "-1"
        checkpoint = -1
