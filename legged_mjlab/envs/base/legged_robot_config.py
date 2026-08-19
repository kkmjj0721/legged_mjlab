from .base_config import BaseConfig



class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_one_step_observations = 45
        num_observations = num_one_step_observations
        num_one_step_privileged_obs = num_one_step_observations + 3 + 187
        num_privileged_obs = num_one_step_privileged_obs
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds


    class commands:
        curriculum = True
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

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
        max_push_vel_xy = 0.4
        max_push_ang_vel = 0.6

        # 随机外力和外力矩
        continuous_push = True
        max_push_force = 0.5
        max_push_torque = 0.5
        # 噪声
        push_force_noise = 0.5
        push_torque_noise = 0.5

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
            termination = -0.0
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.75
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -1.e-5
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -1.0 
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
            hip_pos = -1.0

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

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

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
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
        num_steps_per_env = 24 # per iteration
        max_iterations = 5000 # number of policy updates

        # logging
        save_interval = 1000 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
