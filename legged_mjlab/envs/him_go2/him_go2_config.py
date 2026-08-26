"""HIM-Go2 task configuration."""

from legged_mjlab.envs.base.legged_mjlab_config import LeggedMjlabCfg, LeggedMjlabCfgPPO


class HimGo2RoughCfg(LeggedMjlabCfg):

    class env( LeggedMjlabCfg.env ):
            num_envs = 4096
            num_one_step_observations = 45
            history_length = 6
            num_observations = num_one_step_observations * history_length
            num_privileged_obs = num_one_step_observations + 3 + 187   # if not None, critic uses privileged_obs
            num_actions = 12
            env_spacing = 3.0  # [m] spacing between sub-environments
            send_timeouts = True  # send time out information to algorithm
            episode_length_s = 20.0  # [s] duration of one episode
            seed = 42

    class terrain( LeggedMjlabCfg.terrain ):
            mesh_type = "generator"  # "plane" | "heightfield" | "trimesh" | "generator"
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
            measure_heights = True
            measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 17
            measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5] # 11
            selected = False  # select a unique terrain type and get rid of curriculum
            terrain_kwargs = None  # dict of arguments passed to the specific terrain generator
            max_init_terrain_level = 5  # starting curriculum level

    class commands( LeggedMjlabCfg.commands ):
            curriculum = True
            max_curriculum = 1.0
            num_commands = 4  # [lin_vel_x, lin_vel_y, ang_vel_yaw, heading]
            resampling_time = 10.0  # [s] time before new command is given
            heading_command = False  # if true: compute ang vel command from heading error
            class ranges:
                lin_vel_x = [-1.0, 1.0]  # min max [m/s]
                lin_vel_y = [-1.0, 1.0]  # min max [m/s]
                ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
                heading = [-3.14, 3.14]

    class init_state( LeggedMjlabCfg.init_state ):
        pos = (0.0, 0.0, 0.42)
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control(LeggedMjlabCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 40.0, 'thigh': 40.0, 'calf': 40.0}  # [N*m/rad]
        damping = {'hip': 1.0, 'thigh': 1.0, 'calf': 1.0}     # [N*m*s/rad]
        effort_limit = {'hip': 23.5, 'thigh': 23.5, 'calf': 45}
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 1.0
        

    class asset(LeggedMjlabCfg.asset):
        file = '{LEGGED_MJLAB_ROOT_DIR}/resources/robots/unitree_go2/xmls/go2.xml'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = []
        privileged_contacts_on = ["base", "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class domain_rand( LeggedMjlabCfg.domain_rand ):
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
            # 电机观测延迟
            randomize_obs_motor_latency = True
            range_obs_motor_latency = [1, 4]

            # IMU 观测延迟
            randomize_obs_imu_latency = True
            range_obs_imu_latency = [1, 3]
    
            # ---------------------------------- 动作指令延迟随机化 --------------------------------- #
            # 动作指令延迟
            randomize_cmd_action_latency = True
            range_cmd_action_latency = [1, 4]
            action_hold_prob = 0.3


    class rewards( LeggedMjlabCfg.rewards ):
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -1.5
            ang_vel_xy = -0.05
            orientation = -0.2
            dof_acc = -2.5e-7
            joint_power = -2e-5
            base_height = -1.0
            foot_clearance = -0.01
            action_rate = -0.01
            smoothness = -0.01
            feet_air_time =  0.1
            collision = -0.5
            feet_stumble = -0.0
            stand_still = -1.0
            torques = -0.0
            dof_vel = -0.0
            dof_pos_limits = -10.0
            dof_vel_limits = -0.0
            torque_limits = -1.0
            hip_pos = -1.0

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.85 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.30
        max_contact_force = 100. # forces above this value are penalized
        clearance_height_target = -0.2


class HimGo2CfgPPO(LeggedMjlabCfgPPO):
    seed = 42
    runner_class_name = "HIMOnPolicyRunner"
    
    class policy(LeggedMjlabCfgPPO.policy):
        policy_class_name = "HIMActorCritic"

    class algorithm(LeggedMjlabCfgPPO.algorithm):
        algorithm_class_name = "HIMPPO"
        entropy_coef = 0.01

    class runner(LeggedMjlabCfgPPO.runner):
        policy_class_name = "HIMActorCritic"
        algorithm_class_name = "HIMPPO"
        num_steps_per_env = 100
        max_iterations = 10000
        save_interval = 500
        experiment_name = "him_go2"
