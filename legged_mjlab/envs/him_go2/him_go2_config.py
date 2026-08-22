from legged_mjlab.envs.base.legged_mjlab_config import LeggedMjlabCfg, LeggedMjlabCfgPPO


class HimGo2RoughCfg( LeggedMjlabCfg ):
    class env(LeggedMjlabCfg.env):
        num_envs = 4096
        num_one_step_observations = 45
        history_length = 6
        num_observations = num_one_step_observations * history_length
        num_privileged_obs = num_one_step_observations + 3 + 187        # num_obs + vel + height
        num_actions = 12
        env_spacing = 2.0
        episode_length_s = 20.0
        send_timeouts = True
        seed = 42

    class terrain:
        mesh_type = "plane"
        curriculum = False

    class init_state(LeggedMjlabCfg.init_state):
        pos = [0.0, 0.0, 0.45]
        default_joint_angles = { 
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
        stiffness = {"hip": 40.0, "thigh": 40.0, "calf": 40.0}
        damping = {"hip": 1.0, "thigh": 1.0, "calf": 1.0}
        action_scale = 0.25
        decimation = 4
        hip_reduction = 0.5

    class asset(LeggedMjlabCfg.asset):
        xml = '{LEGGED_MJLAB_ROOT_DIR}/resources/robots/unitree_go2/xmls/go2.xml'
        name = "go2"
        
class HimGo2CfgPPO(LeggedMjlabCfgPPO):
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
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'
        runner_class_name = "HIMOnPolicyRunner"
        num_steps_per_env = 100
        max_iterations = 10000
        save_interval = 500

        experiment_name = "him_go2"
        run_name = ""
        resume = False
        load_run = "-1"
        checkpoint = -1
