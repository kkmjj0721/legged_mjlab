from legged_mjlab.envs.base.legged_mjlab_config import LeggedMjlabCfg, LeggedMjlabCfgPPO


class HimGo2RounghCfg( LeggedMjlabCfg ):
    class env(LeggedMjlabCfg.env):
        num_envs = 4096
        num_one_step_observations = 45
        history_length = 6
        num_observations = num_one_step_observations * history_length
        num_privileged_obs = num_one_step_observations + 3 + 187        # num_obs + vel + height
        num_actions = 12
        env_spacing = 2.0
        episode_length_s = 20.0
        seed = 42

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
        

class Go2VelocityCfgPPO(LeggedMjlabCfgPPO):
    class algorithm( LeggedMjlabCfgPPO.algorithm ):
        entropy_coef = 0.01

    class runner( LeggedMjlabCfgPPO.runner ):
        max_iterations = 5000
        save_interval = 200
        experiment_name = 'him_go2'
        resume = False
