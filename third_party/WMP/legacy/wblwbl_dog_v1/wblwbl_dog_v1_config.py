"""WBLWBL Dog V1 WMP configuration for mixed terrain training."""

from pathlib import Path

from wmp_lab.a1_config import A1AMPCfg, A1AMPCfgPPO

ROOT = Path(__file__).resolve().parents[2]
WBLWBL_MOTION_FILES = sorted(
    str(p) for p in (ROOT / "datasets/mocap_motions_wblwbl").glob("*.txt")
)


class WblwblDogV1AMPCfg(A1AMPCfg):
    class env(A1AMPCfg.env):
        num_envs = 2048
        prop_dim = 33
        action_dim = 12
        privileged_dim = 24 + 26 + 3
        height_dim = 187
        forward_height_dim = 525
        num_observations = prop_dim + privileged_dim + height_dim + action_dim
        num_privileged_obs = num_observations
        reference_state_initialization = False
        amp_motion_files = WBLWBL_MOTION_FILES

    class terrain(A1AMPCfg.terrain):
        # [wave, slope, stairup, stairdown, discrete, flat_gap, pit, tilt, crawl, roughflat, plum_stones, tire_array, high_wall]
        flat_gap = True
        terrain_proportions = [0.0, 0.1, 0.15, 0.1, 0.0, 0.20, 0.05, 0.00, 0.00, 0.05, 0.0, 0.20, 0.15]

    class init_state(A1AMPCfg.init_state):
        # Default stance sole_z is ~ -0.391..-0.399 m relative to base_link; spawn at 0.44 m.
        pos = [0.0, 0.0, 0.44]
        default_joint_angles = {
            "RR_hip_joint": -0.1,
            "RR_thigh_joint": 1.0,
            "RR_calf_joint": -1.5,
            "RL_hip_joint": 0.1,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.5,
            "FR_hip_joint": -0.1,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,
            "FL_hip_joint": 0.1,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
        }

    class control(A1AMPCfg.control):
        control_type = "P"
        stiffness = {"joint": 40.0}
        damping = {"joint": 1.0}
        action_scale = 0.25
        decimation = 4

    class depth(A1AMPCfg.depth):
        use_camera = True
        camera_num_envs = 512
        # Front camera lens face in base_link.STL is at x=0.2846, y=0.0, z=0.060.
        position = [0.285, 0.0, 0.060]
        y_angle = [20.0, 20.0]  # positive pitch down (degrees)
        z_angle = [0.0, 0.0]
        x_angle = [0.0, 0.0]

        update_interval = 5
        original = (64, 64)
        resized = (64, 64)
        horizontal_fov = 58
        buffer_len = 2

        near_clip = 0
        far_clip = 2
        dis_noise = 0.0
        scale = 1
        invert = True

    class asset(A1AMPCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/wblwbl_dog_v1/urdf/wblwbl_dog.urdf"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 1
        armature = 0.01

    class domain_rand(A1AMPCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 2.0]
        randomize_base_mass = True
        added_mass_range = [0.0, 3.0]
        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]
        randomize_com_pos = True
        com_x_pos_range = [-0.05, 0.05]
        com_y_pos_range = [-0.05, 0.05]
        com_z_pos_range = [-0.05, 0.05]
        push_robots = True
        push_interval_s = 15
        min_push_interval_s = 15
        max_push_vel_xy = 1.0

    class normalization(A1AMPCfg.normalization):
        base_height = 0.40  # standing base_link height of wblwbl_dog_v1

    class noise(A1AMPCfg.noise):
        add_noise = False

    class rewards(A1AMPCfg.rewards):
        reward_curriculum = True
        reward_curriculum_term = ["feet_edge"]
        reward_curriculum_schedule = [[4000, 10000, 0.1, 1.0]]

        soft_dof_pos_limit = 0.9
        base_height_target = 0.39
        foot_height_target = 0.15
        tracking_sigma = 0.15
        lin_vel_clip = 0.1

        class scales(A1AMPCfg.rewards.scales):
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.5
            orientation = -0.5
            ang_vel_xy = -0.05
            torques = -0.0001
            dof_acc = -2.5e-7
            base_height = -0.0
            feet_air_time = 0.5
            collision = -1.0
            feet_stumble = -0.1
            action_rate = -0.03
            feet_edge = -1.0
            dof_error = -0.04
            lin_vel_z = -1.0
            cheat = -1.0
            stuck = -1.0

    class commands(A1AMPCfg.commands):
        class ranges(A1AMPCfg.commands.ranges):
            lin_vel_x = [0.0, 0.8]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [0.0, 0.0]
            flat_lin_vel_x = [0.0, 0.8]
            flat_lin_vel_y = [0.0, 0.0]
            flat_ang_vel_yaw = [-1.0, 1.0]
            flat_heading = [-3.14 / 4, 3.14 / 4]


class WblwblDogV1AMPCfgPPO(A1AMPCfgPPO):
    class policy(A1AMPCfgPPO.policy):
        init_noise_std = 1.0
        encoder_hidden_dims = [256, 128]
        wm_encoder_hidden_dims = [64, 64]
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [512, 256, 128]
        latent_dim = 32 + 3
        wm_latent_dim = 32
        activation = "elu"

    class algorithm(A1AMPCfgPPO.algorithm):
        entropy_coef = 0.01
        vel_predict_coef = 1.0
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner(A1AMPCfgPPO.runner):
        run_name = "mixed"
        experiment_name = "wblwbl_dog_v1_amp"
        max_iterations = 20000
        save_interval = 1000
        amp_reward_coef = 0.5 * 0.02
        amp_motion_files = WBLWBL_MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.3
        amp_discr_hidden_dims = [1024, 512]
        min_normalized_std = [0.05, 0.02, 0.05] * 4

    class depth_predictor(A1AMPCfgPPO.depth_predictor):
        lr = 3e-4
        weight_decay = 1e-4
        training_interval = 10
        training_iters = 1000
        batch_size = 1024
        loss_scale = 100
