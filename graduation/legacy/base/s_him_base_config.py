# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

"""HIMLoco A1 blind locomotion defaults."""

from .base_config import BaseConfig


class SHIMBaseCfg(BaseConfig):
    """HIMLoco A1 blind locomotion defaults."""

    class env:
        num_envs = 4096
        num_one_step_observations = 45
        num_observations = num_one_step_observations * 6
        # Actor frame, base velocity, external force, and terrain scan.
        num_one_step_privileged_obs = 45 + 3 + 3 + 187
        num_privileged_obs = num_one_step_privileged_obs  # One frame for the asymmetric critic.
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20

    class terrain:
        mesh_type = "trimesh"  # Supported modes are none, plane, heightfield, and trimesh.
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = None  # select a unique terrain type and pass all arguments ("BarrierTrack", etc.)
        terrain_kwargs = None  # Dict of arguments for selected terrain
        BarrierTrack_kwargs = None  # Dict of arguments for BarrierTrack terrain
        TerrainPerlin_kwargs = None  # Dict of arguments for TerrainPerlin noise
        pad_unavailable_info = False  # If True, pad unavailable terrain info with zeros
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.2, 0.3, 0.3, 0.1]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = True
        max_curriculum = 2.0
        num_commands = 4  # Forward speed, lateral speed, yaw rate, and heading.
        resampling_time = 10.0  # Seconds between commands.
        heading_command = True  # Compute yaw rate from the heading error.

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-3.14, 3.14]  # min max [rad/s]
            heading = [-3.14, 3.14]
        # Opt in to the reset-window velocity curriculum.
        progressive_curriculum = False
        curriculum_init_x = None  # Initial forward half-range (m/s).
        curriculum_init_y = None  # Initial lateral half-range (m/s).
        curriculum_step = 0.2  # Half-range increment (m/s).
        curriculum_threshold = 0.8  # Fraction of maximum tracking reward.

    class init_state:
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # Joint targets [rad] at zero action.
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
        }

    class control:
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40.0}  # [N*m/rad]
        damping = {'joint': 1.0}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 1.0

    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        name = "a1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        disable_gravity = False
        collapse_fixed_joints = False  # Keep bodies connected by fixed joints separate.
        fix_base_link = False
        default_dof_drive_mode = 3  # 0 disables drives, 1 is position, 2 velocity, 3 effort.
        self_collisions = 1  # Legacy bitwise filter. 1 disables self-collision.
        replace_cylinder_with_capsule = True
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01
        privileged_contacts_on = ["base", "thigh", "calf"]

    class domain_rand:
        randomize_payload_mass = True
        payload_mass_range = [-1, 3]
        randomize_com_displacement = True
        com_displacement_range = [-0.1, 0.1]
        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]
        randomize_friction = True
        friction_range = [0.2, 2.75]
        randomize_restitution = True
        restitution_range = [0., 1.0]
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.3]
        randomize_kp = True
        kp_range = [0.8, 1.2]
        randomize_kd = True
        kd_range = [0.8, 1.2]
        randomize_initial_joint_pos = True
        initial_joint_pos_range = [0.5, 1.5]
        disturbance = True
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8
        push_robots = True
        push_interval_s = 16
        max_push_vel_xy = 1.
        delay = True

    class rewards:

        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            dof_acc = -2.5e-7
            joint_power = -2e-5
            base_height = -1.0
            foot_clearance = -0.01
            action_rate = -0.01
            smoothness = -0.01
            feet_air_time = 0.0
            collision = -0.0
            feet_stumble = -0.0
            stand_still = -0.
            torques = -0.0
            dof_vel = -0.0
            dof_pos_limits = -0.0
            dof_vel_limits = -0.0
            torque_limits = -0.0
        only_positive_rewards = False  # Allow negative total rewards.
        dof_pos_hip_weight = 5.0
        dof_pos_thigh_weight = 5.0
        dof_pos_calf_weight = 5.0
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.0  # Fraction of the URDF position limits before penalties apply.
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.30
        max_contact_force = 100.  # forces above this value are penalized
        clearance_height_target = -0.2

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
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class SHIMBaseCfgPPO(BaseConfig):
    """Original HIMLoco PPO defaults with S-HIM and AMP options."""

    seed = 1
    runner_class_name = 'HIMOnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        method = 'himloco'
        actor_type = 'lif'
        actor_norm = 'inherit'
        actor_readout_mode = 'spike'
        actor_surrogate = 'atan'
        actor_threshold = 1.0
        ticks = 4
        actor_widths = None  # Use default_actor_widths for S-HIM.
        default_actor_widths = [128, 64]
        default_critic_widths = [256, 128]  # S-HIM outside paper-training mode.
        population = 5
        en_pop_dim = 10
        en_mean_range = [-5.0, 5.0]
        en_std = 0.3872983346
        freeze_encoder = False
        actor_intra_feedback = False
        actor_output_norm = False
        actor_output_bias_offset = 0.0
        max_action_std = None

    class estimator:
        estimator = 'grsn'
        normalization = 'care'
        width = 64
        layers = 2
        history = 5
        swap_weight = 1.0
        swap_temperature = 0.1
        swap_reduction = 'sum'
        target_mode = 'obs'
        freeze_aux_estimator_stats = False
        balanced_calibration = False
        calibration_interval = 1
        grsn_threshold = None
        grsn_bias = None

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1e-3
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 10.
        aux_learning_rate = 0.001
        max_lr = 0.01
        separate_grad_clip = False
        log_std_lr_scale = 1.0
        actor_weight_decay = 0.0
        paper_training = False
        ppo_rollout_features = False

    class amp:
        enabled = False
        motion_set = 'a1_diagonal'
        obs_mode = "wmp30"
        reward_coef = 2.0
        task_lerp = 0.0
        discr_lr = 0.0001
        discr_updates = 20
        batch_size = 1024
        replay_size = 1000000
        preload = 200000
        discr_widths = [1024, 512]

    class runner:
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'
        num_steps_per_env = 100  # Rollout steps per update.
        max_iterations = 5000  # Policy updates.
        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = 'rough_a1'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
        checkpoint_interval = 0
        diagnostics_interval = 1


class BlindBenchCfgPPO(SHIMBaseCfgPPO):
    """Short runs selected by the original positional-method CLI."""

    class algorithm(SHIMBaseCfgPPO.algorithm):
        learning_rate = 3e-4
        schedule = "fixed"
        max_grad_norm = 1.0
        num_learning_epochs = 4

    class runner(SHIMBaseCfgPPO.runner):
        num_envs = 8
        num_steps_per_env = 16
        max_iterations = 2
