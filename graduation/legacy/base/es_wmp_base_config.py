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

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

"""WMP A1 depth and locomotion defaults."""

from graduation.paths import WMP_A1_MOTIONS

from .base_config import BaseConfig


# Use the same motion library regardless of the working directory.
MOTION_FILES = [str(path) for path in sorted(WMP_A1_MOTIONS.glob("*.txt"))]


class ESWMPBaseCfg(BaseConfig):
    """WMP A1 depth and locomotion defaults."""

    class env:
        num_envs = 4096
        privileged_obs = True  # if True, add the privileged information in the obs
        privileged_dim = 24 + 26 + 3  # Includes the three base linear velocity components.
        height_dim = 187  # privileged_obs[:,-height_dim:] is the heightmap in privileged_obs
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20
        reference_state_initialization = False
        include_history_steps = None  # Number of steps of history to include.
        prop_dim = 33  # proprioception
        action_dim = 12
        num_observations = prop_dim + privileged_dim + height_dim + action_dim
        num_privileged_obs = num_observations
        forward_height_dim = 525  # for depth image prediction
        reference_state_initialization_prob = 0.85
        amp_motion_files = MOTION_FILES

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
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                             0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        # 525 dim, for depth image prediction
        measured_forward_points_x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                     1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                                     2.0]  # 1mx1.6m rectangle (without center line)
        measured_forward_points_y = [-1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.,
                                     0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 0  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [wave, rough slope, stairs up, stairs down, discrete, gap, pit, tilt, crawl, rough_flat]
        terrain_proportions = [0.0, 0.05, 0.15, 0.15, 0.0, 0.25, 0.25, 0.05, 0.05, 0.05]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False
        max_lin_vel_forward_x_curriculum = 1.0
        max_lin_vel_backward_x_curriculum = 0.0
        max_lin_vel_y_curriculum = 0.0
        max_ang_vel_yaw_curriculum = 1.0
        max_flat_lin_vel_forward_x_curriculum = 1.0
        max_flat_lin_vel_backward_x_curriculum = 0.0
        max_flat_lin_vel_y_curriculum = 0.0
        max_flat_ang_vel_yaw_curriculum = 1.0
        num_commands = 4  # Forward speed, lateral speed, yaw rate, and heading.
        resampling_time = 10.0  # Seconds between commands.
        heading_command = True  # Compute yaw rate from the heading error.

        class ranges:
            lin_vel_x = [0.0, 0.8]  # min max [m/s]
            lin_vel_y = [-0., 0.]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            heading = [-0., 0.]
            flat_lin_vel_x = [-0.0, 0.8]  # min max [m/s]
            flat_lin_vel_y = [-0.0, 0.0]  # min max [m/s]
            flat_ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            flat_heading = [-3.14 / 4, 3.14 / 4]

    class init_state:
        pos = [0.0, 0.0, 0.35]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # Joint targets [rad] at zero action.
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.0,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.0,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
        }

    class control:
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40.}  # [N*m/rad]
        damping = {'joint': 1.0}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class depth:
        use_camera = True
        camera_num_envs = 1024
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20
        position = [0.27, 0, 0.03]  # front camera
        y_angle = [-5, 5]  # positive pitch down
        z_angle = [0, 0]
        x_angle = [0, 0]
        update_interval = 5  # Control steps per depth frame.
        original = (64, 64)
        resized = (64, 64)
        horizontal_fov = 58
        buffer_len = 2
        near_clip = 0
        far_clip = 2
        dis_noise = 0.0
        scale = 1
        invert = True

    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        disable_gravity = False
        collapse_fixed_joints = True  # Merge bodies connected by fixed joints.
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

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 2.0]
        randomize_restitution = True
        restitution_range = [0.0, 0.0]
        randomize_base_mass = True
        added_mass_range = [0., 3.]  # kg
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
        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]
        damping_multiplier_range = [0.8, 1.2]
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        randomize_action_latency = True
        latency_range = [0.00, 0.005]

    class rewards:
        reward_curriculum = True
        reward_curriculum_term = ["feet_edge"]
        reward_curriculum_schedule = [[4000, 10000, 0.1, 1.0]]

        class scales:
            termination = -0.0
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.5
            lin_vel_z = -1.0
            ang_vel_xy = 0
            orientation = -0.
            torques = -0.0001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.
            feet_air_time = 0.5
            collision = -1.0
            feet_stumble = -0.1
            action_rate = -0.03
            stand_still = -0.
            feet_edge = -1.0
            dof_error = -0.04
            cheat = -1
            stuck = -1
        only_positive_rewards = True  # Clamp negative total rewards to zero.
        tracking_sigma = 0.15  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.25
        max_contact_force = 100.  # forces above this value are penalized
        foot_height_target = 0.15
        lin_vel_clip = 0.1

    class normalization:

        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            # privileged
            height_measurements = 5.0
            contact_force = 0.005
            com_pos = 20
            pd_gains = 5
        clip_observations = 100.
        clip_actions = 6.0
        base_height = 0.3  # Nominal base height used to normalize terrain measurements.

    class noise:
        add_noise = False
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0  # set lin_vel as privileged information
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0  # only for critic

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class ESWMPBaseCfgPPO(BaseConfig):
    """Event student architecture, distillation, and rollout defaults."""

    seed = 1

    class student:
        channels = [16, 32, 64]
        memory_width = 128
        memory = "grsn"
        normalization = "care"
        action_path = "end-to-end"
        base_channels = 64
        student_normalization = "batch"
        student_threshold = 1.0
        student_residual = "if"

    class distillation:
        lr = 1e-3
        world_weight = 1.0
        cosine_weight = 0.1
        warmup_iterations = 1
        calibration_interval = 1
        threshold = 0.08  # Relative brightness change for event synthesis.

    class runner:
        num_envs = 8
        iterations = 2
        rollout_steps = 20
        episode_seconds = ESWMPBaseCfg.env.episode_length_s
