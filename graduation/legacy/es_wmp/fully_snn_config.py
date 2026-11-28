"""A1 task settings for the independent FullySNN AMP benchmark."""

from .es_wmp_config import ESWMPA1FlatCfg


class FullySNNA1AmpCfg(ESWMPA1FlatCfg):
    class env(ESWMPA1FlatCfg.env):
        num_observations = num_privileged_obs = 48
        privileged_dim = 3
        height_dim = 0

    class terrain(ESWMPA1FlatCfg.terrain):
        measure_heights = False

    class depth(ESWMPA1FlatCfg.depth):
        camera_num_envs = 1

    class init_state(ESWMPA1FlatCfg.init_state):
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = {
            f"{leg}_{joint}_joint": angle
            for leg in ("FL", "FR", "RL", "RR")
            for joint, angle in (("hip", 0.0), ("thigh", 0.9), ("calf", -1.8))
        }

    class control(ESWMPA1FlatCfg.control):
        stiffness = {"joint": 80.0}
        decimation = 6

    class asset(ESWMPA1FlatCfg.asset):
        self_collisions = 0
        terminate_after_contacts_on = ["base", "calf", "thigh"]

    class domain_rand(ESWMPA1FlatCfg.domain_rand):
        friction_range = [0.25, 1.75]
        added_mass_range = [-1.0, 1.0]

    class noise(ESWMPA1FlatCfg.noise):
        class noise_scales(ESWMPA1FlatCfg.noise.noise_scales):
            dof_pos = 0.03
            ang_vel = 0.3

    class commands(ESWMPA1FlatCfg.commands):
        heading_command = False

        class ranges(ESWMPA1FlatCfg.commands.ranges):
            lin_vel_x = flat_lin_vel_x = [-1.0, 2.0]
            lin_vel_y = flat_lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = flat_ang_vel_yaw = [-1.57, 1.57]

    class rewards(ESWMPA1FlatCfg.rewards):
        reward_curriculum = False

        class scales:
            # Legacy reward preparation multiplies these scales by the control period.
            tracking_lin_vel = 1.5 / 0.03
            tracking_ang_vel = 0.5 / 0.03
