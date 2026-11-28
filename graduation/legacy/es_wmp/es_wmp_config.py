"""Inherited WMP depth tasks and ES-Parkour reward profiles."""

from ..base.es_wmp_base_config import ESWMPBaseCfg, ESWMPBaseCfgPPO


class ESWMPA1MixedCfg(ESWMPBaseCfg):
    terrain_name = "mixed"
    reward_profile = None

    class terrain(ESWMPBaseCfg.terrain):
        num_rows = 10
        num_cols = 20
        border_size = 5.0


class ESWMPA1MixedCfgPPO(ESWMPBaseCfgPPO):
    pass


class ESWMPA1FlatCfg(ESWMPA1MixedCfg):
    terrain_name = "flat"

    class terrain(ESWMPA1MixedCfg.terrain):
        mesh_type = "plane"
        curriculum = False


class ESWMPA1FlatCfgPPO(ESWMPA1MixedCfgPPO):
    pass


class ESWMPA1ParkourCfg(ESWMPA1MixedCfg):
    terrain_name = "parkour"

    class terrain(ESWMPA1MixedCfg.terrain):
        num_rows = 6
        num_cols = 4
        max_init_terrain_level = 0
        spawn_jitter_m = 0.15
        terrain_proportions = [0.0] * 10 + [0.25] * 4
        terrain_factory = "graduation.sim.parkour:ParkourTerrain"
        column_types_override = ("gap", "stairup", "hurdle", "parkour")

    class commands(ESWMPA1MixedCfg.commands):
        class ranges(ESWMPA1MixedCfg.commands.ranges):
            lin_vel_x = [0.6, 1.0]
            lin_vel_y = [0.0, 0.0]
            heading = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]


class ESWMPA1ParkourCfgPPO(ESWMPA1MixedCfgPPO):
    class runner(ESWMPA1MixedCfgPPO.runner):
        num_envs = 32
        iterations = 4
        rollout_steps = 25

    class distillation(ESWMPA1MixedCfgPPO.distillation):
        warmup_iterations = 2


class ESWMPA1ParkourSoftCfg(ESWMPA1ParkourCfg):
    reward_profile = "soft"
    core_factory = "graduation.methods.es_parkour.task:ParkourCore"

    class rewards(ESWMPA1ParkourCfg.rewards):
        reward_curriculum = False
        only_positive_rewards = False

        class scales:
            tracking_goal_vel = 1.5
            tracking_yaw = 0.5
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -1.0
            dof_acc = -2.5e-7
            collision = -1.0
            action_rate = -0.1
            delta_torques = -1e-7
            torques = -1e-5
            hip_pos = -0.5
            dof_error = -0.04
            feet_stumble = -0.1
            feet_edge = -0.1


class ESWMPA1ParkourSoftCfgPPO(ESWMPA1ParkourCfgPPO):
    pass


class ESWMPA1ParkourStrictCfg(ESWMPA1ParkourSoftCfg):
    reward_profile = "strict"

    class rewards(ESWMPA1ParkourSoftCfg.rewards):
        only_positive_rewards = True

        class scales(ESWMPA1ParkourSoftCfg.rewards.scales):
            collision = -10.0
            feet_stumble = -1.0
            feet_edge = -1.0


class ESWMPA1ParkourStrictCfgPPO(ESWMPA1ParkourSoftCfgPPO):
    pass
