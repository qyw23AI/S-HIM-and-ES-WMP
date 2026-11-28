"""HIMLoco ANN rough-terrain training with AMP."""

from ..base.s_him_base_config import SHIMBaseCfg, SHIMBaseCfgPPO


class HIMLocoA1AmpRoughCmdCurCfg(SHIMBaseCfg):
    class commands(SHIMBaseCfg.commands):
        progressive_curriculum = True
        curriculum_init_x = 0.2
        curriculum_init_y = 0.2
        curriculum_threshold = 0.70

        class ranges(SHIMBaseCfg.commands.ranges):
            lin_vel_x = [-2.0, 2.0]
            lin_vel_y = [-1.0, 1.0]

    class rewards(SHIMBaseCfg.rewards):
        base_height_target = 0.30

        class scales(SHIMBaseCfg.rewards.scales):
            tracking_lin_vel = 3.0
            tracking_ang_vel = 1.2
            lin_vel_z = -0.5
            ang_vel_xy = -0.03
            orientation = -0.1
            dof_acc = -5e-8
            joint_power = -5e-6
            base_height = -0.5
            foot_clearance = 0.0
            action_rate = 0.0
            smoothness = 0.0
            stumble = -0.1
            collision = -0.25
            stand_still = -0.5


class HIMLocoA1AmpRoughCmdCurCfgPPO(SHIMBaseCfgPPO):
    class policy(SHIMBaseCfgPPO.policy):
        method = "himloco"
        max_action_std = 0.55

    class algorithm(SHIMBaseCfgPPO.algorithm):
        learning_rate = 8e-4
        aux_learning_rate = 3e-4
        desired_kl = 0.045
        max_lr = 8e-4
        max_grad_norm = 5.0
        entropy_coef = 0.003

    class amp(SHIMBaseCfgPPO.amp):
        enabled = True
        motion_set = "a1_diagonal"
        obs_mode = "joint24"
        reward_coef = 0.6
        task_lerp = 0.0
        discr_lr = 5e-5
        discr_updates = 2

    class runner(SHIMBaseCfgPPO.runner):
        diagnostics_interval = 10
        checkpoint_interval = 250
