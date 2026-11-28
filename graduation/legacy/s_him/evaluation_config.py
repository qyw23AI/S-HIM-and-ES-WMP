"""Fixed-command A1 layouts for the evaluation panel."""

from ..base.s_him_base_config import SHIMBaseCfg


class SHIMA1StairEvalCfg(SHIMBaseCfg):
    """Two-flight evaluation layout with a nominal spawn and fixed commands."""

    class terrain(SHIMBaseCfg.terrain):
        num_rows = 1
        max_init_terrain_level = 0
        terrain_length = 7.5
        terrain_width = 2.4
        horizontal_scale = 0.03
        reset_xy_jitter = 0.1

    class commands(SHIMBaseCfg.commands):
        curriculum = False
        resampling_time = SHIMBaseCfg.env.episode_length_s + 1.0

        class ranges(SHIMBaseCfg.commands.ranges):
            heading = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]

    class domain_rand(SHIMBaseCfg.domain_rand):
        push_robots = False
        disturbance = False

    class init_state(SHIMBaseCfg.init_state):
        reset_at_nominal_pose = True


class SHIMA1FlatEvalCfg(SHIMBaseCfg):
    """Flat panel with the same fixed-command evaluation conditions."""

    commands = SHIMA1StairEvalCfg.commands
    domain_rand = SHIMA1StairEvalCfg.domain_rand
    init_state = SHIMA1StairEvalCfg.init_state

    class terrain(SHIMBaseCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        reset_xy_jitter = 0.0
