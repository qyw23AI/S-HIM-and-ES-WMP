"""Independent two-flight A1 staircase evaluation on the native HIM backend."""


# Heights and tread width are from HIMLoco Appendix B.1.
TREAD_M = .27
FLIGHTS = 2
STEPS_PER_FLIGHT = 5
STARTS_M = (1.50, 3.60)
SPAWN_M = .75
FINISH_M = 5.13
LANE_HALF_WIDTH_M = .90


def make_stair_env(args):
    """Create a fixed staircase and capture positions before simulator auto-reset."""
    from himloco_lab.adapter import LegacyRunnerAdapter
    from himloco_lab.config import load_config
    from himloco_lab.env import HIMLabEnv
    from himloco_lab.env_cfg import make_env_cfg
    from himloco_lab.terrain.rough import Terrain
    from himloco_lab.terrain.heightfield import SubTerrain

    class StairTerrain(Terrain):
        def make_terrain(self, choice, difficulty):
            patch = SubTerrain("short_stairs", width=self.length_per_env_pixels, length=self.width_per_env_pixels,
                               vertical_scale=self.cfg.vertical_scale, horizontal_scale=self.cfg.horizontal_scale)
            pixel = self.cfg.horizontal_scale
            step_units = round(args.step_height / self.cfg.vertical_scale)
            for start in STARTS_M:
                for index in range(STEPS_PER_FLIGHT):
                    riser = round((start + TREAD_M * index) / pixel)
                    patch.height_field_raw[riser:] += step_units
            return patch

        def add_terrain_to_map(self, patch, row, col):
            super().add_terrain_to_map(patch, row, col)
            self.env_origins[row, col] = (row * self.env_length + SPAWN_M,
                                          (col + .5) * self.env_width, 0.)

    class StairEnv(HIMLabEnv):
        def _get_dones(self):
            terminated, timeout = super()._get_dones()
            self.pre_reset_position = self.core.root_states[:, :3].clone()
            return terminated, timeout

    legacy, _ = load_config("a1", args.num_envs, 4)
    terrain = legacy.terrain
    terrain.num_rows, terrain.num_cols = 1, min(args.num_envs, 8)
    terrain.max_init_terrain_level = 0
    terrain.terrain_length, terrain.terrain_width = 7.5, 2.4
    terrain.horizontal_scale, terrain.vertical_scale = .03, .005
    terrain.reset_xy_jitter = .10
    terrain.terrain_factory = StairTerrain
    legacy.commands.curriculum = False
    legacy.commands.heading_command = True
    legacy.commands.ranges.heading = [0., 0.]
    legacy.commands.ranges.lin_vel_y = [0., 0.]
    legacy.commands.resampling_time = legacy.env.episode_length_s + 1.
    legacy.domain_rand.push_robots = False
    legacy.domain_rand.disturbance = False
    legacy.init_state.reset_at_nominal_pose = True
    cfg = make_env_cfg(legacy, 4, args.device)
    cfg.seed = args.seed
    env = StairEnv(cfg)
    return env, LegacyRunnerAdapter(env)
