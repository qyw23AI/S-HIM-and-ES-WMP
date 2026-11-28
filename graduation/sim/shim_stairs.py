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
    from himloco_lab.env import HIMLabEnv
    from himloco_lab.env_cfg import make_env_cfg
    from himloco_lab.terrain.rough import Terrain
    from himloco_lab.terrain.heightfield import SubTerrain

    from graduation.legacy.s_him.evaluation_config import SHIMA1StairEvalCfg

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

    legacy = SHIMA1StairEvalCfg()
    legacy.env.num_envs = args.num_envs
    legacy.terrain.num_cols = min(args.num_envs, 8)
    legacy.terrain.terrain_factory = StairTerrain
    cfg = make_env_cfg(legacy, 4, args.device)
    cfg.seed = args.seed
    env = StairEnv(cfg)
    return env, LegacyRunnerAdapter(env)
