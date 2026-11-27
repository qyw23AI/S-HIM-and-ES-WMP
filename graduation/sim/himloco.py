"""Isaac Lab environment factory for the shared rough-terrain benchmark."""


def make_blind_env(args):
    """Build the A1 environment and return the HIMLoco policy defaults."""
    from himloco_lab.adapter import LegacyRunnerAdapter
    from himloco_lab.config import load_config
    from himloco_lab.env import HIMLabEnv
    from himloco_lab.env_cfg import make_env_cfg

    legacy, defaults = load_config("a1", args.num_envs, 4)
    if args.terrain == "flat":
        legacy.terrain.mesh_type = "plane"
        legacy.terrain.curriculum = False
    elif args.terrain in ("stairs", "stairs-high"):
        # Gradual stair curriculum: preserve original step height (0.05 + 0.18 * difficulty)
        # while shrinking the central spawn platform to 2.0 m so stairs begin within ~1.0 m.
        from himloco_lab.terrain import heightfield as terrain_utils
        from himloco_lab.terrain.rough import Terrain

        class CompactPlatformStairTerrain(Terrain):
            """Stair curriculum with original step-height progression and a 2.0 m spawn platform."""

            def make_terrain(self, choice, difficulty):
                if self.proportions[1] <= choice < self.proportions[3]:
                    patch = terrain_utils.SubTerrain(
                        "terrain",
                        width=self.width_per_env_pixels,
                        length=self.width_per_env_pixels,
                        vertical_scale=self.cfg.vertical_scale,
                        horizontal_scale=self.cfg.horizontal_scale,
                    )
                    step_height = 0.05 + 0.18 * difficulty
                    if choice < self.proportions[2]:
                        step_height *= -1
                    terrain_utils.pyramid_stairs_terrain(
                        patch, step_width=0.31, step_height=step_height, platform_size=2.0
                    )
                    return patch
                return super().make_terrain(choice, difficulty)

        legacy.terrain.terrain_proportions = (
            [0.05, 0.10, 0.60, 0.15, 0.10]
            if args.terrain == "stairs"
            else [0.05, 0.10, 0.65, 0.10, 0.10]
        )
        legacy.terrain.terrain_factory = CompactPlatformStairTerrain
    preset = getattr(args, "reward_preset", "default")
    if preset == "climb":
        # Stair-climbing reward preset: stronger forward velocity incentive, gentler vz/height penalty, 16cm clearance target.
        legacy.rewards.scales.tracking_lin_vel = 2.5
        legacy.rewards.scales.lin_vel_z = -0.5
        legacy.rewards.scales.base_height = -0.5
        legacy.rewards.scales.foot_clearance = -0.02
        legacy.rewards.clearance_height_target = -0.12
    elif preset == "snn-amp":
        # SNN+AMP reward preset adapted from FullySNNforLeggedRobots/AMP (strong velocity tracking, no ANN smoothness penalties).
        legacy.rewards.base_height_target = 0.25
        legacy.rewards.scales.tracking_lin_vel = 3.0
        legacy.rewards.scales.tracking_ang_vel = 1.5
        legacy.rewards.scales.lin_vel_z = 0.0
        legacy.rewards.scales.ang_vel_xy = 0.0
        legacy.rewards.scales.orientation = 0.0
        legacy.rewards.scales.dof_acc = 0.0
        legacy.rewards.scales.joint_power = 0.0
        legacy.rewards.scales.base_height = 0.0
        legacy.rewards.scales.foot_clearance = 0.0
        legacy.rewards.scales.action_rate = 0.0
        legacy.rewards.scales.smoothness = 0.0
    elif preset == "snn-climb":
        # SNN stair-climbing preset combining FullySNN/AMP (no discrete spike rate/smoothness penalties) and FullySNN/RMA climb signals.
        legacy.rewards.base_height_target = 0.28
        legacy.rewards.scales.tracking_lin_vel = 2.5
        legacy.rewards.scales.tracking_ang_vel = 1.0
        legacy.rewards.scales.lin_vel_z = -0.2
        legacy.rewards.scales.ang_vel_xy = -0.02
        legacy.rewards.scales.orientation = -0.05
        legacy.rewards.scales.dof_acc = -5e-8
        legacy.rewards.scales.joint_power = -5e-6
        legacy.rewards.scales.base_height = -0.2
        legacy.rewards.scales.foot_clearance = 0.0
        legacy.rewards.scales.action_rate = 0.0
        legacy.rewards.scales.smoothness = 0.0
        legacy.rewards.scales.stumble = -0.5
        legacy.rewards.scales.collision = -1.0
        legacy.rewards.scales.stand_still = -0.5
    elif preset == "snn-stair":
        legacy.rewards.base_height_target = 0.30
        legacy.rewards.scales.tracking_lin_vel = 3.0
        legacy.rewards.scales.tracking_ang_vel = 1.2
        legacy.rewards.scales.lin_vel_z = -0.2
        legacy.rewards.scales.ang_vel_xy = -0.02
        legacy.rewards.scales.orientation = -0.05
        legacy.rewards.scales.dof_acc = -5e-8
        legacy.rewards.scales.joint_power = -5e-6
        legacy.rewards.scales.base_height = -0.5
        legacy.rewards.scales.foot_clearance = 0.0
        legacy.rewards.scales.action_rate = 0.0
        legacy.rewards.scales.smoothness = 0.0
        legacy.rewards.scales.stumble = -0.1
        legacy.rewards.scales.collision = -0.25
        legacy.rewards.scales.stand_still = -0.5
    cfg = make_env_cfg(legacy, 4, args.device)
    cfg.seed = args.seed
    env = HIMLabEnv(cfg)
    return env, LegacyRunnerAdapter(env), defaults
