"""Shared Isaac Lab environment factory for the visual experiments."""


def make_visual_env(args, *, reward_profile=None, capture_outcomes=False):
    """Build the WMP course or enable the ES-Parkour reward profile."""
    from legacy import load_task_config
    from wmp_lab.env import WMPEnv, Adapter
    from wmp_lab.env_cfg import make_env_cfg

    legacy, config = load_task_config("a1_amp")
    legacy.env.num_envs = legacy.depth.camera_num_envs = args.num_envs
    legacy.env.episode_length_s = args.episode_seconds
    legacy.terrain.border_size = 5.0
    if args.terrain == "parkour":
        from graduation.sim.parkour import KINDS, ParkourTerrain

        legacy.terrain.num_rows = 6
        legacy.terrain.num_cols = len(KINDS)
        legacy.terrain.terrain_factory = ParkourTerrain
        legacy.terrain.column_types_override = KINDS
        legacy.terrain.max_init_terrain_level = 0
        legacy.terrain.spawn_jitter_m = .15
        legacy.terrain.terrain_proportions = [0.] * 10 + [0.25] * 4
        legacy.commands.ranges.lin_vel_x = [0.6, 1.0]
        legacy.commands.ranges.lin_vel_y = [0., 0.]
        legacy.commands.ranges.heading = [0., 0.]
        legacy.commands.ranges.ang_vel_yaw = [0., 0.]
    else:
        legacy.terrain.num_rows = 10
        legacy.terrain.num_cols = 20
    if args.terrain == "flat":
        legacy.terrain.mesh_type = "plane"
        legacy.terrain.curriculum = False
    if reward_profile is not None:
        from graduation.methods.es_parkour.task import configure_rewards

        configure_rewards(legacy, reward_profile)
    cfg = make_env_cfg(legacy, 4, args.device)
    cfg.is_finite_horizon = reward_profile is not None
    cfg.seed = args.seed
    cfg.camera_backend = "raycast"
    if capture_outcomes:
        import torch
        from isaaclab.utils.math import matrix_from_quat

        analytic = getattr(args, "flow_mode", "random") == "analytical"

        class EvaluationEnv(WMPEnv):
            optical_sample = None
            depth_source_steps = None

            def _get_observations(self):
                core = self.core
                interval = core.cfg.depth.update_interval
                due = core.global_counter % interval == 0
                fresh = self.depth_age < interval
                observation = super()._get_observations()
                if due:
                    if self.depth_source_steps is None:
                        self.depth_source_steps = torch.full(
                            core.depth_buffer.shape[:2], -1, dtype=torch.long, device=self.device)
                    self.depth_source_steps[:, -2] = self.depth_source_steps[:, -1]
                    self.depth_source_steps[fresh, -2] = core.global_counter
                    self.depth_source_steps[:, -1] = core.global_counter
                    if analytic:
                        data = self.camera.data
                        ids = self.camera_env_ids
                        self.optical_sample = (data.output["distance_to_image_plane"][ids, ..., 0].clone(),
                                               data.intrinsic_matrices[ids].clone(), data.pos_w[ids].clone(),
                                               matrix_from_quat(data.quat_w_ros[ids]))
                return observation

            def _reset_idx(self, ids):
                super()._reset_idx(ids)
                if self.depth_source_steps is not None:
                    self.depth_source_steps[self.camera_lookup[ids]] = -1

            def _get_dones(self):
                terminated, timeout = super()._get_dones()
                core = self.core
                # Isaac resets before returning observations. Capture this transition here.
                self.transition_x = core.root_states[:, 0].clone()
                self.transition_speed = core.base_lin_vel[:, 0].clone()
                self.transition_planar_speed = torch.linalg.vector_norm(core.base_lin_vel[:, :2], dim=1)
                # WMP masks termination on timeout. Recover simultaneous failures.
                collision = torch.any(torch.norm(
                    core.contact_forces[:, core.termination_contact_indices, :], dim=-1) > 1., dim=1)
                self.transition_failed = terminated | (timeout & (collision | core.fall | core.vel_violate))
                return terminated, timeout

        env = EvaluationEnv(cfg)
    else:
        env = WMPEnv(cfg)
    return env, Adapter(env), config
