"""Shared Isaac Lab environment factory for the visual experiments."""


from importlib import import_module

from graduation.legacy import load_config


def _resolve_factory(cfg, name):
    path = getattr(cfg, name, None)
    if isinstance(path, str):
        module, attribute = path.split(":")
        setattr(cfg, name, getattr(import_module(module), attribute))


def make_visual_env(args, *, reward_profile=None, capture_outcomes=False):
    """Build the WMP scene and bind the selected terrain and reward implementations."""
    from wmp_lab.env import Adapter, WMPEnv
    from wmp_lab.env_cfg import make_env_cfg

    task = getattr(args, "task", None)
    terrain = getattr(args, "terrain", None)
    reward_profile = reward_profile or getattr(args, "reward_profile", None)
    legacy, config = load_config(task or f"es_wmp_a1_{terrain or 'mixed'}")
    if terrain is not None and terrain != legacy.terrain_name:
        layout, _ = load_config(f"es_wmp_a1_{terrain}")
        legacy.terrain, legacy.commands = layout.terrain, layout.commands
        legacy.terrain_name = layout.terrain_name

    if reward_profile is not None and reward_profile != legacy.reward_profile:
        rewards, _ = load_config(f"es_wmp_a1_parkour_{reward_profile}")
        legacy.rewards, legacy.core_factory = rewards.rewards, rewards.core_factory
        legacy.reward_profile = rewards.reward_profile

    runner = config["runner"]
    legacy.env.num_envs = legacy.depth.camera_num_envs = getattr(args, "num_envs", runner["num_envs"])
    legacy.env.episode_length_s = getattr(args, "episode_seconds", runner["episode_seconds"])
    _resolve_factory(legacy.terrain, "terrain_factory")
    _resolve_factory(legacy, "core_factory")
    cfg = make_env_cfg(legacy, 4, args.device)
    cfg.is_finite_horizon = hasattr(legacy, "core_factory")
    cfg.seed = getattr(args, "seed", config["seed"])
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
