"""FullySNN AMP reset, expert observations, and Isaac Lab task settings."""

from graduation.legacy.es_wmp.fully_snn_config import FullySNNA1AmpCfg
from graduation.paths import FULLY_SNN_AMP as AMP


def make_amp_env(args):
    """Build the 30 ms A1 task with its author motion files."""
    import torch

    from .motion_loader import AMPLoader
    from wmp_lab.core import Core
    from wmp_lab.env import WMPEnv, Adapter
    from wmp_lab.env_cfg import make_env_cfg
    from wmp_lab.math import quat_apply

    motion_files = sorted(str(p) for p in (AMP / "datasets/mocap_motions").glob("*.txt"))
    motion = AMPLoader(args.device, time_between_frames=.03, preload_transitions=True,
                       num_preload_transitions=args.expert_preload, motion_files=motion_files)

    class AMPCore(Core):
        def compute_observations(self):
            full = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                              self.base_ang_vel * self.obs_scales.ang_vel, self.projected_gravity,
                              self.commands[:, :3] * self.commands_scale,
                              (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                              self.dof_vel * self.obs_scales.dof_vel, self.actions), dim=1)
            if self.add_noise:
                full += (torch.rand_like(full) * 2 - 1) * self.noise_scale_vec
            self.privileged_obs_buf = full.clamp(-100, 100)
            self.obs_buf = self.privileged_obs_buf[:, 6:].clone()

        def get_amp_observations(self):
            feet = self.foot_positions_in_base_frame(self.dof_pos)
            return torch.cat((self.dof_pos, feet, self.base_lin_vel, self.base_ang_vel,
                              self.dof_vel, self.root_states[:, 2:3]), dim=1)

        def _reset_dofs(self, ids):
            if args.eval:
                return super()._reset_dofs(ids)
            self.reference_frames = motion.get_full_frame_batch(len(ids))
            q = AMPLoader.get_joint_pose_batch(self.reference_frames)
            qd = AMPLoader.get_joint_vel_batch(self.reference_frames)
            self.dof_pos[ids], self.dof_vel[ids] = q, qd
            self.backend.robot.write_joint_state_to_sim(q, qd, joint_ids=self.backend.joint_ids, env_ids=ids)

        def _reset_root_states(self, ids):
            if args.eval:
                return super()._reset_root_states(ids)
            frames = self.reference_frames
            root_pos = AMPLoader.get_root_pos_batch(frames) + self.env_origins[ids]
            quat = AMPLoader.get_root_rot_batch(frames)
            lin_vel = quat_apply(quat, AMPLoader.get_linear_vel_batch(frames))
            ang_vel = quat_apply(quat, AMPLoader.get_angular_vel_batch(frames))
            self.root_states[ids] = torch.cat((root_pos, quat, lin_vel, ang_vel), dim=1)
            self.backend.write_root(self.root_states[ids], ids)

    class AMPEnv(WMPEnv):
        def _get_observations(self):
            core = self.core
            self.sync_state(core)
            core.compute_observations()
            core.last_last_actions.copy_(core.last_actions)
            core.last_actions.copy_(core.actions)
            core.last_dof_pos.copy_(core.dof_pos)
            core.last_dof_vel.copy_(core.dof_vel)
            core.last_torques.copy_(core.torques)
            core.last_root_vel.copy_(core.root_states[:, 7:13])
            self.extras = core.extras
            self.extras["time_outs"] = core.time_out_buf.clone()
            return {"policy": core.obs_buf, "critic": core.privileged_obs_buf}

    legacy = FullySNNA1AmpCfg()
    legacy.core_factory = AMPCore
    legacy.env.num_envs = args.num_envs
    cfg = make_env_cfg(legacy, 4, args.device)
    cfg.seed = args.seed
    cfg.camera_backend = "raycast"
    cfg.observation_space = 42
    env = AMPEnv(cfg)
    return env, Adapter(env), motion
