import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from graduation.paths import AMP_A1_MOTIONS, AMP_FOR_HARDWARE, WMP_A1_MOTIONS
from graduation.methods.s_him.export import check_and_export
from graduation.methods.s_him.policy import SHIM
from graduation.rl.amp import (
    AMP_OBS_DIM,
    AmpDiscriminator,
    AmpMotionDataset,
    AmpReplayBuffer,
    RunningMomentNormalizer,
    a1_foot_positions_base,
    amp_obs_dim,
    attach_amp_terminal_hook,
    extract_him_amp_observations,
    resolve_amp_motion_files,
)


class _DummyCore:
    def __init__(self, q_legacy, qd_legacy, lin_vel, ang_vel, base_height):
        self.dof_pos = q_legacy.clone()
        self.dof_vel = qd_legacy.clone()
        self.base_lin_vel = lin_vel.clone()
        self.base_ang_vel = ang_vel.clone()
        self._height = base_height.clone()
        self.root_states = torch.zeros(len(q_legacy), 13)
        self.root_states[:, 2] = base_height

    def _get_base_heights(self):
        return self._height

    def compute_termination_observations(self, env_ids):
        return torch.ones(len(env_ids), 238)


class AmpAndFeedbackTests(unittest.TestCase):
    def test_a1_fk_and_terminal_hook_match_mocap_ordering(self):
        trot0 = AMP_A1_MOTIONS / "trot0.txt"
        self.assertTrue(trot0.is_file())
        self.assertTrue((AMP_FOR_HARDWARE / "datasets/mocap_motions/trot0.txt").is_file())
        payload = json.loads(trot0.read_text())
        frames = torch.tensor(payload["Frames"], dtype=torch.float32)
        q_canon = frames[:, 7:19]
        foot_ref = frames[:, 19:31]
        foot_fk = a1_foot_positions_base(q_canon)
        self.assertLess(float((foot_fk - foot_ref).abs().mean()), 1e-3)
        self.assertLess(float((foot_fk - foot_ref).abs().max()), 3e-3)

        # Convert canonical [FL, FR, RL, RR] into Legacy [FR, FL, RR, RL] for HIMLabEnv parity check.
        canon_to_legacy = torch.tensor((3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8), dtype=torch.long)
        q_legacy = q_canon.index_select(1, canon_to_legacy)
        qd_legacy = frames[:, 37:49].index_select(1, canon_to_legacy)
        core = _DummyCore(q_legacy, qd_legacy, frames[:, 31:34], frames[:, 34:37], frames[:, 2])
        amp_obs = extract_him_amp_observations(core)
        self.assertEqual(amp_obs.shape, (len(frames), AMP_OBS_DIM))
        self.assertTrue(torch.allclose(amp_obs[:, :12], q_canon))
        self.assertLess(float((amp_obs[:, 12:24] - foot_ref).abs().max()), 3e-3)
        self.assertTrue(torch.allclose(amp_obs[:, 24:30], frames[:, 31:37]))
        self.assertTrue(torch.allclose(amp_obs[:, 30:42], frames[:, 37:49]))
        self.assertTrue(torch.allclose(amp_obs[:, 42], frames[:, 2]))

        env = SimpleNamespace(core=core)
        attach_amp_terminal_hook(env)
        ids = torch.tensor([0, 2])
        expected = amp_obs[ids].clone()
        term = core.compute_termination_observations(ids)
        self.assertEqual(term.shape, (2, 238))
        core.dof_pos.zero_()
        self.assertTrue(torch.allclose(core._amp_terminal_obs, expected))

        for mode, dim in (("wmp30", 30), ("joint24", 24), ("full43", 43)):
            with self.subTest(mode=mode):
                attach_amp_terminal_hook(env, obs_mode=mode)
                self.assertEqual(core._amp_terminal_obs.shape, (0, dim))
                expected = extract_him_amp_observations(core, obs_mode=mode)[ids]
                self.assertTrue(torch.equal(core.compute_termination_observations(ids), term))
                torch.testing.assert_close(core._amp_terminal_obs, expected, atol=0, rtol=0)

    def test_diagonal_motion_presets(self):
        forward = [WMP_A1_MOTIONS / name for name in ("trot1.txt", "trot2.txt")]
        self.assertEqual(resolve_amp_motion_files(), (forward, "fl_fr_rl_rr"))
        turns = [
            AMP_FOR_HARDWARE / "datasets/mocap_motions" / name
            for name in ("leftturn0.txt", "rightturn0.txt")
        ]
        for preset, expected in (
            ("a1_diagonal", forward),
            ("a1_diagonal_turn", forward + turns),
        ):
            with self.subTest(preset=preset):
                files, order = resolve_amp_motion_files(preset)
                self.assertEqual(files, expected)
                dataset = AmpMotionDataset(files, preload_transitions=0, leg_order=order)
                mean_feet = torch.cat(dataset.trajectories)[:, [13, 16]].mean(dim=0)
                self.assertGreater(float(mean_feet[0]), 0.05)
                self.assertLess(float(mean_feet[1]), -0.05)
                for path, obs in zip(files, dataset.trajectories):
                    with self.subTest(clip=path.name):
                        error = a1_foot_positions_base(obs[:, :12]) - obs[:, 12:24]
                        self.assertLess(float(error.abs().mean()), 1e-2)
                        if preset == "a1_diagonal":
                            self.assertLess(float(obs[:, 26].std()), 0.08)
                            correlation = torch.corrcoef(obs[:, [14, 23]].T)
                            self.assertGreater(float(correlation[0, 1]), 0.85)

    def test_diagonal_omni_observation_layouts_match_policy_channels(self):
        files, order = resolve_amp_motion_files("a1_diagonal_omni")
        self.assertEqual(order, "auto")
        self.assertEqual(files, [
            WMP_A1_MOTIONS / "trot1.txt",
            WMP_A1_MOTIONS / "trot2.txt",
            AMP_FOR_HARDWARE / "datasets/mocap_motions/leftturn0.txt",
            AMP_FOR_HARDWARE / "datasets/mocap_motions/rightturn0.txt",
            AMP_A1_MOTIONS / "trot_reverse0.txt",
        ])
        # This permutation swaps the left and right legs in either direction.
        swap_legs = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        self.assertEqual(amp_obs_dim(), 30)
        for mode, dim in (("wmp30", 30), ("joint24", 24), ("full43", 43)):
            with self.subTest(mode=mode):
                self.assertEqual(amp_obs_dim(mode), dim)
                dataset = AmpMotionDataset(files, preload_transitions=32, leg_order=order, obs_mode=mode)
                for batch in next(dataset.sample_minibatches(1, 8)):
                    self.assertEqual(batch.shape, (8, dim))
                    self.assertTrue(torch.isfinite(batch).all())

                for path, expert in zip(files, dataset.trajectories, strict=True):
                    with self.subTest(clip=path.name):
                        frames = torch.tensor(json.loads(path.read_text())["Frames"], dtype=torch.float32)
                        q, qd = frames[:, 7:19], frames[:, 37:49]
                        foot_ref = frames[:, 19:31]
                        if path.name in ("leftturn0.txt", "rightturn0.txt"):
                            foot_ref = foot_ref[:, swap_legs]
                        else:
                            q, qd = q[:, swap_legs], qd[:, swap_legs]
                        core = _DummyCore(q, qd, frames[:, 31:34], frames[:, 34:37], frames[:, 2])
                        policy = extract_him_amp_observations(core, obs_mode=mode)
                        self.assertEqual(expert.shape, (len(frames), dim))
                        if mode == "full43":
                            torch.testing.assert_close(expert[:, 12:24], foot_ref, atol=0, rtol=0)
                            # Mocap foot positions differ slightly from runtime forward kinematics.
                            torch.testing.assert_close(policy[:, 12:24], foot_ref, atol=0.016, rtol=0)
                            channels = [*range(12), *range(24, 43)]
                            expert, policy = expert[:, channels], policy[:, channels]
                        torch.testing.assert_close(expert, policy, atol=0, rtol=0)

    def test_amp_dataset_replay_normalizer_and_discriminator_step(self):
        up_files, up_order = resolve_amp_motion_files("upstream_trot")
        up_dataset = AmpMotionDataset(up_files, time_between_frames=0.02, preload_transitions=128,
                                      seed=1, leg_order=up_order)
        up_s, _ = next(up_dataset.sample_minibatches(1, 32))
        self.assertGreater(float(up_s[:, 13].mean()), 0.05)  # Canonical FL foot y > 0
        self.assertLess(float(up_s[:, 16].mean()), -0.05)    # Canonical FR foot y < 0

        files, leg_order = resolve_amp_motion_files("a1_trot")
        self.assertNotIn("right_turn1.txt", [p.name for p in files])
        dataset = AmpMotionDataset(files, time_between_frames=0.02, preload_transitions=256,
                                   seed=1, leg_order=leg_order)
        exp_batches = list(dataset.sample_minibatches(4, 32))
        self.assertEqual(len(exp_batches), 4)
        self.assertEqual(exp_batches[0][0].shape, (32, AMP_OBS_DIM))
        self.assertTrue(torch.isfinite(exp_batches[0][0]).all())

        replay = AmpReplayBuffer(AMP_OBS_DIM, buffer_size=48, device="cpu")
        replay.insert(torch.randn(30, AMP_OBS_DIM), torch.randn(30, AMP_OBS_DIM))
        replay.insert(torch.randn(30, AMP_OBS_DIM), torch.randn(30, AMP_OBS_DIM))
        self.assertEqual(replay.num_samples, 48)
        pol_batches = list(replay.sample_minibatches(4, 32))

        norm = RunningMomentNormalizer(AMP_OBS_DIM)
        disc = AmpDiscriminator(AMP_OBS_DIM, hidden_dims=(64, 32), amp_reward_coef=2.0, task_reward_lerp=0.0, dt=0.02)
        opt = torch.optim.Adam(disc.parameters(), lr=1e-3)
        before = [p.detach().clone() for p in disc.parameters()]
        metrics = disc.update_from_batches(pol_batches, exp_batches, opt, normalizer=norm)
        self.assertTrue(all(k in metrics for k in ("amp_loss", "amp_grad_pen", "amp_expert_d", "amp_policy_d")))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, disc.parameters())))
        self.assertGreater(float(norm.count), 1.0)

        task_r = torch.zeros(32)
        comb, style_sec, _ = disc.predict_reward(exp_batches[0][0], exp_batches[0][1], task_r, normalizer=norm)
        self.assertTrue(torch.allclose(comb, style_sec * 0.02, atol=1e-6))

        disc_lerp = AmpDiscriminator(AMP_OBS_DIM, hidden_dims=(64, 32), amp_reward_coef=2.0, task_reward_lerp=0.3, dt=0.02)
        task_ones = torch.full((32,), 0.05)
        comb_l, style_l, _ = disc_lerp.predict_reward(exp_batches[0][0], exp_batches[0][1], task_ones, normalizer=norm)
        self.assertTrue(torch.allclose(comb_l, 0.7 * (style_l * 0.02) + 0.3 * task_ones, atol=1e-6))

    def test_popsan_intra_feedback_parity_grad_and_export(self):
        torch.manual_seed(0)
        obs = torch.randn(16, 5 * 45)
        base = SHIM(
            45, 238, 12,
            actor_type="popsan", actor_widths=(32, 16), population=6, ticks=5,
            en_pop_dim=8, en_mean_range=(-1.5, 1.5), en_std=0.2,
            freeze_encoder=True, actor_intra_feedback=False,
        )
        fb = SHIM(
            45, 238, 12,
            actor_type="popsan", actor_widths=(32, 16), population=6, ticks=5,
            en_pop_dim=8, en_mean_range=(-1.5, 1.5), en_std=0.2,
            freeze_encoder=True, actor_intra_feedback=True,
        )
        missing, unexpected = fb.load_state_dict(base.state_dict(), strict=False)
        self.assertEqual(set(missing), {"actor.intra_weight", "actor.intra_bias"})
        self.assertEqual(unexpected, [])
        self.assertFalse(fb.actor.encoder.mean.requires_grad)
        self.assertFalse(fb.actor.encoder.std.requires_grad)
        self.assertTrue(torch.allclose(base(obs), fb(obs), atol=1e-6))

        loss = fb(obs).square().mean()
        loss.backward()
        self.assertGreater(float(fb.actor.intra_weight.grad.norm()), 0.0)
        self.assertGreater(float(fb.actor.intra_bias.grad.norm()), 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            metrics = check_and_export(fb, obs, Path(tmp) / "policy.pt")
            self.assertTrue(metrics["fusion_verified"])
            self.assertLessEqual(metrics["fusion_max_error"], 1e-5)


if __name__ == "__main__":
    unittest.main()
