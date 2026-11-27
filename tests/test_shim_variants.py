"""Contract and variant regression checks for single-stage S-HIM."""

import copy
import unittest

import torch

from graduation.methods.s_him.export import check_and_export
from graduation.methods.s_him.policy import SHIM
from graduation.rl.ppo import ppo_update, returns


class SHIMVariantsTest(unittest.TestCase):
    def test_backward_compatible_checkpoint_and_output_norm_fusion(self):
        torch.manual_seed(11)
        legacy = SHIM(45, 238, 12, grsn_threshold=0.6, critic_widths=(512, 256, 128))
        legacy_cfg = {
            "normalization": "care",
            "history": 5,
            "estimator_width": 64,
            "estimator_layers": 2,
            "population": 5,
            "estimator": "grsn",
            "grsn_threshold": 0.6,
            "grsn_bias": None,
            "critic_widths": (512, 256, 128),
        }
        reloaded = SHIM(45, 238, 12, recurrent=legacy_cfg.pop("estimator") == "grsn", **legacy_cfg)
        reloaded.load_state_dict(legacy.state_dict())
        obs = torch.randn(16, 270)
        torch.testing.assert_close(reloaded(obs), legacy(obs))

        out_norm = SHIM(45, 238, 12, grsn_threshold=0.6, critic_widths=(512, 256, 128), actor_output_norm=True)
        out_norm.eval()
        out_norm(obs, "adapt")
        report = check_and_export(out_norm, obs, None)
        self.assertTrue(report["fusion_verified"])
        self.assertLessEqual(report["fusion_max_error"], 1e-5)

    def test_zero_update_logp_ratio_and_microbatch_kl_invariance(self):
        torch.manual_seed(17)
        model = SHIM(45, 238, 12, actor_output_norm=True).eval()
        obs = torch.randn(24, 270)
        critic = torch.randn(24, 238)
        with torch.no_grad():
            feats = model.actor_input(obs)
            old_mu = model.actor(feats)
            old_sigma = model.log_std.clamp(-5, 2).exp().expand_as(old_mu)
            action = old_mu + 0.1 * torch.randn_like(old_mu)
            dist = torch.distributions.Normal(old_mu, old_sigma)
            old_logp = dist.log_prob(action).sum(-1)

        recomputed_mu = model.actor(feats)
        recomputed_dist = torch.distributions.Normal(recomputed_mu, model.log_std.clamp(-5, 2).exp())
        recomputed_logp = recomputed_dist.log_prob(action).sum(-1)
        self.assertLess(float((recomputed_logp - old_logp).abs().max()), 1e-6)
        torch.testing.assert_close((recomputed_logp - old_logp).exp(), torch.ones_like(old_logp))

        # Verify microbatch splitting preserves metrics, weights, and adaptive LR scheduling.
        with torch.no_grad():
            old_value = model.critic(critic).flatten()
        batch = (feats, critic, action, old_logp, old_value, torch.randn(24), torch.randn(24))
        runs = []
        for mb in (None, 5):
            copy_m = copy.deepcopy(model)
            opt = torch.optim.Adam([*copy_m.actor.parameters(), *copy_m.critic.parameters(), copy_m.log_std], lr=1e-3)
            torch.manual_seed(99)
            loss, metrics = ppo_update(
                lambda x, _: copy_m.actor(x),
                lambda _, c: copy_m.critic(c),
                copy_m.log_std,
                opt,
                batch,
                epochs=2,
                minibatches=2,
                microbatch=mb,
                old_mu=old_mu,
                old_sigma=old_sigma,
                desired_kl=0.01,
                max_grad_norm=10.0,
                separate_grad_clip=True,
                actor_params=list(copy_m.actor.parameters()),
                critic_params=list(copy_m.critic.parameters()),
                std_params=[copy_m.log_std],
                return_metrics=True,
            )
            runs.append((loss, metrics, list(copy_m.parameters())))
        self.assertAlmostEqual(runs[0][0], runs[1][0], places=6)
        self.assertAlmostEqual(runs[0][1]["ppo_kl_mean"], runs[1][1]["ppo_kl_mean"], places=6)
        self.assertAlmostEqual(runs[0][1]["learning_rate"], runs[1][1]["learning_rate"], places=9)
        for p0, p1 in zip(runs[0][2], runs[1][2]):
            torch.testing.assert_close(p0, p1, atol=1e-5, rtol=1e-4)

    def test_batch_aux_stats_freeze_buffers_and_gae_termination_bootstrap(self):
        torch.manual_seed(23)
        model = SHIM(45, 238, 12, swap_temperature=3.0, swap_reduction="mean", target_mode="himloco")
        before = {k: v.clone() for k, v in model.named_buffers()}
        obs = torch.randn(32, 270)
        succ = torch.randn(32, 238)
        loss, _, _, metrics = model.auxiliary_loss(obs, succ, 1.0, estimator_stats="batch", return_metrics=True)
        loss.backward()
        self.assertIn("prototype_entropy", metrics)
        for k, v in model.named_buffers():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)

        # Manual 2-step GAE check with one timeout and one hard termination.
        reward = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        value = torch.tensor([[0.5, 0.5], [0.8, 0.8]])
        next_value = torch.tensor([[0.9, 0.0], [0.4, 0.4]])  # env0 step0 timeout bootstrapped=0.9; env1 step0 terminated=0.0
        done = torch.tensor([[True, True], [False, False]])
        target, _ = returns(reward, value, next_value, done, gamma=0.99, lam=0.95)
        expected_step0_env0 = (1.0 + 0.99 * 0.9 - 0.5) + 0.5
        expected_step0_env1 = (1.0 + 0.99 * 0.0 - 0.5) + 0.5
        self.assertAlmostEqual(float(target[0, 0]), expected_step0_env0, places=5)
        self.assertAlmostEqual(float(target[0, 1]), expected_step0_env1, places=5)

    def test_elu_actor_positive_control_and_target_modes(self):
        torch.manual_seed(29)
        obs = torch.randn(8, 270)
        succ = torch.randn(8, 238)
        for mode in ("obs", "zero_cmd", "himloco", "priv_vel_cmd"):
            m = SHIM(45, 238, 12, actor_type="elu", target_mode=mode)
            out = m(obs)
            self.assertEqual(out.shape, (8, 12))
            loss, _, _ = m.auxiliary_loss(obs, succ, 1.0)
            loss.backward()

    def test_lif_actor_layernorm_widths_ticks_and_readout_modes(self):
        import tempfile
        from pathlib import Path
        from graduation.methods.s_him.export import check_and_export
        from graduation.snn.population_actor import LIFActor

        torch.manual_seed(31)
        obs = torch.randn(12, 270)
        feat = torch.randn(12, 64)
        for out_spike, exp_norms in ((True, 3), (False, 2)):
            actor = LIFActor(64, 12, (32, 16), population=6, ticks=8, normalization="ln",
                             output_norm=True, threshold=0.6, output_spike=out_spike)
            self.assertEqual(len(actor.norms), exp_norms)
            self.assertTrue(all(isinstance(n, torch.nn.LayerNorm) for n in actor.norms))
            y = actor(feat, "fixed")
            y_stats, rates, silent = actor.forward_with_stats(feat, "fixed")
            torch.testing.assert_close(y, y_stats)
            self.assertEqual(len(rates), exp_norms)
            self.assertEqual(len(silent), exp_norms)
            y.pow(2).mean().backward()
            self.assertIsNotNone(actor.layers[0].weight.grad)

            model = SHIM(45, 238, 12, actor_widths=(32, 16), population=6, ticks=8,
                         actor_output_norm=True, actor_norm="ln", actor_threshold=0.6,
                         actor_output_spike=out_spike).eval()
            model.calibrate(obs)
            with tempfile.TemporaryDirectory() as tmpdir:
                out_file = Path(tmpdir) / "policy.pt"
                report = check_and_export(model, obs, out_file)
                self.assertTrue(report["fusion_verified"])
                self.assertEqual(report["deployment"], "fused")
                loaded = torch.jit.load(str(out_file))
                torch.testing.assert_close(loaded(obs), model(obs), rtol=1e-5, atol=1e-5)

    def test_popsan_actor_grsn_fusion_spike_inputs_and_export(self):
        import tempfile
        from pathlib import Path
        from graduation.methods.s_him.export import check_and_export
        from graduation.methods.s_him.policy import shim_spike_inputs

        torch.manual_seed(37)
        obs = torch.randn(10, 270)
        model = SHIM(45, 238, 12, actor_widths=(64, 32), population=10, ticks=5,
                     actor_type="popsan", en_pop_dim=8).eval()
        y = model(obs)
        y_stats, rates, silent = model.actor.forward_with_stats(torch.randn(10, 64))
        self.assertEqual(y.shape, (10, 12))
        self.assertEqual(y_stats.shape, (10, 12))
        self.assertEqual(len(rates), 3)
        self.assertEqual(len(silent), 3)
        y.pow(2).mean().backward()
        self.assertIsNotNone(model.actor.encoder.mean.grad)
        self.assertIsNotNone(model.actor.layers[0].weight.grad)

        spike_inputs = shim_spike_inputs(model)
        self.assertIn("actor.layers.0", spike_inputs)
        self.assertIn("actor.layers.1", spike_inputs)
        self.assertIn("actor.layers.2", spike_inputs)

        model.calibrate(obs)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "popsan.pt"
            report = check_and_export(model, obs, out_file)
            self.assertTrue(report["fusion_verified"])
            self.assertEqual(report["deployment"], "fused")
            loaded = torch.jit.load(str(out_file))
            torch.testing.assert_close(loaded(obs), model(obs), rtol=1e-5, atol=1e-5)

    def test_nlif_readout_and_rect_surrogate_compat(self):
        import tempfile
        from pathlib import Path
        from graduation.methods.s_him.export import check_and_export

        torch.manual_seed(41)
        obs = torch.randn(10, 270)
        model = SHIM(45, 238, 12, actor_widths=(64, 32), population=5, ticks=6,
                     actor_norm="none", actor_threshold=0.5,
                     actor_readout_mode="nlif", actor_surrogate="rect").eval()
        self.assertEqual(model.actor.readout_mode, "nlif")
        self.assertEqual(model.actor.surrogate, "rect")
        y = model(obs)
        y_stats, rates, silent = model.actor.forward_with_stats(torch.randn(10, 64))
        self.assertEqual(y.shape, (10, 12))
        self.assertEqual(y_stats.shape, (10, 12))
        self.assertEqual(len(rates), 2)
        self.assertEqual(len(silent), 2)
        y.pow(2).mean().backward()
        self.assertIsNotNone(model.actor.layers[0].weight.grad)
        self.assertIsNotNone(model.actor.layers[-1].weight.grad)

        model.calibrate(obs)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "nlif.pt"
            report = check_and_export(model, obs, out_file)
            self.assertTrue(report["fusion_verified"])
            loaded = torch.jit.load(str(out_file))
            torch.testing.assert_close(loaded(obs), model(obs), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
