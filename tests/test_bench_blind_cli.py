import io
import sys
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.bench_blind import parse_args


class BenchBlindCLITest(unittest.TestCase):
    def parse(self, *extra_args):
        app = SimpleNamespace(AppLauncher=Mock())
        with patch.dict(sys.modules, {"isaaclab.app": app}):
            return parse_args(["--out", "unused.json", *extra_args])

    def parse_max_lr(self, value, *extra_args):
        return self.parse("s-him", "--max-lr", value, *extra_args)

    def test_named_tasks_apply_training_defaults_and_preserve_overrides(self):
        for task, method, readout in (
            ("s_him_a1_rough_cmdcur", "s-him", "nlif"),
            ("s_him_a1_spike_rough_cmdcur", "s-him", "spike"),
            ("himloco_a1_amp_rough_cmdcur", "himloco", "spike"),
        ):
            with self.subTest(task=task):
                args = self.parse("--task", task)
                self.assertEqual((args.method, args.actor_readout_mode), (method, readout))
                self.assertEqual((args.terrain, args.num_envs, args.steps, args.iterations, args.seed),
                                 ("rough", 4096, 100, 5000, 1))
                self.assertEqual((args.cmd_curriculum_init_x, args.cmd_curriculum_init_y, args.cmd_curriculum_max_x),
                                 (0.2, 0.2, 2.0))
                self.assertEqual(args.cmd_curriculum_threshold, 0.70)
                self.assertEqual((args.learning_rate, args.max_lr, args.desired_kl, args.lr_schedule),
                                 (8e-4, 8e-4, 0.045, "adaptive"))
                self.assertEqual((args.max_grad_norm, args.entropy_coef, args.max_action_std), (5.0, 0.003, 0.55))
                self.assertEqual((args.diagnostics_interval, args.checkpoint_interval), (10, 250))
                self.assertTrue(args.amp)
                self.assertEqual((args.amp_motion_set, args.amp_obs_mode, args.amp_reward_coef, args.amp_task_lerp,
                                  args.amp_discr_lr, args.amp_discr_updates),
                                 ("a1_diagonal", "joint24", 0.6, 0.0, 5e-5, 2))
                if method == "s-him":
                    self.assertEqual((args.actor_norm, args.actor_surrogate, args.actor_threshold,
                                      args.ticks, args.actor_widths), ("none", "rect", 0.5, 8, [256, 128]))
                    self.assertTrue(args.freeze_aux_estimator_stats and args.balanced_calibration)
                    self.assertEqual(args.calibration_interval, 5)
                    self.assertTrue(args.paper_training and args.separate_grad_clip and args.ppo_rollout_features)
                else:
                    self.assertFalse(args.paper_training or args.separate_grad_clip or args.ppo_rollout_features)
                    self.assertEqual(args.actor_weight_decay, 0.0)

                override = self.parse("--task", task, "--terrain=flat", "--num-envs", "8",
                                      "--cmd-curriculum-init-y", "0", "--amp-reward-coef", "2",
                                      "--amp-obs-mode", "full43", "--max-lr", "6e-4", "--no-amp")
                self.assertEqual((override.terrain, override.num_envs), ("flat", 8))
                self.assertEqual((override.cmd_curriculum_init_y, override.amp_reward_coef, override.max_lr),
                                 (0.0, 2.0, 6e-4))
                self.assertFalse(override.amp)
                self.assertEqual(override.amp_obs_mode, "full43")

    def test_diagonal_amp_motion_presets(self):
        defaults = self.parse_max_lr("6e-4")
        self.assertEqual((defaults.amp_motion_set, defaults.amp_obs_mode), ("a1_diagonal", "wmp30"))
        for preset in ("a1_diagonal_turn", "a1_diagonal_omni"):
            with self.subTest(preset=preset):
                args = self.parse_max_lr("6e-4", "--amp-motion-set", preset, "--amp-obs-mode", "full43")
                self.assertEqual((args.amp_motion_set, args.amp_obs_mode), (preset, "full43"))

    def test_rejects_nonpositive_max_lr(self):
        for value in ("0", "-0.001"):
            with self.subTest(value=value), redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    self.parse_max_lr(value)
                self.assertEqual(error.exception.code, 2)
                self.assertIn("--max-lr must be positive", stderr.getvalue())

    def test_command_curriculum_defaults(self):
        args = self.parse("s-him")
        self.assertIsNone(args.cmd_curriculum_init_x)
        self.assertIsNone(args.cmd_curriculum_init_y)
        self.assertIsNone(args.cmd_curriculum_max_x)
        self.assertEqual(args.cmd_curriculum_step, 0.2)
        self.assertEqual(args.cmd_curriculum_threshold, 0.8)

    def test_accepts_progressive_commands_for_shared_methods(self):
        for method in ("s-him", "himloco"):
            with self.subTest(method=method):
                args = self.parse(method, "--cmd-curriculum-init-x", "0.2", "--cmd-curriculum-init-y", "0",
                                  "--cmd-curriculum-max-x", "1.5", "--cmd-curriculum-step", "0.1",
                                  "--cmd-curriculum-threshold", "1")
                self.assertEqual(args.terrain, "rough")
                self.assertEqual(args.cmd_curriculum_init_x, 0.2)
                self.assertEqual(args.cmd_curriculum_init_y, 0.0)
                self.assertEqual(args.cmd_curriculum_max_x, 1.5)
                self.assertEqual(args.cmd_curriculum_step, 0.1)
                self.assertEqual(args.cmd_curriculum_threshold, 1.0)

    def test_rejects_invalid_command_curriculum_values(self):
        for name, values in (
            ("init-x", ("0", "-0.2", "nan", "inf")),
            ("init-y", ("-0.2", "nan", "inf")),
            ("max-x", ("0", "-1", "nan", "inf")),
            ("step", ("0", "-0.2", "nan", "inf")),
            ("threshold", ("0", "-0.2", "1.1", "nan", "inf")),
        ):
            flag = f"--cmd-curriculum-{name}"
            for value in values:
                with self.subTest(flag=flag, value=value), redirect_stderr(io.StringIO()) as stderr:
                    with self.assertRaises(SystemExit) as error:
                        self.parse("s-him", flag, value)
                    self.assertEqual(error.exception.code, 2)
                    self.assertIn(f"error: {flag} must be", stderr.getvalue())

    def test_rejects_initial_range_above_forward_ceiling(self):
        for init_x, extra in (("2.1", ()), ("0.3", ("--cmd-curriculum-max-x", "0.2"))):
            with self.subTest(init_x=init_x, extra=extra), redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    self.parse("himloco", "--cmd-curriculum-init-x", init_x, *extra)
                self.assertEqual(error.exception.code, 2)
                self.assertIn("must be at least --cmd-curriculum-init-x", stderr.getvalue())
