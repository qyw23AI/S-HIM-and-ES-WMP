import io
import sys
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.bench_blind import parse_args


class BenchBlindCLITest(unittest.TestCase):
    def parse_max_lr(self, value, *extra_args):
        argv = ["bench_blind.py", "s-him", "--out", "unused.json",
                "--reward-preset", "snn-stair", "--max-lr", value, *extra_args]
        app = SimpleNamespace(AppLauncher=Mock())
        with patch.dict(sys.modules, {"isaaclab.app": app}), patch.object(sys, "argv", argv):
            return parse_args()

    def test_accepts_stair_reward_and_max_lr(self):
        args = self.parse_max_lr("6e-4")
        self.assertEqual(args.reward_preset, "snn-stair")
        self.assertEqual(args.max_lr, 6e-4)

    def test_diagonal_amp_motion_presets(self):
        self.assertEqual(self.parse_max_lr("6e-4").amp_motion_set, "a1_diagonal")
        args = self.parse_max_lr("6e-4", "--amp-motion-set", "a1_diagonal_turn")
        self.assertEqual(args.amp_motion_set, "a1_diagonal_turn")

    def test_rejects_nonpositive_max_lr(self):
        for value in ("0", "-0.001"):
            with self.subTest(value=value), redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    self.parse_max_lr(value)
                self.assertEqual(error.exception.code, 2)
                self.assertIn("--max-lr must be positive", stderr.getvalue())
