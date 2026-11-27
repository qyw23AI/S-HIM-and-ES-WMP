import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from graduation.methods.s_him.export import check_and_export
from graduation.methods.s_him.policy import SHIM


class SHIMExportTest(unittest.TestCase):
    def test_fallback_preserves_reference_policy_if_fusion_changes_actions(self):
        torch.manual_seed(48)
        model = SHIM(45, 238, 12).eval()
        observations = torch.randn(4, 270)
        with torch.no_grad():
            altered = model.fused()
            altered.actor.readout.bias.add_(.1)
        with tempfile.TemporaryDirectory() as directory, patch.object(model, "fused", return_value=altered):
            path = Path(directory) / "policy.pt"
            report = check_and_export(model, observations, path)
            exported = torch.jit.load(str(path))
            torch.testing.assert_close(exported(observations), model(observations))
            self.assertEqual(report["deployment"], "unfused")
            self.assertFalse(report["fusion_verified"])
