"""Regression checks for opt-in spiking variants and exact pulse reuse."""

import copy
import unittest
from unittest.mock import patch

import torch

from graduation.evaluation.energy import SynapticMeter
from graduation.methods.es_parkour.student import ESStudent, es_spike_inputs
from graduation.snn.population_actor import LIFActor, spike


class ModelVariantsTest(unittest.TestCase):
    def test_static_projection_preserves_pulses_gradients_and_calibration_sites(self):
        torch.manual_seed(7)
        actor = LIFActor(9, 3, (16,), 4, 4, "care")
        inputs = torch.randn(5, 9)
        reports = []
        for optimized in (False, True):
            model = copy.deepcopy(actor)
            model.static_projection = optimized
            observed = []
            input_leaf = inputs.clone().requires_grad_()

            def record(voltage):
                pulse = spike(voltage)
                observed.append(pulse.detach())
                return pulse

            with patch("graduation.snn.population_actor.spike", record), SynapticMeter(model) as meter:
                result = model(input_leaf, "calibrate")
                result.square().sum().backward()
            reports.append((result, observed, input_leaf.grad, model.layers[0].weight.grad,
                            model.norms[0].count, meter.report(len(inputs), 50)["layers"]["layers.0"]["mac"]))
        before, after = reports
        torch.testing.assert_close(before[:2], after[:2], rtol=0, atol=0)
        torch.testing.assert_close(before[2:4], after[2:4])
        self.assertEqual(before[4], after[4])
        self.assertEqual(after[4], 20)
        self.assertEqual(before[5], after[5] * 4)

    def test_sew_routes_multivalued_layers_to_dense_counter(self):
        model = ESStudent(44, 12, base=2, normalization="tdbn", threshold=.5, residual="sew-add").eval()
        event = torch.randint(2, (1, 2, 16, 16)).float()
        prop, state = torch.randn(1, 44), torch.zeros(1, 64)
        spike_layers = es_spike_inputs(model)
        self.assertIn("blocks.0.first", spike_layers)
        self.assertNotIn("blocks.1.first", spike_layers)
        self.assertIn("blocks.1.second", spike_layers)
        with torch.no_grad(), SynapticMeter(model, spike_layers, {"stem"}) as meter:
            model(prop, model.observe(event, prop, state))
        self.assertGreater(meter.report(1, 50)["layers"]["blocks.1.first"]["mac"], 0)


if __name__ == "__main__":
    unittest.main()
