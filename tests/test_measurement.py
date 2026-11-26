"""Deterministic contracts for event polarities, synaptic counts and BN folding."""
import unittest

import torch
from torch import nn

from graduation.energy import SynapticMeter
from graduation.es_wmp import EventSpikingWMP
from graduation.events import event_frame


class MeasurementTest(unittest.TestCase):
    def test_signed_depth_events_follow_flow(self):
        depth = torch.tensor([[[.5, .5, 2., 2.]]]).expand(1, 3, 4)
        flow = torch.zeros(1, 2, 3, 4)
        flow[:, 0] = 1
        frame = event_frame(depth, threshold=.08, flow_pixels=flow)
        reverse = event_frame(depth, threshold=.08, flow_pixels=-flow)
        torch.testing.assert_close(frame, reverse.flip(1))
        self.assertEqual(float(frame[:, 1].sum()), 0)
        self.assertGreater(float(frame[:, 0].sum()), 0)
        self.assertEqual(float(event_frame(torch.ones_like(depth), flow_pixels=flow).sum()), 0)

    def test_spike_fanout_and_dense_macs(self):
        model = nn.Sequential(nn.Linear(3, 2, bias=False), nn.Linear(2, 1, bias=False))
        with SynapticMeter(model, {"0"}) as meter:
            model(torch.tensor([[1., 0., 1.], [0., 0., 1.]]))
        result = meter.report(2, 50)
        self.assertEqual(result["mac_per_decision"], 2)
        self.assertEqual(result["ac_per_decision"], 3)
        self.assertAlmostEqual(result["energy_pj_per_decision"], 11.9)
        self.assertEqual(result["layers"]["0"]["input_firing_rate"], .5)

    def test_grouped_convolution_handles_padding(self):
        convolution = nn.Conv2d(2, 4, kernel_size=3, padding=1, groups=2, bias=False)
        data = torch.zeros(1, 2, 3, 3)
        data[0, 0, 0, 0] = data[0, 1, 1, 1] = 1
        with SynapticMeter(convolution, {""}) as meter:
            convolution(data)
        self.assertEqual(meter.report(1, 50)["ac_per_decision"], (4 + 9) * 2)

    def test_event_student_fold_preserves_outputs(self):
        torch.manual_seed(7)
        sample = torch.randint(2, (2, 2, 16, 16)).float()
        proprio = torch.randn(2, 45)
        model = EventSpikingWMP(channels=(8, 8), memory_width=12, latent_dim=16).eval()
        state, _ = model.initial(2)
        with torch.no_grad():
            visual = model.encode(sample, "calibrate")
            model(proprio, visual, state, "calibrate")
            model.commit_norms()
            fused = model.fused()
            for old, new in zip(model(proprio, model.encode(sample), state),
                                fused(proprio, fused.encode(sample), state)):
                torch.testing.assert_close(old, new, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
