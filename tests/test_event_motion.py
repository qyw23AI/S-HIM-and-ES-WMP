"""Projection and per-environment stop/reset contracts."""

import unittest
from types import SimpleNamespace

import torch

from graduation.evaluation.stop_and_go import StopAndGo
from graduation.sensors.events import analytical_flow, event_frame
from graduation.sim.event_camera import EventCamera


class EventMotionTest(unittest.TestCase):
    def setUp(self):
        self.k = torch.tensor([[[6., 0., 3.5], [0., 6., 3.5], [0., 0., 1.]]])
        self.pos = torch.zeros(1, 3)
        self.rot = torch.eye(3)[None]
        self.depth = torch.full((1, 7, 7), 2.)

    def flow(self, pos, rot, depth=None):
        depth = self.depth if depth is None else depth
        return analytical_flow(depth, self.k, pos, rot, depth, self.k, self.pos, self.rot, .1, 5.)

    def test_stationary_flow_and_events_are_exactly_zero_at_depth_edges(self):
        depth = self.depth.clone()
        depth[:, :, :3] = 1.
        flow = self.flow(self.pos, self.rot, depth)
        self.assertEqual(int(flow.count_nonzero()), 0)
        self.assertEqual(int(event_frame(depth, flow_pixels=flow).count_nonzero()), 0)

    def test_horizontal_translation_has_expected_signed_displacement(self):
        flow = self.flow(torch.tensor([[.2, 0., 0.]]), self.rot)
        torch.testing.assert_close(flow[0, :, 3, 3], torch.tensor([-.6, 0.]), atol=1e-6, rtol=0)
        self.assertEqual(int(flow[:, :, :, -1].count_nonzero()), 0)
        clipped = torch.full_like(self.depth, 5.)
        self.assertEqual(int(self.flow(torch.tensor([[.2, 0., 0.]]), self.rot, clipped).count_nonzero()), 0)

    def test_optical_roll_reprojects_without_camera_translation(self):
        roll = torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
        flow = self.flow(self.pos, roll)
        torch.testing.assert_close(flow[0, :, 3, 4], torch.tensor([1., -1.]), atol=1e-6, rtol=0)

    def test_async_reset_clears_only_selected_delayed_history(self):
        depth = self.depth.repeat(2, 1, 1)
        depth[:, :, :3] = 1.
        fake = SimpleNamespace(device="cpu", core=SimpleNamespace(
            depth_buffer=((depth - .1) / 4.9 - .5)[:, None].repeat(1, 2, 1, 1),
            cfg=SimpleNamespace(depth=SimpleNamespace(near_clip=.1, far_clip=5.))))
        camera = EventCamera(fake, threshold=.01, delay_frames=1, flow_mode="analytical")
        k = self.k.repeat(2, 1, 1)
        rot = self.rot.repeat(2, 1, 1)

        def capture(step, x):
            pos = torch.tensor([[x, 0., 0.], [x, 0., 0.]])
            fake.depth_source_steps = torch.full((2, 2), step)
            fake.optical_sample = (depth, k, pos, rot)
            return camera.capture()

        capture(0, 0.)
        capture(5, .15)
        camera.reset(torch.tensor([True, False]))
        events = capture(10, .30)
        self.assertEqual(float(events[0].sum()), 0)
        self.assertGreater(float(events[1].sum()), 0)
        self.assertEqual(camera.source_steps.tolist(), [-1, 5])
        self.assertEqual(float(capture(15, .45)[0].sum()), 0)

    def test_stop_requires_consecutive_stationary_steps_and_restores_command(self):
        commands = torch.tensor([[.8, .1, .2], [.6, 0., 0.]])
        stop = StopAndGo(commands, 0, 2, .1, trigger_distance=.5)
        x = torch.tensor([.6, .2])
        obstacle = torch.ones(2)
        done = torch.zeros(2, dtype=torch.bool)
        action = torch.ones(2, 12)
        stop.prepare(0, commands, x, obstacle)
        self.assertEqual(commands[0].tolist(), [0., 0., 0.])
        for step, speed in enumerate((.16, .08, .12, .09, .07, .06)):
            stop.prepare(step, commands, x, obstacle)
            stop.advance(step, x, torch.tensor([speed, .2]), done, action)
        stopping, _ = stop.prepare(6, commands, x, obstacle)
        self.assertFalse(bool(stopping.any()))
        torch.testing.assert_close(commands[0], torch.tensor([.8, .1, .2]))
        stop.advance(6, x + .1, torch.tensor([.13, .2]), done, action)
        row, = stop.finish()
        self.assertEqual(row["resume_step"], 6)
        self.assertAlmostEqual(row["recovery_distance_m"], .1, places=5)
        self.assertLessEqual(row["hold_peak_planar_speed_mps"], .1)
        self.assertEqual(row["stopping_action_max_abs"], 1.)


if __name__ == "__main__":
    unittest.main()
