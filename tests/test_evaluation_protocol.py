"""Regression cases for the common-course evaluation protocol."""

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from graduation.evaluation.energy import SynapticMeter
from graduation.evaluation.episodes import CourseEpisodes
from graduation.evaluation.pareto import CONDITIONS, ENERGY_COLUMNS, pareto
from graduation.sim.a1_observations import ShimObservation


class EvaluationProtocolTest(unittest.TestCase):
    def test_shim_frame_matches_native_channel_conventions(self):
        source = SimpleNamespace(lin_vel=1., ang_vel=.25, dof_pos=1., dof_vel=.05)
        target = SimpleNamespace(lin_vel=2., ang_vel=.25, dof_pos=1., dof_vel=.05)
        convert = ShimObservation(source, target, [0.] * 12)
        command = torch.tensor([[.8, -.3, .2]])
        gyro, gravity = torch.randn(1, 3), torch.randn(1, 3)
        position, velocity, previous = (torch.randn(1, 12) for _ in range(3))
        wmp = torch.cat((gyro * .25, gravity, command * torch.tensor([1., 1., .25]),
                         position, velocity * .05, previous), dim=1)
        him = torch.cat((command * torch.tensor([2., 2., .25]), gyro * .25, gravity,
                         position, velocity * .05, previous), dim=1)
        torch.testing.assert_close(convert(wmp, command), him)

    def test_first_outcome_wins_and_empty_reset_is_omitted(self):
        spawn = torch.zeros(5)
        labels = torch.zeros(5, dtype=torch.long)
        tracker = CourseEpisodes(spawn, labels, labels, ["gap"], 1.)

        def step(position, failed, done):
            tracker.advance(torch.tensor(position), spawn, True, torch.tensor(failed), torch.tensor(done),
                            spawn, labels, labels)

        # Goal, simultaneous failure and goal, timeout, and two ongoing environments.
        step([1., 1., .5, .2, .5], [False, True, False, False, False], [False, True, True, False, False])
        # A later fall leaves the prior success intact. A goal at timeout is a success.
        step([1., .3, .2, 1., .6], [True, False, False, False, False], [True, False, False, True, False])
        rows = tracker.finish()
        self.assertEqual([(r["env"], r["episode"], r["outcome"]) for r in rows], [
            (0, 0, "success"), (1, 0, "failure"), (2, 0, "timeout"), (3, 0, "success"),
            (1, 1, "censored"), (2, 1, "censored"), (4, 0, "censored")])
        self.assertTrue(all(r["control_steps"] > 0 for r in rows))
        self.assertEqual(sum(r["completed"] for r in rows), 4)

    def test_dual_energy_reclassifies_only_the_event_stem(self):
        model = nn.Sequential(nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
                              nn.Flatten(), nn.Linear(9, 1, bias=False))
        data = torch.zeros(1, 1, 3, 3)
        data[0, 0, 0, 0] = 1.
        with SynapticMeter(model, event_inputs={"0"}) as meter:
            model(data)
        result = meter.report(1, 50)
        conservative = result["conservative_first_layer_mac"]
        strict = result["strict_event_ac"]
        self.assertEqual((conservative["mac_per_decision"], conservative["ac_per_decision"]), (90, 0))
        self.assertEqual((strict["mac_per_decision"], strict["ac_per_decision"]), (9, 4))
        self.assertEqual(result["layers"]["0"]["calls"], 1)

    def test_pareto_accepts_partial_windows_and_pairs_seeds(self):
        def row(method, seed, successes, mac, ac, version=2):
            return dict.fromkeys(CONDITIONS) | {
                "protocol_version": version, "purpose": "formal", "method": method, "seed": seed,
                "finished_episodes": 2, "successes": successes, "censored_episodes": 1,
                ENERGY_COLUMNS["conservative_first_layer_mac"]: mac,
                ENERGY_COLUMNS["strict_event_ac"]: ac,
            }

        rows = [row("wmp", 1, 2, 8., 8.), row("snn", 1, 2, 10., 3.),
                row("snn", 2, 0, 20., 6.), row("legacy", 1, 2, 1., 1., version=1)]
        for mode, winner in (("conservative_first_layer_mac", "wmp"), ("strict_event_ac", "snn")):
            result, = pareto(rows, mode)
            self.assertEqual([point["method"] for point in result["front"]], [winner])
            self.assertTrue(all(p["seeds"] == [1] and p["success_rate"] == 1 for p in result["points"]))


if __name__ == "__main__":
    unittest.main()
