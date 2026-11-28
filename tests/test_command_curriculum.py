import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from graduation.paths import HIM
from graduation.sim.himloco import configure_progressive_command_curriculum, make_blind_env


def make_core(num_envs=10, heading_command=True):
    return SimpleNamespace(
        cfg=SimpleNamespace(commands=SimpleNamespace(curriculum=False, max_curriculum=2.0,
                                                     heading_command=heading_command)),
        command_ranges={"lin_vel_x": [-1.0, 1.0], "lin_vel_y": [-1.0, 1.0],
                        "heading": [0.7, 0.7], "ang_vel_yaw": [-0.3, -0.3]},
        commands=torch.full((num_envs, 4), float("nan")),
        num_envs=num_envs,
        device="cpu",
        max_episode_length=10,
        episode_sums={"tracking_lin_vel": torch.full((num_envs,), 18.0)},
        reward_scales={"tracking_lin_vel": 2.0},
    )


def make_resettable_core():
    core = make_core()
    core.common_step_counter = 0
    core.max_episode_length = 1000
    core.episode_sums["tracking_lin_vel"].zero_()

    def reset_idx(env_ids):
        if len(env_ids) == 0:
            return
        if core.cfg.commands.curriculum and core.common_step_counter % core.max_episode_length == 0:
            core.update_command_curriculum(env_ids)
        core.episode_sums["tracking_lin_vel"][env_ids] = 0.0

    core.reset_idx = reset_idx
    return core


class CommandCurriculumTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(torch.random.fork_rng(devices=[]))
        torch.manual_seed(42)

    def assert_ranges(self, core, x, y):
        self.assertEqual(core.command_ranges["lin_vel_x"], [-x, x])
        self.assertEqual(core.command_ranges["lin_vel_y"], [-y, y])

    def reset_at_step(self, core, step, env_ids, tracking_total):
        core.common_step_counter = step
        core.episode_sums["tracking_lin_vel"][env_ids] = tracking_total
        core.reset_idx(env_ids)

    def test_promotes_on_timeout_resets_after_the_legacy_cadence(self):
        core = make_resettable_core()
        configure_progressive_command_curriculum(core, init_x=0.2)
        env_ids = torch.arange(core.num_envs)
        self.assert_ranges(core, 0.2, 0.2)

        for step, half_range in ((1001, 0.4), (2002, 0.6), (3003, 0.8)):
            with self.subTest(step=step):
                self.reset_at_step(core, step, env_ids, 1800.0)
                self.assert_ranges(core, half_range, half_range)

    def test_early_falls_lower_the_group_average_for_the_current_window(self):
        core = make_resettable_core()
        configure_progressive_command_curriculum(core, init_x=0.2)

        self.reset_at_step(core, 300, torch.tensor([0]), 200.0)
        self.assert_ranges(core, 0.2, 0.2)
        self.reset_at_step(core, 1001, torch.arange(1, core.num_envs), 1800.0)
        self.assert_ranges(core, 0.2, 0.2)
        self.reset_at_step(core, 2002, torch.arange(core.num_envs), 1800.0)
        self.assert_ranges(core, 0.4, 0.4)

    def test_legacy_cadence_does_not_promote_twice(self):
        core = make_resettable_core()
        configure_progressive_command_curriculum(core, init_x=0.2)

        self.reset_at_step(core, 1000, torch.tensor([0, 2]), 1800.0)
        self.assert_ranges(core, 0.4, 0.4)
        env_ids = torch.tensor([1, 3])
        self.reset_at_step(core, 1000, env_ids, 1800.0)
        self.assert_ranges(core, 0.4, 0.4)

        core.episode_sums["tracking_lin_vel"][env_ids] = 1800.0
        core.update_command_curriculum(env_ids)
        self.assert_ranges(core, 0.6, 0.6)

    def test_promotes_both_ranges_to_their_default_ceilings(self):
        core = make_core()
        configure_progressive_command_curriculum(core, init_x=0.2)
        self.assertTrue(core.cfg.commands.curriculum)
        self.assertEqual(core.cfg.commands.max_curriculum, 2.0)
        self.assert_ranges(core, 0.2, 0.2)
        env_ids = torch.arange(core.num_envs)
        for x in (0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.0):
            core.update_command_curriculum(env_ids)
            self.assert_ranges(core, x, min(x, 1.0))

    def test_custom_threshold_step_and_ceiling(self):
        core = make_core()
        core.episode_sums["tracking_lin_vel"].fill_(12.0)
        configure_progressive_command_curriculum(core, init_x=0.2, init_y=0.1, max_x=0.55,
                                                 step=0.3, threshold=0.5)
        self.assertEqual(core.cfg.commands.max_curriculum, 0.55)
        self.assert_ranges(core, 0.2, 0.1)
        env_ids = torch.arange(core.num_envs)
        core.update_command_curriculum(env_ids)
        self.assert_ranges(core, 0.5, 0.4)
        core.update_command_curriculum(env_ids)
        self.assert_ranges(core, 0.55, 0.55)

    def test_initial_resampling_and_high_velocity_split(self):
        core = make_core(num_envs=4096)
        configure_progressive_command_curriculum(core, init_x=0.2)
        env_ids = torch.arange(core.num_envs)
        split = int(core.num_envs * 0.2)
        for cap in (0.2, 1.0):
            with self.subTest(cap=cap):
                if cap == 1.0:
                    for _ in range(4):
                        core.update_command_curriculum(env_ids)
                    core._resample_commands(env_ids)
                self.assertTrue((core.commands[:, :2].abs() <= cap).all())
                self.assertTrue((core.commands[:split, 0].abs() > 0.8 * cap).any())
                self.assertTrue((core.commands[split:, 0].abs() > 0.8 * cap).any())

        for _ in range(5):
            core.update_command_curriculum(env_ids)
        core._resample_commands(env_ids.flip(0))
        x, y = core.commands[:, :2].T
        self.assertTrue((x[split:].abs() <= 1.0).all())
        self.assertTrue((x[:split].abs() <= 2.0).all())
        self.assertGreater(x[:split].abs().max().item(), 1.9)
        self.assertTrue((y[x.abs() >= 1.0] == 0.0).all())
        self.assertGreater(y[split:].abs().max().item(), 0.9)

    def test_deadzone_adapts_and_then_stays_at_point_two(self):
        core = make_core(num_envs=4096)
        configure_progressive_command_curriculum(core, init_x=0.2, step=0.2)
        env_ids = torch.arange(core.num_envs)
        for cutoff in (0.1, 0.2, 0.2):
            norms = core.commands[:, :2].norm(dim=1)
            self.assertTrue(((norms == 0.0) | (norms > cutoff)).all())
            self.assertTrue(((norms > cutoff) & (norms < cutoff + 0.02)).any())
            self.assertGreater((norms == 0.0).float().mean().item(), 0.05)
            self.assertLess((norms == 0.0).float().mean().item(), 0.25)
            core.update_command_curriculum(env_ids)
            core._resample_commands(env_ids)

    def test_empty_groups_and_empty_reset_batch(self):
        core = make_core()
        configure_progressive_command_curriculum(core, init_x=0.2)
        before = core.commands.clone()
        empty = torch.empty(0, dtype=torch.long)
        core._resample_commands(empty)
        torch.testing.assert_close(core.commands, before, equal_nan=True)
        core.update_command_curriculum(empty)
        self.assert_ranges(core, 0.2, 0.2)
        core.update_command_curriculum(torch.tensor([0, 1]))
        self.assert_ranges(core, 0.4, 0.4)
        core.update_command_curriculum(torch.tensor([2, 3]))
        self.assert_ranges(core, 0.6, 0.6)

    def test_each_group_must_exceed_threshold_including_the_split_boundary(self):
        for below_id, total in ((0, 15.0), (2, 15.0), (2, 16.0)):
            with self.subTest(below_id=below_id, total=total):
                core = make_core()
                core.episode_sums["tracking_lin_vel"][below_id] = total
                configure_progressive_command_curriculum(core, init_x=0.2)
                core.update_command_curriculum(torch.tensor([0, 2]))
                self.assert_ranges(core, 0.2, 0.2)

    def test_zero_lateral_range_stays_zero(self):
        core = make_core(num_envs=4)
        configure_progressive_command_curriculum(core, init_x=0.2, init_y=0.0)
        env_ids = torch.arange(core.num_envs)
        for _ in range(10):
            core.update_command_curriculum(env_ids)
            core._resample_commands(env_ids)
        self.assert_ranges(core, 2.0, 0.0)
        self.assertTrue((core.commands[:, 1] == 0.0).all())

    def test_resamples_only_requested_rows_and_angular_column(self):
        for heading_command, column, value in ((True, 3, 0.7), (False, 2, -0.3)):
            with self.subTest(heading_command=heading_command):
                core = make_core(heading_command=heading_command)
                configure_progressive_command_curriculum(core, init_x=0.2)
                core.commands.fill_(10.0)
                core._resample_commands(torch.tensor([0, 2]))
                torch.testing.assert_close(core.commands[[0, 2], column], torch.full((2,), value))
                self.assertTrue((core.commands[[0, 2], 5 - column] == 10.0).all())
                self.assertTrue((core.commands[[1, 3]] == 10.0).all())

    def test_configuration_can_precede_command_buffer_allocation(self):
        core = make_core()
        core.cfg.commands.max_curriculum = 1.5
        del core.commands
        configure_progressive_command_curriculum(core, init_x=1.2)
        self.assert_ranges(core, 1.2, 1.0)
        self.assertEqual(core.cfg.commands.max_curriculum, 1.5)

    def test_factory_enables_curriculum_from_flag_or_named_task(self):
        modules = {
            "himloco_lab.adapter": SimpleNamespace(LegacyRunnerAdapter=Mock()),
            "himloco_lab.env": SimpleNamespace(HIMLabEnv=lambda cfg: SimpleNamespace(cfg=cfg, core=make_core())),
            "himloco_lab.env_cfg": SimpleNamespace(make_env_cfg=lambda legacy, *args: SimpleNamespace(legacy=legacy)),
        }
        args = SimpleNamespace(terrain="flat", num_envs=10, device="cpu", seed=1)
        with patch.object(sys, "path", [str(HIM / "isaaclab_migration"), *sys.path]), patch.dict(sys.modules, modules):
            flat_env, _, flat_defaults = make_blind_env(args)
            args.terrain = "rough"
            legacy_env, _, _ = make_blind_env(args)
            args.cmd_curriculum_init_x = 0.2
            progressive_env, _, _ = make_blind_env(args)
            del args.cmd_curriculum_init_x
            args.task = "s_him_a1_rough_cmdcur"
            named_env, _, _ = make_blind_env(args)
            args.terrain = "flat"
            args.cmd_curriculum_init_y = 0.0
            named_flat_env, _, _ = make_blind_env(args)

        for env in (legacy_env, progressive_env, named_env):
            self.assertEqual(env.cfg.legacy.terrain.terrain_proportions, [0.1, 0.2, 0.3, 0.3, 0.1])
            self.assertEqual(env.cfg.legacy.terrain.mesh_type, "trimesh")
        for env in (flat_env, named_flat_env):
            self.assertEqual(env.cfg.legacy.env.num_envs, 10)
            self.assertEqual(env.cfg.legacy.terrain.mesh_type, "plane")
            self.assertFalse(env.cfg.legacy.terrain.curriculum)
        self.assertEqual(flat_defaults["policy"]["method"], "himloco")
        self.assertEqual(flat_env.cfg.legacy.rewards.scales.tracking_lin_vel, 1.0)
        self.assertFalse(hasattr(flat_env.core, "update_command_curriculum"))
        self.assertFalse(hasattr(legacy_env.core, "update_command_curriculum"))
        self.assert_ranges(legacy_env.core, 1.0, 1.0)
        self.assert_ranges(progressive_env.core, 0.2, 0.2)
        self.assert_ranges(named_env.core, 0.2, 0.2)
        self.assertEqual(named_env.cfg.legacy.rewards.scales.tracking_lin_vel, 3.0)
        self.assert_ranges(named_flat_env.core, 0.2, 0.0)
        self.assertEqual(named_flat_env.cfg.legacy.rewards.scales.tracking_lin_vel, 3.0)
