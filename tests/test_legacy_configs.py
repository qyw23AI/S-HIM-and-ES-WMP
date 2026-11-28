"""Contract tests for simulator-independent task snapshots and benchmark selection."""

import importlib
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from graduation.legacy import TASKS, class_to_dict, list_tasks, load_config
from graduation.legacy.base.s_him_base_config import SHIMBaseCfg, SHIMBaseCfgPPO
from graduation.legacy.es_wmp.es_wmp_config import ESWMPA1ParkourCfg, ESWMPA1ParkourStrictCfg
from graduation.paths import HIM, ROOT, WMP
from graduation.sim.wmp import make_visual_env
from scripts.bench_parkour import parse_args as parse_parkour
from scripts.train_event_wmp import parse_args as parse_wmp


class LegacyConfigTest(unittest.TestCase):
    def test_every_task_loads_without_upstream_or_simulator_imports_and_is_isolated(self):
        subprocess.run(
            [sys.executable, "-c", """
import sys
from importlib.abc import MetaPathFinder

class NoSimulator(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {
            'isaacgym', 'isaaclab', 'isaacsim', 'omni', 'wmp_lab', 'himloco_lab', 'torch',
        }:
            raise AssertionError(f'Unexpected import: {fullname}')

sys.meta_path.insert(0, NoSimulator())
from graduation.legacy import list_tasks, load_config
for task in list_tasks():
    load_config(task)
"""],
            cwd=ROOT,
            check=True,
        )
        self.assertEqual(list_tasks(), list(TASKS))

        for task in list_tasks():
            with self.subTest(task=task):
                cfg, train = load_config(task)
                original_terrain = list(cfg.terrain.terrain_proportions)
                cfg.terrain.terrain_proportions.append(123)
                cfg.init_state.default_joint_angles["FL_hip_joint"] = 123
                group, widths = ("student", "channels") if task.startswith("es_wmp_") else ("policy", "actor_hidden_dims")
                original_widths = list(train[group][widths])
                train[group][widths].append(123)
                cfg, train = load_config(task)
                self.assertEqual(cfg.terrain.terrain_proportions, original_terrain)
                self.assertNotEqual(cfg.init_state.default_joint_angles["FL_hip_joint"], 123)
                self.assertEqual(train[group][widths], original_widths)
                self.assertEqual(type(cfg)().terrain.terrain_proportions, original_terrain)

    def test_himloco_paper_config_matches_upstream(self):
        with patch.object(sys, "path", [str(HIM / "isaaclab_migration"), *sys.path]):
            upstream = importlib.import_module("himloco_lab.legacy.a1.a1_config")
        expected = class_to_dict(upstream.A1RoughCfg())
        actual = class_to_dict(SHIMBaseCfg())
        for name in ("progressive_curriculum", "curriculum_init_x", "curriculum_init_y",
                     "curriculum_step", "curriculum_threshold"):
            del actual["commands"][name]
        self.assertEqual(actual, expected)
        original_train = class_to_dict(upstream.A1RoughCfgPPO())
        train = class_to_dict(SHIMBaseCfgPPO())
        self.assertEqual(train["seed"], original_train["seed"])
        self.assertEqual(train["runner_class_name"], original_train["runner_class_name"])
        for group in ("policy", "algorithm", "runner"):
            self.assertEqual({name: train[group][name] for name in original_train[group]}, original_train[group])

    def test_wmp_a1_mixed_matches_upstream_with_explicit_layout(self):
        with patch.object(sys, "path", [str(WMP / "isaaclab_migration"), *sys.path]):
            upstream = importlib.import_module("wmp_lab.a1_config")
        original = upstream.A1AMPCfg()
        expected = class_to_dict(original)
        cfg, _ = load_config("es_wmp_a1_mixed")
        paths = [Path(path) for path in cfg.env.amp_motion_files]
        self.assertTrue(paths)
        self.assertTrue(all(path.is_file() for path in paths))
        del expected["env"]["amp_motion_files"]
        actual = class_to_dict(cfg)
        del actual["env"]["amp_motion_files"], actual["terrain_name"], actual["reward_profile"]
        expected["terrain"].update(num_rows=10, num_cols=20, border_size=5.0)
        self.assertEqual(actual, expected)

    def test_registry_has_only_the_active_tasks_with_independent_blind_configs(self):
        blind_modules = {
            "s_him_a1_rough_cmdcur": "s_him_rough_cmdcur_config",
            "s_him_a1_spike_rough_cmdcur": "s_him_spike_rough_cmdcur_config",
            "himloco_a1_amp_rough_cmdcur": "himloco_ann_rough_cmdcur_config",
        }
        self.assertEqual(set(TASKS), blind_modules.keys() | {
            "es_wmp_a1_mixed", "es_wmp_a1_flat", "es_wmp_a1_parkour",
            "es_wmp_a1_parkour_soft", "es_wmp_a1_parkour_strict",
        })
        self.assertFalse((ROOT / "graduation/legacy/s_him/s_him_config.py").exists())
        for task, module_name in blind_modules.items():
            with self.subTest(task=task):
                cfg, _ = load_config(task)
                cls = type(cfg)
                self.assertEqual(cls.__module__, f"graduation.legacy.s_him.{module_name}")
                module = importlib.import_module(cls.__module__)
                ppo_cls = getattr(module, f"{cls.__name__}PPO")
                self.assertEqual(ppo_cls.__module__, cls.__module__)
                self.assertEqual(cls.__bases__, (SHIMBaseCfg,))
                self.assertEqual(ppo_cls.__bases__, (SHIMBaseCfgPPO,))

    def test_rough_command_curriculum_and_matching_training_defaults(self):
        cfg, train = load_config("s_him_a1_rough_cmdcur")
        spike, spike_train = load_config("s_him_a1_spike_rough_cmdcur")
        ann, ann_train = load_config("himloco_a1_amp_rough_cmdcur")
        self.assertEqual(class_to_dict(cfg.terrain), class_to_dict(SHIMBaseCfg().terrain))
        self.assertTrue(cfg.commands.progressive_curriculum)
        self.assertEqual((cfg.commands.curriculum_init_x, cfg.commands.curriculum_init_y), (0.2, 0.2))
        self.assertEqual(cfg.commands.ranges.lin_vel_x, [-2.0, 2.0])
        self.assertEqual(cfg.commands.ranges.lin_vel_y, [-1.0, 1.0])
        self.assertEqual((cfg.commands.curriculum_step, cfg.commands.curriculum_threshold), (0.2, 0.8))
        self.assertEqual(cfg.rewards.base_height_target, 0.30)
        self.assertEqual(class_to_dict(cfg.rewards.scales), dict(
            class_to_dict(SHIMBaseCfg().rewards.scales),
            tracking_lin_vel=2.5, tracking_ang_vel=1.0, lin_vel_z=-2.0, ang_vel_xy=-0.05,
            orientation=-0.2, dof_acc=-5e-8, joint_power=-5e-6, base_height=-1.0,
            foot_clearance=0.0, action_rate=0.0, smoothness=-0.01, stumble=-0.1, collision=-0.25, stand_still=-0.5,
        ))
        self.assertEqual(class_to_dict(cfg), class_to_dict(spike))
        self.assertEqual(class_to_dict(cfg), class_to_dict(ann))
        self.assertEqual(spike_train, dict(train, policy=dict(train["policy"], actor_readout_mode="spike")))
        base_train = class_to_dict(SHIMBaseCfgPPO())
        self.assertEqual(train["amp"], dict(
            base_train["amp"], enabled=True, motion_set="a1_diagonal_omni", obs_mode="wmp30",
            reward_coef=1.2, task_lerp=0.0, discr_lr=5e-5, discr_updates=2,
        ))
        self.assertEqual(ann_train, dict(
            train,
            policy=dict(base_train["policy"], method="himloco", max_action_std=0.55),
            estimator=base_train["estimator"],
            algorithm=dict(base_train["algorithm"], learning_rate=8e-4, aux_learning_rate=3e-4,
                           desired_kl=0.045, max_lr=8e-4, max_grad_norm=5.0, entropy_coef=0.002),
        ))

    def test_parkour_strict_inherits_layout_and_changes_only_three_scales(self):
        cfg, _ = load_config("es_wmp_a1_parkour_strict")
        soft, _ = load_config("es_wmp_a1_parkour_soft")
        self.assertIsInstance(cfg, ESWMPA1ParkourStrictCfg)
        self.assertIsInstance(cfg, ESWMPA1ParkourCfg)
        self.assertEqual((cfg.terrain.num_rows, cfg.terrain.num_cols, cfg.terrain.spawn_jitter_m), (6, 4, 0.15))
        self.assertEqual(cfg.terrain.terrain_proportions, [0.0] * 10 + [0.25] * 4)
        self.assertEqual(cfg.commands.ranges.lin_vel_x, [0.6, 1.0])
        expected_scales = dict(class_to_dict(soft.rewards.scales), collision=-10.0, feet_stumble=-1.0, feet_edge=-1.0)
        self.assertEqual(class_to_dict(cfg.rewards.scales), expected_scales)
        self.assertTrue(cfg.rewards.only_positive_rewards)
        self.assertFalse(cfg.rewards.reward_curriculum)

    def test_stage_and_env_overrides_are_local(self):
        cfg, _ = load_config("s_him_a1_rough_cmdcur", num_envs=7, stage=1)
        self.assertEqual(cfg.env.num_envs, 7)
        self.assertEqual(cfg.terrain.mesh_type, "plane")
        self.assertFalse(cfg.noise.add_noise)
        self.assertFalse(cfg.domain_rand.randomize_friction)
        self.assertFalse(cfg.commands.progressive_curriculum)
        cfg, _ = load_config("s_him_a1_rough_cmdcur")
        self.assertTrue(cfg.commands.progressive_curriculum)
        self.assertEqual(cfg.terrain.mesh_type, "trimesh")
        visual, _ = load_config("es_wmp_a1_mixed", num_envs=7)
        self.assertEqual((visual.env.num_envs, visual.depth.camera_num_envs), (7, 7))


class LegacyTaskEntryPointTest(unittest.TestCase):
    def setUp(self):
        launcher = patch.dict(sys.modules, {"isaaclab.app": SimpleNamespace(AppLauncher=Mock())})
        self.addCleanup(launcher.stop)
        launcher.start()

    def test_named_visual_cli_defaults_and_overrides(self):
        args = parse_wmp(["--task", "es_wmp_a1_parkour", "--teacher", "teacher.pt", "--out", "unused",
                          "--terrain", "flat", "--rollout-steps", "10"])
        self.assertEqual((args.terrain, args.num_envs, args.rollout_steps, args.warmup_iterations), ("flat", 32, 10, 2))
        args = parse_parkour(["distill", "--task", "es_wmp_a1_parkour_strict", "--teacher", "teacher.pt",
                              "--out", "unused", "--num-envs", "8"])
        self.assertEqual((args.terrain, args.reward_profile, args.num_envs, args.steps), ("parkour", "strict", 8, 25))

    def test_visual_env_binds_lazy_factories_and_keeps_reward_override_with_flat_terrain(self):
        terrain_factory = Mock()
        core_factory = Mock()
        fake_modules = {
            "wmp_lab.env": SimpleNamespace(WMPEnv=lambda cfg: cfg, Adapter=lambda env: env),
            "wmp_lab.env_cfg": SimpleNamespace(make_env_cfg=lambda legacy, *_: SimpleNamespace(legacy=legacy)),
            "graduation.sim.parkour": SimpleNamespace(ParkourTerrain=terrain_factory),
            "graduation.methods.es_parkour.task": SimpleNamespace(ParkourCore=core_factory),
        }
        args = SimpleNamespace(task="es_wmp_a1_parkour_strict", num_envs=2, device="cpu")
        with patch.dict(sys.modules, fake_modules):
            parkour, _, _ = make_visual_env(args)
            args.terrain = "flat"
            flat, _, _ = make_visual_env(args, reward_profile="soft")

        self.assertIs(parkour.legacy.terrain.terrain_factory, terrain_factory)
        self.assertIs(parkour.legacy.core_factory, core_factory)
        self.assertTrue(parkour.is_finite_horizon)
        self.assertEqual(parkour.legacy.depth.camera_num_envs, 2)
        self.assertEqual(flat.legacy.terrain.mesh_type, "plane")
        self.assertFalse(hasattr(flat.legacy.terrain, "terrain_factory"))
        self.assertEqual(flat.legacy.rewards.scales.collision, -1.0)
        self.assertFalse(flat.legacy.rewards.only_positive_rewards)
