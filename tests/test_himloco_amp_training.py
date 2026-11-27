import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from graduation.experiments.blind import run
from graduation.paths import HIM
from scripts.bench_blind import parse_args
from scripts.eval_shim_panel import load_policy


class FakeHIMEnv:
    device = "cpu"
    step_dt = 0.02
    num_one_step_obs = 45
    num_actions = 12

    def __init__(self, num_envs=2):
        self.num_envs = num_envs
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.tick = 0
        self.core = SimpleNamespace(
            root_states=torch.zeros(num_envs, 13), commands=torch.ones(num_envs, 3),
            base_lin_vel=torch.zeros(num_envs, 3), base_ang_vel=torch.zeros(num_envs, 3),
            dof_pos=torch.zeros(num_envs, 12), dof_vel=torch.zeros(num_envs, 12),
            terrain_levels=torch.ones(num_envs), episode_sums={"tracking": torch.zeros(num_envs)},
            compute_termination_observations=lambda ids: torch.full((len(ids), 238), 0.25),
        )

    def reset(self):
        return torch.zeros(self.num_envs, 225), torch.zeros(self.num_envs, 238)

    def step(self, action):
        self.tick += 1
        self.episode_length_buf.add_(1)
        self.core.dof_pos.add_(action * 0.01)
        self.core.base_lin_vel[:, 0] = 0.5
        self.core.root_states[:, 0].add_(0.01)
        reward = torch.full((self.num_envs,), 0.1)
        self.core.episode_sums["tracking"].add_(reward)
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        done[0] = self.tick % 2 == 0
        ids = done.nonzero(as_tuple=True)[0]
        terminal = self.core.compute_termination_observations(ids)
        self.episode_length_buf[ids] = 0
        self.core.episode_sums["tracking"][ids] = 0
        obs = torch.full((self.num_envs, 225), self.tick * 0.01)
        critic = torch.full((self.num_envs, 238), self.tick * 0.02)
        return obs, critic, reward, done, {"time_outs": done}, ids, terminal

    def close(self):
        pass


class FakeCurriculumEnv(FakeHIMEnv):
    def __init__(self, num_envs):
        super().__init__(num_envs)
        self.core.init_done = True
        self.core.terrain_levels = torch.ones(num_envs, dtype=torch.int32)
        self.core.max_terrain_level = 6
        self.core.terrain_origins = torch.arange(36).reshape(6, 2, 3).float()
        self.core.terrain_types = torch.arange(num_envs) % 2
        self.core.env_origins = torch.zeros(num_envs, 3)
        self.resets = []

    def reset(self):
        if self.core.init_done:
            self.core.terrain_levels.sub_(1).clamp_(min=0)
        self.core.root_states[:, :3].copy_(self.core.env_origins)
        self.resets.append((self.core.init_done, self.core.terrain_levels.clone(),
                            self.core.root_states[:, :3].clone()))
        return super().reset()


class HIMLocoAMPTrainingTest(unittest.TestCase):
    def parse(self, out, extra=()):
        argv = [
            "bench_blind.py", "himloco", "--amp", "--lr-schedule", "adaptive",
            "--desired-kl", "0.01", "--max-lr", "1e-3", "--out", str(out),
            "--terrain", "stairs", "--reward-preset", "snn-stair", "--amp-motion-set", "a1_trot",
            "--num-envs", "2", "--steps", "3", "--learning-rate", "1.1e-3", "--max-action-std", "0.5",
            "--checkpoint-interval", "1", "--amp-discr-updates", "1", "--amp-batch-size", "4",
            "--amp-replay-size", "16", "--amp-preload", "16", "--amp-discr-widths", "8", *extra,
        ]
        app = SimpleNamespace(AppLauncher=Mock())
        with patch.dict(sys.modules, {"isaaclab.app": app}), patch.object(sys, "argv", argv):
            args = parse_args()
        args.device = "cpu"
        return args

    def test_cli_accepts_himloco_amp_and_adaptive_schedule(self):
        args = self.parse(Path("unused"))
        self.assertEqual((args.method, args.amp, args.lr_schedule), ("himloco", True, "adaptive"))
        self.assertEqual((args.desired_kl, args.max_lr), (0.01, 1e-3))

    def test_checkpoint_terrain_curriculum(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            sys, "path", [str(HIM / "isaaclab_migration"), str(HIM / "rsl_rl"), *sys.path]
        ):
            from himloco_lab.config import load_config

            root = Path(tmp)
            env = None

            def make_env(args):
                nonlocal env
                env = FakeCurriculumEnv(args.num_envs)
                _, defaults = load_config("a1", args.num_envs, 4)
                defaults["algorithm"].update(num_learning_epochs=1, num_mini_batches=1)
                return env, env, defaults

            with patch("graduation.sim.himloco.make_blind_env", side_effect=make_env):
                run(self.parse(root / "stairs", ("--iterations", "1")))
                stairs = torch.load(root / "stairs" / "model.pt", weights_only=True)
                torch.testing.assert_close(stairs["terrain_levels"], env.core.terrain_levels)
                run(self.parse(root / "flat", ("--iterations", "1", "--terrain", "flat")))
                flat = torch.load(root / "flat" / "model.pt", weights_only=True)
                self.assertNotIn("terrain_levels", flat)

                # A deliberately out-of-range pair exercises clamping and tiling.
                stairs["terrain_levels"] = torch.tensor([-2, 10], dtype=torch.int64)
                checkpoint = root / "stairs.pt"
                torch.save(stairs, checkpoint)
                for num_envs, expected in ((2, [0, 5]), (5, [0, 5, 0, 5, 0])):
                    with self.subTest(num_envs=num_envs):
                        out = root / f"resumed_{num_envs}"
                        run(self.parse(out, ("--iterations", "1", "--num-envs", str(num_envs),
                                             "--checkpoint", str(checkpoint))))
                        self.assertEqual([reset[0] for reset in env.resets], [True, False])
                        self.assertTrue(env.core.init_done)
                        levels = torch.tensor(expected, dtype=torch.int32)
                        origins = env.core.terrain_origins[levels.long(), env.core.terrain_types]
                        torch.testing.assert_close(env.resets[-1][1], levels)
                        torch.testing.assert_close(env.resets[-1][2], origins)
                        torch.testing.assert_close(env.core.env_origins, origins)
                        resumed = torch.load(out / "model.pt", weights_only=True)
                        torch.testing.assert_close(resumed["terrain_levels"], levels)
                        self.assertEqual(resumed["iteration"], 2)

                legacy_out = root / "legacy_resume"
                run(self.parse(legacy_out, ("--iterations", "1", "--checkpoint", str(root / "flat" / "model.pt"))))
                self.assertEqual([reset[0] for reset in env.resets], [True])
                self.assertEqual(torch.load(legacy_out / "model.pt", weights_only=True)["iteration"], 2)

    def test_trains_saves_deployable_weights_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            sys, "path", [str(HIM / "isaaclab_migration"), str(HIM / "rsl_rl"), *sys.path]
        ):
            from himloco_lab.config import load_config

            _, defaults = load_config("a1", 2, 4)
            defaults["algorithm"].update(num_learning_epochs=1, num_mini_batches=1)
            out = Path(tmp) / "train"
            resume = Path(tmp) / "resume"

            def make_env(_):
                env = FakeHIMEnv()
                return env, env, defaults

            with patch("graduation.sim.himloco.make_blind_env", side_effect=make_env):
                run(self.parse(out, ("--iterations", "2")))
                run(self.parse(resume, ("--iterations", "1", "--checkpoint", str(out / "model.pt"))))

            first = torch.load(out / "model_000001.pt", weights_only=True)
            trained = torch.load(out / "model.pt", weights_only=True)
            resumed = torch.load(resume / "model.pt", weights_only=True)
            self.assertEqual((trained["method"], trained["iteration"], resumed["iteration"]), ("himloco", 2, 3))
            obs, critic = FakeHIMEnv().reset()
            deploy = SimpleNamespace(method="himloco", num_envs=2, checkpoint=resume / "model.pt")
            policy, iteration = load_policy(deploy, obs, critic, 45, 12, "cpu")
            self.assertEqual(iteration, 3)
            with torch.no_grad():
                self.assertTrue(torch.isfinite(policy(obs)).all())
            self.assertFalse(torch.equal(first["model_state_dict"]["actor.0.weight"],
                                         trained["model_state_dict"]["actor.0.weight"]))
            self.assertFalse(torch.equal(trained["model_state_dict"]["actor.0.weight"],
                                         resumed["model_state_dict"]["actor.0.weight"]))
            for before, after in (
                (trained["optimizer"], resumed["optimizer"]),
                (trained["estimator_optimizer"], resumed["estimator_optimizer"]),
                (trained["amp"]["optimizer"], resumed["amp"]["optimizer"]),
            ):
                self.assertEqual(max(state["step"].item() for state in before["state"].values()), 2)
                self.assertEqual(max(state["step"].item() for state in after["state"].values()), 3)
            self.assertGreater(resumed["amp"]["normalizer"]["count"], trained["amp"]["normalizer"]["count"])
            self.assertLessEqual(resumed["optimizer"]["param_groups"][0]["lr"], 1e-3)
            self.assertLessEqual(float(resumed["model_state_dict"]["std"].max()), 0.5)

            results = json.loads((out / "validation.json").read_text())["results"]
            self.assertEqual([row["iteration"] for row in results], [1, 2])
            for row in results:
                self.assertEqual(row["samples_per_iteration"], 6)
                self.assertEqual(row["terrain_level"], 1.0)
                self.assertGreater(row["resets"], 0)
                self.assertAlmostEqual(row["reward_per_second"]["tracking"], 5.0, places=5)
                for metric in ("tracking_rmse_mps", "command_aligned_velocity_mps", "velocity_loss",
                               "swap_loss", "ppo_loss", "action_std", "iteration_seconds", "amp_loss"):
                    self.assertTrue(math.isfinite(row[metric]), metric)
                self.assertAlmostEqual(row["mean_reward"],
                                       row["task_reward_mean"] + FakeHIMEnv.step_dt * row["amp_style_reward_per_sec"])
