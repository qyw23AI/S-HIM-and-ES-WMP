#!/usr/bin/env python3
"""Flat-ground walk/directional checks and staged two-flight stair milestones for S-HIM and HIMLoco."""

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graduation.paths import HIM
from graduation.sim.launch import launch


def wilson_ci(successes, total, z=1.96):
    if total <= 0:
        return [0.0, 0.0]
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def load_policy(args, obs, critic, width, actions, device):
    import torch

    cfg = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if args.method == "himloco":
        from graduation.legacy import class_to_dict
        from graduation.legacy.base.s_him_base_config import SHIMBaseCfgPPO
        from graduation.sim.himloco import himloco_policy_kwargs
        from rsl_rl.modules.him_actor_critic import HIMActorCritic

        defaults = class_to_dict(SHIMBaseCfgPPO())
        baseline = HIMActorCritic(obs.shape[1], critic.shape[1], width, actions, **himloco_policy_kwargs(defaults)).to(device).eval()
        baseline.load_state_dict(cfg["model_state_dict"])
        return baseline.act_inference, int(cfg.get("iteration", cfg.get("iter", -1)))
    from graduation.methods.s_him.policy import SHIM

    if cfg.get("method") != "s-him":
        raise ValueError("Expected an S-HIM checkpoint")
    options = dict(cfg.get("model_config", {}))
    options["recurrent"] = options.pop("estimator", "grsn") == "grsn"
    policy = SHIM(width, critic.shape[1], actions, **options).to(device).eval()
    policy.load_state_dict(cfg["model"])
    return policy, int(cfg.get("iteration", cfg.get("iter", -1)))


def make_flat_env(args):
    from himloco_lab.adapter import LegacyRunnerAdapter
    from graduation.legacy.s_him.evaluation_config import SHIMA1FlatEvalCfg
    from himloco_lab.env import HIMLabEnv
    from himloco_lab.env_cfg import make_env_cfg

    class FlatEnv(HIMLabEnv):
        def _get_dones(self):
            terminated, timeout = super()._get_dones()
            self.pre_reset_position = self.core.root_states[:, :3].clone()
            return terminated, timeout

    legacy = SHIMA1FlatEvalCfg()
    legacy.env.num_envs = args.num_envs
    cfg = make_env_cfg(legacy, 4, args.device)
    cfg.seed = args.seed
    env = FlatEnv(cfg)
    return env, LegacyRunnerAdapter(env)


def run_flat_panel(args):
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env, adapter = make_flat_env(args)
    try:
        obs, critic = adapter.reset()
        width = adapter.num_one_step_obs
        policy, iteration = load_policy(args, obs, critic, width, adapter.num_actions, env.device)
        commands = [float(x) for x in args.flat_commands]
        panel_report = {"method": args.method, "checkpoint": str(args.checkpoint), "iteration": iteration,
                        "seed": args.seed, "trials": args.num_envs, "steps": args.flat_steps, "commands": {}}
        for cmd_x in commands:
            obs, _ = adapter.reset()
            start_pos = env.core.root_states[:, :2].clone()
            q = env.core.root_states[:, 3:7]
            yaw0 = torch.atan2(2.0 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
                               1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2))
            heading0 = torch.stack((torch.cos(yaw0), torch.sin(yaw0)), dim=-1)
            outcome = torch.zeros(args.num_envs, device=env.device, dtype=torch.int64)
            fail_steps = torch.full((args.num_envs,), args.flat_steps, device=env.device, dtype=torch.int64)
            net_forward = torch.zeros(args.num_envs, device=env.device)
            vel_sum = torch.zeros(args.num_envs, device=env.device)
            active_steps = torch.zeros(args.num_envs, device=env.device)

            def observe(raw_obs):
                env.core.commands[:, 0] = cmd_x
                env.core.commands[:, 1] = 0.0
                env.core.commands[:, 2] = 0.0
                env.core.commands[:, 3] = 0.0
                frame = raw_obs[:, :width].clone()
                frame[:, :3] = env.core.commands[:, :3] * env.core.commands_scale
                return frame

            frame = observe(obs)
            history = frame[:, None].repeat(1, obs.shape[1] // width, 1)
            with torch.no_grad():
                for tick in range(1, args.flat_steps + 1):
                    action = policy(history.flatten(1))
                    # Read pre-step body velocity, then step simulator.
                    body_vx = env.core.base_lin_vel[:, 0].clone()
                    obs, _, _, done, extras, *_ = adapter.step(action)
                    pending = outcome == 0
                    vel_sum[pending] += body_vx[pending]
                    active_steps[pending] += 1.0
                    delta_xy = env.pre_reset_position[:, :2] - start_pos
                    proj = (delta_xy * heading0).sum(-1)
                    net_forward[pending] = proj[pending]
                    failed = done & ~extras["time_outs"]
                    timeout = done & extras["time_outs"]
                    newly_failed = pending & failed
                    fail_steps[newly_failed] = tick
                    outcome[newly_failed] = 2
                    outcome[pending & timeout] = 3
                    if bool((outcome != 0).all()):
                        break
                    frame = observe(obs)
                    history = torch.cat((frame[:, None], history[:, :-1]), dim=1)
                    history[done] = frame[done, None]

            mean_window_vx = (vel_sum / args.flat_steps).tolist()
            mean_active_vx = (vel_sum / active_steps.clamp_min(1.0)).tolist()
            signed_forward = net_forward.tolist()
            labels = ("window_censored", "success", "contact_fail", "timeout")
            ep_rows = []
            for i in range(args.num_envs):
                ep_rows.append({
                    "outcome": labels[int(outcome[i])],
                    "net_forward_m": signed_forward[i],
                    "window_mean_body_vx_mps": mean_window_vx[i],
                    "active_mean_body_vx_mps": mean_active_vx[i],
                    "active_steps": int(active_steps[i]),
                    "failure_time_s": None if int(outcome[i]) != 2 else float(fail_steps[i] * env.step_dt),
                })
            pass_1m = sum(1 for v in signed_forward if v >= 1.0)
            median_window_vx = float(statistics.median(mean_window_vx))
            panel_report["commands"][str(cmd_x)] = {
                "pass_1m_count": pass_1m,
                "pass_1m_rate": pass_1m / args.num_envs,
                "pass_1m_wilson_95ci": wilson_ci(pass_1m, args.num_envs),
                "median_window_body_vx_mps": median_window_vx,
                "mean_net_forward_m": float(sum(signed_forward) / len(signed_forward)),
                "contact_failures": int((outcome == 2).sum()),
                "walk_gate_passed": (pass_1m >= math.ceil(0.75 * args.num_envs) and median_window_vx >= 0.25) if cmd_x > 0 else None,
                "episodes": ep_rows,
            }
        return panel_report
    finally:
        env.close()


def run_stairs_panel(args, step_height):
    import numpy as np
    import torch
    from graduation.sim.shim_stairs import FINISH_M, LANE_HALF_WIDTH_M, SPAWN_M, STARTS_M, STEPS_PER_FLIGHT, TREAD_M, make_stair_env

    stair_args = argparse.Namespace(**vars(args), step_height=step_height)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env, adapter = make_stair_env(stair_args)
    try:
        obs, critic = adapter.reset()
        width = adapter.num_one_step_obs
        policy, iteration = load_policy(args, obs, critic, width, adapter.num_actions, env.device)
        terrain_origin = env.core.env_origins.clone()
        spawn_world = env.core.root_states[:, :2].clone()
        progress = torch.zeros(args.num_envs, device=env.device)
        outcome = torch.zeros_like(progress, dtype=torch.int64)
        finish_steps = torch.zeros_like(outcome)
        forward_displacement = torch.zeros_like(progress)
        max_height = env.core.root_states[:, 2].clone()
        # History buffer for +-0.5s around first failure
        trace_steps = []

        def observe(raw_obs):
            env.core.commands[:, 0] = args.forward_speed
            env.core.commands[:, 1] = 0.0
            env.core.commands[:, 2] = 0.0
            env.core.commands[:, 3] = 0.0
            frame = raw_obs[:, :width].clone()
            frame[:, :3] = env.core.commands[:, :3] * env.core.commands_scale
            return frame

        frame = observe(obs)
        history = frame[:, None].repeat(1, obs.shape[1] // width, 1)
        with torch.no_grad():
            for tick in range(1, args.steps + 1):
                action = policy(history.flatten(1))
                # capture pre-step orientation/foot state
                q = env.core.base_quat
                sinp = (2.0 * (q[:, 3] * q[:, 1] - q[:, 2] * q[:, 0])).clamp(-1.0, 1.0)
                pitch = torch.asin(sinp)
                body_vx = env.core.base_lin_vel[:, 0].clone()
                min_foot_z = env.core.feet_pos[:, :, 2].min(dim=1).values.clone()
                foot_fz = env.core.contact_forces[:, env.core.feet_indices, 2].max(dim=1).values.clone()

                obs, _, _, done, extras, *_ = adapter.step(action)
                position = env.pre_reset_position - terrain_origin
                x = position[:, 0] + SPAWN_M
                pending = outcome == 0
                forward_displacement[pending] = env.pre_reset_position[pending, 0] - spawn_world[pending, 0]
                max_height[pending] = torch.maximum(max_height[pending], env.pre_reset_position[pending, 2])
                progress[pending] = torch.maximum(progress[pending], x[pending])
                # Exclusive classification priority: contact_fail > lateral_exit > success > timeout
                contact_fail = done & ~extras["time_outs"]
                lateral_exit = position[:, 1].abs() > LANE_HALF_WIDTH_M
                goal = x >= FINISH_M
                outcome[pending & contact_fail] = 2
                outcome[pending & ~contact_fail & lateral_exit] = 4
                outcome[pending & ~contact_fail & ~lateral_exit & goal] = 1
                outcome[pending & ~contact_fail & ~lateral_exit & ~goal & done] = 3
                finish_steps[pending] = tick

                if len(trace_steps) < args.steps:
                    trace_steps.append({
                        "tick": tick,
                        "x": position[:, 0].cpu().numpy() + SPAWN_M,
                        "z": env.pre_reset_position[:, 2].cpu().numpy(),
                        "vx": body_vx.cpu().numpy(),
                        "pitch_rad": pitch.cpu().numpy(),
                        "min_foot_z": min_foot_z.cpu().numpy(),
                        "max_foot_fz": foot_fz.cpu().numpy(),
                        "act_rms": action.square().mean(-1).sqrt().cpu().numpy(),
                    })
                if bool((outcome != 0).all()):
                    break
                frame = observe(obs)
                history = torch.cat((frame[:, None], history[:, :-1]), dim=1)
                history[done] = frame[done, None]

        flight1_top = STARTS_M[0] + STEPS_PER_FLIGHT * TREAD_M  # 2.85 m
        flight2_start = STARTS_M[1]  # 3.60 m
        reached_step1 = int((progress >= STARTS_M[0]).sum())
        reached_f1_top = int((progress >= flight1_top).sum())
        reached_f2_start = int((progress >= flight2_start).sum())
        successes = int((outcome == 1).sum())

        def safe_cond(num, den):
            return None if den == 0 else num / den

        labels = ("window_censored", "success", "contact_fail", "timeout", "lateral_exit")
        failure_snippets = []
        half_win = max(1, int(round(0.5 / env.step_dt)))
        for idx in range(args.num_envs):
            if outcome[idx].item() in (2, 4) and len(failure_snippets) < 2:
                f_step = int(finish_steps[idx])
                lo = max(0, f_step - 1 - half_win)
                hi = min(len(trace_steps), f_step + half_win)
                failure_snippets.append({
                    "env": idx,
                    "outcome": labels[int(outcome[idx])],
                    "failure_step": f_step,
                    "furthest_course_x_m": float(progress[idx]),
                    "window": [{k: float(row[k][idx]) if k != "tick" else int(row[k]) for k in row}
                               for row in trace_steps[lo:hi]],
                })

        return {
            "method": args.method,
            "checkpoint": str(args.checkpoint),
            "iteration": iteration,
            "seed": args.seed,
            "step_height_m": step_height,
            "trials": args.num_envs,
            "steps": args.steps,
            "forward_speed_mps": args.forward_speed,
            "successes": successes,
            "success_rate": successes / args.num_envs,
            "success_wilson_95ci": wilson_ci(successes, args.num_envs),
            "milestones": {
                "first_step_1p50m": {"count": reached_step1, "cum_rate": reached_step1 / args.num_envs},
                "flight1_top_2p85m": {"count": reached_f1_top, "cum_rate": reached_f1_top / args.num_envs,
                                      "cond_rate": safe_cond(reached_f1_top, reached_step1)},
                "flight2_start_3p60m": {"count": reached_f2_start, "cum_rate": reached_f2_start / args.num_envs,
                                        "cond_rate": safe_cond(reached_f2_start, reached_f1_top)},
                "finish_5p13m": {"count": successes, "cum_rate": successes / args.num_envs,
                                 "cond_rate": safe_cond(successes, reached_f2_start)},
            },
            "outcomes": {
                "success": int((outcome == 1).sum()),
                "contact_fail": int((outcome == 2).sum()),
                "lateral_exit": int((outcome == 4).sum()),
                "timeout": int((outcome == 3).sum()),
                "window_censored": int((outcome == 0).sum()),
            },
            "failure_windows": failure_snippets,
            "episodes": [{"outcome": labels[c], "furthest_course_x_m": x, "steps": s,
                          "net_forward_m": d, "max_base_height_m": h}
                         for c, x, s, d, h in zip(
                             outcome.tolist(), progress.tolist(), finish_steps.tolist(),
                             forward_displacement.tolist(), max_height.tolist())],
        }
    finally:
        env.close()


def run(args):
    report = {}
    if args.panel == "flat":
        report["flat"] = run_flat_panel(args)
    elif args.panel == "stairs":
        report["stairs"] = [run_stairs_panel(args, args.step_heights[0])]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    summary = {}
    if "flat" in report:
        summary["flat"] = {cmd: {k: v for k, v in vals.items() if k != "episodes"}
                           for cmd, vals in report["flat"]["commands"].items()}
    if "stairs" in report:
        summary["stairs"] = [{k: v for k, v in item.items() if k not in ("episodes", "failure_windows")}
                             for item in report["stairs"]]
    print(json.dumps(summary, indent=2), flush=True)


def main():
    import subprocess
    import tempfile

    sys.path[:0] = [str(HIM / "isaaclab_migration"), str(HIM / "rsl_rl")]
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("s-him", "himloco"), default="s-him")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--panel", choices=("flat", "stairs", "all"), default="flat")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--flat-steps", type=int, default=250)
    parser.add_argument("--flat-commands", type=float, nargs="+", default=[0.6, 0.0, -0.6])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--step-heights", type=float, nargs="+", default=[0.16])
    parser.add_argument("--forward-speed", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=21)
    AppLauncher.add_app_launcher_args(parser)
    args, unknown = parser.parse_known_args()
    # Isaac Lab permits only one env lifecycle per SimulationApp; fan out multi-env panels cleanly.
    if args.panel == "all" or (args.panel == "stairs" and len(args.step_heights) > 1):
        merged = {}
        extra_flags = ["--headless"] if args.headless else []
        extra_flags += ["--device", args.device] if getattr(args, "device", None) else []
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            if args.panel in ("flat", "all"):
                flat_out = tmp_dir / "flat.json"
                subprocess.run([
                    sys.executable, __file__, *extra_flags, "--method", args.method,
                    "--checkpoint", str(args.checkpoint), "--out", str(flat_out),
                    "--panel", "flat", "--num-envs", str(args.num_envs),
                    "--flat-steps", str(args.flat_steps),
                    "--flat-commands", *[str(x) for x in args.flat_commands],
                    "--seed", str(args.seed),
                ], check=True)
                merged.update(json.loads(flat_out.read_text()))
            if args.panel in ("stairs", "all"):
                merged["stairs"] = []
                for idx, h in enumerate(args.step_heights):
                    st_out = tmp_dir / f"stairs_{idx}.json"
                    subprocess.run([
                        sys.executable, __file__, *extra_flags, "--method", args.method,
                        "--checkpoint", str(args.checkpoint), "--out", str(st_out),
                        "--panel", "stairs", "--num-envs", str(args.num_envs),
                        "--steps", str(args.steps), "--step-heights", str(h),
                        "--forward-speed", str(args.forward_speed), "--seed", str(args.seed),
                    ], check=True)
                    merged["stairs"].extend(json.loads(st_out.read_text())["stairs"])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(merged, indent=2) + "\n")
        return

    args = parser.parse_args()
    launch(AppLauncher, args, run)


if __name__ == "__main__":
    main()
