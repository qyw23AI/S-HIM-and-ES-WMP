#!/usr/bin/env python3
"""Evaluate one blind S-HIM checkpoint on two flights of five stairs."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graduation.paths import HIM
from graduation.sim.launch import launch


def run(args):
    import random

    import numpy as np
    import torch

    from graduation.methods.s_him.policy import SHIM
    from graduation.sim.shim_stairs import FINISH_M, LANE_HALF_WIDTH_M, SPAWN_M, make_stair_env

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env, adapter = make_stair_env(args)
    try:
        obs, critic = adapter.reset()
        cfg = torch.load(args.checkpoint, map_location=env.device, weights_only=True)
        width = adapter.num_one_step_obs
        if args.method == "himloco":
            from graduation.legacy import class_to_dict
            from graduation.legacy.base.s_him_base_config import SHIMBaseCfgPPO
            from graduation.sim.himloco import himloco_policy_kwargs
            from rsl_rl.modules.him_actor_critic import HIMActorCritic

            defaults = class_to_dict(SHIMBaseCfgPPO())
            baseline = HIMActorCritic(obs.shape[1], critic.shape[1], width, adapter.num_actions,
                                     **himloco_policy_kwargs(defaults)).to(env.device).eval()
            baseline.load_state_dict(cfg["model_state_dict"])
            policy = baseline.act_inference
        else:
            if cfg["method"] != "s-him":
                raise ValueError("Expected a S-HIM checkpoint")
            options = dict(cfg.get("model_config", {}))
            options["recurrent"] = options.pop("estimator", "grsn") == "grsn"
            policy = SHIM(width, critic.shape[1], adapter.num_actions, **options).to(env.device).eval()
            policy.load_state_dict(cfg["model"])
        terrain_origin = env.core.env_origins.clone()
        progress = torch.zeros(args.num_envs, device=env.device)
        outcome = torch.zeros_like(progress, dtype=torch.int64)
        finish_steps = torch.zeros_like(outcome)
        forward_displacement = torch.zeros_like(progress)
        previous_x = env.core.root_states[:, 0].clone()
        max_height = env.core.root_states[:, 2].clone()
        initial_action = None

        def observe(obs):
            env.core.commands[:, 0] = args.forward_speed
            obs = obs[:, :width].clone()
            obs[:, :3] = env.core.commands[:, :3] * env.core.commands_scale
            return obs

        frame = observe(obs)
        history = frame[:, None].repeat(1, obs.shape[1] // width, 1)
        with torch.no_grad():
            if args.diagnostics:
                from graduation.evaluation.energy import SynapticMeter
                from graduation.methods.s_him.policy import shim_spike_inputs

                zero_command = history.clone()
                zero_command[..., :3] = 0
                output_rates = []
                handle = policy.actor.readout.register_forward_pre_hook(
                    lambda _, inputs: output_rates.append(float(inputs[0].mean())))
                try:
                    with SynapticMeter(policy, shim_spike_inputs(policy)) as meter:
                        moving = policy(history.flatten(1))
                finally:
                    handle.remove()
                stationary = policy(zero_command.flatten(1))
                layer_stats = meter.report(args.num_envs, 1 / env.step_dt)["layers"]
                initial_action = {
                    "actor_population_firing_rate": output_rates[0],
                    "command_mean_action_abs_delta": float((moving - stationary).abs().mean()),
                    "command_max_action_abs_delta": float((moving - stationary).abs().max()),
                    "actor_pulse_inputs": {name: stats["input_firing_rate"] for name, stats in layer_stats.items()
                                           if name in ("actor.layers.1", "actor.layers.2")},
                }
            for tick in range(1, args.steps + 1):
                action = policy(history.flatten(1))
                obs, _, _, done, extras, *_ = adapter.step(action)
                position = env.pre_reset_position - terrain_origin
                x = position[:, 0] + SPAWN_M
                pending = outcome == 0
                forward_displacement[pending] += env.pre_reset_position[pending, 0] - previous_x[pending]
                previous_x = env.core.root_states[:, 0].clone()
                max_height[pending] = torch.maximum(max_height[pending], env.pre_reset_position[pending, 2])
                progress[pending] = torch.maximum(progress[pending], x[pending])
                failed = (done & ~extras["time_outs"]) | (position[:, 1].abs() > LANE_HALF_WIDTH_M)
                goal = x >= FINISH_M
                outcome[pending & failed] = 2
                outcome[pending & goal & ~failed] = 1
                outcome[pending & done & ~failed & ~goal] = 3
                finish_steps[pending] = tick
                if bool((outcome != 0).all()):
                    break
                frame = observe(obs)
                history = torch.cat((frame[:, None], history[:, :-1]), dim=1)
                history[done] = frame[done, None]
        labels = ("window_censored", "success", "failed", "timeout")
        result = {"method": args.method, "checkpoint": str(args.checkpoint),
                  "iteration": int(cfg.get("iteration", cfg.get("iter", -1))), "seed": args.seed,
                  "step_height_m": args.step_height, "step_depth_m": .27, "stairs": 10, "finish_x_m": FINISH_M,
                  "forward_speed_mps": args.forward_speed, "trials": args.num_envs,
                  "successes": int((outcome == 1).sum()), "success_rate": float((outcome == 1).float().mean()),
                  "episodes": [{"outcome": labels[code], "furthest_course_x_m": x, "steps": steps,
                                "net_forward_m": displacement, "max_base_height_m": height}
                               for code, x, steps, displacement, height in zip(
                                   outcome.tolist(), progress.tolist(), finish_steps.tolist(),
                                   forward_displacement.tolist(), max_height.tolist())]}
        if initial_action is not None:
            result["initial_action_diagnostics"] = initial_action
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print({key: value for key, value in result.items() if key != "episodes"}, flush=True)
    finally:
        env.close()


def main():
    sys.path[:0] = [str(HIM / "isaaclab_migration"), str(HIM / "rsl_rl")]
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("s-him", "himloco"), default="s-him")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--step-height", type=float, default=.16)
    parser.add_argument("--forward-speed", type=float, default=.6)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--diagnostics", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.steps, args.step_height, args.forward_speed) <= 0:
        parser.error("environment count, steps, height and speed must be positive")
    if args.diagnostics and args.method != "s-him":
        parser.error("--diagnostics requires s-him")
    launch(AppLauncher, args, run)


if __name__ == "__main__":
    main()
