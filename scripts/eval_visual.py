#!/usr/bin/env python3
"""Evaluate WMP, Event-Spiking-WMP, ES-Parkour and S-HIM on one A1 course contract."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graduation.experiments.visual_suite import run
from graduation.legacy import load_config
from graduation.paths import HIM, WMP
from graduation.sim.launch import launch



def main():
    sys.path[:0] = [str(WMP), str(WMP / "isaaclab_migration")]
    from isaaclab.app import AppLauncher

    cfg, train = load_config("es_wmp_a1_mixed")
    runner = train["runner"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("wmp", "es-parkour", "s-him", "event-spiking-wmp"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--purpose", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--terrain", choices=("parkour", "mixed", "flat"), default=cfg.terrain_name)
    parser.add_argument("--num-envs", type=int, default=runner["num_envs"])
    parser.add_argument("--steps", type=int, default=runner["rollout_steps"])
    parser.add_argument("--episode-seconds", type=float, default=runner["episode_seconds"])
    parser.add_argument("--goal-distance", type=float, default=3.)
    parser.add_argument("--threshold", type=float, default=train["distillation"]["threshold"])
    parser.add_argument("--flow-mode", choices=("random", "analytical"), default="random")
    parser.add_argument("--mask-events-on-pause", action="store_true")
    parser.add_argument("--pause-trigger-distance", type=float, help="Approach distance to the first parkour obstacle")
    parser.add_argument("--stop-speed", type=float, default=.15, help="Planar speed limit in m/s for the hold")
    parser.add_argument("--memory-control", choices=("checkpoint", "zero-state"), default="checkpoint")
    parser.add_argument("--legacy-actor-projection", action="store_true")
    parser.add_argument("--delay-frames", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1, help="Retain every Nth native camera frame")
    parser.add_argument("--blur", type=int, default=1)
    parser.add_argument("--pause-at", type=int, default=5, help="Trigger step or earliest distance-trigger step")
    parser.add_argument("--pause-steps", type=int, default=0, help="Consecutive control steps inside the stop speed band")
    parser.add_argument("--resume-distance", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=train["seed"])
    parser.add_argument("--energy", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.steps, args.num_envs, args.episode_seconds, args.goal_distance, args.threshold, args.resume_distance, args.frame_stride) <= 0:
        parser.error("counts, durations and thresholds must be positive")
    if args.blur < 1 or args.blur % 2 == 0 or min(args.delay_frames, args.pause_steps, args.pause_at) < 0:
        parser.error("use an odd blur width and nonnegative delays")
    if args.pause_steps and args.pause_at + args.pause_steps >= args.steps:
        parser.error("pause validation requires a subsequent observation interval")
    if not 0 < args.stop_speed < float("inf"):
        parser.error("--stop-speed must be positive and finite")
    if args.pause_trigger_distance is not None and (
            args.terrain != "parkour" or not args.pause_steps or not 0 < args.pause_trigger_distance < float("inf")):
        parser.error("Distance-triggered stopping requires parkour terrain, a positive distance and --pause-steps")
    if args.mask_events_on_pause and (not args.pause_steps or args.method not in ("es-parkour", "event-spiking-wmp")):
        parser.error("--mask-events-on-pause requires an event policy and --pause-steps")
    if args.memory_control != "checkpoint" and args.method != "event-spiking-wmp":
        parser.error("--memory-control applies to event-spiking-wmp")
    if args.method == "s-him":
        sys.path.append(str(HIM / "isaaclab_migration"))
    launch(AppLauncher, args, run)


if __name__ == "__main__":
    main()
