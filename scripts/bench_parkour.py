#!/usr/bin/env python3
"""Train and evaluate ES-Parkour, or record frozen WMP A1 event rollouts."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graduation.experiments.parkour import run
from graduation.paths import WMP
from graduation.sim.launch import launch


def parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("teacher", "distill", "eval", "wmp-events"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--terrain", choices=("parkour", "mixed", "flat"), default=None)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--student-normalization", choices=("batch", "tdbn", "care"))
    parser.add_argument("--student-threshold", type=float)
    parser.add_argument("--student-residual", choices=("if", "sew-add"))
    parser.add_argument("--visual-response-probe", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.08)
    parser.add_argument("--episode-seconds", type=float, default=20)
    parser.add_argument("--reward-profile", choices=("soft", "strict"), default="soft")
    parser.add_argument("--energy", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.terrain is None:
        args.terrain = "mixed" if args.mode == "wmp-events" else "parkour"
    if args.mode in ("eval", "wmp-events") and args.checkpoint is None:
        parser.error("evaluation and frozen WMP require --checkpoint")
    if args.mode == "distill" and args.teacher is None:
        parser.error("distill requires --teacher")
    if min(args.steps, args.iterations, args.num_envs, args.base_channels) < 1 or args.episode_seconds <= 0:
        parser.error("counts, channels and episode length must be positive")
    if args.student_threshold is not None and not 0 < args.student_threshold < float("inf"):
        parser.error("--student-threshold must be positive and finite")
    if any(item is not None for item in (args.student_normalization, args.student_threshold, args.student_residual)):
        if args.mode != "distill" or args.checkpoint:
            parser.error("Student architecture switches require fresh distillation")
    if args.visual_response_probe and args.mode not in ("distill", "eval"):
        parser.error("--visual-response-probe requires distill or eval")
    if args.energy and args.mode not in ("eval", "wmp-events"):
        parser.error("--energy requires eval or wmp-events")
    return args


def main():
    sys.path[:0] = [str(WMP), str(WMP / "isaaclab_migration")]
    from isaaclab.app import AppLauncher

    args = parse_args()
    launch(AppLauncher, args, run)


if __name__ == "__main__":
    main()
