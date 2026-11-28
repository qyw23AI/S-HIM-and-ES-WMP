#!/usr/bin/env python3
"""Train and evaluate ES-Parkour, or record frozen WMP A1 event rollouts."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graduation.experiments.parkour import run
from graduation.legacy import list_tasks, load_config
from graduation.legacy.cli import apply_defaults, parse_overrides, visual_defaults
from graduation.paths import WMP
from graduation.sim.launch import launch


def parse_args(argv=None):
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("teacher", "distill", "eval", "wmp-events"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--terrain", choices=("parkour", "mixed", "flat"))
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--warmup-iterations", type=int)
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--student-normalization", choices=("batch", "tdbn", "care"))
    parser.add_argument("--student-threshold", type=float)
    parser.add_argument("--student-residual", choices=("if", "sew-add"))
    parser.add_argument("--visual-response-probe", action="store_true")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--lr", type=float, help="Student distillation learning rate")
    parser.add_argument("--episode-seconds", type=float)
    parser.add_argument("--reward-profile", choices=("soft", "strict"))
    parser.add_argument("--energy", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--task", choices=[task for task in list_tasks() if task.startswith("es_wmp_")])
    AppLauncher.add_app_launcher_args(parser)
    defaults = visual_defaults(*load_config("es_wmp_a1_parkour_soft"), parkour=True)
    args = parse_overrides(parser, defaults, argv)
    architecture = ("student_normalization", "student_threshold", "student_residual")
    architecture_override = any(hasattr(args, name) for name in architecture)
    if args.task:
        defaults = visual_defaults(*load_config(args.task), parkour=True)
    elif args.mode == "wmp-events":
        defaults["terrain"] = "mixed"
        args.reward_profile = None
    apply_defaults(args, defaults)
    if args.mode in ("eval", "wmp-events") and args.checkpoint is None:
        parser.error("evaluation and frozen WMP require --checkpoint")
    if args.mode == "distill" and args.teacher is None:
        parser.error("distill requires --teacher")
    if min(args.steps, args.iterations, args.num_envs, args.base_channels) < 1 or min(args.episode_seconds, args.lr) <= 0:
        parser.error("counts, channels, episode length and learning rate must be positive")
    if args.student_threshold is not None and not 0 < args.student_threshold < float("inf"):
        parser.error("--student-threshold must be positive and finite")
    if architecture_override:
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
