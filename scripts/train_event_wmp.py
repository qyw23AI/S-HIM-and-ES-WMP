#!/usr/bin/env python3
"""Train Event-Spiking-WMP with frozen WMP labels and closed-loop rollouts."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graduation.experiments.event_wmp import run
from graduation.legacy import list_tasks, load_config
from graduation.legacy.cli import apply_defaults, parse_overrides, visual_defaults
from graduation.paths import WMP
from graduation.sim.launch import launch


def parse_args(argv=None):
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--warmup-iterations", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--episode-seconds", type=float)
    parser.add_argument("--terrain", choices=("parkour", "mixed", "flat"))
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--channels", type=int, nargs="+")
    parser.add_argument("--memory-width", type=int)
    parser.add_argument("--memory", choices=("grsn", "lif", "gru"))
    parser.add_argument("--normalization", choices=("care", "tdbn", "none"))
    parser.add_argument("--world-weight", type=float)
    parser.add_argument("--cosine-weight", type=float)
    parser.add_argument("--action-path", choices=("end-to-end", "actor"))
    parser.add_argument("--calibration-interval", type=int)
    parser.add_argument("--task", choices=[task for task in list_tasks() if task.startswith("es_wmp_")])
    AppLauncher.add_app_launcher_args(parser)
    task = "es_wmp_a1_mixed"
    defaults = visual_defaults(*load_config(task))
    args = parse_overrides(parser, defaults, argv)
    apply_defaults(args, visual_defaults(*load_config(args.task)) if args.task else defaults)
    if min(args.num_envs, args.iterations, args.rollout_steps, args.memory_width,
           args.calibration_interval, *args.channels) <= 0:
        parser.error("counts and widths must be positive")
    if min(args.world_weight, args.cosine_weight, args.warmup_iterations) < 0:
        parser.error("loss weights and warm-up iterations must be nonnegative")
    return args


def main():
    sys.path[:0] = [str(WMP), str(WMP / "isaaclab_migration")]
    from isaaclab.app import AppLauncher

    launch(AppLauncher, parse_args(), run)


if __name__ == "__main__":
    main()
