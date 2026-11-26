#!/usr/bin/env python3
"""Train Event-Spiking-WMP with frozen WMP labels and closed-loop rollouts."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graduation.experiments.event_wmp import run
from graduation.paths import WMP
from graduation.sim.launch import launch



def main():
    sys.path[:0] = [str(WMP), str(WMP / "isaaclab_migration")]
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--episode-seconds", type=float, default=20)
    parser.add_argument("--terrain", choices=("parkour", "mixed", "flat"), default="mixed")
    parser.add_argument("--threshold", type=float, default=.08)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--channels", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--memory-width", type=int, default=128)
    parser.add_argument("--memory", choices=("grsn", "lif", "gru"), default="grsn")
    parser.add_argument("--normalization", choices=("care", "tdbn", "none"), default="care")
    parser.add_argument("--world-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=.1)
    parser.add_argument("--action-path", choices=("end-to-end", "actor"), default="end-to-end")
    parser.add_argument("--calibration-interval", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.iterations, args.rollout_steps, args.memory_width,
           args.calibration_interval, *args.channels) <= 0:
        parser.error("counts and widths must be positive")
    if min(args.world_weight, args.cosine_weight, args.warmup_iterations) < 0:
        parser.error("loss weights and warm-up iterations must be nonnegative")
    launch(AppLauncher, args, run)


if __name__ == "__main__":
    main()
