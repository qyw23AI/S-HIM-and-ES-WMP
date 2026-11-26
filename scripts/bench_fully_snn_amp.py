#!/usr/bin/env python3
"""Migrate the authors' dedicated FullySNN A1 AMP path to Isaac Lab PhysX."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graduation.experiments.fully_snn_amp import run
from graduation.paths import WMP, FULLY_SNN_AMP
from graduation.sim.launch import launch



def main():
    sys.path[:0] = [str(FULLY_SNN_AMP / "rsl_rl"), str(WMP), str(WMP / "isaaclab_migration")]
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--expert-preload", type=int, default=256)
    parser.add_argument("--replay", type=int, default=4096)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.eval and not args.checkpoint:
        parser.error("--eval requires --checkpoint")
    if min(args.num_envs, args.steps, args.iterations, args.expert_preload, args.replay) <= 0:
        parser.error("training counts must be positive")
    launch(AppLauncher, args, run)


if __name__ == "__main__":
    main()
