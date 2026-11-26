#!/usr/bin/env python3
"""Run shared-terrain A1 PPO, FullySNN RMA, and S-HIM training or short evaluation."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graduation.experiments.blind import run
from graduation.paths import HIM
from graduation.sim.launch import launch


def parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("himloco", "ppo-mlp", "fully-snn", "s-him"))
    parser.add_argument("--rma-phase", choices=("joint", "policy", "adaptation"), default="joint")
    parser.add_argument("--success-distance-m", type=float, default=None,
                        help="Evaluation gate relative to the episode spawn along world +X")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--steps", type=int, default=16, help="Rollout steps per update or evaluation length")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--terrain", choices=("rough", "flat"), default="rough")
    parser.add_argument("--normalization", choices=("care", "tdbn", "none"), default="care")
    parser.add_argument("--swap-weight", type=float, default=1.0)
    parser.add_argument("--history", type=int, choices=range(1, 7), default=5)
    parser.add_argument("--estimator-width", type=int, default=64)
    parser.add_argument("--estimator-layers", type=int, default=2)
    parser.add_argument("--population", type=int, default=5)
    parser.add_argument("--estimator", choices=("grsn", "lif"), default="grsn")
    grsn = parser.add_mutually_exclusive_group()
    grsn.add_argument("--grsn-threshold", type=float)
    grsn.add_argument("--grsn-bias", type=float, help="Initial GRSN normalization bias for a fresh run")
    parser.add_argument("--grsn-diagnostics", action="store_true")
    parser.add_argument("--legacy-actor-projection", action="store_true")
    parser.add_argument("--imu-noise", type=float, default=0.0, help="Additional uniform noise in normalized gyro channels")
    parser.add_argument("--impulse", type=float, default=0.0, help="Lateral body-frame impulse in N s")
    parser.add_argument("--impulse-step", type=int, default=5)
    parser.add_argument("--calibration-interval", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, help="Load weights for evaluation or resume optimization")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--energy", action="store_true", help="Profile an evaluation rollout")
    parser.add_argument("--out", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.steps, args.iterations, args.calibration_interval,
           args.estimator_width, args.estimator_layers, args.population) < 1:
        parser.error("counts and widths must be positive")
    if (args.grsn_threshold is not None or args.grsn_bias is not None):
        if args.method != "s-him" or args.estimator != "grsn" or args.checkpoint:
            parser.error("GRSN tuning requires a fresh S-HIM GRSN run")
        if args.grsn_threshold is not None and not 0 < args.grsn_threshold < float("inf"):
            parser.error("--grsn-threshold must be positive and finite")
        if args.grsn_bias is not None and (
                args.normalization == "none" or not abs(args.grsn_bias) < float("inf")):
            parser.error("--grsn-bias requires active normalization and a finite value")
    if args.energy and not args.eval:
        parser.error("--energy requires --eval")
    if args.method == "himloco" and not args.eval:
        parser.error("train HIMLoco using its upstream migration entry point")
    if args.success_distance_m is not None and (not args.eval or args.success_distance_m <= 0):
        parser.error("--success-distance-m requires evaluation and a positive distance")
    if args.rma_phase != "joint" and args.method != "fully-snn":
        parser.error("--rma-phase requires fully-snn")
    if args.rma_phase == "adaptation" and not args.checkpoint:
        parser.error("adaptation requires a trained base policy as --checkpoint")
    if args.impulse and not 0 <= args.impulse_step < args.steps:
        parser.error("the impulse control step must lie in the rollout")
    if args.eval and not args.checkpoint:
        parser.error("--eval requires --checkpoint")
    return args


def main():
    sys.path[:0] = [str(HIM / "isaaclab_migration"), str(HIM / "rsl_rl")]
    from isaaclab.app import AppLauncher

    args = parse_args()
    launch(AppLauncher, args, run)


if __name__ == "__main__":
    main()
