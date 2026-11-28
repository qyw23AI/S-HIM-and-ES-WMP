#!/usr/bin/env python3
"""Run shared-terrain A1 PPO, HIMLoco, FullySNN RMA, and S-HIM training or short evaluation."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graduation.experiments.blind import run
from graduation.legacy import list_tasks
from graduation.legacy.cli import configure_blind_cli
from graduation.paths import HIM
from graduation.sim.launch import launch


def parse_args(argv=None):
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[name for name in list_tasks() if name.startswith(("s_him_", "himloco_"))],
                        help="Load a named environment and training profile")
    parser.add_argument("--rma-phase", choices=("joint", "policy", "adaptation"), default="joint")
    parser.add_argument("--success-distance-m", type=float, default=None,
                        help="Evaluation gate relative to the episode spawn along world +X")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--steps", type=int, help="Rollout steps per update or evaluation length")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--terrain", choices=("rough", "flat"), help="Terrain layout (default: rough)")
    parser.add_argument("--cmd-curriculum-init-x", type=float,
                        help="Enable progressive velocity commands with this initial x half-range (m/s)")
    parser.add_argument("--cmd-curriculum-init-y", type=float,
                        help="Initial y half-range (m/s), defaults to min(init-x, 1.0), zero disables lateral commands")
    parser.add_argument("--cmd-curriculum-max-x", type=float,
                        help="Maximum x half-range (m/s), defaults to the A1 limit of 2.0")
    parser.add_argument("--cmd-curriculum-step", type=float,
                        help="Increase in velocity half-range per promotion (m/s)")
    parser.add_argument("--cmd-curriculum-threshold", type=float,
                        help="Tracking reward fraction that each sampled group must exceed for promotion")
    parser.add_argument("--normalization", choices=("care", "tdbn", "none"))
    parser.add_argument("--swap-weight", type=float)
    parser.add_argument("--history", type=int, choices=range(1, 7))
    parser.add_argument("--estimator-width", type=int)
    parser.add_argument("--estimator-layers", type=int)
    parser.add_argument("--population", type=int)
    parser.add_argument("--en-pop-dim", type=int, help="PopSAN Gaussian receptive field encoder width")
    parser.add_argument("--en-mean-range", type=float, nargs=2,
                        help="Initial [min, max] receptive field centers for PopEncoder")
    parser.add_argument("--en-std", type=float,
                        help="Initial Gaussian receptive field width for PopEncoder")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze PopEncoder mean and std parameters inside PopSANActor")
    parser.add_argument("--actor-intra-feedback", action="store_true",
                        help="Enable ILC-SAN per-joint output population intra-step feedback inside PopSANActor")
    parser.add_argument("--actor-weight-decay", type=float,
                        help="Optional decoupled weight decay on S-HIM actor parameters")
    parser.add_argument("--ticks", type=int, help="SNN actor simulation ticks per control step")
    parser.add_argument("--actor-widths", type=int, nargs="+",
                        help="Hidden widths for S-HIM actor MLP/LIF backbone")
    parser.add_argument("--estimator", choices=("grsn", "lif"))
    grsn = parser.add_mutually_exclusive_group()
    grsn.add_argument("--grsn-threshold", type=float)
    grsn.add_argument("--grsn-bias", type=float, help="Initial GRSN normalization bias for a fresh run")
    parser.add_argument("--grsn-diagnostics", action="store_true")
    parser.add_argument("--legacy-actor-projection", action="store_true")
    parser.add_argument("--imu-noise", type=float, default=0.0, help="Additional uniform noise in normalized gyro channels")
    parser.add_argument("--impulse", type=float, default=0.0, help="Lateral body-frame impulse in N s")
    parser.add_argument("--impulse-step", type=int, default=5)
    parser.add_argument("--paper-training", action=argparse.BooleanOptionalAction, help="Use the S-HIM thesis optimizer and critic settings")
    parser.add_argument("--actor-output-bias-offset", type=float,
                        help="Optional initialization offset for the final LIF current")
    parser.add_argument("--actor-output-norm", action="store_true",
                        help="Apply CaRe-BN to the output population layer before spiking")
    parser.add_argument("--actor-norm", choices=("inherit", "care", "tdbn", "ln", "none"),
                        help="Normalization layer inside LIFActor (defaults to --normalization)")
    parser.add_argument("--actor-threshold", type=float, help="LIFActor firing threshold")
    parser.add_argument("--actor-readout-mode", choices=("spike", "nlif", "linear"),
                        help="Final layer mode: spiking LIF (spike), non-spiking LIF membrane (nlif), or rate-linear")
    parser.add_argument("--actor-surrogate", choices=("atan", "rect"),
                        help="Surrogate gradient shape inside LIFActor")
    parser.add_argument("--actor-type", choices=("lif", "elu", "popsan"),
                        help="Actor architecture (lif default, elu positive control, or FullySNN-style popsan)")
    parser.add_argument("--freeze-aux-estimator-stats", action=argparse.BooleanOptionalAction,
                        help="Freeze CaRe-BN running buffers during auxiliary estimator updates")
    parser.add_argument("--balanced-calibration", action=argparse.BooleanOptionalAction,
                        help="Use balanced random sampling across envs and ticks for calibration")
    parser.add_argument("--learning-rate", type=float, help="PPO optimizer learning rate")
    parser.add_argument("--aux-learning-rate", type=float, help="Auxiliary estimator learning rate")
    parser.add_argument("--lr-schedule", choices=("fixed", "adaptive"),
                        help="PPO optimizer LR schedule")
    parser.add_argument("--desired-kl", type=float, help="Target KL for adaptive PPO LR schedule")
    parser.add_argument("--max-lr", type=float,
                        help="Upper bound for adaptive PPO learning rate schedule")
    parser.add_argument("--max-grad-norm", type=float, help="PPO gradient clip norm")
    parser.add_argument("--separate-grad-clip", action=argparse.BooleanOptionalAction,
                        help="Clip actor, critic, and log_std gradient norms independently")
    parser.add_argument("--log-std-lr-scale", type=float,
                        help="Relative LR multiplier for log_std parameter group")
    parser.add_argument("--entropy-coef", type=float,
                        help="Entropy regularization coefficient in PPO")
    parser.add_argument("--max-action-std", type=float,
                        help="Optional upper bound on Gaussian action std")
    parser.add_argument("--swap-temperature", type=float, help="SwAV softmax temperature")
    parser.add_argument("--swap-reduction", choices=("sum", "mean"),
                        help="SwAV cross-entropy reduction across prototypes")
    parser.add_argument("--target-mode", choices=("obs", "zero_cmd", "himloco", "priv_vel_cmd"),
                        help="Successor input layout for SwAV target network")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction,
                        help="Enable training-time Adversarial Motion Priors (AMP) style reward")
    parser.add_argument("--amp-motion-set",
                        choices=("a1_diagonal", "a1_diagonal_turn", "a1_diagonal_omni", "upstream_trot", "upstream",
                                 "a1_trot", "a1_trot_fwd", "a1_locomotion"),
                        help="Preset local third_party A1 mocap subset when --amp-motions is omitted")
    parser.add_argument("--amp-obs-mode", choices=("wmp30", "joint24", "full43"), default="wmp30",
                        help="Observation layout for policy and expert AMP transitions")
    parser.add_argument("--amp-motions", type=Path, nargs="+", default=None,
                        help="Explicit AMP mocap .txt files overriding --amp-motion-set")
    parser.add_argument("--amp-reward-coef", type=float,
                        help="AMP style reward scale per second (scaled by dt to match HIMLoco)")
    parser.add_argument("--amp-task-lerp", type=float,
                        help="0 gives additive task + style; in (0, 1] interpolates (1-lerp)*style + lerp*task")
    parser.add_argument("--amp-discr-lr", type=float, help="AMP discriminator Adam LR")
    parser.add_argument("--amp-discr-updates", type=int, help="Discriminator steps per PPO iteration")
    parser.add_argument("--amp-batch-size", type=int, help="AMP discriminator minibatch size")
    parser.add_argument("--amp-replay-size", type=int, help="AMP policy replay capacity")
    parser.add_argument("--amp-preload", type=int, help="Preloaded expert transitions")
    parser.add_argument("--amp-discr-widths", type=int, nargs="+",
                        help="AMP discriminator hidden widths")
    parser.add_argument("--warmstart-teacher", type=Path, help="Existing HIMLoco A1 checkpoint for action warm-start")
    parser.add_argument("--warmstart-iterations", type=int, default=0)
    parser.add_argument("--ppo-rollout-features", action=argparse.BooleanOptionalAction,
                        help="Use the estimator features recorded when PPO rollout actions were sampled")
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--diagnostics-interval", type=int)
    parser.add_argument("--calibration-interval", type=int)
    parser.add_argument("--checkpoint", type=Path, help="Load weights for evaluation or resume optimization")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--energy", action="store_true", help="Profile an evaluation rollout")
    parser.add_argument("--out", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("method", nargs="?", choices=("himloco", "ppo-mlp", "fully-snn", "s-him"))
    args, smoke, default_max_x = configure_blind_cli(parser, argv)
    init_x = args.cmd_curriculum_init_x
    init_y = args.cmd_curriculum_init_y
    max_x = args.cmd_curriculum_max_x
    if init_x is not None and not 0 < init_x < float("inf"):
        parser.error("--cmd-curriculum-init-x must be positive and finite")
    if init_y is not None and not 0 <= init_y < float("inf"):
        parser.error("--cmd-curriculum-init-y must be non-negative and finite")
    if max_x is not None and not 0 < max_x < float("inf"):
        parser.error("--cmd-curriculum-max-x must be positive and finite")
    if init_x is not None and init_x > (default_max_x if max_x is None else max_x):
        parser.error(f"--cmd-curriculum-max-x (A1 default {default_max_x}) must be at least --cmd-curriculum-init-x")
    if not 0 < args.cmd_curriculum_step < float("inf"):
        parser.error("--cmd-curriculum-step must be positive and finite")
    if not 0 < args.cmd_curriculum_threshold <= 1:
        parser.error("--cmd-curriculum-threshold must be in (0, 1]")
    if not args.max_lr > 0:
        parser.error("--max-lr must be positive")
    if min(args.num_envs, args.steps, args.iterations, args.calibration_interval,
           args.estimator_width, args.estimator_layers, args.population, args.en_pop_dim, args.ticks) < 1:
        parser.error("counts and widths must be positive")
    if args.actor_widths is not None and min(args.actor_widths) < 1:
        parser.error("--actor-widths entries must be positive")
    if not 0 < args.actor_threshold < float("inf"):
        parser.error("--actor-threshold must be positive and finite")
    if (args.grsn_threshold is not None or args.grsn_bias is not None):
        if args.method != "s-him" or args.estimator != "grsn" or args.checkpoint:
            parser.error("GRSN tuning requires a fresh S-HIM GRSN run")
        if args.grsn_threshold is not None and not 0 < args.grsn_threshold < float("inf"):
            parser.error("--grsn-threshold must be positive and finite")
        if args.grsn_bias is not None and (
                args.normalization == "none" or not abs(args.grsn_bias) < float("inf")):
            parser.error("--grsn-bias requires active normalization and a finite value")
    if args.actor_output_bias_offset and (
            not args.paper_training or args.checkpoint or not abs(args.actor_output_bias_offset) < float("inf")):
        parser.error("Actor output bias offset requires a fresh paper-training run and a finite offset")
    if args.warmstart_iterations < 0 or (args.warmstart_iterations > args.iterations and not args.checkpoint):
        parser.error("warm-start iterations must be within the run length")
    if bool(args.warmstart_teacher) != bool(args.warmstart_iterations):
        parser.error("warm-start requires a teacher checkpoint and positive warm-start iterations")
    if args.warmstart_teacher and not args.paper_training:
        parser.error("warm-start requires paper training")
    if args.ppo_rollout_features and not args.paper_training:
        parser.error("--ppo-rollout-features requires --paper-training")
    if args.paper_training and (args.method != "s-him" or args.eval):
        parser.error("--paper-training requires S-HIM training")
    shim_only = (
        "actor_output_norm", "actor_type", "freeze_aux_estimator_stats", "balanced_calibration", "separate_grad_clip",
        "log_std_lr_scale", "swap_temperature", "swap_reduction", "target_mode", "ticks", "en_pop_dim",
        "actor_widths", "actor_norm", "actor_threshold", "actor_readout_mode", "actor_surrogate", "en_mean_range",
        "en_std", "freeze_encoder", "actor_intra_feedback", "actor_weight_decay",
    )
    if args.method != "s-him" and (
        any(getattr(args, key) != smoke[key] for key in shim_only if key != "en_std")
        or abs(args.en_std - smoke["en_std"]) > 1e-9
        or (args.method != "himloco" and args.lr_schedule != smoke["lr_schedule"])
    ):
        parser.error("S-HIM variant flags require method s-him")
    if (args.actor_intra_feedback or args.freeze_encoder) and args.actor_type != "popsan":
        parser.error("--actor-intra-feedback and --freeze-encoder require --actor-type popsan")
    if args.en_mean_range[0] >= args.en_mean_range[1] or args.en_std <= 0:
        parser.error("--en-mean-range must be strictly increasing and --en-std must be positive")
    if args.actor_weight_decay < 0:
        parser.error("--actor-weight-decay must be non-negative")
    if args.amp:
        if args.eval:
            parser.error("--amp is a training option and cannot be used with --eval")
        if args.amp_reward_coef < 0 or not 0.0 <= args.amp_task_lerp <= 1.0:
            parser.error("--amp-reward-coef must be >= 0 and --amp-task-lerp must be in [0, 1]")
        if min(args.amp_discr_lr, args.amp_discr_updates, args.amp_batch_size, args.amp_replay_size, args.amp_preload) <= 0:
            parser.error("AMP discriminator LR, updates, batch, replay, and preload counts must be positive")
        if min(args.amp_discr_widths) < 1:
            parser.error("--amp-discr-widths entries must be positive")
    if min(args.learning_rate, args.aux_learning_rate, args.desired_kl, args.max_grad_norm,
           args.log_std_lr_scale, args.swap_temperature) <= 0:
        parser.error("learning rates, clip norm, KL target, and temperature must be positive")
    if args.checkpoint_interval < 0 or args.diagnostics_interval < 1:
        parser.error("invalid checkpoint or diagnostics interval")
    if args.energy and not args.eval:
        parser.error("--energy requires --eval")
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
