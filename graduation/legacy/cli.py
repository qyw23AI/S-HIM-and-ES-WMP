"""Translate nested task defaults to the benchmark CLI namespaces."""

from . import class_to_dict
from .base.s_him_base_config import BlindBenchCfgPPO, SHIMBaseCfg


def parse_overrides(parser, defaults, argv=None):
    """Parse config-backed options only when the user supplies them."""
    omitted = object()
    parser.set_defaults(**dict.fromkeys(defaults, omitted))
    args = parser.parse_args(argv)
    for name in defaults:
        if getattr(args, name) is omitted:
            delattr(args, name)
    return args


def apply_defaults(args, defaults):
    for name, value in defaults.items():
        vars(args).setdefault(name, value)


def blind_defaults(cfg, train, *, smoke=False):
    """Flatten environment and training fields consumed by bench_blind."""
    commands = cfg.commands
    defaults = {
        "num_envs": train["runner"].get("num_envs", cfg.env.num_envs),
        "iterations": train["runner"]["max_iterations"],
        "steps": train["runner"]["num_steps_per_env"],
        "seed": train["seed"],
        "terrain": "flat" if cfg.terrain.mesh_type == "plane" else "rough",
        "cmd_curriculum_init_x": commands.curriculum_init_x,
        "cmd_curriculum_init_y": commands.curriculum_init_y,
        "cmd_curriculum_max_x": None if smoke else commands.max_curriculum,
        "cmd_curriculum_step": commands.curriculum_step,
        "cmd_curriculum_threshold": commands.curriculum_threshold,
        "estimator_width": train["estimator"]["width"],
        "estimator_layers": train["estimator"]["layers"],
        "lr_schedule": train["algorithm"]["schedule"],
    }
    for group, fields in (
        ("policy", "actor_type actor_norm actor_readout_mode actor_surrogate actor_threshold ticks actor_widths "
                   "population en_pop_dim en_mean_range en_std freeze_encoder actor_intra_feedback "
                   "actor_output_norm actor_output_bias_offset max_action_std"),
        ("estimator", "estimator normalization history swap_weight swap_temperature swap_reduction target_mode "
                      "freeze_aux_estimator_stats balanced_calibration calibration_interval grsn_threshold grsn_bias"),
        ("algorithm", "learning_rate aux_learning_rate desired_kl max_lr max_grad_norm separate_grad_clip "
                      "log_std_lr_scale actor_weight_decay entropy_coef paper_training ppo_rollout_features"),
        ("runner", "checkpoint_interval diagnostics_interval"),
    ):
        defaults.update({name: train[group][name] for name in fields.split()})
    defaults.update({"amp" if name == "enabled" else f"amp_{name}": value for name, value in train["amp"].items()})
    return defaults


def configure_blind_cli(parser, argv=None):
    """Apply task defaults while preserving the original no-task CLI."""
    from . import load_config

    baseline = SHIMBaseCfg()
    smoke = blind_defaults(baseline, class_to_dict(BlindBenchCfgPPO()), smoke=True)
    args = parse_overrides(parser, smoke, argv)
    if args.task is None:
        if args.method is None:
            parser.error("method is required when --task is omitted")
        defaults = smoke
    else:
        cfg, train = load_config(args.task)
        args.method = args.method or train["policy"]["method"]
        defaults = blind_defaults(cfg, train)
        if args.method != "s-him" and train["policy"]["method"] == "s-him":
            for name in (
                "actor_type", "actor_norm", "actor_readout_mode", "actor_surrogate", "actor_threshold",
                "ticks", "actor_widths", "freeze_aux_estimator_stats", "balanced_calibration", "calibration_interval",
                "separate_grad_clip", "log_std_lr_scale", "actor_weight_decay", "paper_training",
                "ppo_rollout_features",
            ):
                defaults[name] = smoke[name]
        if args.method not in ("s-him", "himloco"):
            defaults["lr_schedule"] = smoke["lr_schedule"]
        if args.eval:
            defaults.update(amp=False, paper_training=False)
        if not getattr(args, "paper_training", defaults["paper_training"]):
            defaults["ppo_rollout_features"] = False
    apply_defaults(args, defaults)
    return args, smoke, baseline.commands.max_curriculum


def visual_defaults(cfg, train, *, parkour=False):
    """Flatten shared visual options and the corresponding student options."""
    runner = train["runner"]
    student = train["student"]
    dist = train["distillation"]
    defaults = {
        "terrain": cfg.terrain_name,
        "seed": train["seed"],
        "num_envs": runner["num_envs"],
        "iterations": runner["iterations"],
        "episode_seconds": runner["episode_seconds"],
        "threshold": dist["threshold"],
        "warmup_iterations": dist["warmup_iterations"],
        "lr": dist["lr"],
    }
    if parkour:
        defaults.update(
            steps=runner["rollout_steps"], reward_profile=cfg.reward_profile,
            **{name: student[name] for name in (
                "base_channels", "student_normalization", "student_threshold", "student_residual",
            )},
        )
    else:
        defaults.update(
            rollout_steps=runner["rollout_steps"],
            **{name: student[name] for name in ("channels", "memory_width", "memory", "normalization", "action_path")},
            **{name: dist[name] for name in ("world_weight", "cosine_weight", "calibration_interval")},
        )
    return defaults
