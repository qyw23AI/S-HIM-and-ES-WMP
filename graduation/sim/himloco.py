"""Isaac Lab environment factory for the shared rough-terrain benchmark."""


def make_blind_env(args):
    """Build the A1 environment and return the HIMLoco policy defaults."""
    from himloco_lab.adapter import LegacyRunnerAdapter
    from himloco_lab.config import load_config
    from himloco_lab.env import HIMLabEnv
    from himloco_lab.env_cfg import make_env_cfg

    legacy, defaults = load_config("a1", args.num_envs, 4)
    if args.terrain == "flat":
        legacy.terrain.mesh_type = "plane"
        legacy.terrain.curriculum = False
    cfg = make_env_cfg(legacy, 4, args.device)
    cfg.seed = args.seed
    env = HIMLabEnv(cfg)
    return env, LegacyRunnerAdapter(env), defaults
