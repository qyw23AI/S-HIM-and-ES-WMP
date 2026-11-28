"""Load A1 configuration snapshots without simulator imports."""

from copy import deepcopy
from importlib import import_module


TASKS = {
    "s_him_a1_rough_cmdcur": ("s_him.s_him_rough_cmdcur_config", "SHIMA1RoughCmdCurCfg"),
    "s_him_a1_spike_rough_cmdcur": ("s_him.s_him_spike_rough_cmdcur_config", "SHIMA1SpikeRoughCmdCurCfg"),
    "himloco_a1_amp_rough_cmdcur": ("s_him.himloco_ann_rough_cmdcur_config", "HIMLocoA1AmpRoughCmdCurCfg"),
    "es_wmp_a1_mixed": ("es_wmp.es_wmp_config", "ESWMPA1MixedCfg"),
    "es_wmp_a1_flat": ("es_wmp.es_wmp_config", "ESWMPA1FlatCfg"),
    "es_wmp_a1_parkour": ("es_wmp.es_wmp_config", "ESWMPA1ParkourCfg"),
    "es_wmp_a1_parkour_soft": ("es_wmp.es_wmp_config", "ESWMPA1ParkourSoftCfg"),
    "es_wmp_a1_parkour_strict": ("es_wmp.es_wmp_config", "ESWMPA1ParkourStrictCfg"),
}


def class_to_dict(obj):
    """Convert public data attributes and nested sections to plain containers."""
    if isinstance(obj, dict):
        return {key: class_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [class_to_dict(value) for value in obj]
    if not hasattr(obj, "__dict__"):
        return obj
    fields = (name for name in dir(obj) if not name.startswith("_"))
    values = ((name, getattr(obj, name)) for name in fields)
    return {name: class_to_dict(value) for name, value in values if not callable(value)}


def list_tasks():
    """Return the registered task names in declaration order."""
    return list(TASKS)


def load_config(task, num_envs=None, stage=4):
    """Return a fresh environment config and its training dictionary."""
    module, name = TASKS[task]
    config_module = import_module(f".{module}", __package__)
    cfg = deepcopy(getattr(config_module, name)())
    train = class_to_dict(getattr(config_module, f"{name}PPO")())

    if num_envs is not None:
        cfg.env.num_envs = num_envs
        if hasattr(cfg, "depth"):
            cfg.depth.camera_num_envs = num_envs
    if stage < 4:
        cfg.terrain.mesh_type = "plane"
        cfg.terrain.selected = None
        cfg.terrain.curriculum = False
    if stage == 1:
        cfg.noise.add_noise = False
        cfg.noise.noise_level = 0.0
        for key, value in vars(cfg.domain_rand).items():
            if isinstance(value, bool):
                setattr(cfg.domain_rand, key, False)
        cfg.commands.curriculum = False
        if hasattr(cfg.commands, "progressive_curriculum"):
            cfg.commands.progressive_curriculum = False
    return cfg, train
