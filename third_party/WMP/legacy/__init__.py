"""Task registry and config inheritance for WMP training (HIMLoco/SuperMimic legacy style)."""

from __future__ import annotations

import copy
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TASKS = {
    "a1_amp": ("a1.a1_amp_config", "A1AMPCfg", "A1AMPCfgPPO"),
    "wblwbl_dog_v1": (
        "wblwbl_dog_v1.wblwbl_dog_v1_config",
        "WblwblDogV1AMPCfg",
        "WblwblDogV1AMPCfgPPO",
    ),
    "wblwbl_dog_v1_amp": (
        "wblwbl_dog_v1.wblwbl_dog_v1_config",
        "WblwblDogV1AMPCfg",
        "WblwblDogV1AMPCfgPPO",
    ),
}


def list_tasks() -> tuple[str, ...]:
    return tuple(sorted(TASKS))


def class_to_dict(obj):
    if isinstance(obj, dict):
        return {k: class_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [class_to_dict(v) for v in obj]
    if not hasattr(obj, "__dict__"):
        return obj
    return {
        k: class_to_dict(getattr(obj, k))
        for k in dir(obj)
        if not k.startswith("_") and not callable(getattr(obj, k))
    }


def resolve_repo_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else (ROOT / p).resolve()


def resolve_motion_files(files) -> list[str]:
    if not files:
        return []
    resolved = []
    for item in files:
        p = Path(str(item).format(LEGGED_GYM_ROOT_DIR=str(ROOT)))
        resolved.append(str(p if p.is_absolute() else (ROOT / p).resolve()))
    return sorted(resolved)


def load_task_config(task_name: str, num_envs: int | None = None):
    if task_name not in TASKS:
        raise KeyError(f"Unknown task {task_name!r}; available: {list_tasks()}")
    module_name, env_cls_name, train_cls_name = TASKS[task_name]
    mod = import_module(f"legacy.{module_name}")
    env_cfg = copy.deepcopy(getattr(mod, env_cls_name)())
    train_cfg = copy.deepcopy(getattr(mod, train_cls_name)())
    if hasattr(env_cfg.env, "amp_motion_files"):
        env_cfg.env.amp_motion_files = resolve_motion_files(env_cfg.env.amp_motion_files)
    if hasattr(train_cfg.runner, "amp_motion_files"):
        train_cfg.runner.amp_motion_files = resolve_motion_files(train_cfg.runner.amp_motion_files)
    if num_envs is not None:
        env_cfg.env.num_envs = int(num_envs)
    return env_cfg, train_cfg
