"""Simulator-independent A1 environment and training configurations."""

from .base import BaseConfig
from .config import TASKS, class_to_dict, list_tasks, load_config

__all__ = ["BaseConfig", "TASKS", "class_to_dict", "list_tasks", "load_config"]
