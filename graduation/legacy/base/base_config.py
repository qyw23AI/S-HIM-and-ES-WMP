"""Nested-class configuration with instance-owned values."""

from copy import deepcopy
from inspect import isclass


class BaseConfig:
    """Instantiate inherited configuration sections recursively."""

    def __init__(self):
        self.init_member_classes(self)

    @staticmethod
    def init_member_classes(obj):
        for name in dir(obj):
            if name == "__class__":
                continue
            value = getattr(obj, name)
            if isclass(value):
                section = value()
                setattr(obj, name, section)
                BaseConfig.init_member_classes(section)
            elif not name.startswith("__") and not callable(value):
                # deepcopy(instance) alone would leave mutable class defaults shared.
                setattr(obj, name, deepcopy(value))
