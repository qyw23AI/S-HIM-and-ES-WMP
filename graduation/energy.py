"""Compatibility imports for the original module path."""

from .evaluation.energy import SynapticMeter
from .methods.s_him.policy import shim_spike_inputs

__all__ = ["SynapticMeter", "shim_spike_inputs"]
