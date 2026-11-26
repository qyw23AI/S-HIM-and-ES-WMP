"""Compatibility imports for the original module path."""

from .methods.s_him.policy import SHIM
from .snn.neurons import GRSN, spike
from .snn.normalization import CaReBN
from .snn.population_actor import LIFActor

__all__ = ["CaReBN", "GRSN", "LIFActor", "SHIM", "spike"]
