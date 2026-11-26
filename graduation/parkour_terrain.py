"""Compatibility imports for the original module path."""

from .methods.es_parkour.task import ParkourCore
from .sim.parkour import KINDS, ParkourTerrain

__all__ = ["KINDS", "ParkourCore", "ParkourTerrain"]
