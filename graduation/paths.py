"""Paths for the local Isaac Lab ports and author source trees."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"
HIM = THIRD_PARTY / "MyHIMLoco"
WMP = THIRD_PARTY / "WMP"
FULLY_SNN = THIRD_PARTY / "upstream/FullySNNforLeggedRobots"
FULLY_SNN_AMP = FULLY_SNN / "AMP/AMP_for_hardware-main"
