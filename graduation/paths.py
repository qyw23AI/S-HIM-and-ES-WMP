"""Paths for the local Isaac Lab ports and author source trees."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"
HIM = THIRD_PARTY / "MyHIMLoco"
WMP = THIRD_PARTY / "WMP"
WMP_A1_MOTIONS = WMP / "datasets/mocap_motions"
FULLY_SNN = THIRD_PARTY / "upstream/FullySNNforLeggedRobots"
FULLY_SNN_AMP = FULLY_SNN / "AMP/AMP_for_hardware-main"
AMP_FOR_HARDWARE = THIRD_PARTY / "upstream/AMP_for_hardware"
AMP_A1_MOTIONS = THIRD_PARTY / "amp_motions_a1"
