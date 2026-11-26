"""A1 WMP configuration registered under legacy/."""

from pathlib import Path

from wmp_lab.a1_config import A1AMPCfg as _BaseA1AMPCfg
from wmp_lab.a1_config import A1AMPCfgPPO as _BaseA1AMPCfgPPO

ROOT = Path(__file__).resolve().parents[2]
A1_MOTION_FILES = sorted(str(p) for p in (ROOT / "datasets/mocap_motions").glob("*.txt"))


class A1AMPCfg(_BaseA1AMPCfg):
    class env(_BaseA1AMPCfg.env):
        amp_motion_files = A1_MOTION_FILES


class A1AMPCfgPPO(_BaseA1AMPCfgPPO):
    class runner(_BaseA1AMPCfgPPO.runner):
        amp_motion_files = A1_MOTION_FILES
