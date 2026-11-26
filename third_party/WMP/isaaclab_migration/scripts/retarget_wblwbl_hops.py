#!/usr/bin/env python3
"""Retarget WMP A1 pronk/hop AMP clips (hop1, hop2) to wblwbl_dog_v1 and render MuJoCo comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

WMP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_A1_URDF = WMP_ROOT / "resources/robots/a1/urdf/a1.urdf"
DEFAULT_A1_MESH_DIR = WMP_ROOT / "resources/robots/a1/meshes"
DEFAULT_A1_MJCF_DIR = WMP_ROOT / "tmp/a1_mjcf"
DEFAULT_WBL_MJCF = Path("/home/qyw/robocon2027_sim_control/deploy_cpp/robot/wblwbl_dog_v1/xml/wblwbl_dog_v1.xml")
DEFAULT_SOURCE_DIR = WMP_ROOT / "datasets/mocap_motions"
DEFAULT_OUTPUT_DIR = WMP_ROOT / "datasets/mocap_motions_wblwbl"
DEFAULT_REVIEW_DIR = WMP_ROOT / "tmp/wblwbl_hop_review"

TARGET_LEGS = ("RR", "RL", "FR", "FL")
SOURCE_LEGS_WMP = ("FL", "FR", "RL", "RR")
A1_TO_WBL_LEG_ORDER = [SOURCE_LEGS_WMP.index(leg) for leg in TARGET_LEGS]  # [3, 2, 1, 0]
SIDES = np.array([-1.0, 1.0, -1.0, 1.0])
JOINT_SUFFIXES = ("hip", "thigh", "calf")

WBL_NEUTRAL_Q = np.array([
    -0.1, 1.0, -1.5,  # RR
    0.1, 1.0, -1.5,   # RL
    -0.1, 0.8, -1.5,  # FR
    0.1, 0.8, -1.5,   # FL
], dtype=float)

A1_STAND_Q = np.array([
    -0.1, 1.0, -1.5,  # RR
    0.1, 1.0, -1.5,   # RL
    -0.1, 0.8, -1.5,  # FR
    0.1, 0.8, -1.5,   # FL
], dtype=float)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_a1_mjcf(urdf_path: Path, mesh_dir: Path, out_dir: Path) -> Path:
    """Create a MuJoCo-loadable Unitree A1 XML from a1.urdf and Collada meshes if absent."""
    xml_path = out_dir / "a1.xml"
    stl_dir = out_dir / "meshes"
    needed_stls = ["trunk.stl", "hip.stl", "thigh.stl", "thigh_mirror.stl", "calf.stl"]
    if xml_path.is_file() and all((stl_dir / name).is_file() for name in needed_stls):
        return xml_path

    import trimesh

    stl_dir.mkdir(parents=True, exist_ok=True)
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    for dae_name in ["trunk.dae", "hip.dae", "thigh.dae", "thigh_mirror.dae", "calf.dae"]:
        root = ET.parse(mesh_dir / dae_name).getroot()
        meshes = []
        for geom in root.findall(".//c:geometry", ns):
            mesh_el = geom.find("c:mesh", ns)
            if mesh_el is None:
                continue
            sources = {}
            for src in mesh_el.findall("c:source", ns):
                fa = src.find("c:float_array", ns)
                if fa is not None and fa.text:
                    sources[src.attrib["id"]] = np.fromstring(fa.text, sep=" ")
            verts_el = mesh_el.find("c:vertices", ns)
            pos_id = next(
                inp.attrib["source"].lstrip("#")
                for inp in verts_el.findall("c:input", ns)
                if inp.attrib.get("semantic") == "POSITION"
            )
            verts_id = verts_el.attrib["id"]
            v_raw = sources[pos_id].reshape(-1, 3)
            for tris in mesh_el.findall("c:triangles", ns):
                inputs = tris.findall("c:input", ns)
                stride = max(int(i.attrib["offset"]) for i in inputs) + 1
                v_off = next(
                    int(i.attrib["offset"])
                    for i in inputs
                    if i.attrib.get("semantic") == "VERTEX" or i.attrib["source"].lstrip("#") == verts_id
                )
                p_el = tris.find("c:p", ns)
                if p_el is not None and p_el.text:
                    p = np.fromstring(p_el.text, sep=" ", dtype=np.int64)
                    f_idx = p.reshape(-1, 3, stride)[:, :, v_off]
                    meshes.append(trimesh.Trimesh(vertices=v_raw, faces=f_idx, process=False))
        trimesh.util.concatenate(meshes).export(stl_dir / (Path(dae_name).stem + ".stl"))

    urdf_joints = {j.attrib["name"]: j for j in ET.parse(urdf_path).findall("joint")}
    urdf_links = {l.attrib["name"]: l for l in ET.parse(urdf_path).findall("link")}
    leg_xml = []
    for leg in SOURCE_LEGS_WMP:
        hj = urdf_joints[f"{leg}_hip_joint"]
        tj = urdf_joints[f"{leg}_thigh_joint"]
        cj = urdf_joints[f"{leg}_calf_joint"]
        fj = urdf_joints[f"{leg}_foot_fixed"]

        h_xyz = hj.find("origin").attrib["xyz"]
        t_xyz = tj.find("origin").attrib["xyz"]
        c_xyz = cj.find("origin").attrib["xyz"]
        f_xyz = fj.find("origin").attrib["xyz"]

        h_lim = hj.find("limit").attrib
        t_lim = tj.find("limit").attrib
        c_lim = cj.find("limit").attrib

        h_vis = urdf_links[f"{leg}_hip"].find("visual")
        h_rpy = np.fromstring(h_vis.find("origin").attrib["rpy"], sep=" ")
        h_quat = " ".join(f"{v:.8f}" for v in Rotation.from_euler("xyz", h_rpy).as_quat()[[3, 0, 1, 2]])
        th_mesh = Path(urdf_links[f"{leg}_thigh"].find("visual/geometry/mesh").attrib["filename"]).stem

        leg_xml.append(
            f"""
            <body name="{leg}_hip" pos="{h_xyz}">
                <joint name="{leg}_hip_joint" axis="1 0 0" range="{h_lim['lower']} {h_lim['upper']}" />
                <geom type="mesh" contype="0" conaffinity="0" rgba="0.30 0.35 0.42 1" quat="{h_quat}" mesh="hip" />
                <body name="{leg}_thigh" pos="{t_xyz}">
                    <joint name="{leg}_thigh_joint" axis="0 1 0" range="{t_lim['lower']} {t_lim['upper']}" />
                    <geom type="mesh" contype="0" conaffinity="0" rgba="0.22 0.25 0.30 1" mesh="{th_mesh}" />
                    <body name="{leg}_calf" pos="{c_xyz}">
                        <joint name="{leg}_calf_joint" axis="0 1 0" range="{c_lim['lower']} {c_lim['upper']}" />
                        <geom type="mesh" contype="0" conaffinity="0" rgba="0.22 0.25 0.30 1" mesh="calf" />
                        <geom name="{leg}_foot" type="sphere" size="0.02" pos="{f_xyz}" rgba="0.90 0.45 0.12 1" />
                    </body>
                </body>
            </body>"""
        )

    xml_text = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="unitree_a1">
    <compiler angle="radian" meshdir="meshes/" autolimits="true" />
    <default>
        <geom contype="0" conaffinity="0" />
    </default>
    <asset>
        <mesh name="trunk" file="trunk.stl" />
        <mesh name="hip" file="hip.stl" />
        <mesh name="thigh" file="thigh.stl" />
        <mesh name="thigh_mirror" file="thigh_mirror.stl" />
        <mesh name="calf" file="calf.stl" />
        <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="512" height="512" />
        <texture name="plane" type="2d" builtin="flat" rgb1="0.96 0.96 0.96" rgb2="0.96 0.96 0.96" width="512" height="512" mark="cross" markrgb="0.65 0.65 0.65" />
        <material name="plane" reflectance="0.0" texture="plane" texrepeat="3 3" texuniform="true" />
    </asset>
    <worldbody>
        <light directional="true" diffuse=".8 .8 .8" pos="0 0 10" dir="0 0 -10" />
        <camera name="track" mode="trackcom" pos="0 -1.3 1.0" xyaxes="1 0 0 0 0.707 0.707" />
        <geom name="floor" type="plane" material="plane" pos="0 0 0" size="0 0 1" />
        <body name="trunk" pos="0 0 0.3">
            <freejoint />
            <geom type="mesh" rgba="0.18 0.20 0.24 1" mesh="trunk" />
            {''.join(leg_xml)}
        </body>
    </worldbody>
</mujoco>
"""
    xml_path.write_text(xml_text)
    return xml_path


class RobotKinematics:
    """Forward kinematics and foot-sphere metadata wrapper around a quadruped MJCF."""

    def __init__(self, mjcf_path: Path, leg_order: tuple[str, ...]) -> None:
        self.path = Path(mjcf_path)
        self.model = mj.MjModel.from_xml_path(str(self.path))
        self.data = mj.MjData(self.model)
        self.leg_order = leg_order
        joint_names = [f"{leg}_{suffix}_joint" for leg in leg_order for suffix in JOINT_SUFFIXES]
        self.joint_ids = np.array([self.model.joint(n).id for n in joint_names], dtype=int)
        self.q_idx = np.array([self.model.joint(n).qposadr[0] for n in joint_names], dtype=int)
        self.limits = self.model.jnt_range[self.joint_ids].T.copy()

        self.calf_ids = np.array([self.model.body(f"{leg}_calf").id for leg in leg_order], dtype=int)
        foot_geoms = []
        for b in self.calf_ids:
            gid = next(
                i
                for i in range(self.model.ngeom)
                if self.model.geom_bodyid[i] == b and self.model.geom_type[i] == mj.mjtGeom.mjGEOM_SPHERE
            )
            foot_geoms.append(gid)
        self.foot_geom_ids = np.array(foot_geoms, dtype=int)
        self.foot_offsets = np.array([self.model.geom(i).pos.copy() for i in self.foot_geom_ids], dtype=float)
        self.radius = np.array([self.model.geom(i).size[0] for i in self.foot_geom_ids], dtype=float)

    def fk(self, q: np.ndarray, quat_xyzw: np.ndarray | None = None, pos: np.ndarray | None = None) -> np.ndarray:
        self.data.qpos[:3] = np.zeros(3) if pos is None else pos
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0]) if quat_xyzw is None else quat_xyzw[[3, 0, 1, 2]]
        self.data.qpos[self.q_idx] = q
        mj.mj_kinematics(self.model, self.data)
        return self.data.xpos[self.calf_ids] + np.einsum(
            "nij,nj->ni", self.data.xmat[self.calf_ids].reshape(-1, 3, 3), self.foot_offsets
        )


def resample_trajectory(rows: np.ndarray, source_dt: float, time_stretch: float, target_fps: int = 50) -> np.ndarray:
    """Resample [pos(3), quat_xyzw(4), q(12)] to fixed 50 Hz under optional time stretch."""
    n = len(rows)
    src_t = np.arange(n, dtype=float) * source_dt * time_stretch
    duration = src_t[-1]
    out_t = np.arange(int(np.round(duration * target_fps)) + 1, dtype=float) / target_fps
    out_t[-1] = min(out_t[-1], duration)

    out = np.empty((len(out_t), rows.shape[1]), dtype=float)
    for col in (*range(3), *range(7, rows.shape[1])):
        out[:, col] = np.interp(out_t, src_t, rows[:, col])
    rot = Rotation.from_quat(rows[:, 3:7])
    out[:, 3:7] = Slerp(src_t, rot)(out_t).as_quat()
    return out


def standardize_quaternions_xyzw(quats: np.ndarray) -> np.ndarray:
    """Normalize quaternions and keep hemisphere continuous across frames."""
    q = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    if q[0, 3] < 0:
        q[0] = -q[0]
    for k in range(1, len(q)):
        if np.dot(q[k - 1], q[k]) < 0:
            q[k] = -q[k]
    return q


def solve_retarget(
    a1_kin: RobotKinematics,
    wbl_kin: RobotKinematics,
    raw_frames: np.ndarray,
    source_dt: float,
    scale_xyz: tuple[float, float, float] = (1.21, 0.65, 1.21),
    time_stretch: float = 1.10,
    min_width: float = 0.145,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Retarget an A1 WMP clip onto wblwbl_dog_v1 using Froude-balanced foot/root scaling."""
    q_a1_raw = raw_frames[:, 7:19].reshape(-1, 4, 3)[:, A1_TO_WBL_LEG_ORDER].reshape(-1, 12)
    stored_toes_raw = raw_frames[:, 19:31].reshape(-1, 4, 3)[:, A1_TO_WBL_LEG_ORDER]

    a1_fk_raw = np.array([a1_kin.fk(x) for x in q_a1_raw])
    com_delta_std = float(np.std(stored_toes_raw - a1_fk_raw, axis=(0, 1)).max())
    com_delta_mean = (stored_toes_raw - a1_fk_raw).mean(axis=(0, 1))

    # Check original A1 sole height using exact A1 FK (touches floor z=0 cleanly).
    a1_world_raw = np.array([a1_kin.fk(q_a1_raw[k], raw_frames[k, 3:7], raw_frames[k, :3]) for k in range(len(raw_frames))])
    a1_sole_raw = a1_world_raw[:, :, 2] - a1_kin.radius

    rows_19 = np.c_[raw_frames[:, :7], q_a1_raw]
    rows_res = resample_trajectory(rows_19, source_dt, time_stretch)
    rows_res[:, 3:7] = standardize_quaternions_xyzw(rows_res[:, 3:7])

    stand_a1 = a1_kin.fk(A1_STAND_Q)
    stand_wbl = wbl_kin.fk(WBL_NEUTRAL_Q)
    z0_a1 = float(-np.min(stand_a1[:, 2] - a1_kin.radius))
    z0_wbl = float(-np.min(stand_wbl[:, 2] - wbl_kin.radius))

    sx, sy, sz = scale_xyz
    fk_a1 = np.array([a1_kin.fk(x) for x in rows_res[:, 7:19]])
    rot = Rotation.from_quat(rows_res[:, 3:7])
    local = stand_wbl + (fk_a1 - stand_a1) * np.array([sx, sy, sz])

    root = rows_res[:, :7].copy()
    root[:, :2] = sx * (root[:, :2] - root[0, :2])
    root[:, 2] = z0_wbl + sz * (root[:, 2] - z0_a1)

    targets = np.einsum("tij,tkj->tki", rot.as_matrix(), local) + root[:, None, :3]
    targets[:, :, 2] = np.maximum(targets[:, :, 2], wbl_kin.radius + 0.001)

    lo, hi = wbl_kin.limits.copy()
    hi[2::3] = np.minimum(hi[2::3], -0.85)

    # Converge frame 0 cleanly from neutral before sequential smooth IK.
    def res0(q: np.ndarray) -> np.ndarray:
        loc = wbl_kin.fk(q)
        wld = wbl_kin.fk(q, root[0, 3:7], root[0, :3])
        inward = np.minimum(0.0, SIDES * loc[:, 1] - min_width)
        return np.r_[12.0 * (wld - targets[0]).ravel(), 12.0 * inward, 0.02 * (q - WBL_NEUTRAL_Q)]

    prev = least_squares(res0, WBL_NEUTRAL_Q, bounds=(lo + 1e-6, hi - 1e-6), max_nfev=80).x
    pose = []
    fit_errors = []
    for k, (row, feet) in enumerate(zip(root, targets)):
        if k == 0:
            q = prev
        else:
            def res(q_var: np.ndarray) -> np.ndarray:
                loc = wbl_kin.fk(q_var)
                wld = wbl_kin.fk(q_var, row[3:7], row[:3])
                inward = np.minimum(0.0, SIDES * loc[:, 1] - min_width)
                return np.r_[
                    12.0 * (wld - feet).ravel(),
                    12.0 * inward,
                    0.055 * (q_var - WBL_NEUTRAL_Q),
                    0.065 * (q_var - prev),
                ]

            q = least_squares(res, prev, bounds=(lo + 1e-6, hi - 1e-6), max_nfev=36).x
        prev = q
        pose.append(np.r_[row, q])
        fit_errors.append(np.linalg.norm(wbl_kin.fk(q, row[3:7], row[:3]) - feet, axis=1))

    pose = np.asarray(pose)
    fit_errors = np.asarray(fit_errors)

    # Smooth floor clearance guard so all 4 foot spheres remain >= +1 mm above z=0.
    soles = np.array([wbl_kin.fk(r[7:], r[3:7], r[:3])[:, 2] - wbl_kin.radius for r in pose])
    required_lift = np.maximum(0.0, 0.001 - soles.min(axis=1))
    smooth_lift = gaussian_filter1d(maximum_filter1d(required_lift, 9, mode="nearest"), 1.0, mode="nearest")
    pose[:, 2] += np.maximum(required_lift, smooth_lift)

    # Centered A1 trajectory starting at origin for side-by-side comparison.
    a1_pose = rows_19.copy()
    a1_pose[:, :2] -= a1_pose[0, :2]
    a1_pose[:, 3:7] = standardize_quaternions_xyzw(a1_pose[:, 3:7])

    info = {
        "source_a1_com_offset_mean_m": np.round(com_delta_mean, 6).tolist(),
        "source_a1_com_offset_std_m": com_delta_std,
        "source_a1_sole_min_m": float(a1_sole_raw.min()),
        "source_a1_airborne_frames_gt2cm": int((a1_sole_raw.min(axis=1) > 0.02).sum()),
        "source_a1_airborne_peak_m": float(a1_sole_raw.min(axis=1).max()),
    }
    return pose, a1_pose, fit_errors, info


def to_wmp_frames(wbl_kin: RobotKinematics, pose: np.ndarray, dt: float = 0.02) -> np.ndarray:
    """Convert retargeted 19-col [pos, quat_xyzw, q] into WMP 61-col frame array."""
    pos = pose[:, :3]
    quat_xyzw = standardize_quaternions_xyzw(pose[:, 3:7])
    q = pose[:, 7:19]
    rot = Rotation.from_quat(quat_xyzw)

    toes_local = np.array([wbl_kin.fk(x) for x in q]).reshape(-1, 12)
    vel_world = np.gradient(pos, dt, axis=0)
    lin_vel_body = np.array([rot[k].inv().apply(vel_world[k]) for k in range(len(pose))])

    ang_vel_body = np.zeros((len(pose), 3), dtype=float)
    drot_step = (rot[:-1].inv() * rot[1:]).as_rotvec() / dt
    if len(pose) >= 2:
        ang_vel_body[0] = drot_step[0]
        ang_vel_body[-1] = drot_step[-1]
    if len(pose) >= 3:
        ang_vel_body[1:-1] = (rot[:-2].inv() * rot[2:]).as_rotvec() / (2.0 * dt)

    dof_vel = np.gradient(q, dt, axis=0)
    toe_vel = np.zeros((len(pose), 12), dtype=float)
    frames = np.hstack([pos, quat_xyzw, q, toes_local, lin_vel_body, ang_vel_body, dof_vel, toe_vel])
    if frames.shape[1] != 61 or not np.isfinite(frames).all():
        raise ValueError(f"Invalid WMP frame shape/values: {frames.shape}")
    return frames


def audit_motion(
    name: str,
    wbl_kin: RobotKinematics,
    pose: np.ndarray,
    frames_61: np.ndarray,
    fit_errors: np.ndarray,
    source_info: dict,
    dt: float = 0.02,
) -> dict:
    """Compute kinematic, physical, contact, and limit metrics for review."""
    wfeet = np.array([wbl_kin.fk(r[7:], r[3:7], r[:3]) for r in pose])
    soles = wfeet[:, :, 2] - wbl_kin.radius
    vfeet_xy = np.linalg.norm(np.diff(wfeet[:, :, :2], axis=0) / dt, axis=-1)
    stance_mask = (soles[:-1] < 0.005) & (soles[1:] < 0.005)
    air_mask = soles.min(axis=1) > 0.02
    az = np.diff(pose[:, 2], n=2) / (dt ** 2)

    q = pose[:, 7:19]
    dq = np.diff(q, axis=0) / dt
    qt_centered = q[:, 7] - q[:, 7].mean()
    freqs = np.fft.rfftfreq(len(qt_centered), d=dt)
    dom_freq_hz = float(freqs[np.argmax(np.abs(np.fft.rfft(qt_centered)))])
    lo, hi = wbl_kin.limits
    hip_margin = float(np.minimum(q[:, 0::3] - lo[0::3], hi[0::3] - q[:, 0::3]).min())
    thigh_margin = float(np.minimum(q[:, 1::3] - lo[1::3], hi[1::3] - q[:, 1::3]).min())
    calf_margin = float(np.minimum(q[:, 2::3] - lo[2::3], hi[2::3] - q[:, 2::3]).min())

    # Self-collision check using active WBL XML collision boxes.
    m_col = mj.MjModel.from_xml_path(str(wbl_kin.path))
    for i in range(m_col.ngeom):
        if m_col.geom_bodyid[i] == 0 and m_col.geom(i).name != "floor":
            m_col.geom_contype[i] = m_col.geom_conaffinity[i] = 0
    d_col = mj.MjData(m_col)
    qids = [m_col.joint(f"{l}_{s}_joint").qposadr[0] for l in TARGET_LEGS for s in JOINT_SUFFIXES]
    self_col_frames = 0
    for r in pose:
        d_col.qpos[:3] = r[:3]
        d_col.qpos[3:7] = r[[6, 3, 4, 5]]
        d_col.qpos[qids] = r[7:]
        mj.mj_forward(m_col, d_col)
        if any(m_col.geom_bodyid[c.geom1] > 0 and m_col.geom_bodyid[c.geom2] > 0 for c in d_col.contact):
            self_col_frames += 1

    return {
        "clip": name,
        "frames": int(len(pose)),
        "duration_s": float((len(pose) - 1) * dt),
        "root_displacement_xy_m": [float(pose[-1, 0] - pose[0, 0]), float(pose[-1, 1] - pose[0, 1])],
        "mean_lin_vel_x_m_s": float(frames_61[:, 31].mean()),
        "root_z_range_m": [float(pose[:, 2].min()), float(pose[:, 2].max())],
        "target_foot_fit_p95_mm": float(np.quantile(fit_errors, 0.95) * 1000.0),
        "target_foot_fit_max_mm": float(fit_errors.max() * 1000.0),
        "sole_min_m": float(soles.min()),
        "below_floor_frames": int((soles < 0.0).sum()),
        "swing_max_clearance_cm": float(soles.max() * 100.0),
        "swing_p95_clearance_cm": float(np.quantile(soles, 0.95) * 100.0),
        "dominant_gait_freq_hz": dom_freq_hz,
        "airborne_frames_gt2cm": int(air_mask.sum()),
        "airborne_peak_clearance_cm": float(soles.min(axis=1).max() * 100.0),
        "airborne_median_az_m_s2": float(np.median(az[air_mask[1:-1]])) if air_mask[1:-1].any() else None,
        "stance_foot_slip_mean_m_s": float(vfeet_xy[stance_mask].mean()) if stance_mask.any() else None,
        "knee_range_rad": [float(q[:, 2::3].min()), float(q[:, 2::3].max())],
        "joint_limit_margin_rad": {"hip": hip_margin, "thigh": thigh_margin, "calf": calf_margin},
        "joint_vel_p99_rad_s": float(np.quantile(np.abs(dq), 0.99)),
        "joint_vel_max_rad_s": float(np.abs(dq).max()),
        "self_collision_frames": int(self_col_frames),
        **source_info,
    }


def _prep_render_model(xml_path: Path) -> mj.MjModel:
    model = mj.MjModel.from_xml_path(str(xml_path))
    for i in range(model.ngeom):
        if model.geom_bodyid[i] == 0 and model.geom(i).name != "floor":
            model.geom_rgba[i, 3] = 0.0
            model.geom_contype[i] = 0
            model.geom_conaffinity[i] = 0
        elif model.geom(i).name == "floor":
            # Crisp neutral floor so foot shadows and ground contact remain clear.
            model.geom_rgba[i] = np.array([0.86, 0.87, 0.89, 1.0], dtype=np.float32)
    # Use consistent directional + fill light and ground shadows in EGL renders.
    if model.nlight >= 1:
        model.light_castshadow[0] = 1
        model.light_diffuse[0] = np.array([0.75, 0.75, 0.75], dtype=np.float32)
    return model


def _render_panel(
    renderer: mj.Renderer,
    model: mj.MjModel,
    data: mj.MjData,
    joint_qpos_idx: np.ndarray,
    row19: np.ndarray,
    azimuth: float,
    elevation: float,
    distance: float,
    lookat_z: float,
) -> np.ndarray:
    data.qpos[:3] = row19[:3]
    data.qpos[3:7] = row19[[6, 3, 4, 5]]
    data.qpos[joint_qpos_idx] = row19[7:19]
    mj.mj_forward(model, data)

    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [row19[0], row19[1], lookat_z]
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    renderer.update_scene(data, camera=cam)
    return renderer.render()


def render_comparison(
    clip_name: str,
    a1_xml: Path,
    wbl_xml: Path,
    a1_pose: np.ndarray,
    wbl_pose: np.ndarray,
    a1_kin: RobotKinematics,
    wbl_kin: RobotKinematics,
    review_dir: Path,
    dt: float = 0.02,
) -> tuple[Path, Path, Path]:
    """Render phase-locked 2x2 MuJoCo video + side and 3/4 keyframe sheets."""
    review_dir.mkdir(parents=True, exist_ok=True)
    m_a1 = _prep_render_model(a1_xml)
    m_wbl = _prep_render_model(wbl_xml)
    d_a1 = mj.MjData(m_a1)
    d_wbl = mj.MjData(m_wbl)

    q_a1_idx = np.array([m_a1.joint(f"{l}_{s}_joint").qposadr[0] for l in TARGET_LEGS for s in JOINT_SUFFIXES])
    q_wbl_idx = np.array([m_wbl.joint(f"{l}_{s}_joint").qposadr[0] for l in TARGET_LEGS for s in JOINT_SUFFIXES])

    # Precompute sole clearances for labels (a1_pose and wbl_pose are both in TARGET_LEGS order).
    a1_soles = np.array([
        a1_kin.fk(a1_pose[k, 7:19], a1_pose[k, 3:7], a1_pose[k, :3])[:, 2] - a1_kin.radius
        for k in range(len(a1_pose))
    ])
    wbl_soles = np.array([
        wbl_kin.fk(wbl_pose[k, 7:19], wbl_pose[k, 3:7], wbl_pose[k, :3])[:, 2] - wbl_kin.radius
        for k in range(len(wbl_pose))
    ])
    a1_sole_min = a1_soles.min(axis=1)
    wbl_sole_min = wbl_soles.min(axis=1)
    a1_sole_max = a1_soles.max(axis=1)
    wbl_sole_max = wbl_soles.max(axis=1)

    pw, ph = 560, 340
    r_a1 = mj.Renderer(m_a1, ph, pw)
    r_wbl = mj.Renderer(m_wbl, ph, pw)

    is_hop = bool(wbl_sole_min.max() > 0.02)
    if is_hop:
        wbl_peak_idx = int(np.argmax(wbl_sole_min[80:-80])) + 80
        wbl_picks = [
            max(0, wbl_peak_idx - 13),
            max(0, wbl_peak_idx - 6),
            wbl_peak_idx,
            min(len(wbl_pose) - 1, wbl_peak_idx + 6),
            min(len(wbl_pose) - 1, wbl_peak_idx + 13),
        ]
        phase_labels = ["1. Crouch", "2. Push-off", "3. Apex (Airborne)", "4. Descent", "5. Landing"]
    else:
        # Pick one complete trot gait cycle starting at a clean FR swing peak.
        qt = wbl_pose[:, 14] - wbl_pose[:, 14].mean()
        freqs = np.fft.rfftfreq(len(qt), d=dt)
        dom_f = max(0.5, float(freqs[np.argmax(np.abs(np.fft.rfft(qt)))]))
        period = max(20, int(round(1.0 / (dom_f * dt))))
        search_hi = max(120, len(wbl_pose) - period - 40)
        fr_peak = int(np.argmax(wbl_soles[80:search_hi, 2])) + 80
        wbl_picks = [min(len(wbl_pose) - 1, fr_peak + int(round(i * period / 4))) for i in range(5)]
        phase_labels = ["1. FR/RL Swing", "2. Crossover A", "3. FL/RR Swing", "4. Crossover B", "5. Next FR/RL Swing"]

    def compose_sheet(azimuth: float, elevation: float, suffix: str) -> Path:
        cols = len(wbl_picks)
        sheet = Image.new("RGB", (pw * cols, ph * 2 + 36), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        for col, (lbl, wi) in enumerate(zip(phase_labels, wbl_picks)):
            ai = int(round(wi * (len(a1_pose) - 1) / max(1, len(wbl_pose) - 1)))
            a1_yaw = np.degrees(Rotation.from_quat(a1_pose[ai, 3:7]).as_euler("xyz")[2])
            wbl_yaw = np.degrees(Rotation.from_quat(wbl_pose[wi, 3:7]).as_euler("xyz")[2])
            img_a1 = Image.fromarray(_render_panel(r_a1, m_a1, d_a1, q_a1_idx, a1_pose[ai], azimuth + a1_yaw, elevation, 1.35, 0.25))
            img_wbl = Image.fromarray(_render_panel(r_wbl, m_wbl, d_wbl, q_wbl_idx, wbl_pose[wi], azimuth + wbl_yaw, elevation, 1.45, 0.31))
            a1_metric = f"sole={a1_sole_min[ai]*100:.1f}cm" if is_hop else f"swing={a1_sole_max[ai]*100:.1f}cm"
            wbl_metric = f"sole={wbl_sole_min[wi]*100:.1f}cm" if is_hop else f"swing={wbl_sole_max[wi]*100:.1f}cm"
            d1 = ImageDraw.Draw(img_a1)
            d1.text((10, 8), f"Original A1 | {lbl} | f={ai} z={a1_pose[ai,2]:.3f}m {a1_metric}", fill=(20, 20, 20))
            d2 = ImageDraw.Draw(img_wbl)
            d2.text((10, 8), f"Retargeted WBL | {lbl} | f={wi} z={wbl_pose[wi,2]:.3f}m {wbl_metric}", fill=(10, 40, 120))
            sheet.paste(img_a1, (col * pw, 36))
            sheet.paste(img_wbl, (col * pw, 36 + ph))
        draw.text((14, 10), f"{clip_name} MuJoCo Keyframe Review ({suffix}) - Top: Unitree A1 | Bottom: WBLWBL Dog V1 (Phase-Locked)", fill=(15, 15, 15))
        out_png = review_dir / f"{clip_name}_keyframes_{suffix}.png"
        sheet.save(out_png)
        return out_png

    side_png = compose_sheet(azimuth=90.0, elevation=-3.0, suffix="side")
    obl_png = compose_sheet(azimuth=135.0, elevation=-16.0, suffix="34view")

    # Render phase-locked 2x2 comparison MP4 (50 fps real-time cadence for WBL, A1 phase-matched)
    mp4_path = review_dir / f"{clip_name}_compare.mp4"
    grid_w, grid_h = pw * 2, ph * 2
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{grid_w}x{grid_h}", "-r", "50", "-i", "-",
        "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(mp4_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    canvas = Image.new("RGB", (grid_w, grid_h))
    try:
        for wi in range(len(wbl_pose)):
            ai = int(round(wi * (len(a1_pose) - 1) / max(1, len(wbl_pose) - 1)))
            a1_yaw = np.degrees(Rotation.from_quat(a1_pose[ai, 3:7]).as_euler("xyz")[2])
            wbl_yaw = np.degrees(Rotation.from_quat(wbl_pose[wi, 3:7]).as_euler("xyz")[2])

            a1_obl = _render_panel(r_a1, m_a1, d_a1, q_a1_idx, a1_pose[ai], 135.0 + a1_yaw, -16.0, 1.35, 0.25)
            wbl_obl = _render_panel(r_wbl, m_wbl, d_wbl, q_wbl_idx, wbl_pose[wi], 135.0 + wbl_yaw, -16.0, 1.45, 0.31)
            a1_side = _render_panel(r_a1, m_a1, d_a1, q_a1_idx, a1_pose[ai], 90.0 + a1_yaw, -3.0, 1.35, 0.25)
            wbl_side = _render_panel(r_wbl, m_wbl, d_wbl, q_wbl_idx, wbl_pose[wi], 90.0 + wbl_yaw, -3.0, 1.45, 0.31)

            top = np.concatenate([a1_obl, wbl_obl], axis=1)
            bot = np.concatenate([a1_side, wbl_side], axis=1)
            frame_img = Image.fromarray(np.concatenate([top, bot], axis=0))
            d = ImageDraw.Draw(frame_img)
            a1_tag = f"min_sole={a1_sole_min[ai]*100:.1f}cm" if is_hop else f"max_swing={a1_sole_max[ai]*100:.1f}cm"
            wbl_tag = f"min_sole={wbl_sole_min[wi]*100:.1f}cm" if is_hop else f"max_swing={wbl_sole_max[wi]*100:.1f}cm"
            d.text((12, 10), f"{clip_name} | Original A1 (3/4) | t={ai*dt:.2f}s | z={a1_pose[ai,2]:.3f}m | {a1_tag}", fill=(20, 20, 20))
            d.text((pw + 12, 10), f"{clip_name} | Retargeted WBL (3/4) | t={wi*dt:.2f}s | z={wbl_pose[wi,2]:.3f}m | {wbl_tag}", fill=(10, 40, 120))
            d.text((12, ph + 10), "Original A1 (Side Profile)", fill=(20, 20, 20))
            d.text((pw + 12, ph + 10), "Retargeted WBL (Side Profile)", fill=(10, 40, 120))
            proc.stdin.write(frame_img.tobytes())
    finally:
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed ({rc}): {stderr}")

    return mp4_path, side_png, obl_png


def _prefix_subtree(elem: ET.Element, prefix: str, mesh_map: dict[str, str]) -> ET.Element:
    """Deep-rename body/joint/geom/site/mesh references so two robots coexist in one MJCF."""
    clone = ET.fromstring(ET.tostring(elem, encoding="unicode"))
    for node in clone.iter():
        if node.tag in ("body", "joint", "geom", "site") and "name" in node.attrib:
            node.attrib["name"] = prefix + node.attrib["name"]
        if node.tag == "geom" and "mesh" in node.attrib:
            node.attrib["mesh"] = mesh_map[node.attrib["mesh"]]
        if node.tag == "geom":
            node.attrib["contype"] = "0"
            node.attrib["conaffinity"] = "0"
            if "class" in node.attrib:
                del node.attrib["class"]
            if node.attrib.get("type") == "box" and node.attrib.get("rgba") == "0 0 0 0":
                node.attrib["group"] = "4"
        if node.tag == "body" and "childclass" in node.attrib:
            del node.attrib["childclass"]
    return clone


def build_dual_scene_xml(a1_xml: Path, wbl_xml: Path, out_xml: Path) -> Path:
    """Create a combined side-by-side dual-robot MJCF for interactive 3D comparison."""
    a1_root = ET.parse(a1_xml).getroot()
    wbl_root = ET.parse(wbl_xml).getroot()

    a1_mesh_dir = (a1_xml.parent / a1_root.find("compiler").attrib.get("meshdir", "")).resolve()
    wbl_mesh_dir = (wbl_xml.parent / wbl_root.find("compiler").attrib.get("meshdir", "")).resolve()

    assets_xml = []
    a1_mesh_map: dict[str, str] = {}
    for m in a1_root.findall("./asset/mesh"):
        old_n = m.attrib["name"]
        new_n = f"a1_{old_n}"
        a1_mesh_map[old_n] = new_n
        fp = (a1_mesh_dir / m.attrib["file"]).resolve()
        assets_xml.append(f'<mesh name="{new_n}" file="{fp}" />')

    wbl_mesh_map: dict[str, str] = {}
    for m in wbl_root.findall("./asset/mesh"):
        old_n = m.attrib["name"]
        new_n = f"wbl_{old_n}"
        wbl_mesh_map[old_n] = new_n
        fp = (wbl_mesh_dir / m.attrib["file"]).resolve()
        assets_xml.append(f'<mesh name="{new_n}" file="{fp}" />')

    a1_body = _prefix_subtree(a1_root.find("./worldbody/body"), "a1_", a1_mesh_map)
    wbl_body = _prefix_subtree(wbl_root.find("./worldbody/body"), "wbl_", wbl_mesh_map)

    scene_text = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="a1_vs_wblwbl_dual">
    <compiler angle="radian" autolimits="true" />
    <asset>
        {''.join(assets_xml)}
        <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="512" height="512" />
        <texture name="plane" type="2d" builtin="flat" rgb1="0.88 0.89 0.91" rgb2="0.88 0.89 0.91" width="512" height="512" mark="cross" markrgb="0.55 0.55 0.58" />
        <material name="plane" reflectance="0.0" texture="plane" texrepeat="3 3" texuniform="true" />
    </asset>
    <worldbody>
        <light directional="true" castshadow="true" diffuse=".8 .8 .8" pos="0 0 10" dir="0 0 -10" />
        <geom name="floor" type="plane" material="plane" pos="0 0 0" size="0 0 1" />
        {ET.tostring(a1_body, encoding="unicode")}
        {ET.tostring(wbl_body, encoding="unicode")}
    </worldbody>
</mujoco>
"""
    out_xml.write_text(scene_text)
    return out_xml


def launch_interactive_viewer(
    clip_name: str,
    robot: str,
    xml_path: Path,
    a1_pose: np.ndarray,
    wbl_pose: np.ndarray,
    dt: float = 0.02,
    speed: float = 0.5,
    dual_offset_y: float = 0.42,
) -> None:
    """Play original A1, retargeted WBLWBL, or both side-by-side in MuJoCo GUI."""
    import time
    import mujoco.viewer

    model = _prep_render_model(xml_path)
    data = mj.MjData(model)

    if robot == "dual":
        a1_j = np.array([model.joint(f"a1_{l}_{s}_joint").qposadr[0] for l in TARGET_LEGS for s in JOINT_SUFFIXES])
        wbl_j = np.array([model.joint(f"wbl_{l}_{s}_joint").qposadr[0] for l in TARGET_LEGS for s in JOINT_SUFFIXES])
        a1_r = int(model.body("a1_trunk").jntadr[0])
        wbl_r = int(model.body("wbl_base_link").jntadr[0])
        a1_q0 = int(model.jnt_qposadr[a1_r])
        wbl_q0 = int(model.jnt_qposadr[wbl_r])
    else:
        pose = wbl_pose if robot == "wbl" else a1_pose
        q_idx = np.array([model.joint(f"{l}_{s}_joint").qposadr[0] for l in TARGET_LEGS for s in JOINT_SUFFIXES])

    print(f"Launching MuJoCo GUI for {robot} / {clip_name} ({len(wbl_pose)} frames @ {speed}x speed). Close window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.9 if robot == "dual" else 1.5
        viewer.cam.azimuth = 125
        viewer.cam.elevation = -18
        k = 0
        while viewer.is_running():
            if robot == "dual":
                wi = k
                ai = int(round(wi * (len(a1_pose) - 1) / max(1, len(wbl_pose) - 1)))
                arow, wrow = a1_pose[ai], wbl_pose[wi]
                data.qpos[a1_q0 : a1_q0 + 3] = arow[:3] + np.array([0.0, dual_offset_y, 0.0])
                data.qpos[a1_q0 + 3 : a1_q0 + 7] = arow[[6, 3, 4, 5]]
                data.qpos[a1_j] = arow[7:19]
                data.qpos[wbl_q0 : wbl_q0 + 3] = wrow[:3] + np.array([0.0, -dual_offset_y, 0.0])
                data.qpos[wbl_q0 + 3 : wbl_q0 + 7] = wrow[[6, 3, 4, 5]]
                data.qpos[wbl_j] = wrow[7:19]
                mj.mj_forward(model, data)
                mid = 0.5 * (arow[:2] + wrow[:2])
                viewer.cam.lookat[:] = [mid[0], mid[1], 0.30]
                k = (k + 1) % len(wbl_pose)
            else:
                row = pose[k]
                data.qpos[:3] = row[:3]
                data.qpos[3:7] = row[[6, 3, 4, 5]]
                data.qpos[q_idx] = row[7:19]
                mj.mj_forward(model, data)
                viewer.cam.lookat[:] = [row[0], row[1], 0.30]
                k = (k + 1) % len(pose)
            viewer.sync()
            time.sleep(dt / max(speed, 1e-3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", nargs="+", default=["trot1", "trot2", "hop1", "hop2"])
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--wbl-mjcf", type=Path, default=DEFAULT_WBL_MJCF)
    parser.add_argument("--scale-x", type=float, default=1.21)
    parser.add_argument("--scale-y", type=float, default=0.65)
    parser.add_argument("--scale-z", type=float, default=1.21)
    parser.add_argument("--time-stretch", type=float, default=1.10)
    parser.add_argument("--min-width", type=float, default=0.145)
    parser.add_argument("--interactive", choices=["none", "dual", "wbl", "a1"], default="none")
    parser.add_argument("--interactive-clip", default="hop1")
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    a1_xml = ensure_a1_mjcf(DEFAULT_A1_URDF, DEFAULT_A1_MESH_DIR, DEFAULT_A1_MJCF_DIR)
    dual_xml = build_dual_scene_xml(a1_xml, args.wbl_mjcf, args.review_dir / "dual_scene.xml")
    a1_kin = RobotKinematics(a1_xml, TARGET_LEGS)
    wbl_kin = RobotKinematics(args.wbl_mjcf, TARGET_LEGS)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.review_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    exported_files = []
    cached_poses = {}

    for clip in args.clips:
        src = args.source_dir / f"{clip}.txt"
        doc = json.loads(src.read_text())
        raw = np.asarray(doc["Frames"], dtype=float)
        src_dt = float(doc["FrameDuration"])

        pose, a1_pose, fit_errors, src_info = solve_retarget(
            a1_kin,
            wbl_kin,
            raw,
            src_dt,
            scale_xyz=(args.scale_x, args.scale_y, args.scale_z),
            time_stretch=args.time_stretch,
            min_width=args.min_width,
        )
        frames_61 = to_wmp_frames(wbl_kin, pose, dt=0.02)
        metrics = audit_motion(clip, wbl_kin, pose, frames_61, fit_errors, src_info, dt=0.02)
        metrics["source_file"] = str(src)
        metrics["source_sha256"] = sha256_file(src)

        out_txt = args.output_dir / f"{clip}.txt"
        payload = {
            "LoopMode": doc.get("LoopMode", "Wrap"),
            "FrameDuration": 0.02,
            "MotionWeight": float(doc.get("MotionWeight", 1.0)),
            "Frames": np.round(frames_61, 6).tolist(),
        }
        out_txt.write_text(json.dumps(payload, separators=(",", ": ")))
        exported_files.append(str(out_txt))
        cached_poses[clip] = (a1_pose, pose)

        if not args.skip_render and args.interactive == "none":
            mp4, side_png, obl_png = render_comparison(
                clip, a1_xml, args.wbl_mjcf, a1_pose, pose, a1_kin, wbl_kin, args.review_dir, dt=0.02
            )
            metrics["review_mp4"] = str(mp4)
            metrics["review_keyframes_side_png"] = str(side_png)
            metrics["review_keyframes_34_png"] = str(obl_png)

        summary.append(metrics)

    # Verify all exported WMP files through rsl_rl.datasets.motion_loader.AMPLoader.
    if str(WMP_ROOT) not in sys.path:
        sys.path.insert(0, str(WMP_ROOT))
    from rsl_rl.datasets.motion_loader import AMPLoader

    loader = AMPLoader(
        device="cpu",
        time_between_frames=0.02,
        preload_transitions=True,
        num_preload_transitions=2000,
        motion_files=exported_files,
    )
    report = {
        "params": {
            "scale_xyz": [args.scale_x, args.scale_y, args.scale_z],
            "time_stretch": args.time_stretch,
            "min_width_m": args.min_width,
            "target_joint_order": list(TARGET_LEGS),
        },
        "amp_loader_check": {
            "num_motions": int(loader.num_motions),
            "observation_dim": int(loader.observation_dim),
            "trajectory_lens_s": [float(x) for x in loader.trajectory_lens],
        },
        "dual_scene_xml": str(dual_xml),
        "clips": summary,
    }
    (args.output_dir / "retarget_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.review_dir / "retarget_report.json").write_text(json.dumps(report, indent=2) + "\n")
    readme_text = (
        "# WBLWBL Dog V1 AMP Expert Motions (A1-Matched 4-Clip Pool)\n\n"
        f"Retargeted from `datasets/mocap_motions/{{trot1,trot2,hop1,hop2}}.txt` using "
        f"`isaaclab_migration/scripts/retarget_wblwbl_hops.py` (scale_xyz=[{args.scale_x}, {args.scale_y}, {args.scale_z}], "
        f"Froude time_stretch={args.time_stretch}, min_width={args.min_width} m, joint order `RR, RL, FR, FL`, 50 Hz, 61 columns).\n"
    )
    (args.output_dir / "README.md").write_text(readme_text)
    print(json.dumps(report, indent=2))

    if args.interactive != "none":
        a1_p, wbl_p = cached_poses[args.interactive_clip]
        target_xml = dual_xml if args.interactive == "dual" else (args.wbl_mjcf if args.interactive == "wbl" else a1_xml)
        launch_interactive_viewer(args.interactive_clip, args.interactive, target_xml, a1_p, wbl_p, speed=args.speed)


if __name__ == "__main__":
    main()
