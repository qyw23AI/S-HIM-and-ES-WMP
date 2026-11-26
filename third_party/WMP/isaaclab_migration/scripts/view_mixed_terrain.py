#!/usr/bin/env python3
"""Open the configured mixed training terrain (e.g. wblwbl_dog_v1) standalone in Isaac Lab.

Uses a visual-only UsdGeom.Mesh (no PhysX BVH collider cooking) and sim.render()
so the full 10x20 training terrain opens quickly for interactive inspection.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "isaaclab_migration"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="wblwbl_dog_v1", help="Task name registered in legacy/")
parser.add_argument("--rows", type=int, default=10, help="Number of difficulty rows (0..rows-1)")
parser.add_argument("--cols", type=int, default=20, help="Number of terrain columns")
parser.add_argument("--focus_type", type=str, default=None, help="Terrain type to focus camera on (e.g. tire_array, high_wall, gap, pit)")
parser.add_argument("--focus_level", type=int, default=6, help="Difficulty row (0..9) for --focus_type")
parser.add_argument("--seconds", type=float, default=0.0, help="0: stay open until window is closed")
parser.add_argument("--screenshot", type=Path, default=None, help="Optional PNG path to save initial camera view (GUI mode)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
app = launcher.app


def _build_visual_mesh(stage, prim_path: str, vertices, faces) -> None:
    import numpy as np
    from pxr import Gf, UsdGeom, Vt

    verts = np.asarray(vertices, dtype=np.float32)
    tris = np.asarray(faces, dtype=np.int32)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts))
    mesh.CreateFaceVertexCountsAttr(np.full(len(tris), 3, dtype=np.int32))
    mesh.CreateFaceVertexIndicesAttr(tris.ravel())
    mesh.CreateSubdivisionSchemeAttr("none")
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, vmin)), Gf.Vec3f(*map(float, vmax))]))


def _add_spawn_marker(stage, prim_path: str, xyz, color) -> None:
    from pxr import Gf, UsdGeom, Vt

    cyl = UsdGeom.Cylinder.Define(stage, prim_path)
    cyl.CreateRadiusAttr(0.12)
    cyl.CreateHeightAttr(0.04)
    cyl.CreateAxisAttr("Z")
    cyl.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, color))]))
    UsdGeom.XformCommonAPI(cyl).SetTranslate(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2] + 0.03)))


def _focus_camera(sim, terrain, rows: int, cols: int, focus_type: str | None, level: int) -> None:
    if focus_type is None:
        mid_x = 0.5 * rows * terrain.env_length
        mid_y = 0.5 * cols * terrain.env_width
        sim.set_camera_view(eye=(-18.0, mid_y, 34.0), target=(mid_x, mid_y, 0.0))
        return

    matching = [idx for idx, name in enumerate(terrain.column_types) if name == focus_type]
    if not matching:
        raise ValueError(f"Unknown --focus_type {focus_type!r}; available: {sorted(set(terrain.column_types))}")
    col = matching[len(matching) // 2]
    row = max(0, min(rows - 1, int(level)))
    tx, ty, tz = terrain.env_origins[row, col]
    sim.set_camera_view(
        eye=(float(tx - 2.6), float(ty - 5.8), float(tz + 4.2)),
        target=(float(tx + 1.8), float(ty), float(tz + 0.15)),
    )


try:
    import time
    import isaaclab.sim as sim_utils
    from legacy import load_task_config
    from wmp_lab.terrain import Terrain

    t0 = time.monotonic()
    cfg, _ = load_task_config(args.task)
    cfg.terrain.num_rows = args.rows
    cfg.terrain.num_cols = args.cols
    cfg.terrain.border_size = 5.0
    terrain = Terrain(cfg.terrain, max(4, args.rows * args.cols))
    print("TERRAIN_COLUMNS", list(enumerate(terrain.column_types)), flush=True)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.02, device=args.device))
    light = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.82, 0.84, 0.88))
    light.func("/World/Light", light)
    sun = sim_utils.DistantLightCfg(intensity=2200.0, angle=2.0)
    sun.func("/World/Sun", sun, translation=(0.0, 0.0, 20.0))

    v = terrain.vertices.copy()
    v[:, :2] -= cfg.terrain.border_size
    _build_visual_mesh(sim.stage, "/World/ground", v, terrain.triangles)
    shader = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.40, 0.45), roughness=0.85)
    shader.func("/World/ground/material", shader)
    sim_utils.bind_visual_material("/World/ground", "/World/ground/material")

    lane_colors = {
        "tire_array": (0.15, 0.75, 0.95),
        "high_wall": (0.95, 0.35, 0.15),
        "gap": (0.25, 0.85, 0.35),
        "pit": (0.95, 0.85, 0.20),
    }
    for col_idx, kind in enumerate(terrain.column_types):
        color = lane_colors.get(kind)
        if color is None:
            continue
        for row_idx in range(args.rows):
            ox, oy, oz = terrain.env_origins[row_idx, col_idx]
            _add_spawn_marker(sim.stage, f"/World/spawns/c{col_idx}_r{row_idx}", (ox, oy, oz), color)

    sim.reset()
    _focus_camera(sim, terrain, args.rows, args.cols, args.focus_type, args.focus_level)

    for _ in range(4):
        sim.render()

    ui_window = None
    if not args.headless:
        try:
            import omni.ui as ui

            state = {"focus": args.focus_type, "level": max(0, min(args.rows - 1, args.focus_level))}
            available_types = list(dict.fromkeys(terrain.column_types))
            ui_window = ui.Window("Mixed Terrain Viewer", width=260, height=340)
            with ui_window.frame:
                with ui.VStack(spacing=6):
                    ui.Label(f"Task: {args.task} ({args.rows}x{args.cols})", height=22)
                    lvl_label = ui.Label(f"Level: {state['level']}", height=20)
                    slider = ui.IntSlider(min=0, max=max(0, args.rows - 1), height=22)
                    slider.model.set_value(state["level"])

                    def _on_level(model):
                        state["level"] = model.get_value_as_int()
                        lvl_label.text = f"Level: {state['level']}"
                        if state["focus"] is not None:
                            _focus_camera(sim, terrain, args.rows, args.cols, state["focus"], state["level"])

                    slider.model.add_value_changed_fn(_on_level)

                    def _jump(kind):
                        state["focus"] = kind
                        _focus_camera(sim, terrain, args.rows, args.cols, kind, state["level"])

                    ui.Button("Overview", height=26, clicked_fn=lambda: _jump(None))
                    for kind in available_types:
                        ui.Button(f"Focus: {kind}", height=24, clicked_fn=lambda k=kind: _jump(k))
        except Exception as exc:
            print(f"[view_mixed_terrain] UI panel skipped: {exc}", flush=True)

    if args.screenshot is not None and not args.headless:
        import omni.kit.viewport.utility as vp_util

        vp = vp_util.get_active_viewport_window()
        if vp is not None:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            vp_util.capture_viewport_to_file(vp.viewport_api, str(args.screenshot))
            for _ in range(4):
                sim.render()

    print(f"READY in {time.monotonic() - t0:.2f}s (focus_type={args.focus_type}, level={args.focus_level})", flush=True)
    start = time.monotonic()
    while app.is_running() and (args.seconds <= 0.0 or time.monotonic() - start < args.seconds):
        sim.render()
        time.sleep(0.02)
    print("VIEWER_LOOP_DONE", flush=True)
finally:
    try:
        from isaaclab.sim import SimulationContext
        SimulationContext.clear_instance()
    except Exception:
        pass
    app.close()
    print("VIEWER_CLOSED", flush=True)
