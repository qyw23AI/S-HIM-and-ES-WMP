#!/usr/bin/env python3
"""Interactive GUI viewer and video/keyframe recorder for S-HIM and HIMLoco policies."""

import argparse
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graduation.paths import HIM
from graduation.sim.launch import launch


def _encode_media(capture_dir: Path, step_dt: float, capture_every: int,
                  video_out: Path | None, keyframes_out: Path | None, labels: dict[str, str]):
    """Convert captured PNG frames into an MP4 video and an annotated keyframe strip."""
    from PIL import Image, ImageDraw

    frames = sorted(capture_dir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError(f"No frame_*.png captured in {capture_dir}")
    fps = max(1, int(round(1.0 / (step_dt * capture_every))))
    if video_out is not None:
        video_out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps), "-pattern_type", "glob", "-i", str(capture_dir / "frame_*.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
            str(video_out),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("VIDEO_SAVED", video_out, flush=True)

    if keyframes_out is not None:
        keyframes_out.parent.mkdir(parents=True, exist_ok=True)
        count = min(6, len(frames))
        indices = [round(i * (len(frames) - 1) / max(count - 1, 1)) for i in range(count)]
        chosen = [frames[i] for i in indices]
        cols = 3 if count >= 3 else count
        rows = (count + cols - 1) // cols
        with Image.open(chosen[0]) as im0:
            tw, th = 480, max(1, round(480 * im0.height / im0.width))
        sheet = Image.new("RGB", (cols * tw, rows * th), (18, 20, 24))
        draw = ImageDraw.Draw(sheet)
        for idx, frame_path in enumerate(chosen):
            r, c = divmod(idx, cols)
            x0, y0 = c * tw, r * th
            with Image.open(frame_path) as im:
                tile = im.convert("RGB").resize((tw, th), Image.Resampling.LANCZOS)
            sheet.paste(tile, (x0, y0))
            caption = labels.get(frame_path.name, frame_path.stem)
            draw.rectangle([x0, y0, x0 + tw, y0 + 22], fill=(0, 0, 0))
            draw.text((x0 + 8, y0 + 5), caption, fill=(240, 240, 240))
        sheet.save(keyframes_out)
        print("KEYFRAMES_SAVED", keyframes_out, flush=True)


def run(args):
    import torch
    from himloco_lab.visualization import ViewportCapture, configure_viewport
    from graduation.sim.himloco import make_blind_env
    from graduation.sim.shim_stairs import SPAWN_M, make_stair_env
    from scripts.eval_shim_panel import load_policy, make_flat_env

    if args.terrain == "flat":
        env, adapter = make_flat_env(args)
    elif args.terrain == "stairs":
        env, adapter = make_stair_env(args)
    else:
        args.reward_preset = "default"
        env, adapter, _ = make_blind_env(args)

    labels: dict[str, str] = {}
    step_dt = 0.02
    try:
        obs, critic = adapter.reset()
        if args.viewer_resolution and not args.headless:
            configure_viewport(args.viewer_resolution)
        if args.capture_dir:
            adapter.capture = ViewportCapture(args.capture_dir, args.capture_every)
        step_dt = float(env.step_dt)
        width = adapter.num_one_step_obs
        policy, iteration = load_policy(args, obs, critic, width, adapter.num_actions, env.device)
        origin = env.core.env_origins.clone()

        def observe(raw_obs):
            env.core.commands[:, 0] = args.command[0]
            env.core.commands[:, 1] = args.command[1]
            env.core.commands[:, 2] = args.command[2]
            env.core.commands[:, 3] = 0.0
            frame = raw_obs[:, :width].clone()
            frame[:, :3] = env.core.commands[:, :3] * env.core.commands_scale
            return frame

        frame = observe(obs)
        history = frame[:, None].repeat(1, obs.shape[1] // width, 1)
        started = time.monotonic()
        deadline = started + args.seconds if args.seconds > 0 else float("inf")
        step = 0
        print(f"PLAY_SHIM_STARTED iter={iteration} terrain={args.terrain} cmd={args.command}", flush=True)
        with torch.inference_mode():
            while step < args.steps if args.steps is not None else time.monotonic() < deadline:
                tick = time.monotonic()
                if args.follow_camera:
                    pos0 = env.core.root_states[args.camera_env, :3].tolist()
                    eye = (pos0[0] - 1.5, pos0[1] - 2.0, pos0[2] + 0.85)
                    target = (pos0[0] + 0.15, pos0[1], pos0[2] + 0.10)
                    env.sim.set_camera_view(eye, target)
                action = policy(history.flatten(1))
                obs, _, _, done, *_ = adapter.step(action)
                step += 1
                if args.capture_dir and step % args.capture_every == 0:
                    pre_pos = getattr(env, "pre_reset_position", env.core.root_states[:, :3])
                    pos_rel = (pre_pos[args.camera_env] - origin[args.camera_env]).tolist()
                    vx = float(env.core.base_lin_vel[args.camera_env, 0])
                    x_val = pos_rel[0] + (SPAWN_M if args.terrain == "stairs" else 0.0)
                    fname = f"frame_{step:06d}.png"
                    labels[fname] = f"t={step * step_dt:.2f}s  x={x_val:.2f}m  z={pos_rel[2]:.2f}m  vx={vx:.2f}m/s"
                frame = observe(obs)
                history = torch.cat((frame[:, None], history[:, :-1]), dim=1)
                history[done] = frame[done, None]
                if args.realtime:
                    time.sleep(max(0.0, step_dt - (time.monotonic() - tick)))

        if adapter.capture is not None:
            adapter.capture.capture_final(env.sim, "final.png")
    finally:
        env.close()

    if args.capture_dir and (args.video_out or args.keyframes_out):
        _encode_media(args.capture_dir, step_dt, args.capture_every, args.video_out, args.keyframes_out, labels)


def main():
    sys.path[:0] = [str(HIM / "isaaclab_migration"), str(HIM / "rsl_rl")]
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("s-him", "himloco"), default="s-him")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--terrain", choices=("flat", "stairs", "rough"), default="flat")
    parser.add_argument("--step-height", type=float, default=0.08)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--command", type=float, nargs=3, default=[0.6, 0.0, 0.0], metavar=("VX", "VY", "YAW"))
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--camera-env", type=int, default=0)
    parser.add_argument("--follow-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--viewer-resolution", type=int, nargs=2, default=[960, 540], metavar=("W", "H"))
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--capture-every", type=int, default=5)
    parser.add_argument("--video-out", type=Path)
    parser.add_argument("--keyframes-out", type=Path)
    parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=False)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.seconds > 0 and args.steps == 300:
        args.steps = None
    launch(AppLauncher, args, run)


if __name__ == "__main__":
    main()
