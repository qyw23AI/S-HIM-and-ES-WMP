"""Replay every saved policy with the same GUI setup and save comparable clips."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('checkpoint_dir', type=Path)
parser.add_argument('--output_dir', type=Path)
parser.add_argument('--task', type=str, default=None, help='Task name in legacy/; defaults to checkpoint_dir/config.json args.task')
parser.add_argument('--terrain', type=str, default=None, help='Terrain name; defaults to checkpoint_dir/config.json args.terrain')
parser.add_argument('--camera_backend', choices=['raycast', 'rtx'], default=None)
parser.add_argument('--play_steps', type=int, default=1000, help='Control steps; 1000 steps = 20 simulated seconds')
parser.add_argument('--num_envs', type=int, default=4)
parser.add_argument('--terrain_col', type=int, default=10, help='Viewed terrain column; 10=gap, 15=climb, 17=tilt, 18=crawl')
parser.add_argument('--terrain_level', type=int, default=6)
parser.add_argument('--forward_speed', type=float, default=0.5)
parser.add_argument('--capture_fps', type=int, default=20,
                    help='Frame rate matched to 50 Hz simulated time')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--only', type=int, nargs='+', help='Record only these checkpoint iteration numbers')
args = parser.parse_args()

cfg_path = args.checkpoint_dir.resolve() / 'config.json'
saved_args = json.loads(cfg_path.read_text()).get('args', {}) if cfg_path.is_file() else {}
task_name = args.task or saved_args.get('task') or 'a1_amp'
terrain_name = args.terrain or saved_args.get('terrain') or 'mixed'
camera_backend = args.camera_backend or saved_args.get('camera_backend') or 'raycast'

checkpoints = sorted(args.checkpoint_dir.resolve().glob('model_*.pt'),
                     key=lambda p: int(p.stem.removeprefix('model_')))
if args.only:
    checkpoints = [p for p in checkpoints if int(p.stem.removeprefix('model_')) in args.only]
if not checkpoints:
    parser.error('No model_*.pt checkpoints found')
output = (args.output_dir or args.checkpoint_dir / 'checkpoint_videos').resolve()
output.mkdir(parents=True, exist_ok=True)
for index, checkpoint in enumerate(checkpoints, 1):
    clip = output / checkpoint.stem
    if (clip / 'preview.mp4').exists() and not args.overwrite:
        print(f'[{index}/{len(checkpoints)}] Already recorded: {checkpoint.name}', flush=True)
        continue
    clip.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for pattern in ('frame_*.png', 'depth_*.png', 'preview.mp4', 'preview.gif', 'playback.png'):
            for old in clip.glob(pattern):
                old.unlink()
    print(
        f'[{index}/{len(checkpoints)}] Recording {checkpoint.name} '
        f'(task={task_name}, terrain={terrain_name}, num_envs={args.num_envs})',
        flush=True,
    )
    command = [str(ROOT / 'isaaclab_migration/run.sh'), '--play', '--task', task_name,
               '--terrain', terrain_name, '--camera_backend', camera_backend,
               '--num_envs', str(args.num_envs), '--camera_envs', str(args.num_envs),
               '--play_steps', str(args.play_steps), '--capture_fps', str(args.capture_fps),
               '--eval_terrain_col', str(args.terrain_col), '--eval_terrain_level', str(args.terrain_level),
               '--fixed_forward_speed', str(args.forward_speed), '--follow_camera',
               '--checkpoint', str(checkpoint), '--log_dir', str(clip / 'log'),
               '--capture_dir', str(clip)]
    with (clip / 'run.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'isaaclab_migration/scripts/make_preview.py'), str(clip),
                    '--fps', str(args.capture_fps), '--mp4_only'],
                   cwd=ROOT, check=True)
