"""Poll one active local training process and evaluate its durable checkpoints."""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("pid", type=int)
parser.add_argument("run", type=Path)
parser.add_argument("--interval", type=float, default=1800., help="Seconds between training snapshots")
args = parser.parse_args()
if args.pid < 1 or not 0. < args.interval < float("inf"):
    parser.error("PID and interval must be positive")
pid, run = args.pid, args.run
project = Path(__file__).resolve().parents[1]
process = Path(f"/proc/{pid}")
log = run / "console.log"
eval_dir = run / "stairs"
eval_dir.mkdir(exist_ok=True)
def sample():
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 131072))
        recent = stream.read().decode(errors="replace").splitlines()
    rows = []
    for line in recent:
        if line.startswith("{'iteration':"):
            try:
                rows.append(ast.literal_eval(line))
            except (ValueError, SyntaxError):
                continue
    row = rows[-1] if rows else {}
    latest_fr = next((x for x in reversed(rows) if "grsn_layers" in x), {})
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
        text=True).splitlines()[0]
    used, utilization, temp = map(int, gpu.split(","))
    entry = {"utc": datetime.now(timezone.utc).isoformat(), "training_pid": pid, "gpu_mib": used,
             "gpu_percent": utilization, "gpu_c": temp, **row,
             "last_grsn_iteration": latest_fr.get("iteration"), "last_grsn_layers": latest_fr.get("grsn_layers")}
    with (run / "monitor.jsonl").open("a") as stream:
        stream.write(json.dumps(entry) + "\n")
    print(json.dumps(entry), flush=True)

def check_checkpoint(path, trials, seed):
    filename = f"{path.stem}-seed{seed}"
    dest = eval_dir / f"{filename}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    with (eval_dir / f"{filename}.log").open("w") as stream:
        subprocess.run([sys.executable, str(project / "scripts/eval_shim_stairs.py"), "--headless", "--checkpoint", str(path),
                        "--out", str(dest), "--num-envs", str(trials), "--seed", str(seed)],
                       stdout=stream, stderr=subprocess.STDOUT, check=False)
    if not dest.exists():
        print(f"evaluation failed for {filename}", flush=True)
        return {}
    result = json.loads(dest.read_text())
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}), flush=True)
    return result

def training_alive():
    try:
        command = (process / "cmdline").read_bytes().decode(errors="replace").split("\0")
    except FileNotFoundError:
        command = []
    return "s-him" in command and any(arg.endswith("bench_blind.py") for arg in command)


checked = set()
while True:
    live = training_alive()
    next_check = time.monotonic() + args.interval
    sample()
    for path in sorted(run.glob("model_*.pt")):
        step = int(re.search(r"_(\d+)", path.stem)[1])
        if step not in checked and (step == 25 or step % 100 == 0):
            checked.add(step)
            result = check_checkpoint(path, 16, 21)
            if result.get("success_rate", 0) >= .8:
                for seed in (31, 41):
                    check_checkpoint(path, 50, seed)
    if not live:
        complete = run / "model.pt"
        if not complete.is_file():
            complete = max(run.glob("model_*.pt"), default=None)
        if complete is not None:
            check_checkpoint(complete, 50, 31)
        break
    while training_alive() and (remaining := next_check - time.monotonic()) > 0:
        # Watch process liveness only so that a terminal run is evaluated promptly.
        time.sleep(min(10, remaining))
