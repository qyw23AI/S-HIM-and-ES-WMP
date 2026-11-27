#!/usr/bin/env python3
"""Summarize key metrics across iterations from a blind.py run directory."""

import ast
import json
from pathlib import Path
import statistics
import sys


def load_rows(run_dir: Path):
    val = run_dir / "validation.json"
    if val.exists():
        data = json.loads(val.read_text())
        if data.get("results"):
            return data["results"]
    log = run_dir / "console.log"
    rows = []
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            s = line.strip()
            if s.startswith("{") and s.endswith("}") and "iteration" in s:
                try:
                    rows.append(ast.literal_eval(s))
                except Exception:
                    try:
                        rows.append(json.loads(s))
                    except Exception:
                        pass
    return rows


def main():
    for arg in sys.argv[1:]:
        p = Path(arg)
        rows = load_rows(p)
        print(f"\n=== {p.name} (iters={len(rows)}) ===")
        if not rows:
            continue
        vel = [r["command_aligned_velocity_mps"] for r in rows if "command_aligned_velocity_mps" in r]
        if vel:
            print(f"  cmd_vel_mps: mean={statistics.mean(vel):.5f} std={statistics.pstdev(vel):.5f} last5_mean={statistics.mean(vel[-5:]):.5f}")
        keys = [
            "iteration", "mean_reward", "resets", "command_aligned_velocity_mps",
            "terrain_level", "action_std", "learning_rate",
            "ppo_kl_mean", "ppo_kl_p95", "clip_fraction",
            "aux_update_kl_mean", "aux_update_kl_p95",
            "adapt_kl_mean", "commit_kl_mean",
            "speed_rmse_mps", "zero_speed_rmse_mps",
            "actor_grad_norm", "critic_grad_norm", "std_grad_norm",
            "actor_rel_update", "std_rel_update",
            "cmd_delta_0_to_pos06", "cmd_delta_0_to_neg06",
            "actor_layer_firing_rates", "actor_layer_silent_ratios",
        ]
        n = len(rows)
        pick = {1, 5, 10, 25, 50, 75, 100, 125, rows[-1].get("iteration")}
        for r in rows:
            if r.get("iteration") in pick or ("adapt_kl_mean" in r and r.get("iteration", 0) in (1, 5, 25, 50, 75, 100, 125)):
                sub = {}
                for k in keys:
                    if k in r and r[k] is not None:
                        v = r[k]
                        if isinstance(v, float):
                            sub[k] = round(v, 5)
                        elif isinstance(v, list):
                            sub[k] = [round(x, 4) if isinstance(x, float) else x for x in v]
                        else:
                            sub[k] = v
                print(" ", sub)


if __name__ == "__main__":
    main()
