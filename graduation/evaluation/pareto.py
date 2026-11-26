"""Summarize matched A1 runs and compute a per-condition performance/energy Pareto front."""
from collections import defaultdict
import csv
import json


ENERGY_MODES = ("conservative_first_layer_mac", "strict_event_ac")
CONDITIONS = ("protocol_version", "terrain", "goal_distance_m", "event_threshold", "camera_hz",
              "delay_frames", "frame_stride", "blur_pixels", "pause_at", "pause_steps", "resume_distance_m",
              "episode_seconds", "num_envs", "steps", "flow_mode", "mask_events_on_pause",
              "pause_trigger_distance_m", "stop_speed_mps")
ENERGY_COLUMNS = {mode: f"{mode}_pj_per_decision" for mode in ENERGY_MODES}
FIELDS = ("purpose", "method", "seed", *CONDITIONS, "successes", "finished_episodes", "censored_episodes",
          "timeout_episodes", "memory_control", "checkpoint_memory", "success_rate", "tracking_rmse_mps", "energy_pj_per_decision",
          *ENERGY_COLUMNS.values(), "source")


def pareto(rows, mode):
    """Build one front per condition using only seeds shared by every method."""
    column = ENERGY_COLUMNS[mode]
    groups = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if (row["purpose"] == "formal" and row["protocol_version"] == 2 and row["finished_episodes"]
                and row[column] is not None):
            conditions = tuple(row[key] for key in CONDITIONS)
            method = row["method"]
            if method == "event-spiking-wmp":
                memory = row.get("checkpoint_memory") or "grsn"
                control = row.get("memory_control") or "checkpoint"
                if memory != "grsn" or control != "checkpoint":
                    method = f"{method} ({memory}, {control})"
            trials = groups[conditions][method]
            # Repeated results for a seed form one seed-level point.
            trials.setdefault(row["seed"], []).append(row)

    fronts = []
    for conditions, methods in groups.items():
        if len(methods) < 2:
            continue
        seeds = sorted(set.intersection(*(set(trials) for trials in methods.values())))
        if not seeds:
            continue
        points = []
        for method, trials in methods.items():
            matched = [row for seed in seeds for row in trials[seed]]
            rate = sum(row["successes"] for row in matched) / sum(row["finished_episodes"] for row in matched)
            energy = sum(sum(row[column] for row in trials[seed]) / len(trials[seed]) for seed in seeds) / len(seeds)
            points.append({"method": method, "success_rate": rate, "energy_pj_per_decision": energy,
                           "runs": len(matched), "seeds": seeds})
        frontier = [
            point for point in sorted(points, key=lambda row: row["energy_pj_per_decision"])
            if not any(other["energy_pj_per_decision"] <= point["energy_pj_per_decision"]
                       and other["success_rate"] >= point["success_rate"]
                       and (other["energy_pj_per_decision"] < point["energy_pj_per_decision"]
                            or other["success_rate"] > point["success_rate"]) for other in points)
        ]
        fronts.append({"energy_mode": mode, "conditions": dict(zip(CONDITIONS, conditions)), "points": points, "front": frontier})
    return fronts


class NoUsableReports(ValueError):
    """Raised when the inputs contain no compatible visual reports."""


def summarize(reports, out):
    """Aggregate visual reports and write the CSV and Pareto outputs."""
    files = {child.resolve() for path in reports
             for child in (path.rglob("validation.json") if path.is_dir() else [path])}
    rows = []
    for path in sorted(files):
        payload = json.loads(path.read_text())
        if payload.get("status") != "passed" or "episodes" not in payload or "purpose" not in payload:
            continue
        row = {key: payload.get(key) for key in FIELDS}
        row["frame_stride"] = payload.get("frame_stride", 1)
        row["flow_mode"] = payload.get("flow_mode", "random")
        row["mask_events_on_pause"] = payload.get("mask_events_on_pause", False)
        if payload.get("pause_steps") and "stop_and_go" not in payload:
            # Old pauses forced actions to zero. Keep them in CSV under their own condition.
            row["stop_speed_mps"] = "legacy_action_zero"
        energy = payload.get("energy", {})
        row["energy_pj_per_decision"] = energy.get("energy_pj_per_decision")
        for mode, column in ENERGY_COLUMNS.items():
            row[column] = energy.get(mode, {}).get("energy_pj_per_decision")
        row["source"] = str(path)
        rows.append(row)
    if not rows:
        raise NoUsableReports("no usable visual evaluation reports")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    fronts = [front for mode in ENERGY_MODES for front in pareto(rows, mode)]
    (out / "pareto.json").write_text(json.dumps(fronts, indent=2) + "\n")
    if fronts:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for index, result in enumerate(fronts, 1):
            fig, ax = plt.subplots(figsize=(7.6, 4.5), layout="constrained")
            for point in result["points"]:
                x = point["energy_pj_per_decision"] / 1e6
                y = point["success_rate"] * 100
                ax.scatter(x, y)
                ax.annotate(point["method"], (x, y), xytext=(5, 5), textcoords="offset points")
            front = result["front"]
            ax.plot([pt["energy_pj_per_decision"] / 1e6 for pt in front],
                    [pt["success_rate"] * 100 for pt in front], alpha=.75)
            ax.set(xlabel="Estimated synaptic energy (µJ/control step)", ylabel="Obstacle success (%)",
                   title=f"A1 {result['conditions']['terrain']} · {result['energy_mode'].replace('_', ' ')}")
            ax.set_xscale("log")
            ax.grid(alpha=.15)
            fig.savefig(out / f"pareto-{index:02}.png", dpi=180)
            plt.close(fig)
    print(f"Summarized {len(rows)} runs across {len(fronts)} formal Pareto conditions.")
