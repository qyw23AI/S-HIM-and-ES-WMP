"""Record the first course outcome for each physical episode."""

import torch


class CourseEpisodes:
    """Track goal, failure, timeout, and right censoring for parallel courses."""

    def __init__(self, spawn, terrain_id, level, terrain_names, goal_distance):
        self.spawn = spawn.clone()
        self.terrain_id = terrain_id.clone()
        self.level = level.clone()
        self.terrain_names = terrain_names
        self.goal_distance = goal_distance
        self.number = torch.zeros_like(terrain_id)
        self.steps = torch.zeros_like(terrain_id)
        self.speed_samples = torch.zeros_like(terrain_id)
        self.progress = torch.zeros_like(spawn)
        self.speed_error = torch.zeros_like(spawn)
        self.resolved = torch.zeros_like(terrain_id, dtype=torch.bool)
        self.rows = []

    def _append(self, index, outcome):
        self.rows.append({
            "env": index, "episode": int(self.number[index]),
            "terrain": self.terrain_names[int(self.terrain_id[index])], "level": int(self.level[index]),
            "outcome": outcome, "success": outcome == "success", "completed": outcome != "censored",
            "fall": outcome == "failure", "distance_m": float(self.progress[index]),
            "control_steps": int(self.steps[index]), "speed_sq_error": float(self.speed_error[index]),
            "speed_samples": int(self.speed_samples[index]),
        })

    def advance(self, end_x, speed_error, sample_speed, failed, done, reset_x, terrain_id, level):
        active = ~self.resolved
        self.steps += active
        self.progress = torch.where(active, torch.maximum(self.progress, end_x - self.spawn), self.progress)
        valid_speed = active & sample_speed
        self.speed_error += torch.where(valid_speed, speed_error.square(), 0.)
        self.speed_samples += valid_speed

        reached = self.progress >= self.goal_distance
        resolved = active & (done | reached)
        for index in resolved.nonzero().flatten().tolist():
            outcome = "failure" if failed[index] else "success" if reached[index] else "timeout"
            self._append(index, outcome)
        self.resolved |= resolved
        self.resolved[done] = False
        self.number += done
        self.spawn[done] = reset_x[done]
        self.terrain_id[done] = terrain_id[done]
        self.level[done] = level[done]
        self.progress[done] = self.speed_error[done] = 0
        self.steps[done] = self.speed_samples[done] = 0

    def finish(self):
        pending = (~self.resolved) & (self.steps > 0)
        for index in pending.nonzero().flatten().tolist():
            self._append(index, "censored")
        return self.rows
