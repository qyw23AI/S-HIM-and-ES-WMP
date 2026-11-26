"""Command-only stopping and recovery for parallel course evaluation."""

import torch


class StopAndGo:
    APPROACH, BRAKE, HOLD, RESUME = range(4)

    def __init__(self, commands, pause_at, hold_steps, speed_limit, trigger_distance=None):
        self.pause_at = pause_at
        self.hold_steps = hold_steps
        self.speed_limit = speed_limit
        self.trigger_distance = trigger_distance
        count = len(commands)
        self.phase = torch.zeros(count, device=commands.device, dtype=torch.long)
        self.hold = torch.zeros_like(self.phase)
        self.episode = torch.zeros_like(self.phase)
        self.trigger_step = torch.full_like(self.phase, -1)
        self.settled_step = torch.full_like(self.phase, -1)
        self.resume_step = torch.full_like(self.phase, -1)
        self.original = torch.zeros_like(commands[:, :3])
        self.resume_x = commands[:, 0].clone().zero_()
        self.recovery = self.resume_x.clone()
        self.peak_hold_speed = self.resume_x.clone()
        self.stop_action_peak = self.resume_x.clone()
        self.min_stop_speed = torch.full_like(self.resume_x, float("inf"))
        self.trigger_offset = torch.full_like(self.resume_x, float("nan"))
        self.raw_events = torch.zeros((count, 4), device=commands.device, dtype=torch.long)
        self.delivered_events = torch.zeros_like(self.raw_events)
        self.event_sites = torch.zeros_like(self.raw_events)
        self.records = []

    def prepare(self, step, commands, x, obstacle_x=None):
        """Select and apply the command seen by both controller and simulator."""
        if self.trigger_distance is None:
            reached = step == self.pause_at
        else:
            gap = obstacle_x - x
            reached = (step >= self.pause_at) & (gap >= 0) & (gap <= self.trigger_distance)
        trigger = (self.phase == self.APPROACH) & reached
        self.phase[trigger] = self.BRAKE
        self.original[trigger] = commands[trigger, :3]
        self.trigger_step[trigger] = step
        if obstacle_x is not None:
            self.trigger_offset[trigger] = (obstacle_x - x)[trigger]

        release = (self.phase == self.HOLD) & (self.hold >= self.hold_steps)
        self.phase[release] = self.RESUME
        self.resume_step[release] = step
        self.resume_x[release] = x[release]
        commands[release, :3] = self.original[release]
        stopping = (self.phase == self.BRAKE) | (self.phase == self.HOLD)
        commands[stopping, :3] = 0.
        return stopping, trigger | stopping | release

    def events(self, frame, stopping, mask):
        """Count raw and delivered events per phase and optionally mask stop frames."""
        rows = torch.arange(len(frame), device=frame.device)
        raw = frame.flatten(1).count_nonzero(dim=1)
        self.raw_events[rows, self.phase] += raw
        self.event_sites[rows, self.phase] += frame[0].numel()
        self.delivered_events[rows, self.phase] += torch.where(stopping & mask, 0, raw)
        return frame.masked_fill(stopping[:, None, None, None], 0) if mask else frame

    def advance(self, step, end_x, speed, done, action):
        stopping = (self.phase == self.BRAKE) | (self.phase == self.HOLD)
        settled = stopping & (speed <= self.speed_limit) & ~done
        self.min_stop_speed[stopping] = torch.minimum(self.min_stop_speed[stopping], speed[stopping])
        first = settled & (self.settled_step < 0)
        self.settled_step[first] = step + 1
        held = settled & (self.phase == self.HOLD)
        self.hold[held] += 1
        action_peak = action[stopping].abs().amax(dim=1)
        self.stop_action_peak[stopping] = torch.maximum(self.stop_action_peak[stopping], action_peak)
        self.hold[stopping & ~settled] = 0
        self.peak_hold_speed[settled] = torch.maximum(self.peak_hold_speed[settled], speed[settled])
        self.peak_hold_speed[stopping & ~settled] = 0.
        self.phase[settled] = self.HOLD
        self.phase[stopping & ~settled] = self.BRAKE
        resumed = self.phase == self.RESUME
        self.recovery[resumed] = end_x[resumed] - self.resume_x[resumed]
        self._record(done, completed=True)
        self.episode += done
        self.phase[done] = self.APPROACH
        self.hold[done] = 0
        self.trigger_step[done] = self.settled_step[done] = self.resume_step[done] = -1
        self.trigger_offset[done] = float("nan")
        self.min_stop_speed[done] = float("inf")
        self.peak_hold_speed[done] = self.recovery[done] = self.stop_action_peak[done] = 0.
        self.raw_events[done] = self.delivered_events[done] = self.event_sites[done] = 0

    def _record(self, selected, completed):
        for index in (selected & (self.trigger_step >= 0)).nonzero().flatten().tolist():
            stages = {}
            for phase, name in enumerate(("approach", "brake", "hold", "resume")):
                sites = int(self.event_sites[index, phase])
                stages[name] = {"sampled_sites": sites,
                                "raw_density": int(self.raw_events[index, phase]) / sites if sites else None,
                                "delivered_density": int(self.delivered_events[index, phase]) / sites if sites else None}
            offset = float(self.trigger_offset[index])
            resume = int(self.resume_step[index])
            min_speed = float(self.min_stop_speed[index])
            self.records.append({
                "env": index, "episode": int(self.episode[index]), "completed_physical_episode": completed,
                "trigger_step": int(self.trigger_step[index]), "trigger_distance_m": offset if offset == offset else None,
                "original_command": self.original[index].tolist(), "settled_step": int(self.settled_step[index]),
                "resume_step": resume, "hold_start_step": resume - self.hold_steps if resume >= 0 else None,
                "min_planar_speed_mps": min_speed if min_speed < float("inf") else None,
                "hold_peak_planar_speed_mps": float(self.peak_hold_speed[index]),
                "stopping_action_max_abs": float(self.stop_action_peak[index]),
                "recovery_distance_m": float(self.recovery[index]), "event_density": stages,
            })

    def finish(self):
        self._record(self.trigger_step >= 0, completed=False)
        return self.records
