"""Meter the executed WMP deployment path and its camera-rate updates."""

from collections import defaultdict

from torch import nn

from graduation.evaluation.energy import SynapticMeter


class WMPDeploymentMeter(SynapticMeter):
    """Count the policy and world model with the same accounting in all WMP entry points."""

    def __init__(self, teacher):
        super().__init__(nn.ModuleDict({"policy": teacher.policy, "world": teacher.world}))
        self.teacher = teacher
        self.phase = None
        self.updates = defaultdict(lambda: {"batches": 0, "frames": 0, "reset_frames": 0, "mac": 0})

    def act(self, obs, step, **kwargs):
        teacher = self.teacher
        if step % teacher.period == 0:
            reset_frames = int(teacher.first.count_nonzero())
            self.phase = "startup" if teacher.state is None else "reset_batch" if reset_frames else "steady"
            row = self.updates[self.phase]
            row["batches"] += 1
            row["frames"] += len(obs)
            row["reset_frames"] += reset_frames
        try:
            return teacher.act(obs, step, **kwargs)
        finally:
            self.phase = None

    def _count(self, name, module, inputs, output):
        before = self.rows[name]["mac"]
        super()._count(name, module, inputs, output)
        if self.phase and name.startswith("world."):
            self.updates[self.phase]["mac"] += self.rows[name]["mac"] - before

    def report(self, decisions, policy_hz):
        report = super().report(decisions, policy_hz)
        subsystems = {
            "policy": "policy.", "cnn": "world.encoder._cnn.", "prop_mlp": "world.encoder._mlp.",
            "rssm": "world.dynamics.",
        }
        report["subsystems"] = {
            label: self.totals([row for name, row in self.rows.items() if name.startswith(prefix)],
                               decisions, policy_hz, conservative=True)
            for label, prefix in subsystems.items()
        }
        report["scope"] = "policy_and_world_model"
        report["world_model_hz"] = policy_hz / self.teacher.period
        report["world_updates"] = {key: dict(self.updates[key]) for key in ("startup", "reset_batch", "steady")}
        return report
