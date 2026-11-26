"""S-HIM estimator, auxiliary objectives, and actor-critic."""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from graduation.snn.neurons import GRSN
from graduation.snn.normalization import CaReBN
from graduation.snn.population_actor import LIFActor


class SHIM(nn.Module):
    def __init__(self, obs_dim, critic_dim, actions, normalization="care", history=5,
                 estimator_width=64, estimator_layers=2, actor_widths=(128, 64),
                 population=5, ticks=4, recurrent=True, grsn_threshold=1.0, grsn_bias=None):
        super().__init__()
        self.obs_dim = obs_dim
        self.history = history
        self.estimator = nn.ModuleList(
            GRSN(obs_dim if index == 0 else estimator_width, estimator_width, normalization, recurrent,
                 threshold=grsn_threshold, initial_bias=grsn_bias)
            for index in range(estimator_layers)
        )
        self.estimate = nn.Linear(estimator_width, 19)
        self.target = nn.Sequential(nn.Linear(obs_dim, 64), nn.ELU(), nn.Linear(64, 16))
        self.prototypes = nn.Parameter(torch.randn(32, 16))
        self.actor = LIFActor(obs_dim + 19, actions, actor_widths, population, ticks, normalization)
        self.critic = nn.Sequential(nn.Linear(critic_dim, 256), nn.ELU(), nn.Linear(256, 128),
                                    nn.ELU(), nn.Linear(128, 1))
        self.log_std = nn.Parameter(torch.zeros(actions))

    def estimate_history(self, obs, statistics="fixed"):
        sequence = obs.reshape(obs.shape[0], -1, self.obs_dim)[:, :self.history].flip(1)
        for layer in self.estimator:
            sequence = layer(sequence, statistics)
        state = self.estimate(sequence[:, -1])
        return state[:, :3], F.normalize(state[:, 3:], dim=-1)

    def forward(self, obs, statistics="fixed"):
        with torch.no_grad():
            speed, latent = self.estimate_history(obs, statistics)
        return self.actor(torch.cat((obs[:, :self.obs_dim], speed, latent), dim=-1), statistics)

    @torch.no_grad()
    def estimator_diagnostics(self, obs):
        sequence = obs.reshape(obs.shape[0], -1, self.obs_dim)[:, :self.history].flip(1)
        rows = []
        for index, cell in enumerate(self.estimator):
            sequence, voltages = cell.sequence(sequence, "fixed", record_voltage=True)
            voltage = torch.stack(voltages).float().flatten()
            quantiles = torch.quantile(voltage, voltage.new_tensor([.05, .5, .95]))
            rows.append({"layer": index, "threshold": cell.threshold, "firing_rate": float(sequence.mean()),
                         "membrane_p05_p50_p95": quantiles.tolist()})
        return rows

    def auxiliary_loss(self, obs, successor, swap_weight):
        speed, latent = self.estimate_history(obs, "adapt")
        target = F.normalize(self.target(successor[:, :self.obs_dim]), dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        student, teacher = latent @ prototypes.T, target @ prototypes.T

        def assignments(logits):
            with torch.no_grad():
                matrix = (logits.float() / 0.05).exp().T
                matrix /= matrix.sum().clamp_min(1e-12)
                for _ in range(3):
                    matrix /= matrix.sum(dim=1, keepdim=True).clamp_min(1e-12) * matrix.shape[0]
                    matrix /= matrix.sum(dim=0, keepdim=True).clamp_min(1e-12) * matrix.shape[1]
                return matrix.T * matrix.shape[1]

        swap = -0.5 * (
            (assignments(teacher) * (student / 0.1).log_softmax(-1)).sum(-1).mean()
            + (assignments(student) * (teacher / 0.1).log_softmax(-1)).sum(-1).mean()
        )
        regression = F.mse_loss(speed, successor[:, self.obs_dim:self.obs_dim + 3])
        return regression + swap_weight * swap, regression.detach(), swap.detach()

    def calibrate(self, obs):
        with torch.no_grad():
            self(obs, "calibrate")
            for item in self.modules():
                if isinstance(item, CaReBN):
                    item.commit()

    def fused(self):
        model = copy.deepcopy(self).eval()
        for block in model.estimator:
            block.norm.fuse(block.projection)
        for layer, norm in zip(model.actor.layers, model.actor.norms):
            norm.fuse(layer)
        return model


def shim_spike_inputs(model):
    names = {f"actor.layers.{index}" for index in range(1, len(model.actor.layers))}
    names.update(("estimate",))
    for index, block in enumerate(model.estimator):
        if index:
            names.add(f"estimator.{index}.projection")
        if block.forget is not None:
            names.update((f"estimator.{index}.forget", f"estimator.{index}.input_gate"))
    return names
