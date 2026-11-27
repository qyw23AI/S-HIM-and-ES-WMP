"""S-HIM estimator, auxiliary objectives, and actor-critic."""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from graduation.snn.neurons import GRSN
from graduation.snn.normalization import CaReBN
from graduation.snn.population_actor import LIFActor, PopSANActor


class ELUActor(nn.Module):
    """Equal-width continuous ELU positive control sharing the S-HIM estimator interface."""

    def __init__(self, inputs, actions, widths):
        super().__init__()
        layers = []
        for a, b in zip((inputs, *widths), widths):
            layers.extend((nn.Linear(a, b), nn.ELU()))
        layers.append(nn.Linear(widths[-1], actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs, statistics="fixed"):
        return self.net(obs)


class SHIM(nn.Module):
    def __init__(self, obs_dim, critic_dim, actions, normalization="care", history=5,
                 estimator_width=64, estimator_layers=2, actor_widths=(128, 64),
                 population=5, ticks=4, recurrent=True, grsn_threshold=1.0, grsn_bias=None, critic_widths=(256, 128),
                 actor_output_norm=False, actor_type="lif", swap_temperature=0.1,
                 swap_reduction="sum", target_mode="obs", actor_norm=None,
                 actor_threshold=1.0, actor_output_spike=True, en_pop_dim=10,
                 actor_readout_mode=None, actor_surrogate="atan",
                 en_mean_range=(-5.0, 5.0), en_std=0.3872983346,
                 freeze_encoder=False, actor_intra_feedback=False):
        super().__init__()
        self.obs_dim = obs_dim
        self.history = history
        self.actor_output_norm = actor_output_norm
        self.actor_type = actor_type
        self.actor_norm = actor_norm
        self.actor_threshold = actor_threshold
        self.actor_output_spike = actor_output_spike
        self.actor_readout_mode = actor_readout_mode
        self.actor_surrogate = actor_surrogate
        self.en_pop_dim = en_pop_dim
        self.en_mean_range = tuple(en_mean_range)
        self.en_std = float(en_std)
        self.freeze_encoder = freeze_encoder
        self.actor_intra_feedback = actor_intra_feedback
        self.swap_temperature = swap_temperature
        self.swap_reduction = swap_reduction
        self.target_mode = target_mode
        self.estimator = nn.ModuleList(
            GRSN(obs_dim if index == 0 else estimator_width, estimator_width, normalization, recurrent,
                 threshold=grsn_threshold, initial_bias=grsn_bias)
            for index in range(estimator_layers)
        )
        self.estimate = nn.Linear(estimator_width, 19)
        self.target = nn.Sequential(nn.Linear(obs_dim, 64), nn.ELU(), nn.Linear(64, 16))
        self.prototypes = nn.Parameter(torch.randn(32, 16))
        if actor_type == "elu":
            self.actor = ELUActor(obs_dim + 19, actions, actor_widths)
        elif actor_type == "popsan":
            self.actor = PopSANActor(obs_dim + 19, actions, actor_widths, population=population,
                                     ticks=ticks, en_pop_dim=en_pop_dim,
                                     mean_range=self.en_mean_range, std=self.en_std,
                                     threshold=actor_threshold,
                                     freeze_encoder=freeze_encoder,
                                     intra_feedback=actor_intra_feedback)
        else:
            effective_actor_norm = normalization if actor_norm is None else actor_norm
            self.actor = LIFActor(
                obs_dim + 19, actions, actor_widths, population, ticks, effective_actor_norm,
                output_norm=actor_output_norm, threshold=actor_threshold,
                output_spike=actor_output_spike, readout_mode=actor_readout_mode,
                surrogate=actor_surrogate,
            )
        critic = []
        for inputs, outputs in zip((critic_dim, *critic_widths), critic_widths):
            critic.extend((nn.Linear(inputs, outputs), nn.ELU()))
        self.critic = nn.Sequential(*critic, nn.Linear(critic_widths[-1], 1))
        self.log_std = nn.Parameter(torch.zeros(actions))

    def estimate_history(self, obs, statistics="fixed"):
        sequence = obs.reshape(obs.shape[0], -1, self.obs_dim)[:, :self.history].flip(1)
        for layer in self.estimator:
            sequence = layer(sequence, statistics)
        state = self.estimate(sequence[:, -1])
        return state[:, :3], F.normalize(state[:, 3:], dim=-1)

    def actor_input(self, obs, statistics="fixed"):
        with torch.no_grad():
            speed, latent = self.estimate_history(obs, statistics)
        return torch.cat((obs[:, :self.obs_dim], speed, latent), dim=-1)

    def forward(self, obs, statistics="fixed"):
        return self.actor(self.actor_input(obs, statistics), statistics)

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

    def _target_input(self, successor):
        if self.target_mode == "zero_cmd":
            target_obs = successor[:, :self.obs_dim].clone()
            target_obs[:, :3] = 0
            return target_obs
        if self.target_mode == "himloco":
            return successor[:, 3:self.obs_dim + 3]
        if self.target_mode == "priv_vel_cmd":
            target_obs = successor[:, :self.obs_dim].clone()
            target_obs[:, :3] = successor[:, self.obs_dim:self.obs_dim + 3]
            return target_obs
        return successor[:, :self.obs_dim]

    def auxiliary_loss(self, obs, successor, swap_weight, estimator_stats="adapt", return_metrics=False):
        speed, latent = self.estimate_history(obs, estimator_stats)
        target = F.normalize(self.target(self._target_input(successor)), dim=-1)
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

        temp = self.swap_temperature
        term_a = assignments(teacher) * (student / temp).log_softmax(-1)
        term_b = assignments(student) * (teacher / temp).log_softmax(-1)
        if self.swap_reduction == "mean":
            swap = -0.5 * (term_a.mean() + term_b.mean())
        else:
            swap = -0.5 * (term_a.sum(-1).mean() + term_b.sum(-1).mean())
        regression = F.mse_loss(speed, successor[:, self.obs_dim:self.obs_dim + 3])
        loss = regression + swap_weight * swap
        if not return_metrics:
            return loss, regression.detach(), swap.detach()
        with torch.no_grad():
            probs = (student / temp).softmax(-1).mean(0)
            proto_entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum())
            proto_active = float((probs > 1.0 / (2.0 * probs.numel())).float().mean())
        return loss, regression.detach(), swap.detach(), {"prototype_entropy": proto_entropy,
                                                          "prototype_active_ratio": proto_active}

    def calibrate(self, obs, commit=True):
        with torch.no_grad():
            self(obs, "calibrate")
            if commit:
                for item in self.modules():
                    if isinstance(item, CaReBN):
                        item.commit()

    def fused(self):
        model = copy.deepcopy(self).eval()
        for block in model.estimator:
            block.norm.fuse(block.projection)
        if hasattr(model.actor, "layers") and hasattr(model.actor, "norms"):
            for layer, norm in zip(model.actor.layers, model.actor.norms):
                if isinstance(norm, CaReBN):
                    norm.fuse(layer)
        return model


def shim_spike_inputs(model):
    names = set()
    if hasattr(model.actor, "layers"):
        start = 0 if hasattr(model.actor, "encoder") else 1
        names.update(f"actor.layers.{index}" for index in range(start, len(model.actor.layers)))
    names.add("estimate")
    for index, block in enumerate(model.estimator):
        if index:
            names.add(f"estimator.{index}.projection")
        if block.forget is not None:
            names.update((f"estimator.{index}.forget", f"estimator.{index}.input_gate"))
    return names
