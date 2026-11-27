"""Population-coded actor shared by S-HIM and ES-WMP."""

import torch
from torch import nn

from .neurons import spike
from .normalization import CaReBN


def _encoder_spike(voltage, threshold=0.999):
    return (voltage > threshold).to(voltage.dtype) + (voltage - voltage.detach())


def _rect_spike(voltage, threshold=0.5, window=0.5):
    surrogate = voltage.clamp(threshold - window, threshold + window)
    return (voltage > threshold).to(voltage.dtype) + (surrogate - surrogate.detach())


class LIFActor(nn.Module):
    def __init__(self, inputs, actions, widths, population, ticks, normalization,
                 output_norm=False, threshold=1.0, output_spike=True,
                 readout_mode=None, surrogate="atan"):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(
            (inputs, *widths), (*widths, actions * population)
        ))
        resolved_mode = ("spike" if output_spike else "linear") if readout_mode is None else readout_mode
        if resolved_mode not in ("spike", "nlif", "linear"):
            raise ValueError(f"Unsupported readout_mode: {resolved_mode}")
        if surrogate not in ("atan", "rect"):
            raise ValueError(f"Unsupported surrogate: {surrogate}")
        self.normalization = normalization
        self.output_norm = output_norm
        self.threshold = threshold
        self.output_spike = resolved_mode == "spike"
        self.readout_mode = resolved_mode
        self.surrogate = surrogate
        norm_widths = (*widths, actions * population) if (output_norm and resolved_mode == "spike") else widths
        if normalization == "ln":
            self.norms = nn.ModuleList(nn.LayerNorm(width) for width in norm_widths)
        else:
            self.norms = nn.ModuleList(CaReBN(width, normalization) for width in norm_widths)
        self.readout = nn.Conv1d(actions, actions, population, groups=actions)
        self.actions = actions
        self.ticks = ticks
        self.static_projection = True

    def _apply_norm(self, index, current, statistics):
        if index >= len(self.norms):
            return current
        norm = self.norms[index]
        return norm(current) if isinstance(norm, nn.LayerNorm) else norm(current, statistics)

    def _run(self, obs, statistics="fixed", record_stats=False):
        value = obs[:, None].expand(-1, self.ticks, -1)
        rates, silent = [], []
        last = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            if index == last and self.readout_mode == "linear":
                projected = layer(value.mean(dim=1)).reshape(obs.shape[0], self.actions, -1)
                return self.readout(projected).squeeze(-1), rates, silent
            if index == 0 and self.static_projection:
                # Apply LayerNorm on the single frame cleanly, while keeping repeated ticks visible to CaReBN calibration.
                if index < len(self.norms) and isinstance(self.norms[index], nn.LayerNorm):
                    current = self.norms[index](layer(obs))[:, None].expand(-1, self.ticks, -1)
                else:
                    current = self._apply_norm(
                        index, layer(obs)[:, None].expand(-1, self.ticks, -1), statistics
                    )
            else:
                current = self._apply_norm(index, layer(value), statistics)
            if index == last and self.readout_mode == "nlif":
                # Non-spiking LIF readout integrates binary hidden spikes across ticks without firing or resetting.
                voltage = torch.zeros_like(current[:, 0])
                membranes = []
                for tick in current.unbind(1):
                    voltage = 0.5 * voltage + tick
                    membranes.append(voltage)
                membrane_mean = torch.stack(membranes, dim=1).mean(dim=1)
                return self.readout(membrane_mean.reshape(obs.shape[0], self.actions, -1)).squeeze(-1), rates, silent
            voltage = torch.zeros_like(current[:, 0])
            pulses = []
            for tick in current.unbind(1):
                voltage = 0.5 * voltage + tick
                fired = (
                    _rect_spike(voltage, self.threshold, 0.5)
                    if self.surrogate == "rect" else spike(voltage - self.threshold)
                )
                voltage = voltage - self.threshold * fired
                pulses.append(fired)
            value = torch.stack(pulses, dim=1)
            if record_stats:
                with torch.no_grad():
                    rates.append(float(value.mean()))
                    silent.append(float((value.mean(dim=(0, 1)) == 0).float().mean()))
        rate = value.mean(dim=1).reshape(obs.shape[0], self.actions, -1)
        return self.readout(rate).squeeze(-1), rates, silent

    def forward_with_stats(self, obs, statistics="fixed"):
        return self._run(obs, statistics=statistics, record_stats=True)

    def forward(self, obs, statistics="fixed"):
        return self._run(obs, statistics=statistics, record_stats=False)[0]


class PopEncoder(nn.Module):
    """Learnable Gaussian receptive-field population spike encoder."""

    def __init__(self, obs_dim, en_pop_dim=10, ticks=5, mean_range=(-5.0, 5.0), std=0.3872983346):
        super().__init__()
        self.obs_dim = obs_dim
        self.en_pop_dim = en_pop_dim
        self.ticks = ticks
        centers = torch.linspace(mean_range[0], mean_range[1], en_pop_dim).view(1, 1, en_pop_dim)
        self.mean = nn.Parameter(centers.expand(1, obs_dim, en_pop_dim).clone())
        self.std = nn.Parameter(torch.full((1, obs_dim, en_pop_dim), float(std)))

    def forward(self, obs):
        std = self.std.abs().clamp_min(1e-4)
        activity = torch.exp(-0.5 * ((obs.unsqueeze(-1) - self.mean) / std).pow(2)).reshape(obs.shape[0], -1)
        volt = torch.zeros_like(activity)
        pulses = []
        for _ in range(self.ticks):
            volt = volt + activity
            fired = _encoder_spike(volt)
            volt = volt - fired * 0.999
            pulses.append(fired)
        return pulses


class PopSANActor(nn.Module):
    """Gaussian-encoded, current-decaying population SNN actor adapted from FullySNNforLeggedRobots."""

    def __init__(self, inputs, actions, widths, population=10, ticks=5, en_pop_dim=10,
                 mean_range=(-5.0, 5.0), std=0.3872983346, threshold=0.5, window=0.5,
                 freeze_encoder=False, intra_feedback=False):
        super().__init__()
        self.encoder = PopEncoder(inputs, en_pop_dim=en_pop_dim, ticks=ticks,
                                  mean_range=mean_range, std=std)
        if freeze_encoder:
            self.encoder.mean.requires_grad_(False)
            self.encoder.std.requires_grad_(False)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(
            (inputs * en_pop_dim, *widths), (*widths, actions * population)
        ))
        self.readout = nn.Conv1d(actions, actions, population, groups=actions)
        self.actions = actions
        self.population = population
        self.ticks = ticks
        self.threshold = threshold
        self.window = window
        self.intra_feedback = intra_feedback
        if intra_feedback:
            self.intra_weight = nn.Parameter(torch.zeros(actions, population, population))
            self.intra_bias = nn.Parameter(torch.zeros(actions, population))
        else:
            self.register_parameter("intra_weight", None)
            self.register_parameter("intra_bias", None)

    def _run(self, obs, record_stats=False):
        in_pulses = self.encoder(obs)
        batch = obs.shape[0]
        currents = [obs.new_zeros(batch, layer.out_features) for layer in self.layers]
        volts = [obs.new_zeros(batch, layer.out_features) for layer in self.layers]
        prev_spikes = [obs.new_zeros(batch, layer.out_features) for layer in self.layers]
        spike_sums = [obs.new_zeros(batch, layer.out_features) for layer in self.layers] if record_stats else None
        out_sum = obs.new_zeros(batch, self.layers[-1].out_features)
        last = len(self.layers) - 1
        for step in range(self.ticks):
            pre = in_pulses[step]
            for index, layer in enumerate(self.layers):
                trans = layer(pre)
                if index == last and step > 0 and self.intra_weight is not None and self.intra_bias is not None:
                    prev_group = prev_spikes[index].reshape(batch, self.actions, self.population)
                    trans = trans + (
                        torch.einsum("bjp,jqp->bjq", prev_group, self.intra_weight) + self.intra_bias
                    ).reshape(batch, -1)
                currents[index] = currents[index] * 0.5 + trans
                volts[index] = volts[index] * 0.75 * (1.0 - prev_spikes[index]) + currents[index]
                spk = _rect_spike(volts[index], self.threshold, self.window)
                prev_spikes[index] = spk
                pre = spk
                if spike_sums is not None:
                    spike_sums[index] = spike_sums[index] + spk.detach()
            out_sum = out_sum + prev_spikes[-1]
        rates, silent = [], []
        if spike_sums is not None:
            with torch.no_grad():
                for total in spike_sums:
                    mean_rate = total / self.ticks
                    rates.append(float(mean_rate.mean()))
                    silent.append(float((mean_rate.mean(dim=0) == 0).float().mean()))
        rate = (out_sum / self.ticks).reshape(batch, self.actions, -1)
        return self.readout(rate).squeeze(-1), rates, silent

    def forward_with_stats(self, obs, statistics="fixed"):
        return self._run(obs, record_stats=True)

    def forward(self, obs, statistics="fixed"):
        return self._run(obs, record_stats=False)[0]
