"""Population-coded actor shared by S-HIM and ES-WMP."""

import torch
from torch import nn

from .neurons import spike
from .normalization import CaReBN


class LIFActor(nn.Module):
    def __init__(self, inputs, actions, widths, population, ticks, normalization):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(
            (inputs, *widths), (*widths, actions * population)
        ))
        self.norms = nn.ModuleList(CaReBN(width, normalization) for width in widths)
        self.readout = nn.Conv1d(actions, actions, population, groups=actions)
        self.actions = actions
        self.ticks = ticks
        self.static_projection = True

    def forward(self, obs, statistics="fixed"):
        value = obs[:, None].expand(-1, self.ticks, -1)
        for index, layer in enumerate(self.layers):
            if index == 0 and self.static_projection:
                current = layer(obs)[:, None].expand(-1, self.ticks, -1)
            else:
                current = layer(value)
            # Keep all repeated sites visible to adaptive and recalibration statistics.
            if index < len(self.norms):
                current = self.norms[index](current, statistics)
            voltage = torch.zeros_like(current[:, 0])
            pulses = []
            for tick in current.unbind(1):
                voltage = 0.5 * voltage + tick
                fired = spike(voltage - 1)
                voltage = voltage - fired
                pulses.append(fired)
            value = torch.stack(pulses, dim=1)
        rate = value.mean(dim=1).reshape(obs.shape[0], self.actions, -1)
        return self.readout(rate).squeeze(-1)
