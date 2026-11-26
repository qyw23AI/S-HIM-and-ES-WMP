"""Spiking neurons and explicit recurrent state."""

import math

import torch
from torch import nn

from .normalization import CaReBN


def spike(voltage):
    surrogate = torch.atan(math.pi * voltage) / math.pi
    return (voltage >= 0).to(voltage.dtype) + (surrogate - surrogate.detach())


class GRSN(nn.Module):
    def __init__(self, inputs, width, normalization, recurrent=True, threshold=1.0, initial_bias=None):
        super().__init__()
        self.projection = nn.Linear(inputs, width)
        self.forget = nn.Linear(width, width) if recurrent else None
        self.input_gate = nn.Linear(width, width) if recurrent else None
        self.beta = nn.Parameter(torch.zeros(width))
        self.norm = CaReBN(width, normalization)
        self.threshold = threshold
        if initial_bias is not None:
            if normalization == "none":
                raise ValueError("GRSN normalization bias requires an active normalization")
            with torch.no_grad():
                self.norm.bias.fill_(initial_bias)

    def empty_state(self, batch):
        return self.beta.new_zeros(batch, 3 * self.beta.numel())

    def step(self, value, state):
        pulse, state, _ = self.transition(value, state)
        return pulse, state

    def transition(self, value, state):
        voltage, current, previous = state.chunk(3, dim=-1)
        if self.forget is not None:
            forget = self.forget(previous).sigmoid()
            current = forget * current + (1 - forget) * self.input_gate(previous).relu()
        beta = self.beta.sigmoid()
        voltage = beta * voltage + (1 - beta) * (value + current)
        pulse = spike(voltage - self.threshold)
        return pulse, torch.cat((voltage - self.threshold * pulse, current, pulse), dim=-1), voltage

    def forward(self, sequence, statistics):
        return self.sequence(sequence, statistics)[0]

    def sequence(self, sequence, statistics, record_voltage=False):
        # Unroll one cell update per historical control frame, oldest first.
        state = self.empty_state(sequence.shape[0])
        rows = []
        voltages = []
        input_current = self.norm(self.projection(sequence), statistics)
        for value in input_current.unbind(1):
            pulse, state, voltage = self.transition(value, state)
            rows.append(pulse)
            if record_voltage:
                voltages.append(voltage)
        return torch.stack(rows, dim=1), voltages


def if_sequence(current, threshold=1.0):
    voltage = torch.zeros_like(current[0])
    result = []
    for tick in current.unbind(0):
        voltage = voltage + tick
        fired = spike(voltage - threshold)
        voltage = voltage * (1 - fired.detach())
        result.append(fired)
    return torch.stack(result)


def sequence_layer(layer, sequence):
    return layer(sequence.flatten(0, 1)).unflatten(0, sequence.shape[:2])
