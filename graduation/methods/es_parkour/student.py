"""ES-Parkour IF ResNet and configurable spiking student variants."""

import torch
from torch import nn
from torch.nn import functional as F

from graduation.snn.neurons import if_sequence, sequence_layer
from graduation.snn.normalization import CaReBN


def make_norm(channels, normalization, threshold):
    if normalization == "batch":
        return nn.BatchNorm2d(channels)
    norm = CaReBN(channels, normalization, threshold)
    if normalization == "tdbn":
        with torch.no_grad():
            norm.bias.fill_(threshold / 2)
    return norm


def normalize_spatial(norm, current):
    if isinstance(norm, CaReBN):
        return norm(current.movedim(2, -1), "adapt" if norm.training else "fixed").movedim(-1, 2)
    return sequence_layer(norm, current)


class SpikeBlock(nn.Module):
    def __init__(self, inputs, channels, stride, normalization="batch", threshold=1.0, residual="if"):
        super().__init__()
        self.threshold = threshold
        self.residual = residual
        self.first = nn.Conv2d(inputs, channels, 3, stride, 1, bias=False)
        self.norm1 = make_norm(channels, normalization, threshold)
        self.second = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.norm2 = make_norm(channels, normalization, threshold)
        self.shortcut = (nn.Sequential(nn.Conv2d(inputs, channels, 1, stride, bias=False),
                                      make_norm(channels, normalization, threshold))
                         if stride != 1 or inputs != channels else nn.Identity())

    def forward(self, value):
        pulses = if_sequence(normalize_spatial(self.norm1, sequence_layer(self.first, value)), self.threshold)
        current = normalize_spatial(self.norm2, sequence_layer(self.second, pulses))
        shortcut = value
        if isinstance(self.shortcut, nn.Sequential):
            shortcut = normalize_spatial(self.shortcut[1], sequence_layer(self.shortcut[0], value))
            if self.residual == "sew-add":
                shortcut = if_sequence(shortcut, self.threshold)
        if self.residual == "sew-add":
            return if_sequence(current, self.threshold) + shortcut
        return if_sequence(current + shortcut, self.threshold)


class ESStudent(nn.Module):
    def __init__(self, proprio, actions, base=64, latent=64, ticks=4,
                 normalization="batch", threshold=1.0, residual="if"):
        super().__init__()
        self.ticks = ticks
        self.threshold = threshold
        self.stem = nn.Conv2d(2, base, 7, 2, 3, bias=False)
        self.stem_norm = make_norm(base, normalization, threshold)
        blocks = []
        width = base
        for stage in range(4):
            out = base * 2 ** stage
            opts = (normalization, threshold, residual)
            blocks.extend((SpikeBlock(width, out, 2 if stage else 1, *opts), SpikeBlock(out, out, 1, *opts)))
            width = out
        self.blocks = nn.Sequential(*blocks)
        self.fuse = nn.GRUCell(width + proprio, latent)
        self.direction = nn.Linear(latent, 1)
        self.actor = nn.ModuleList(nn.Linear(a, b) for a, b in zip(
            (proprio + latent + 1, 512, 256, 128), (512, 256, 128, actions)
        ))
        self.actor_norms = nn.ModuleList(
            make_norm(width, normalization, threshold) for width in (512, 256, 128)
        ) if normalization != "batch" else nn.ModuleList()

    def observe(self, events, proprio, memory, trace=None):
        sequence = events[None].expand(self.ticks, -1, -1, -1, -1)
        current = normalize_spatial(self.stem_norm, sequence_layer(self.stem, sequence))
        pulses = if_sequence(current, self.threshold)
        pulses = F.max_pool2d(pulses.flatten(0, 1), 3, 2, 1).unflatten(0, pulses.shape[:2])
        for index, block in enumerate(self.blocks):
            pulses = block(pulses)
            if trace is not None:
                trace[f"blocks.{index}"] = pulses
        features = pulses.mean(dim=(0, 3, 4))
        return self.fuse(torch.cat((features, proprio), dim=-1), memory)

    def forward(self, proprio, memory, trace=None):
        yaw = self.direction(memory)
        state = torch.cat((proprio, memory, yaw), dim=-1)
        pulse = state[None].expand(self.ticks, -1, -1)
        for index, layer in enumerate(self.actor[:-1]):
            current = layer(pulse)
            if len(self.actor_norms):
                norm = self.actor_norms[index]
                current = norm(current, "adapt" if self.training else "fixed")
            pulse = if_sequence(current, self.threshold)
            if trace is not None:
                trace[f"actor.{index + 1}.input"] = pulse
        return self.actor[-1](pulse).mean(dim=0), yaw

    @torch.no_grad()
    def visual_response(self, event, proprio, memory):
        """Hold proprioception and memory fixed while replacing only the event frame."""
        training = self.training
        self.eval()
        try:
            traces = [{}, {}]
            hidden, action = [], []
            for frame, trace in zip((event, torch.zeros_like(event)), traces):
                hidden.append(self.observe(frame, proprio, memory, trace))
                action.append(self(proprio, hidden[-1], trace)[0])
            layers = {}
            for name, actual in traces[0].items():
                zero = traces[1][name]
                layers[name] = {
                    "active_fraction": float((actual != 0).float().mean()),
                    "zero_frame_active_fraction": float((zero != 0).float().mean()),
                    "max_abs_delta": float((actual - zero).abs().max()),
                }
            return {"active_event_pixels": int(event.sum()), "gru_max_abs_delta": float((hidden[0] - hidden[1]).abs().max()),
                    "action_max_abs_delta": float((action[0] - action[1]).abs().max()), "layers": layers}
        finally:
            self.train(training)


def es_spike_inputs(model):
    names = {f"actor.{i}" for i in range(1, len(model.actor))}
    for i, block in enumerate(model.blocks):
        names.add(f"blocks.{i}.second")
        # SEW addition produces multi-valued activations after its first block.
        if i == 0 or block.residual != "sew-add":
            names.add(f"blocks.{i}.first")
            if isinstance(block.shortcut, nn.Sequential):
                names.add(f"blocks.{i}.shortcut.0")
    return names
