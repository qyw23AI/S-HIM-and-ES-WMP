"""Event-Spiking-WMP student, with per-control-tick GRSN state."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from graduation.snn.neurons import GRSN, spike
from graduation.snn.normalization import CaReBN
from graduation.snn.population_actor import LIFActor


class EventSpikingWMP(nn.Module):
    def __init__(self, proprio_dim=45, latent_dim=512, actions=12, memory_width=128,
                 channels=(16, 32, 64), normalization="care", memory="grsn", grsn_threshold=1.0, grsn_bias=None):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv2d(inputs, outputs, 3, 2, 1)
                                   for inputs, outputs in zip((2, *channels[:-1]), channels))
        self.conv_norms = nn.ModuleList(CaReBN(width, normalization) for width in channels)
        self.vision_dim = channels[-1]
        self.memory = (nn.GRUCell(self.vision_dim + proprio_dim, memory_width)
                       if memory == "gru" else GRSN(self.vision_dim + proprio_dim, memory_width,
                                                   normalization, recurrent=memory == "grsn",
                                                   threshold=grsn_threshold, initial_bias=grsn_bias))
        self.memory_width = memory_width
        self.world = nn.Linear(memory_width, latent_dim)
        self.actor = LIFActor(latent_dim + proprio_dim, actions, (256, 128), 5, 4, normalization)

    def initial(self, batch):
        if isinstance(self.memory, GRSN):
            state = self.memory.empty_state(batch)
        else:
            state = self.world.weight.new_zeros(batch, self.memory_width)
        return state, self.world.weight.new_zeros(batch, self.vision_dim)

    def encode(self, events, statistics="fixed"):
        value = events
        for conv, norm in zip(self.convs, self.conv_norms):
            current = conv(value).movedim(1, -1)
            value = spike(norm(current, statistics) - 0.5).movedim(-1, 1)
        return value.mean(dim=(2, 3))

    def forward(self, proprio, visual, state, statistics="fixed", detach_action=False):
        inputs = torch.cat((visual, proprio), dim=-1)
        if isinstance(self.memory, GRSN):
            current = self.memory.norm(self.memory.projection(inputs), statistics)
            feature, state = self.memory.step(current, state)
        else:
            state = self.memory(inputs, state)
            feature = state
        latent = self.world(feature)
        actor_latent = latent.detach() if detach_action else latent
        action = self.actor(torch.cat((proprio, actor_latent), dim=-1), statistics)
        return action, latent, state

    def commit_norms(self):
        for layer in self.modules():
            if isinstance(layer, CaReBN):
                layer.commit()

    def fused(self):
        model = copy.deepcopy(self).eval()
        for conv, norm in zip(model.convs, model.conv_norms):
            norm.fuse(conv)
        if isinstance(model.memory, GRSN):
            model.memory.norm.fuse(model.memory.projection)
        for layer, norm in zip(model.actor.layers, model.actor.norms):
            norm.fuse(layer)
        return model

    def spike_layers(self):
        names = {f"convs.{index}" for index in range(len(self.convs))}
        names.update(f"actor.layers.{index}" for index in range(1, len(self.actor.layers)))
        if isinstance(self.memory, GRSN):
            names.add("world")
            if self.memory.forget is not None:
                names.update(("memory.forget", "memory.input_gate"))
        return names


def distillation_losses(action, teacher_action, latent, teacher_latent):
    action_loss = F.mse_loss(action, teacher_action.detach())
    mse_loss = F.mse_loss(latent, teacher_latent.detach())
    cosine_loss = (1 - F.cosine_similarity(latent, teacher_latent.detach(), dim=-1)).mean()
    return action_loss, mse_loss, cosine_loss
