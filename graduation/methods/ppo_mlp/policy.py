"""Feed-forward PPO baseline for the blind benchmark."""

import torch
from torch import nn


def build_mlp(obs_dim, critic_dim, actions, device):
    model = nn.Module()
    model.actor = nn.Sequential(nn.Linear(obs_dim, 256), nn.ELU(), nn.Linear(256, 128),
                                nn.ELU(), nn.Linear(128, actions))
    model.critic = nn.Sequential(nn.Linear(critic_dim, 256), nn.ELU(),
                                 nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1))
    model.log_std = nn.Parameter(torch.zeros(actions))
    model.to(device)
    return model
