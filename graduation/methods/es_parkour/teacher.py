"""Privileged ES-Parkour teacher with scan and oracle heading."""

import torch
from torch import nn


class Teacher(nn.Module):
    def __init__(self, proprio_dim, scan_dim, actions):
        super().__init__()
        self.scan = nn.Sequential(nn.Linear(scan_dim, 128), nn.ELU(), nn.Linear(128, 32))
        self.actor = nn.Sequential(nn.Linear(proprio_dim + 33, 256), nn.ELU(), nn.Linear(256, 128),
                                   nn.ELU(), nn.Linear(128, actions))
        self.critic = nn.Sequential(nn.Linear(proprio_dim + scan_dim + 1, 256), nn.ELU(),
                                    nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1))
        self.log_std = nn.Parameter(torch.zeros(actions))

    def mean(self, proprio, privileged):
        return self.actor(torch.cat((proprio, self.scan(privileged[:, :-1]), privileged[:, -1:]), dim=-1))

    def value(self, proprio, privileged):
        return self.critic(torch.cat((proprio, privileged), dim=-1))
