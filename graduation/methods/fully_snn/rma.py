"""Isaac Lab adapter for the author's FullySNN A1 RMA actor."""

import importlib.util

import torch
from torch import nn
from torch.nn import functional as F

from graduation.paths import FULLY_SNN

SOURCE = (FULLY_SNN
          / "RMA/quadruped-robot-master/rl-robotics/rsl_rl/rsl_rl/modules/actor_critic.py")


def build_rma(obs_dim, critic_dim, history_dim, actions, device):
    spec = importlib.util.spec_from_file_location("fully_snn_upstream_actor", SOURCE)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    actor_critic = upstream.ActorCritic(obs_dim, critic_dim - obs_dim, history_dim, actions)
    actor_critic.actor.encoder.device = device
    actor_critic.actor.snn.device = device
    return RMA(actor_critic, obs_dim).to(device)


class RMA(nn.Module):
    def __init__(self, upstream, obs_dim):
        super().__init__()
        self.upstream = upstream
        self.obs_dim = obs_dim

    @property
    def log_std(self):
        return self.upstream.actor.log_std

    def mean(self, obs, critic=None):
        current = obs[:, :self.obs_dim]
        factor = (self.upstream.adaptation_module(obs) if critic is None
                  else self.upstream.env_factor_encoder(critic[:, self.obs_dim:]))
        return self.upstream.actor(torch.cat((current, factor), dim=-1))[0]

    def value(self, obs, critic):
        return self.upstream.evaluate(obs[:, :self.obs_dim], critic[:, self.obs_dim:])

    def adaptation_loss(self, obs, critic):
        with torch.no_grad():
            target = self.upstream.env_factor_encoder(critic[:, self.obs_dim:])
        return F.mse_loss(self.upstream.adaptation_module(obs), target)
