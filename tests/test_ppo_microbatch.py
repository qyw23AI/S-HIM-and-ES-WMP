import copy
import unittest

import torch
from torch import nn

from graduation.rl.ppo import ppo_update


class PPOMicrobatchTest(unittest.TestCase):
    def test_adaptive_learning_rate_clamps_to_max(self):
        actor = nn.Linear(1, 1)
        critic = nn.Linear(1, 1)
        log_std = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters(), log_std], lr=4e-4)
        obs = action = torch.zeros(4, 1)
        with torch.no_grad():
            old_mu = actor(obs) + 0.1
            old_sigma = torch.ones_like(old_mu)
            old_logp = torch.distributions.Normal(old_mu, old_sigma).log_prob(action).sum(-1)
            old_value = critic(obs).flatten()
        batch = (obs, obs, action, old_logp, old_value, old_value, torch.ones_like(old_value))

        ppo_update(lambda x, _: actor(x), lambda _, x: critic(x), log_std, optimizer, batch,
                   epochs=1, minibatches=1, old_mu=old_mu, old_sigma=old_sigma, desired_kl=10.0, lr_max=5e-4)

        self.assertEqual(optimizer.param_groups[0]["lr"], 5e-4)

    def test_chunks_preserve_minibatch_update_weighting(self):
        torch.manual_seed(41)
        actor = nn.Linear(5, 2)
        critic = nn.Linear(3, 1)
        obs, privileged = torch.randn(28, 5), torch.randn(28, 3)
        action = torch.randn(28, 2)
        with torch.no_grad():
            old_value = critic(privileged).flatten()
            old_logp = torch.distributions.Normal(actor(obs), 1.).log_prob(action).sum(-1)
        batch = (obs, privileged, action, old_logp, old_value, torch.randn(28), torch.randn(28))
        results = []
        for microbatch in (None, 3):
            policy, value = copy.deepcopy(actor), copy.deepcopy(critic)
            log_std = nn.Parameter(torch.zeros(2))
            params = [*policy.parameters(), *value.parameters(), log_std]
            optimizer = torch.optim.Adam(params, lr=3e-4)
            torch.manual_seed(9)
            loss = ppo_update(lambda x, _: policy(x), lambda _, x: value(x),
                              log_std, optimizer, batch, epochs=2, microbatch=microbatch)
            results.append((loss, params))
        self.assertAlmostEqual(results[0][0], results[1][0], places=6)
        torch.testing.assert_close(results[0][1], results[1][1], atol=1e-7, rtol=1e-6)
