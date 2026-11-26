"""Bounded batched PPO updates for the shared A1 simulation interfaces."""
import torch
from torch.distributions import Normal
from torch.nn import functional as F


def returns(reward, value, next_value, done, gamma=0.99, lam=0.95):
    advantage = torch.zeros_like(reward)
    carry = torch.zeros_like(reward[0])
    for tick in reversed(range(len(reward))):
        delta = reward[tick] + gamma * next_value[tick] - value[tick]
        carry = delta + gamma * lam * carry * (~done[tick])
        advantage[tick] = carry
    return advantage + value, (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)


def ppo_update(mean, value, log_std, optimizer, batch, epochs=4, minibatches=4):
    obs, critic, action, old_logp, old_value, target, advantage = batch
    loss_total = torch.zeros((), device=obs.device)
    updates = 0
    for _ in range(epochs):
        for rows in torch.randperm(len(obs), device=obs.device).chunk(minibatches):
            distribution = Normal(mean(obs[rows], critic[rows]), log_std.clamp(-5, 2).exp())
            ratio = (distribution.log_prob(action[rows]).sum(-1) - old_logp[rows]).exp()
            policy_loss = -torch.minimum(ratio * advantage[rows], ratio.clamp(0.8, 1.2) * advantage[rows]).mean()
            prediction = value(obs[rows], critic[rows]).reshape(-1)
            clipped = old_value[rows] + (prediction - old_value[rows]).clamp(-0.2, 0.2)
            critic_loss = torch.maximum((prediction - target[rows]).square(), (clipped - target[rows]).square()).mean()
            loss = policy_loss + critic_loss - 0.01 * distribution.entropy().sum(-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            loss_total += loss.detach()
            updates += 1
    return float(loss_total / updates)
