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


def ppo_update(mean, value, log_std, optimizer, batch, epochs=4, minibatches=4, microbatch=None, *,
               old_mu=None, old_sigma=None, desired_kl=None, lr_min=1e-5, lr_max=1e-2,
               max_grad_norm=1.0, separate_grad_clip=False, actor_params=None, critic_params=None,
               std_params=None, return_metrics=False, entropy_coef=0.01, max_log_std=2.0,
               clip_param=0.2, value_loss_coef=1.0, use_clipped_value_loss=True):
    obs, critic, action, old_logp, old_value, target, advantage = batch
    loss_total = torch.zeros((), device=obs.device)
    updates = 0
    kl_means, kl_p95s, clip_fracs, entropies = [], [], [], []
    actor_gns, critic_gns, std_gns = [], [], []
    actor_upds, critic_upds, std_upds = [], [], []
    preds_all, targets_all = [], []

    def _norm(params):
        vals = [p.detach().square().sum() for p in params]
        return float(torch.stack(vals).sum().sqrt()) if vals else 0.0

    def _grad_norm(params):
        vals = [p.grad.detach().square().sum() for p in params if p.grad is not None]
        return float(torch.stack(vals).sum().sqrt()) if vals else 0.0

    for _ in range(epochs):
        for rows in torch.randperm(len(obs), device=obs.device).chunk(minibatches):
            optimizer.zero_grad(set_to_none=True)
            mb_kls, mb_clips, mb_ents = [], [], []
            for indices in rows.split(microbatch or len(rows)):
                distribution = Normal(mean(obs[indices], critic[indices]), log_std.clamp(-5, max_log_std).exp())
                ratio = (distribution.log_prob(action[indices]).sum(-1) - old_logp[indices]).exp()
                clipped_ratio = ratio.clamp(1.0 - clip_param, 1.0 + clip_param)
                policy_loss = -torch.minimum(ratio * advantage[indices], clipped_ratio * advantage[indices]).mean()
                prediction = value(obs[indices], critic[indices]).reshape(-1)
                value_error = (prediction - target[indices]).square()
                if use_clipped_value_loss:
                    clipped = old_value[indices] + (prediction - old_value[indices]).clamp(-clip_param, clip_param)
                    value_error = torch.maximum(value_error, (clipped - target[indices]).square())
                entropy_val = distribution.entropy().sum(-1)
                loss = policy_loss + value_loss_coef * value_error.mean() - entropy_coef * entropy_val.mean()
                weighted = loss * (len(indices) / len(rows))
                weighted.backward()
                loss_total += weighted.detach()
                if return_metrics or desired_kl is not None:
                    with torch.no_grad():
                        if old_mu is not None and old_sigma is not None:
                            mu_b = distribution.mean
                            sigma_b = distribution.stddev
                            kl_b = (
                                torch.log(sigma_b / old_sigma[indices] + 1e-5)
                                + (old_sigma[indices].square() + (old_mu[indices] - mu_b).square())
                                / (2.0 * sigma_b.square())
                                - 0.5
                            ).sum(-1)
                            mb_kls.append(kl_b)
                        if return_metrics:
                            mb_clips.append(((ratio - 1.0).abs() > clip_param).float())
                            mb_ents.append(entropy_val)
                            preds_all.append(prediction.detach())
                            targets_all.append(target[indices].detach())
            if mb_kls:
                kl_all = torch.cat(mb_kls)
                kl_mean = float(kl_all.mean())
                kl_means.append(kl_mean)
                if return_metrics:
                    kl_p95s.append(float(torch.quantile(kl_all.float(), 0.95)))
                if desired_kl is not None:
                    base_lr = optimizer.param_groups[0]["lr"] / optimizer.param_groups[0].get("lr_scale", 1.0)
                    if kl_mean > desired_kl * 2.0:
                        base_lr = max(lr_min, base_lr / 1.5)
                    elif 0.0 < kl_mean < desired_kl / 2.0:
                        base_lr = min(lr_max, base_lr * 1.5)
                    for group in optimizer.param_groups:
                        group["lr"] = base_lr * group.get("lr_scale", 1.0)
            if return_metrics:
                if actor_params:
                    actor_gns.append(_grad_norm(actor_params))
                if critic_params:
                    critic_gns.append(_grad_norm(critic_params))
                if std_params:
                    std_gns.append(_grad_norm(std_params))
                before_a = [p.detach().clone() for p in (actor_params or ())]
                before_c = [p.detach().clone() for p in (critic_params or ())]
                before_s = [p.detach().clone() for p in (std_params or ())]
            if separate_grad_clip and (actor_params or critic_params or std_params):
                for group_params in (actor_params or (), critic_params or (), std_params or ()):
                    if group_params:
                        torch.nn.utils.clip_grad_norm_(group_params, max_grad_norm, error_if_nonfinite=True)
            else:
                parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
                torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            if max_log_std < 2.0:
                with torch.no_grad():
                    log_std.clamp_(max=max_log_std)
            if return_metrics:
                if mb_clips:
                    clip_fracs.append(float(torch.cat(mb_clips).mean()))
                if mb_ents:
                    entropies.append(float(torch.cat(mb_ents).mean()))
                with torch.no_grad():
                    if actor_params:
                        delta = float(torch.stack([(p - b).square().sum() for p, b in zip(actor_params, before_a)]).sum().sqrt())
                        actor_upds.append(delta / max(_norm(before_a), 1e-12))
                    if critic_params:
                        delta = float(torch.stack([(p - b).square().sum() for p, b in zip(critic_params, before_c)]).sum().sqrt())
                        critic_upds.append(delta / max(_norm(before_c), 1e-12))
                    if std_params:
                        delta = float(torch.stack([(p - b).square().sum() for p, b in zip(std_params, before_s)]).sum().sqrt())
                        std_upds.append(delta / max(_norm(before_s), 1e-12))
            updates += 1
    avg_loss = float(loss_total / updates)
    if not return_metrics:
        return avg_loss
    explained_var = None
    if preds_all:
        p_cat, t_cat = torch.cat(preds_all).float(), torch.cat(targets_all).float()
        t_var = float(t_cat.var(unbiased=False))
        explained_var = float(1.0 - (t_cat - p_cat).var(unbiased=False) / t_var) if t_var > 1e-12 else 0.0
    metrics = {
        "ppo_kl_mean": float(sum(kl_means) / len(kl_means)) if kl_means else None,
        "ppo_kl_p95": float(sum(kl_p95s) / len(kl_p95s)) if kl_p95s else None,
        "clip_fraction": float(sum(clip_fracs) / len(clip_fracs)) if clip_fracs else None,
        "entropy": float(sum(entropies) / len(entropies)) if entropies else None,
        "explained_variance": explained_var,
        "actor_grad_norm": float(sum(actor_gns) / len(actor_gns)) if actor_gns else None,
        "critic_grad_norm": float(sum(critic_gns) / len(critic_gns)) if critic_gns else None,
        "std_grad_norm": float(sum(std_gns) / len(std_gns)) if std_gns else None,
        "actor_rel_update": float(sum(actor_upds) / len(actor_upds)) if actor_upds else None,
        "critic_rel_update": float(sum(critic_upds) / len(critic_upds)) if critic_upds else None,
        "std_rel_update": float(sum(std_upds) / len(std_upds)) if std_upds else None,
        "learning_rate": float(optimizer.param_groups[0]["lr"] / optimizer.param_groups[0].get("lr_scale", 1.0)),
    }
    return avg_loss, metrics
