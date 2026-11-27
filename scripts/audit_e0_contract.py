#!/usr/bin/env python3
"""E0 zero-training behavioral and data-contract verification on real Isaac Lab rollouts."""

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graduation.paths import HIM
from graduation.sim.launch import launch


def run(args):
    import numpy as np
    import torch
    from torch.distributions import Normal

    from graduation.methods.s_him.policy import SHIM
    from graduation.rl.ppo import returns
    from graduation.sim.himloco import make_blind_env

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env, adapter, _ = make_blind_env(args)
    try:
        obs, critic_obs = adapter.reset()
        width = adapter.num_one_step_obs
        actions = adapter.num_actions
        model = SHIM(width, critic_obs.shape[1], actions, "care", 5, 64, 2,
                     grsn_threshold=0.6, critic_widths=(512, 256, 128),
                     actor_output_norm=True).to(env.device).eval()

        def history_observation(observation, previous=None, reset_ids=None):
            frames = observation.reshape(args.num_envs, -1, width)
            clean = frames[:, :1].clone()
            if previous is None:
                return clean.expand(-1, frames.shape[1], -1).flatten(1).contiguous()
            older = previous.reshape_as(frames)[:, :-1]
            result = torch.cat((clean, older), dim=1)
            result[reset_ids] = clean[reset_ids].expand(-1, frames.shape[1], -1)
            return result.flatten(1)

        obs = history_observation(obs)
        # Force one environment near timeout and check reset/timeout bootstrapping
        env.episode_length_buf[0] = env.core.max_episode_length

        rows = []
        with torch.no_grad():
            for tick in range(args.steps):
                current_value = model.critic(critic_obs).reshape(-1)
                features = model.actor_input(obs)
                mu = model.actor(features)
                sigma = model.log_std.clamp(-5, 2).exp().expand_as(mu)
                dist = Normal(mu, sigma)
                act = dist.sample()
                logp = dist.log_prob(act).sum(-1)

                # Force env 1 hard contact termination on tick 2 by lowering root z? Or naturally wait: set reset condition cleanly on a copy?
                nxt, successor, reward, done, extras, ids, terminal = adapter.step(act)
                nxt = history_observation(nxt, obs, ids)
                terminal_batch = successor.clone()
                terminal_batch[ids] = terminal
                next_value = model.critic(successor).reshape(-1)
                next_value[done] = 0
                timeout = extras["time_outs"][ids]
                if timeout.any():
                    next_value[ids[timeout]] = model.critic(terminal[timeout]).reshape(-1)
                rows.append(tuple(t.clone() for t in (
                    obs, critic_obs, features, act, mu, sigma, logp, current_value,
                    reward, next_value, done, successor, terminal_batch
                )))
                obs, critic_obs = nxt, successor

        (b_obs, b_critic, b_feat, b_act, b_mu, b_sigma, b_logp, b_val,
         b_rew, b_nval, b_done, b_succ, b_term) = [torch.stack(items) for items in zip(*rows)]

        flat = lambda x: x.flatten(0, 1)
        f_obs, f_feat, f_act, f_mu, f_sigma, f_logp = map(flat, (b_obs, b_feat, b_act, b_mu, b_sigma, b_logp))

        def check_recompute(label, mb=None, permute=False):
            n = len(f_obs)
            order = torch.randperm(n, device=env.device) if permute else torch.arange(n, device=env.device)
            chunks = order.split(mb or n)
            diffs_feat, diffs_obs, kls_feat, kls_obs, ratios_feat = [], [], [], [], []
            with torch.no_grad():
                for idx in chunks:
                    mu_f = model.actor(f_feat[idx])
                    d_f = Normal(mu_f, f_sigma[idx])
                    lp_f = d_f.log_prob(f_act[idx]).sum(-1)
                    diffs_feat.append((lp_f - f_logp[idx]).abs())
                    kls_feat.append(0.5 * ((mu_f - f_mu[idx]) / f_sigma[idx]).square().sum(-1))
                    ratios_feat.append((lp_f - f_logp[idx]).exp())

                    mu_o = model(f_obs[idx])
                    d_o = Normal(mu_o, f_sigma[idx])
                    lp_o = d_o.log_prob(f_act[idx]).sum(-1)
                    diffs_obs.append((lp_o - f_logp[idx]).abs())
                    kls_obs.append(0.5 * ((mu_o - f_mu[idx]) / f_sigma[idx]).square().sum(-1))
            return {
                "label": label,
                "max_abs_logp_diff_feat": float(torch.cat(diffs_feat).max()),
                "max_abs_logp_diff_obs": float(torch.cat(diffs_obs).max()),
                "mean_kl_feat": float(torch.cat(kls_feat).mean()),
                "mean_kl_obs": float(torch.cat(kls_obs).mean()),
                "ratio_feat_min_max": [float(torch.cat(ratios_feat).min()), float(torch.cat(ratios_feat).max())],
            }

        checks = [
            check_recompute("full_batch_order"),
            check_recompute("microbatch_17_permuted", mb=17, permute=True),
            check_recompute("microbatch_512_permuted", mb=512, permute=True),
        ]
        # Manual GAE timeout check on env 0 at tick 0
        target, adv = returns(b_rew, b_val, b_nval, b_done)
        env0_done0 = bool(b_done[0, 0])
        with torch.no_grad():
            term_diff = float((b_term[0, 0] - b_succ[0, 0]).abs().max()) if env0_done0 else 0.0
            manual_carry_step0 = float(b_rew[0, 0] + 0.99 * b_nval[0, 0] - b_val[0, 0])
            gae_step0 = float(target[0, 0] - b_val[0, 0])

        # Split train vs held-out probe envs cleanly and save
        env_ids = torch.arange(args.num_envs)
        split = {"update_envs": env_ids[env_ids % 2 == 0].tolist(), "probe_envs": env_ids[env_ids % 2 == 1].tolist()}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"obs": b_obs.cpu(), "term": b_term.cpu(), "split": split}, args.out.with_suffix(".pt"))
        report = {
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "checks": checks,
            "env0_tick0_timeout_done": env0_done0,
            "env0_tick0_terminal_vs_postreset_max_diff": term_diff,
            "manual_vs_gae_timeout_diff": abs(manual_carry_step0 - gae_step0),
            "passed": all(c["max_abs_logp_diff_feat"] < 1e-4 and c["max_abs_logp_diff_obs"] < 1e-4 for c in checks)
            and env0_done0 and term_diff > 1e-3 and abs(manual_carry_step0 - gae_step0) < 1e-5,
        }
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


def main():
    sys.path[:0] = [str(HIM / "isaaclab_migration"), str(HIM / "rsl_rl")]
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--terrain", default="rough")
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--out", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launch(AppLauncher, args, run)


if __name__ == "__main__":
    main()
