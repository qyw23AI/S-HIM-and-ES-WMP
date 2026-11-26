"""Run shared-terrain A1 PPO, FullySNN RMA, and S-HIM training or short evaluation."""

import json
from contextlib import nullcontext
from pathlib import Path


def run(args):
    import numpy as np
    import torch
    from torch import nn
    from torch.distributions import Normal

    from graduation.evaluation.energy import SynapticMeter
    from graduation.methods.fully_snn.rma import build_rma
    from graduation.methods.ppo_mlp.policy import build_mlp
    from graduation.methods.s_him.policy import SHIM, shim_spike_inputs
    from graduation.rl.ppo import ppo_update, returns
    from graduation.sim.himloco import make_blind_env

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env, adapter, defaults = make_blind_env(args)
    try:
        obs, critic_obs = adapter.reset()
        model_config_keys = ("normalization", "history", "estimator_width", "estimator_layers", "population", "estimator",
                             "grsn_threshold", "grsn_bias")
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True) if args.checkpoint else None
        if checkpoint and args.method == "s-him":
            for key, val in checkpoint.get("model_config", {}).items():
                setattr(args, key, val)
        args.grsn_threshold = 1.0 if args.grsn_threshold is None else args.grsn_threshold
        width = adapter.num_one_step_obs
        actions = adapter.num_actions
        if args.method == "himloco":
            from rsl_rl.modules.him_actor_critic import HIMActorCritic
            model = HIMActorCritic(obs.shape[1], critic_obs.shape[1], width, actions,
                                   **defaults["policy"]).to(args.device)
        elif args.method == "s-him":
            model = SHIM(width, critic_obs.shape[1], actions, args.normalization, args.history,
                         args.estimator_width, args.estimator_layers, population=args.population,
                         recurrent=args.estimator == "grsn", grsn_threshold=args.grsn_threshold,
                         grsn_bias=args.grsn_bias).to(args.device)
            model.actor.static_projection = not args.legacy_actor_projection
        elif args.method == "fully-snn":
            model = build_rma(width, critic_obs.shape[1], obs.shape[1], actions, args.device)
        else:
            model = build_mlp(obs.shape[1], critic_obs.shape[1], actions, args.device)
        shim = args.method == "s-him"
        rma = args.method == "fully-snn"
        if shim:
            aux_params = [*model.estimator.parameters(), *model.estimate.parameters(),
                          *model.target.parameters(), model.prototypes]
            ppo_params = [*model.actor.parameters(), *model.critic.parameters(), model.log_std]
        elif rma:
            aux_params = list(model.upstream.adaptation_module.parameters())
            ppo_params = [*model.upstream.actor.parameters(), *model.upstream.critic.parameters(),
                          *model.upstream.env_factor_encoder.parameters()]
        else:
            aux_params = []
            ppo_params = list(model.parameters())
        optimizer = torch.optim.Adam(ppo_params, lr=3e-4)
        aux_optimizer = torch.optim.Adam(aux_params, lr=1e-3) if aux_params else None

        def mean(observation, privileged=None):
            if args.method == "himloco":
                return model.act_inference(observation)
            if shim:
                return model(observation)
            if rma:
                return model.mean(observation, privileged)
            return model.actor(observation)

        def value(observation, privileged):
            return model.value(observation, privileged) if rma else model.critic(privileged)

        completed = 0
        if args.checkpoint:
            if args.method == "himloco":
                model.load_state_dict(checkpoint["model_state_dict"])
            elif checkpoint["method"] != args.method:
                raise ValueError("Checkpoint method does not match requested method")
            else:
                model.load_state_dict(checkpoint["model"])
            if not args.eval:
                optimizer.load_state_dict(checkpoint["optimizer"])
                if aux_optimizer:
                    aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
                completed = checkpoint["iteration"]

        noise_source = torch.Generator(device=args.device).manual_seed(args.seed + 1)

        def history_observation(observation, previous=None, reset_ids=None):
            frames = observation.reshape(args.num_envs, -1, width)
            clean = frames[:, :1].clone()
            if args.imu_noise:
                gyro = clean[:, :, 3:6]
                gyro.add_((torch.rand(gyro.shape, device=gyro.device, dtype=gyro.dtype, generator=noise_source) * 2 - 1) * args.imu_noise)
            if previous is None:
                return clean.expand(-1, frames.shape[1], -1).flatten(1).contiguous()
            older = previous.reshape_as(frames)[:, :-1]
            result = torch.cat((clean, older), dim=1)
            result[reset_ids] = clean[reset_ids].expand(-1, frames.shape[1], -1)
            return result.flatten(1)

        obs = history_observation(obs)
        meter = None
        if args.energy:
            if shim:
                meter = SynapticMeter(model, shim_spike_inputs(model))
            elif args.method == "himloco":
                meter = SynapticMeter(model)
            elif rma:
                spike_layers = {name for name, submodule in model.named_modules()
                                if isinstance(submodule, nn.Linear) and ".snn." in name}
                meter = SynapticMeter(model, spike_layers)
            else:
                meter = SynapticMeter(model.actor)
        results = []
        args.out.mkdir(parents=True, exist_ok=True)
        episode_start_x = env.core.root_states[:, 0].clone()
        for iteration in range(1 if args.eval else args.iterations):
            rows = []
            tracking_errors = []
            forward_offsets = []
            applied = 0
            impulse_velocity = None
            perturbation_enabled = args.rma_phase == "adaptation" or args.eval
            scale = torch.zeros(actions, device=args.device) if args.method == "himloco" else model.log_std
            completed_episodes = 0
            successes = 0
            episode_success = torch.zeros(args.num_envs, device=args.device, dtype=torch.bool)
            with torch.no_grad():
                for tick in range(args.steps):
                    applying = tick == args.impulse_step and args.impulse != 0
                    if applying:
                        lateral_before = env.core.base_lin_vel[:, 1].clone()
                        env.pending_force[:, env.core.base_body_id, 1] += args.impulse / env.cfg.sim.dt
                        applied += 1
                    location = env.core.root_states[:, :2].clone()
                    command = env.core.commands[:, :2].clone()
                    current_value = value(obs, critic_obs).reshape(-1)
                    with meter if meter else nullcontext():
                        action_mean = mean(obs, None if perturbation_enabled else critic_obs)
                    distribution = Normal(action_mean,
                                          scale.clamp(-5, 2).exp())
                    action = distribution.mean if perturbation_enabled else distribution.sample()
                    nxt, successor, reward, done, extras, ids, terminal = adapter.step(action)
                    if applying:
                        impulse_velocity = float((env.core.base_lin_vel[:, 1] - lateral_before).mean())
                    nxt = history_observation(nxt, obs, ids)
                    terminal_batch = successor.clone()
                    terminal_batch[ids] = terminal
                    next_value = value(nxt, successor).reshape(-1)
                    next_value[done] = 0
                    timeout = extras["time_outs"][ids]
                    if timeout.any():
                        next_value[ids[timeout]] = value(terminal[timeout, :width] if rma else nxt[ids[timeout]],
                                                        terminal[timeout]).reshape(-1)
                    rows.append(tuple(t.clone() for t in (
                        obs, critic_obs, action, distribution.log_prob(action).sum(-1), current_value,
                        reward, next_value, done, terminal_batch
                    )))
                    if args.success_distance_m is not None:
                        progress = env.core.root_states[:, 0] - episode_start_x
                        goal = (progress >= args.success_distance_m) & ~done
                        episode_success |= goal
                        completed_episodes += int(done.sum())
                        successes += int(episode_success[done].sum())
                        episode_success[done] = False
                        episode_start_x[done] = env.core.root_states[done, 0]
                    tracking_errors.append((command - env.core.base_lin_vel[:, :2]).square().sum(-1))
                    displacement = env.core.root_states[:, :2] - location
                    forward_offsets.append(torch.where(done, 0, displacement[:, 0]))
                    obs, critic_obs = nxt, successor
            buffer = [torch.stack(items) for items in zip(*rows)]
            histories, privileged, sampled, old_logp, old_value, reward, next_value, done, successor = buffer
            flatten = lambda data: data.flatten(0, 1)
            statistics = {"iteration": completed + iteration + 1, "mean_reward": float(reward.mean()),
                          "resets": int(done.sum()), "tracking_rmse_mps": float(torch.stack(tracking_errors).mean().sqrt()),
                          "mean_forward_displacement_m": float(torch.stack(forward_offsets).sum(0).mean()),
                          "impulses_per_env": applied}
            if impulse_velocity is not None:
                statistics["mean_lateral_velocity_change_mps"] = impulse_velocity
            if args.success_distance_m is not None:
                ongoing_successes = int(episode_success.sum())
                finished = completed_episodes + ongoing_successes
                success_count = successes + ongoing_successes
                statistics.update(successes=success_count, finished_episodes=finished,
                                  terminated_episodes=completed_episodes,
                                  censored_episodes=args.num_envs - ongoing_successes,
                                  success_rate=success_count / finished if finished else None)
            if not args.eval:
                if aux_optimizer and args.rma_phase != "policy":
                    for batch in torch.randperm(histories.shape[0] * histories.shape[1], device=args.device).split(512):
                        h, p, s = (flatten(data)[batch] for data in (histories, privileged, successor))
                        if shim:
                            auxiliary, regression, swap = model.auxiliary_loss(h, s, args.swap_weight)
                            statistics.update(velocity_loss=float(regression), swap_loss=float(swap))
                        else:
                            auxiliary = model.adaptation_loss(h, p)
                        aux_optimizer.zero_grad(set_to_none=True)
                        auxiliary.backward()
                        nn.utils.clip_grad_norm_(aux_params, 1.0, error_if_nonfinite=True)
                        aux_optimizer.step()
                    statistics["auxiliary_loss"] = float(auxiliary.detach())
                # Historical behavior log-probabilities remain fixed while the updated estimator is frozen.
                if args.rma_phase != "adaptation":
                    target, advantage = returns(reward, old_value, next_value, done)
                    statistics["ppo_loss"] = ppo_update(mean, value, model.log_std, optimizer,
                        tuple(map(flatten, (histories, privileged, sampled, old_logp, old_value, target, advantage))))
                if shim:
                    with torch.no_grad():
                        sample = flatten(histories)[::max(1, histories.shape[0] * histories.shape[1] // 1024)][:1024]
                        model(sample, "adapt")
                    if (completed + iteration + 1) % args.calibration_interval == 0:
                        model.calibrate(sample)
            if shim and (args.grsn_diagnostics or args.grsn_threshold != 1.0 or args.grsn_bias is not None):
                statistics["grsn_layers"] = model.estimator_diagnostics(flatten(histories)[:1024])
            results.append(statistics)
            print(statistics, flush=True)
        with torch.no_grad():
            if shim:
                fused = model.fused()
                error = float((fused(obs) - model(obs)).abs().max())
                if error > 1e-5:
                    raise RuntimeError(f"CaRe-BN fusion round trip failed: {error}")
                statistics["fusion_max_error"] = error
                if not args.eval:
                    inference = torch.jit.freeze(torch.jit.trace(fused, obs, check_trace=False))
                    output = inference(obs[:1])
                    torch.testing.assert_close(output, model(obs[:1]), rtol=1e-5, atol=1e-5)
                    inference.save(str(args.out / "policy.pt"))
            final_action = mean(obs)
            if not torch.isfinite(final_action).all():
                raise RuntimeError("Nonfinite deployment action")
        if not args.eval:
            torch.save({"method": args.method, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer else None,
                        "iteration": completed + args.iterations,
                        "model_config": {key: getattr(args, key) for key in model_config_keys} if shim else {}},
                       args.out / "model.pt")
        report = {"status": "passed", "method": args.method, "mode": "eval" if args.eval else "train",
                  "action_shape": list(final_action.shape), "dt": env.step_dt,
                  "arguments": {key: str(val) if isinstance(val, Path) else val for key, val in vars(args).items()},
                  "results": results}
        if meter:
            report["energy"] = meter.report(args.steps * args.num_envs, 1 / env.step_dt)
        (args.out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()
