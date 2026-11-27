"""Run shared-terrain A1 PPO, HIMLoco, FullySNN RMA, and S-HIM training or short evaluation."""

import json
import math
from contextlib import nullcontext
from pathlib import Path


def run(args):
    import random
    from time import monotonic

    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.distributions import Normal

    from graduation.evaluation.energy import SynapticMeter
    from graduation.methods.fully_snn.rma import build_rma
    from graduation.methods.ppo_mlp.policy import build_mlp
    from graduation.methods.s_him.policy import SHIM, shim_spike_inputs
    from graduation.rl.amp import (
        AMP_OBS_DIM, AmpDiscriminator, AmpMotionDataset, AmpReplayBuffer,
        RunningMomentNormalizer, attach_amp_terminal_hook, extract_him_amp_observations,
        resolve_amp_motion_files,
    )
    from graduation.rl.ppo import ppo_update, returns
    from graduation.sim.himloco import make_blind_env

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env, adapter, defaults = make_blind_env(args)
    try:
        obs, critic_obs = adapter.reset()
        model_config_keys = ("normalization", "history", "estimator_width", "estimator_layers", "actor_widths",
                             "population", "ticks", "en_pop_dim", "estimator", "grsn_threshold", "grsn_bias",
                             "critic_widths", "actor_output_norm", "actor_type", "actor_norm", "actor_threshold",
                             "actor_output_spike", "actor_readout_mode", "actor_surrogate",
                             "en_mean_range", "en_std", "freeze_encoder", "actor_intra_feedback",
                             "swap_temperature", "swap_reduction", "target_mode")
        args.critic_widths = (512, 256, 128) if args.paper_training else (256, 128)
        raw_widths = getattr(args, "actor_widths", None)
        args.actor_widths = (128, 64) if raw_widths is None else tuple(raw_widths)
        args.ticks = getattr(args, "ticks", 4)
        args.en_pop_dim = getattr(args, "en_pop_dim", 10)
        args.en_mean_range = tuple(getattr(args, "en_mean_range", (-5.0, 5.0)))
        args.en_std = float(getattr(args, "en_std", 0.3872983346))
        args.freeze_encoder = getattr(args, "freeze_encoder", False)
        args.actor_intra_feedback = getattr(args, "actor_intra_feedback", False)
        args.actor_weight_decay = float(getattr(args, "actor_weight_decay", 0.0))
        args.amp = getattr(args, "amp", False)
        args.amp_motion_set = getattr(args, "amp_motion_set", "a1_diagonal")
        args.amp_motions = getattr(args, "amp_motions", None)
        args.amp_reward_coef = float(getattr(args, "amp_reward_coef", 2.0))
        args.amp_task_lerp = float(getattr(args, "amp_task_lerp", 0.0))
        args.amp_discr_lr = float(getattr(args, "amp_discr_lr", 1e-4))
        args.amp_discr_updates = int(getattr(args, "amp_discr_updates", 20))
        args.amp_batch_size = int(getattr(args, "amp_batch_size", 1024))
        args.amp_replay_size = int(getattr(args, "amp_replay_size", 1000000))
        args.amp_preload = int(getattr(args, "amp_preload", 200000))
        args.amp_discr_widths = tuple(getattr(args, "amp_discr_widths", (1024, 512)))
        args.actor_output_norm = getattr(args, "actor_output_norm", False)
        args.actor_type = getattr(args, "actor_type", "lif")
        raw_actor_norm = getattr(args, "actor_norm", None)
        args.actor_norm = None if raw_actor_norm in (None, "inherit") else raw_actor_norm
        args.actor_threshold = getattr(args, "actor_threshold", 1.0)
        args.actor_readout_mode = getattr(args, "actor_readout_mode", "spike")
        args.actor_surrogate = getattr(args, "actor_surrogate", "atan")
        args.actor_output_spike = getattr(args, "actor_output_spike", args.actor_readout_mode == "spike")
        args.freeze_aux_estimator_stats = getattr(args, "freeze_aux_estimator_stats", False)
        args.balanced_calibration = getattr(args, "balanced_calibration", False)
        args.learning_rate = getattr(args, "learning_rate", 3e-4)
        args.aux_learning_rate = getattr(args, "aux_learning_rate", 1e-3)
        args.lr_schedule = getattr(args, "lr_schedule", "fixed")
        args.desired_kl = getattr(args, "desired_kl", 0.01)
        args.max_grad_norm = getattr(args, "max_grad_norm", 1.0)
        args.separate_grad_clip = getattr(args, "separate_grad_clip", False)
        args.log_std_lr_scale = getattr(args, "log_std_lr_scale", 1.0)
        args.entropy_coef = float(getattr(args, "entropy_coef", 0.01))
        args.max_action_std = getattr(args, "max_action_std", None)
        max_log_std = math.log(float(args.max_action_std)) if args.max_action_std else 2.0
        args.swap_temperature = getattr(args, "swap_temperature", 0.1)
        args.swap_reduction = getattr(args, "swap_reduction", "sum")
        args.target_mode = getattr(args, "target_mode", "obs")
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True) if args.checkpoint else None
        if checkpoint and args.method == "s-him":
            for key, val in checkpoint.get("model_config", {}).items():
                setattr(args, key, val)
        args.grsn_threshold = 1.0 if args.grsn_threshold is None else args.grsn_threshold
        width = adapter.num_one_step_obs
        actions = adapter.num_actions
        himloco = args.method == "himloco"
        alg = None
        if himloco:
            from rsl_rl.modules.him_actor_critic import HIMActorCritic
            model = HIMActorCritic(obs.shape[1], critic_obs.shape[1], width, actions,
                                   **defaults["policy"]).to(args.device)
            if not args.eval:
                from rsl_rl.algorithms.him_ppo import HIMPPO

                algorithm = dict(
                    defaults["algorithm"], learning_rate=args.learning_rate, schedule=args.lr_schedule,
                    desired_kl=args.desired_kl, entropy_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
                )
                alg = HIMPPO(model, device=args.device, **algorithm)
                alg.init_storage(args.num_envs, args.steps, [obs.shape[1]], [critic_obs.shape[1]], [actions])
        elif args.method == "s-him":
            model = SHIM(width, critic_obs.shape[1], actions, args.normalization, args.history,
                         args.estimator_width, args.estimator_layers, actor_widths=tuple(args.actor_widths),
                         population=args.population, ticks=args.ticks,
                         recurrent=args.estimator == "grsn", grsn_threshold=args.grsn_threshold,
                         grsn_bias=args.grsn_bias, critic_widths=args.critic_widths,
                         actor_output_norm=args.actor_output_norm, actor_type=args.actor_type,
                         swap_temperature=args.swap_temperature, swap_reduction=args.swap_reduction,
                         target_mode=args.target_mode, actor_norm=args.actor_norm,
                         actor_threshold=args.actor_threshold,
                         actor_output_spike=args.actor_output_spike,
                         en_pop_dim=args.en_pop_dim,
                         actor_readout_mode=args.actor_readout_mode,
                         actor_surrogate=args.actor_surrogate,
                         en_mean_range=args.en_mean_range,
                         en_std=args.en_std,
                         freeze_encoder=args.freeze_encoder,
                         actor_intra_feedback=args.actor_intra_feedback).to(args.device)
            if hasattr(model.actor, "static_projection"):
                model.actor.static_projection = not args.legacy_actor_projection
            if args.actor_output_bias_offset and hasattr(model.actor, "layers"):
                with torch.no_grad():
                    model.actor.layers[-1].bias.add_(args.actor_output_bias_offset)
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
        if himloco:
            optimizer = alg.optimizer if alg is not None else None
        elif shim and (args.log_std_lr_scale != 1.0 or args.actor_weight_decay > 0.0):
            optimizer = torch.optim.AdamW([
                {"params": list(model.actor.parameters()), "lr": args.learning_rate, "lr_scale": 1.0,
                 "weight_decay": args.actor_weight_decay},
                {"params": list(model.critic.parameters()), "lr": args.learning_rate, "lr_scale": 1.0,
                 "weight_decay": 0.0},
                {"params": [model.log_std], "lr": args.learning_rate * args.log_std_lr_scale,
                 "lr_scale": args.log_std_lr_scale, "weight_decay": 0.0},
            ], lr=args.learning_rate)
        else:
            optimizer = torch.optim.Adam(ppo_params, lr=args.learning_rate)
        aux_optimizer = torch.optim.Adam(aux_params, lr=args.aux_learning_rate) if aux_params else None

        amp_enabled = bool(args.amp and not args.eval)
        amp_dataset = amp_replay = amp_norm = amp_disc = amp_optimizer = None
        if amp_enabled:
            motion_paths, amp_leg_order = resolve_amp_motion_files(args.amp_motion_set, args.amp_motions)
            amp_dataset = AmpMotionDataset(
                motion_paths, time_between_frames=env.step_dt,
                preload_transitions=args.amp_preload, device=args.device, seed=args.seed,
                leg_order=amp_leg_order,
            )
            amp_replay = AmpReplayBuffer(AMP_OBS_DIM, args.amp_replay_size, args.device)
            amp_norm = RunningMomentNormalizer(AMP_OBS_DIM).to(args.device)
            amp_disc = AmpDiscriminator(
                AMP_OBS_DIM, hidden_dims=args.amp_discr_widths,
                amp_reward_coef=args.amp_reward_coef, task_reward_lerp=args.amp_task_lerp, dt=env.step_dt,
            ).to(args.device)
            amp_optimizer = torch.optim.Adam(amp_disc.parameters(), lr=args.amp_discr_lr, weight_decay=1e-4)
            attach_amp_terminal_hook(env)

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
                if alg is not None:
                    optimizer.load_state_dict(checkpoint.get("optimizer", checkpoint.get("optimizer_state_dict")))
                    alg.learning_rate = optimizer.param_groups[0]["lr"]
                    estimator_state = checkpoint.get("estimator_optimizer", checkpoint.get("estimator_optimizer_state_dict"))
                    if estimator_state is not None:
                        model.estimator.optimizer.load_state_dict(estimator_state)
                else:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if aux_optimizer:
                    aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
                if amp_enabled and checkpoint.get("amp") is not None:
                    amp_state = checkpoint["amp"]
                    amp_disc.load_state_dict(amp_state["discriminator"])
                    amp_norm.load_state_dict(amp_state["normalizer"])
                    amp_optimizer.load_state_dict(amp_state["optimizer"])
                if himloco:
                    completed = int(checkpoint.get("iteration", checkpoint.get("iter", 0)))
                else:
                    completed = checkpoint["iteration"]
                core = getattr(env, "core", None)
                saved = checkpoint.get("terrain_levels")
                target = getattr(core, "terrain_levels", None)
                if args.terrain != "flat" and saved is not None and saved.numel() and target is not None:
                    levels = saved.flatten().to(device=target.device, dtype=target.dtype)
                    if levels.numel() != target.numel():
                        levels = levels[torch.arange(target.numel(), device=target.device) % levels.numel()]
                    if getattr(core, "max_terrain_level", None) is not None:
                        levels = levels.clamp(0, core.max_terrain_level - 1)
                    target.copy_(levels.reshape_as(target))
                    if all(getattr(core, name, None) is not None
                           for name in ("terrain_origins", "terrain_types", "env_origins")):
                        core.env_origins[:] = core.terrain_origins[target.long(), core.terrain_types.long()]
                    # Respawn on the restored terrain without applying the curriculum to this reset.
                    had_init_done = hasattr(core, "init_done")
                    init_done = getattr(core, "init_done", None)
                    core.init_done = False
                    try:
                        obs, critic_obs = adapter.reset()
                    finally:
                        if had_init_done:
                            core.init_done = init_done
                        else:
                            del core.init_done
        if shim and args.max_action_std:
            with torch.no_grad():
                model.log_std.clamp_(max=max_log_std)

        teacher = warm_optimizer = warmstart_spec = None
        if args.warmstart_teacher:
            from rsl_rl.modules.him_actor_critic import HIMActorCritic

            warmstart_spec = {"teacher": str(args.warmstart_teacher.resolve()), "iterations": args.warmstart_iterations}
            if checkpoint and checkpoint.get("warmstart_spec") != warmstart_spec:
                raise ValueError("Resume imitation with the same teacher and warm-start iteration cutoff")
            weights = torch.load(args.warmstart_teacher, map_location=args.device, weights_only=True)
            teacher = HIMActorCritic(obs.shape[1], critic_obs.shape[1], width, actions,
                                     **defaults["policy"]).to(env.device).eval()
            teacher.load_state_dict(weights["model_state_dict"])
            teacher.requires_grad_(False)
            warm_optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-3)
            if checkpoint:
                warm_optimizer.load_state_dict(checkpoint["warm_optimizer"])
            else:
                model.critic.load_state_dict(teacher.critic.state_dict())
                with torch.no_grad():
                    model.log_std.copy_(teacher.std.detach().clamp_min(1e-6).log())

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
        def save_checkpoint(path, iteration):
            temp = path.with_suffix(".tmp")
            if himloco:
                state = {"method": "himloco", "model_state_dict": model.state_dict(),
                         "optimizer": optimizer.state_dict(), "iteration": iteration,
                         "estimator_optimizer": model.estimator.optimizer.state_dict()}
            else:
                state = {"method": args.method, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer else None,
                            "iteration": iteration,
                            "model_config": {key: getattr(args, key) for key in model_config_keys} if shim else {}}
            if amp_enabled:
                state["amp"] = {
                    "discriminator": amp_disc.state_dict(),
                    "normalizer": amp_norm.state_dict(),
                    "optimizer": amp_optimizer.state_dict(),
                }
            if warm_optimizer is not None:
                state.update(warmstart_spec=warmstart_spec, warm_optimizer=warm_optimizer.state_dict())
            terrain_levels = getattr(getattr(env, "core", None), "terrain_levels", None)
            if args.terrain != "flat" and terrain_levels is not None:
                state["terrain_levels"] = terrain_levels.detach().cpu().clone()
            torch.save(state, temp)
            temp.replace(path)

        def update_amp(statistics, task_rewards, amp_styles):
            statistics["task_reward_mean"] = float(torch.stack(task_rewards).mean())
            statistics["amp_style_reward_per_sec"] = float(torch.stack(amp_styles).mean())
            statistics.update(amp_disc.update_from_batches(
                amp_replay.sample_minibatches(args.amp_discr_updates, args.amp_batch_size),
                amp_dataset.sample_minibatches(args.amp_discr_updates, args.amp_batch_size),
                amp_optimizer,
                normalizer=amp_norm,
            ))

        def action_kl_stats(after_act, before_act):
            sigma = model.log_std.clamp(-5, max_log_std).exp()
            kl = 0.5 * ((after_act - before_act) / sigma).square().sum(-1)
            return float(kl.mean()), float(torch.quantile(kl.float(), 0.95))

        def actor_population_stats(obs_batch):
            with torch.no_grad():
                features = model.actor_input(obs_batch)
                zero_obs = obs_batch.reshape(obs_batch.shape[0], -1, width).clone()
                zero_obs[..., 0] = 0.0
                pos_obs = zero_obs.clone()
                pos_obs[..., 0] = 0.6 * float(env.core.commands_scale[0])
                neg_obs = zero_obs.clone()
                neg_obs[..., 0] = -0.6 * float(env.core.commands_scale[0])
                act_zero = model(zero_obs.flatten(1))
                act_pos = model(pos_obs.flatten(1))
                act_neg = model(neg_obs.flatten(1))
                base_stats = {
                    "cmd_delta_0_to_pos06": float((act_pos - act_zero).abs().mean()),
                    "cmd_delta_0_to_neg06": float((act_neg - act_zero).abs().mean()),
                }
                if not hasattr(model.actor, "layers"):
                    act = model.actor(features)
                    base_stats["actor_action_std_mean"] = float(act.std(0).mean())
                    return base_stats
                act_base, rates, silent = model.actor.forward_with_stats(features, "fixed")
                base_stats.update({
                    "actor_layer_firing_rates": rates,
                    "actor_layer_silent_ratios": silent,
                    "actor_action_std_mean": float(act_base.std(0).mean()),
                })
                return base_stats

        episode_start_x = env.core.root_states[:, 0].clone()
        amp_obs = extract_him_amp_observations(env.core) if amp_enabled else None
        him_training = alg is not None
        log_training_details = args.paper_training or him_training
        log_aligned_speeds = args.ppo_rollout_features or him_training
        for iteration in range(1 if args.eval else args.iterations):
            started = monotonic()
            lengths = []
            rollouts = env.episode_length_buf
            rows = []
            him_rewards, him_dones = [], []
            warm = completed + iteration < args.warmstart_iterations
            due = iteration == 0 or (completed + iteration + 1) % args.diagnostics_interval == 0
            teacher_actions = []
            behavior_features = []
            aligned_speeds = []
            tracking_errors = []
            forward_offsets = []
            task_rewards = []
            amp_styles = []
            reward_term_totals = {}
            reward_term_steps = 0
            has_episode_sums = hasattr(env.core, "episode_sums")
            log_reward_terms = has_episode_sums and (him_training or (args.paper_training and due))
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
                    if him_training:
                        if args.max_action_std is not None:
                            model.std.clamp_(max=args.max_action_std)
                        action = alg.act(obs, critic_obs)
                    else:
                        current_value = value(obs, critic_obs).reshape(-1)
                        with meter if meter else nullcontext():
                            if args.ppo_rollout_features and not warm:
                                features = model.actor_input(obs)
                                behavior_features.append(features)
                                action_mean = model.actor(features)
                            else:
                                action_mean = mean(obs, None if perturbation_enabled else critic_obs)
                        action_sigma = scale.clamp(-5, max_log_std).exp().expand_as(action_mean)
                        distribution = Normal(action_mean, action_sigma)
                        action = distribution.mean if perturbation_enabled else distribution.sample()
                        if warm:
                            target_action = teacher.act_inference(obs)
                            teacher_actions.append(target_action)
                            teacher_fraction = max(.25, 1. - (completed + iteration) / max(1, args.warmstart_iterations - 1))
                            use_teacher = torch.rand((args.num_envs, 1), device=env.device) < teacher_fraction
                            action = torch.where(use_teacher, target_action, action_mean)
                    if log_training_details:
                        step_lengths = rollouts + 1
                    if log_reward_terms:
                        prev_sums = {k: v.clone() for k, v in env.core.episode_sums.items()}
                    nxt, successor, reward, done, extras, ids, terminal = adapter.step(action)
                    if log_reward_terms:
                        keep = ~done
                        if keep.any():
                            reward_term_steps += int(keep.sum())
                            for k, v in env.core.episode_sums.items():
                                reward_term_totals[k] = reward_term_totals.get(k, 0.0) + float((v[keep] - prev_sums[k][keep]).sum())
                    if amp_enabled:
                        task_rewards.append(reward.clone())
                        next_amp = extract_him_amp_observations(env.core)
                        after_amp = next_amp.clone()
                        if len(ids):
                            after_amp[ids] = env.core._amp_terminal_obs
                        amp_replay.insert(amp_obs, after_amp)
                        reward, style_per_sec, _ = amp_disc.predict_reward(
                            amp_obs, after_amp, reward, normalizer=amp_norm
                        )
                        amp_styles.append(style_per_sec)
                        amp_obs = next_amp
                    if log_training_details:
                        lengths.append(step_lengths[ids])
                    if applying:
                        impulse_velocity = float((env.core.base_lin_vel[:, 1] - lateral_before).mean())
                    nxt = history_observation(nxt, obs, ids)
                    terminal_batch = successor.clone()
                    terminal_batch[ids] = terminal
                    if him_training:
                        alg.process_env_step(reward, done, extras, terminal_batch)
                        him_rewards.append(reward.clone())
                        him_dones.append(done.clone())
                    else:
                        next_value = value(nxt, successor).reshape(-1)
                        next_value[done] = 0
                        timeout = extras["time_outs"][ids]
                        if timeout.any():
                            next_value[ids[timeout]] = value(terminal[timeout, :width] if rma else nxt[ids[timeout]],
                                                            terminal[timeout]).reshape(-1)
                        rows.append(tuple(t.clone() for t in (
                            obs, critic_obs, action, distribution.log_prob(action).sum(-1), current_value,
                            reward, next_value, done, terminal_batch, action_mean, action_sigma
                        )))
                    if args.success_distance_m is not None:
                        progress = env.core.root_states[:, 0] - episode_start_x
                        goal = (progress >= args.success_distance_m) & ~done
                        episode_success |= goal
                        completed_episodes += int(done.sum())
                        successes += int(episode_success[done].sum())
                        episode_success[done] = False
                        episode_start_x[done] = env.core.root_states[done, 0]
                    velocity = env.core.base_lin_vel[:, :2]
                    tracking_errors.append((command - velocity).square().sum(-1))
                    if log_aligned_speeds:
                        projection = (command * velocity).sum(-1) / command.norm(dim=-1).clamp_min(.1)
                        aligned_speeds.append(torch.where(done, 0., projection))
                    displacement = env.core.root_states[:, :2] - location
                    forward_offsets.append(torch.where(done, 0, displacement[:, 0]))
                    obs, critic_obs = nxt, successor
            if him_training:
                reward, done = torch.stack(him_rewards), torch.stack(him_dones)
            else:
                buffer = [torch.stack(items) for items in zip(*rows)]
                histories, privileged, sampled, old_logp, old_value, reward, next_value, done, successor, old_mu, old_sigma = buffer
            flatten = lambda data: data.flatten(0, 1)
            statistics = {"iteration": completed + iteration + 1, "mean_reward": float(reward.mean()),
                          "resets": int(done.sum()), "tracking_rmse_mps": float(torch.stack(tracking_errors).mean().sqrt()),
                          "mean_forward_displacement_m": float(torch.stack(forward_offsets).sum(0).mean()),
                          "impulses_per_env": applied}
            if reward_term_steps > 0:
                denom = reward_term_steps * env.step_dt
                statistics["reward_per_second"] = {k: v / denom for k, v in reward_term_totals.items()}
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
            if log_aligned_speeds:
                statistics["command_aligned_velocity_mps"] = float(torch.stack(aligned_speeds).mean())
            if him_training:
                with torch.no_grad():
                    alg.compute_returns(critic_obs)
                value_loss, surrogate_loss, estimation_loss, swap_loss = alg.update()
                max_lr = getattr(args, "max_lr", 1e-2)
                if alg.learning_rate > max_lr:
                    alg.learning_rate = max_lr
                    for group in optimizer.param_groups:
                        group["lr"] = max_lr
                if args.max_action_std is not None:
                    with torch.no_grad():
                        model.std.clamp_(max=args.max_action_std)
                statistics.update(value_loss=value_loss, surrogate_loss=surrogate_loss,
                                  velocity_loss=estimation_loss, swap_loss=swap_loss,
                                  auxiliary_loss=estimation_loss + swap_loss,
                                  ppo_loss=surrogate_loss + alg.value_loss_coef * value_loss,
                                  learning_rate=alg.learning_rate)
                if amp_enabled:
                    update_amp(statistics, task_rewards, amp_styles)
            elif not args.eval:
                probe_obs = probe_succ = probe_before = None
                if shim and args.paper_training and due:
                    flat_h = flatten(histories)
                    flat_s = flatten(successor)
                    stride = max(1, len(flat_h) // 2048)
                    probe_obs = flat_h[::stride][:2048]
                    probe_succ = flat_s[::stride][:2048]
                    with torch.no_grad():
                        probe_before = model(probe_obs)
                if aux_optimizer and args.rma_phase != "policy":
                    batch_indices = torch.randperm(histories.shape[0] * histories.shape[1], device=args.device)
                    if args.paper_training:
                        batch_indices = batch_indices[:8192]
                    est_stats = "batch" if args.freeze_aux_estimator_stats else "adapt"
                    batches = list(batch_indices.split(2048 if args.paper_training else 512))
                    for bi, batch in enumerate(batches):
                        h, p, s = (flatten(data)[batch] for data in (histories, privileged, successor))
                        if shim:
                            res = model.auxiliary_loss(h, s, args.swap_weight, estimator_stats=est_stats,
                                                       return_metrics=(bi == len(batches) - 1 and due))
                            auxiliary, regression, swap = res[:3]
                            statistics.update(velocity_loss=float(regression), swap_loss=float(swap))
                            if len(res) == 4:
                                statistics.update(res[3])
                        else:
                            auxiliary = model.adaptation_loss(h, p)
                        aux_optimizer.zero_grad(set_to_none=True)
                        auxiliary.backward()
                        nn.utils.clip_grad_norm_(aux_params, 1.0, error_if_nonfinite=True)
                        aux_optimizer.step()
                    statistics["auxiliary_loss"] = float(auxiliary.detach())
                    if probe_obs is not None:
                        with torch.no_grad():
                            aux_kl_m, aux_kl_p95 = action_kl_stats(model(probe_obs), probe_before)
                            spd, _ = model.estimate_history(probe_obs, "fixed")
                            gt_vel = probe_succ[:, width:width + 3] / 2.0
                            statistics.update(
                                aux_update_kl_mean=aux_kl_m,
                                aux_update_kl_p95=aux_kl_p95,
                                speed_rmse_mps=float(F.mse_loss(spd / 2.0, gt_vel).sqrt()),
                                zero_speed_rmse_mps=float(gt_vel.square().mean().sqrt()),
                            )
                # Historical behavior log-probabilities remain fixed while the updated estimator is frozen.
                if warm:
                    policy_obs = flatten(histories)
                    labels = flatten(torch.stack(teacher_actions))
                    selected = torch.randperm(len(labels), device=env.device)[:32768]
                    total = torch.zeros((), device=env.device)
                    for batch in selected.split(2048):
                        predicted = model(policy_obs[batch])
                        loss = F.mse_loss(predicted, labels[batch])
                        warm_optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1., error_if_nonfinite=True)
                        warm_optimizer.step()
                        total += loss.detach() * len(batch)
                    targets, _ = returns(reward, old_value, next_value, done)
                    critic_inputs, targets = flatten(privileged), flatten(targets)
                    critic_total = torch.zeros((), device=env.device)
                    for batch in selected[:8192].split(2048):
                        critic_error = F.mse_loss(model.critic(critic_inputs[batch]).flatten(), targets[batch])
                        optimizer.zero_grad(set_to_none=True)
                        critic_error.backward()
                        nn.utils.clip_grad_norm_(model.critic.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                        critic_total += critic_error.detach() * len(batch)
                    statistics.update(phase="teacher_warmstart", teacher_action_fraction=teacher_fraction,
                                      imitation_mse=float(total / len(selected)),
                                      warmstart_value_mse=float(critic_total / min(len(selected), 8192)))
                elif args.rma_phase != "adaptation":
                    target, advantage = returns(reward, old_value, next_value, done)
                    policy_obs = flatten(histories)
                    policy_mean = mean
                    if args.paper_training:
                        with torch.no_grad():
                            if args.ppo_rollout_features:
                                original = torch.cat(behavior_features)
                                if due:
                                    stride = max(1, len(original) // 2048)
                                    behavior = model.actor(original[::stride])
                                    updated = model.actor(model.actor_input(policy_obs[::stride]))
                                    kl = ((updated - behavior) / model.log_std.clamp(-5, max_log_std).exp()).square()
                                    statistics["estimator_update_policy_kl"] = float(.5 * kl.sum(-1).mean())
                                policy_obs = original
                            else:
                                policy_obs = torch.cat([model.actor_input(chunk) for chunk in policy_obs.split(4096)])
                        policy_mean = lambda features, _: model.actor(features)
                    ppo_loss, ppo_metrics = ppo_update(
                        policy_mean, value, model.log_std, optimizer,
                        (policy_obs, *map(flatten, (privileged, sampled, old_logp, old_value, target, advantage))),
                        epochs=5 if args.paper_training else 4, microbatch=4096 if args.paper_training else None,
                        old_mu=flatten(old_mu), old_sigma=flatten(old_sigma),
                        desired_kl=args.desired_kl if args.lr_schedule == "adaptive" else None,
                        lr_max=getattr(args, "max_lr", 1e-2),
                        max_grad_norm=args.max_grad_norm,
                        separate_grad_clip=args.separate_grad_clip,
                        actor_params=list(model.actor.parameters()) if shim else None,
                        critic_params=list(model.critic.parameters()),
                        std_params=[model.log_std] if shim else None,
                        return_metrics=True,
                        entropy_coef=args.entropy_coef,
                        max_log_std=max_log_std,
                    )
                    statistics["ppo_loss"] = ppo_loss
                    statistics.update(ppo_metrics)
                    if amp_enabled:
                        update_amp(statistics, task_rewards, amp_styles)
                if teacher is not None and not warm:
                    statistics["phase"] = "paper_ppo"
                if shim:
                    with torch.no_grad():
                        size = 2048 if args.paper_training else 1024
                        if args.balanced_calibration:
                            steps_n, envs_n = histories.shape[:2]
                            env_perm = torch.randperm(envs_n, device=args.device)
                            env_idx = env_perm.repeat((size + envs_n - 1) // envs_n)[:size]
                            tick_idx = torch.randint(steps_n, (size,), device=args.device)
                            sample = histories[tick_idx, env_idx]
                        else:
                            sample = flatten(histories)[::max(1, histories.shape[0] * histories.shape[1] // size)][:size]
                        if probe_obs is not None:
                            pre_adapt = model(probe_obs)
                        model(sample, "adapt")
                        if probe_obs is not None:
                            adapt_kl_m, adapt_kl_p95 = action_kl_stats(model(probe_obs), pre_adapt)
                            statistics.update(adapt_kl_mean=adapt_kl_m, adapt_kl_p95=adapt_kl_p95)
                    if args.calibration_interval > 0:
                        is_commit_step = (completed + iteration + 1) % args.calibration_interval == 0
                        if is_commit_step and probe_obs is not None:
                            with torch.no_grad():
                                pre_commit = model(probe_obs)
                        model.calibrate(sample, commit=is_commit_step)
                        if is_commit_step and probe_obs is not None:
                            with torch.no_grad():
                                commit_kl_m, commit_kl_p95 = action_kl_stats(model(probe_obs), pre_commit)
                                statistics.update(commit_kl_mean=commit_kl_m, commit_kl_p95=commit_kl_p95)
                    if probe_obs is not None:
                        statistics.update(actor_population_stats(probe_obs))
            if log_training_details:
                finished_lengths = torch.cat(lengths)
                action_std = model.std.detach() if him_training else model.log_std.detach().exp()
                statistics.update(iteration_seconds=monotonic() - started, samples_per_iteration=args.num_envs * args.steps,
                                  episode_length_steps=float(finished_lengths.float().mean()) if len(finished_lengths) else None,
                                  action_std=float(action_std.mean()),
                                  terrain_level=float(env.core.terrain_levels.float().mean()) if args.terrain != "flat" else None)
            if shim and due and (args.grsn_diagnostics or args.grsn_threshold != 1.0 or args.grsn_bias is not None):
                statistics["grsn_layers"] = model.estimator_diagnostics(flatten(histories)[:1024])
            results.append(statistics)
            print(statistics, flush=True)
            if not args.eval and args.checkpoint_interval and statistics["iteration"] % args.checkpoint_interval == 0:
                save_checkpoint(args.out / f'model_{statistics["iteration"]:06d}.pt', statistics["iteration"])
        if not args.eval:
            save_checkpoint(args.out / "model.pt", completed + args.iterations)
        with torch.no_grad():
            if shim:
                from graduation.methods.s_him.export import check_and_export

                statistics.update(check_and_export(model, obs, None if args.eval else args.out / "policy.pt"))
            final_action = mean(obs)
            if not torch.isfinite(final_action).all():
                raise RuntimeError("Nonfinite deployment action")
        report = {"status": "passed", "method": args.method, "mode": "eval" if args.eval else "train",
                  "action_shape": list(final_action.shape), "dt": env.step_dt,
                  "arguments": {key: str(val) if isinstance(val, Path) else val for key, val in vars(args).items()},
                  "results": results}
        if meter:
            report["energy"] = meter.report(args.steps * args.num_envs, 1 / env.step_dt)
        (args.out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()
