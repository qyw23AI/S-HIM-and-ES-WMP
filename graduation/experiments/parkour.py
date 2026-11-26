"""Train and evaluate ES-Parkour, or record frozen WMP A1 event rollouts."""

import json
import random
from contextlib import nullcontext

from graduation.methods.es_parkour.teacher import Teacher
from graduation.methods.wmp.rollout import frozen_wmp
from graduation.sim.wmp import make_visual_env


def run(args):
    import numpy as np
    import torch
    from torch import nn
    from torch.distributions import Normal
    from torch.nn import functional as F

    from graduation.evaluation.energy import SynapticMeter
    from graduation.methods.es_parkour.student import ESStudent, es_spike_inputs
    from graduation.sensors.events import event_frame
    from graduation.rl.ppo import ppo_update, returns

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    env, adapter, _ = make_visual_env(args, reward_profile=args.reward_profile if args.mode != "wmp-events" else None)
    try:
        observation, _ = adapter.reset()
        core = env.core
        period = core.cfg.depth.update_interval
        proprio_dim = 44
        scan_dim = core.height_dim
        actions = core.num_actions

        def observations():
            offset = core.privileged_dim
            local = core.obs_buf[:, offset:-scan_dim]
            # Retain body observation and velocity commands. Yaw is a separate teacher signal.
            proprio = torch.cat((local[:, :8], local[:, 9:]), dim=-1)
            heading = core.heading_error()[:, None]
            scan = core.obs_buf[:, -scan_dim:]
            return proprio, heading, torch.cat((scan, heading), dim=1)


        def events():
            near, far = core.cfg.depth.near_clip, core.cfg.depth.far_clip
            depth = (core.depth_buffer[:, -1] + .5) * (far - near) + near
            return event_frame(depth, threshold=args.threshold)

        if args.mode == "wmp-events":
            report = frozen_wmp(args, env, adapter, observation, events)
        elif args.mode == "teacher":
            teacher = Teacher(proprio_dim, scan_dim, actions).to(env.device)
            optimizer = torch.optim.Adam(teacher.parameters(), lr=3e-4)
            if args.checkpoint:
                saved = torch.load(args.checkpoint, map_location=env.device, weights_only=True)
                teacher.load_state_dict(saved["teacher"])
                optimizer.load_state_dict(saved["optimizer"])
            report = {"updates": []}
            for _ in range(args.iterations):
                trajectory = []
                with torch.no_grad():
                    for _ in range(args.steps):
                        prop, _, privileged = observations()
                        old = teacher.value(prop, privileged).squeeze(-1)
                        distribution = Normal(teacher.mean(prop, privileged), teacher.log_std.clamp(-5, 2).exp())
                        action = distribution.sample()
                        _, _, reward, done, *_ = adapter.step(action)
                        next_prop, _, next_privileged = observations()
                        next_value = teacher.value(next_prop, next_privileged).squeeze(-1) * (~done)
                        trajectory.append((prop, privileged, action, distribution.log_prob(action).sum(-1),
                                           old, reward.clone(), next_value, done.clone()))
                prop, prv, action, logp, old, rewards, nxt, done = map(torch.stack, zip(*trajectory))
                target, advantage = returns(rewards, old, nxt, done)
                loss = ppo_update(teacher.mean, teacher.value, teacher.log_std, optimizer,
                                  tuple(t.flatten(0, 1) for t in (prop, prv, action, logp, old, target, advantage)))
                report["updates"].append({"ppo_loss": loss, "mean_reward": float(rewards.mean()),
                                          "reward_std": float(rewards.std(unbiased=False))})
            torch.save({"teacher": teacher.state_dict(), "optimizer": optimizer.state_dict()},
                       args.out / "teacher.pt")
        else:
            saved = (torch.load(args.checkpoint, map_location=env.device, weights_only=True)
                     if args.checkpoint else None)
            channels = saved["base_channels"] if saved else args.base_channels
            student_config = (saved.get("student_config", {}) if saved else {
                "normalization": args.student_normalization or "batch", "threshold": args.student_threshold or 1.0,
                "residual": args.student_residual or "if",
            })
            student = ESStudent(proprio_dim, actions, base=channels, **student_config).to(env.device)
            probe_inputs = None
            optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
            if saved:
                student.load_state_dict(saved["student"])
                if args.mode == "distill":
                    optimizer.load_state_dict(saved["optimizer"])
            teacher = None
            meter = None
            if args.mode == "distill":
                teacher = Teacher(proprio_dim, scan_dim, actions).to(env.device)
                teacher.load_state_dict(torch.load(args.teacher, map_location=env.device, weights_only=True)["teacher"])
                teacher.eval().requires_grad_(False)
            else:
                student.eval().requires_grad_(False)
                if args.energy:
                    meter = SynapticMeter(student, es_spike_inputs(student), event_inputs={"stem"})
            memory = torch.zeros(args.num_envs, student.fuse.hidden_size, device=env.device)
            iteration_start = saved.get("iteration", 0) if saved else 0
            report = {"updates": [], "resets": 0, "event_frames": 0, "spike_pixels": 0,
                      "student_config": student_config}
            global_step = 0
            with meter if meter else nullcontext():
                for iteration in range(args.iterations if teacher else 1):
                    losses = []
                    warmup = teacher is not None and iteration + iteration_start < args.warmup_iterations
                    phase = "teacher_rollout" if warmup else ("student_rollout" if teacher else "eval")
                    for _ in range(args.steps):
                        prop, heading, privileged = observations()
                        if global_step % period == 0:
                            frame = events()
                            if global_step == 0:
                                np.save(args.out / "event_frame.npy", frame[0].cpu().numpy())
                            if args.visual_response_probe and (probe_inputs is None or not probe_has_events):
                                probe_inputs = (frame.detach().clone(), prop.detach().clone(), memory.detach().clone())
                                probe_has_events = bool(frame.any())
                            memory = student.observe(frame, prop, memory)
                            report["event_frames"] += args.num_envs
                            report["spike_pixels"] += int(frame.sum())
                        action, predicted_heading = student(prop, memory)
                        if teacher:
                            with torch.no_grad():
                                target_action = teacher.mean(prop, privileged)
                            losses.append((F.mse_loss(action, target_action), F.mse_loss(predicted_heading, heading)))
                        executed = target_action if warmup else action
                        if not torch.isfinite(executed).all():
                            raise RuntimeError("Nonfinite inference action")
                        _, _, _, done, *_ = adapter.step(executed.detach())
                        memory = memory * (~done[:, None])
                        report["resets"] += int(done.sum())
                        global_step += 1
                    if losses:
                        action_loss, yaw_loss = (torch.stack(t).mean() for t in zip(*losses))
                        optimizer.zero_grad(set_to_none=True)
                        (action_loss + yaw_loss).backward()
                        nn.utils.clip_grad_norm_(student.parameters(), 1.0, error_if_nonfinite=True)
                        optimizer.step()
                        report["updates"].append({"phase": phase, "action_mse": float(action_loss.detach()),
                                                  "yaw_mse": float(yaw_loss.detach())})
                    memory = memory.detach()
            if probe_inputs is not None:
                report["visual_response"] = student.visual_response(*probe_inputs)
            if teacher:
                torch.save({"student": student.state_dict(), "optimizer": optimizer.state_dict(),
                            "base_channels": channels, "student_config": student_config, "iteration": iteration_start + args.iterations},
                           args.out / "student.pt")
            if meter:
                report["energy"] = meter.report(args.steps * args.num_envs, 1 / env.step_dt)
        report.update(mode=args.mode, status="passed", terrain=args.terrain, seed=args.seed,
                      num_envs=args.num_envs, steps=args.steps, policy_hz=1 / env.step_dt,
                      event_hz=1 / (period * env.step_dt), reward_profile=args.reward_profile if args.mode != "wmp-events" else None)
        (args.out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print("VALIDATION", report, flush=True)
    finally:
        env.close()
