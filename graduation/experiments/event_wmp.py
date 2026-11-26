"""Train Event-Spiking-WMP with frozen WMP labels and closed-loop rollouts."""

import json
import random

from graduation.sim.wmp import make_visual_env


def run(args):
    import numpy as np
    import torch

    from graduation.methods.es_wmp.student import EventSpikingWMP, distillation_losses
    from graduation.sim.a1_observations import body_observation
    from graduation.sim.event_camera import EventCamera
    from graduation.methods.wmp.teacher import FrozenWMP

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    env, adapter, _ = make_visual_env(args)
    try:
        obs, _ = adapter.reset()
        teacher = FrozenWMP(args.teacher, env, adapter, args.out)
        # Use FP32 for the student so that nearest-threshold spikes match after BN fusion.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        prop_dim = body_observation(env, obs).shape[-1]
        config = dict(proprio_dim=prop_dim, latent_dim=teacher.latent_size, actions=adapter.num_actions,
                      memory_width=args.memory_width, memory=args.memory,
                      channels=tuple(args.channels), normalization=args.normalization)
        saved = torch.load(args.resume, map_location=env.device, weights_only=True) if args.resume else None
        if saved and saved["config"] != config:
            raise ValueError("Resume requires the saved student architecture")
        student = EventSpikingWMP(**config).to(env.device)
        optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
        completed = 0
        if saved:
            student.load_state_dict(saved["student"])
            optimizer.load_state_dict(saved["optimizer"])
            completed = saved["iteration"]
        camera = EventCamera(env, threshold=args.threshold, seed=args.seed)
        state, visual = student.initial(args.num_envs)
        global_step = 0
        updates = []
        report = {"actions": adapter.num_actions, "teacher_latent": teacher.latent_size,
                  "teacher_iteration": teacher.iteration, "checkpoint_verified": teacher.checkpoint_verified,
                  "camera_hz": 1 / (teacher.period * env.step_dt), "policy_hz": 1 / env.step_dt,
                  "steps_per_update": args.rollout_steps, "envs": args.num_envs,
                  "config": config, "seed": args.seed, "updates": updates,
                  "action_path": args.action_path, "world_weight": args.world_weight,
                  "cosine_weight": args.cosine_weight, "resets": 0}
        for iteration in range(completed, completed + args.iterations):
            losses = []
            phase = "teacher_rollout" if iteration < args.warmup_iterations else "student_rollout"
            first_conv_before = student.convs[0].weight.detach().clone()
            first_actor_before = student.actor.layers[0].weight.detach().clone()
            for _ in range(args.rollout_steps):
                target_action, target_latent = teacher.act(obs, global_step)
                if global_step % teacher.period == 0:
                    frame = camera.capture()
                    visual = student.encode(frame, "adapt")
                prop = body_observation(env, obs)
                action, latent, state = student(prop, visual, state, "adapt",
                                                detach_action=args.action_path == "actor")
                losses.append(distillation_losses(action, target_action, latent, target_latent))
                executed = target_action if phase == "teacher_rollout" else action.detach()
                if not torch.isfinite(executed).all():
                    raise RuntimeError("Nonfinite distillation action")
                obs, _, _, done, *_ = adapter.step(executed)
                teacher.advance(executed, done)
                state = state * (~done[:, None])
                visual = visual * (~done[:, None])
                camera.reset(done)
                report["resets"] += int(done.sum())
                global_step += 1
            action_mse, latent_mse, latent_cosine = (torch.stack(items).mean() for items in zip(*losses))
            total = action_mse + args.world_weight * (latent_mse + args.cosine_weight * latent_cosine)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad = torch.nn.utils.clip_grad_norm_(student.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                # Refresh deployment statistics on aligned control decisions periodically.
                if (iteration + 1) % args.calibration_interval == 0:
                    calibration_visual = student.encode(frame, "calibrate")
                    student(prop, calibration_visual, state.detach(), "calibrate")
                    student.commit_norms()
            state, visual = state.detach(), visual.detach()
            row = {"iteration": iteration + 1, "phase": phase, "action_mse": float(action_mse.detach()),
                   "latent_mse": float(latent_mse.detach()), "latent_cosine_distance": float(latent_cosine.detach()),
                   "gradient_norm": float(grad),
                   "visual_weight_max_change": float((student.convs[0].weight.detach() - first_conv_before).abs().max()),
                   "actor_weight_max_change": float((student.actor.layers[0].weight.detach() - first_actor_before).abs().max())}
            updates.append(row)
            print(row, flush=True)

        student.eval()
        fused = student.fused()
        with torch.no_grad():
            fused_visual = fused.encode(frame)
            regular = student(prop, student.encode(frame), state)
            combined = fused(prop, fused_visual, state)
            errors = [float((a - b).abs().max()) for a, b in zip(regular, combined)]
            for a, b in zip(regular, combined):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
            traced = torch.jit.freeze(torch.jit.trace_module(fused, {
                "forward": (prop, fused_visual, state), "encode": (frame,),
            }, check_trace=False), preserved_attrs=["encode"])
            output = traced(prop[:1], traced.encode(frame[:1]), state[:1])
            for cached, value in zip(output, combined):
                torch.testing.assert_close(cached, value[:1], rtol=1e-5, atol=1e-5)
            traced.save(str(args.out / "policy.pt"))

        torch.save({"config": config, "student": student.state_dict(), "optimizer": optimizer.state_dict(),
                    "iteration": completed + args.iterations}, args.out / "student.pt")
        report.update(status="passed", fusion_max_errors=errors)
        (args.out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()
