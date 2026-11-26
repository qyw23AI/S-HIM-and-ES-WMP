"""Record frozen teacher actions, features, and event frames."""

from contextlib import nullcontext


def frozen_wmp(args, env, adapter, observation, event_callback):
    import numpy as np
    import torch

    from graduation.methods.wmp.profiling import WMPDeploymentMeter
    from graduation.methods.wmp.teacher import FrozenWMP

    teacher = FrozenWMP(args.checkpoint, env, adapter, args.out)
    saved_events, saved_actions, saved_features, saved_proprio, saved_done, event_steps = [], [], [], [], [], []
    resets = 0
    meter = WMPDeploymentMeter(teacher) if args.energy else None
    act = meter.act if meter else teacher.act
    with torch.no_grad(), meter if meter else nullcontext():
        for step in range(args.steps):
            prop = observation[:, teacher.offset:teacher.offset + 33]
            action, feature = act(observation, step)
            if step % teacher.period == 0:
                saved_events.append(event_callback().cpu())
                event_steps.append(step)
            if not (torch.isfinite(action).all() and torch.isfinite(feature).all()):
                raise RuntimeError("Nonfinite frozen WMP outputs")
            saved_actions.append(action.cpu())
            saved_proprio.append(prop.cpu())
            saved_features.append(feature.cpu())
            observation, _, _, done, *_ = adapter.step(action)
            saved_done.append(done.cpu())
            teacher.advance(action, done)
            resets += int(done.sum())
    event_array = torch.stack(saved_events).numpy().astype(np.uint8)
    np.savez_compressed(args.out / "rollout.npz", events=event_array, event_steps=event_steps,
                        actions=torch.stack(saved_actions).numpy(), rssm=torch.stack(saved_features).numpy(),
                        proprio=torch.stack(saved_proprio).numpy(), done=torch.stack(saved_done).numpy())
    result = {"teacher_checkpoint": str(args.checkpoint), "teacher_iterations": teacher.iteration,
              "checkpoint_verified": teacher.checkpoint_verified, "event_shape": list(event_array.shape),
              "active_event_pixels": int(event_array.sum()), "resets": resets,
              "rssm_shape": list(torch.stack(saved_features).shape)}
    if meter:
        report = meter.report(env.num_envs * args.steps, 1 / env.step_dt)
        result["energy"] = report
        result["actor_affine_energy"] = {
            "decisions": report["decisions"], "policy_hz": report["policy_hz"],
            **report["subsystems"]["policy"],
            "layers": {key.removeprefix("policy."): row for key, row in report["layers"].items()
                       if key.startswith("policy.")},
        }
    return result
