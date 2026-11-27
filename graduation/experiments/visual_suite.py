"""Evaluate WMP, Event-Spiking-WMP, ES-Parkour and S-HIM on one A1 course contract."""

import json
import random
from contextlib import nullcontext

from graduation.sim.wmp import make_visual_env


def run(args):
    import numpy as np
    import torch

    from graduation.evaluation.energy import SynapticMeter
    from graduation.evaluation.episodes import CourseEpisodes
    from graduation.evaluation.stop_and_go import StopAndGo
    from graduation.methods.es_parkour.student import ESStudent, es_spike_inputs
    from graduation.methods.es_wmp.student import EventSpikingWMP
    from graduation.methods.s_him.policy import SHIM, shim_spike_inputs
    from graduation.methods.wmp.teacher import FrozenWMP
    from graduation.methods.wmp.profiling import WMPDeploymentMeter
    from graduation.sim.a1_observations import body_observation, shim_observation_converter
    from graduation.sim.event_camera import EventCamera

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    env, adapter, _ = make_visual_env(args, capture_outcomes=True)
    try:
        obs, _ = adapter.reset()
        core = env.core
        period = core.cfg.depth.update_interval
        camera = EventCamera(env, args.threshold, args.delay_frames, args.blur, args.seed, args.flow_mode)
        count = args.num_envs
        method = args.method
        event_inputs = set()
        observation_probe = None
        if method == "wmp":
            teacher = FrozenWMP(args.checkpoint, env, adapter, args.out)
            meter = WMPDeploymentMeter(teacher) if args.energy else None
            act_wmp = meter.act if meter else teacher.act
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            saved = torch.load(args.checkpoint, map_location=env.device, weights_only=True)
            if method == "event-spiking-wmp":
                model = EventSpikingWMP(**saved["config"]).to(env.device)
                model.load_state_dict(saved["student"])
                model.eval()
                model = model.fused()
                model.actor.static_projection = not args.legacy_actor_projection
                state, visual = model.initial(count)
                spike_inputs = model.spike_layers()
                event_inputs = {"convs.0"}
            elif method == "es-parkour":
                model = ESStudent(44, adapter.num_actions, base=saved["base_channels"],
                                  **saved.get("student_config", {})).to(env.device)
                model.load_state_dict(saved["student"])
                model.eval()
                state = torch.zeros(count, model.fuse.hidden_size, device=env.device)
                spike_inputs = es_spike_inputs(model)
                event_inputs = {"stem"}
            else:
                config = saved.get("model_config", {})
                # The 238 critic input is used only to load checkpoint tensors. Actor inference is blind.
                model = SHIM(45, 238, adapter.num_actions, normalization=config.get("normalization", "care"),
                             history=config.get("history", 5), estimator_width=config.get("estimator_width", 64),
                             estimator_layers=config.get("estimator_layers", 2), population=config.get("population", 5),
                             recurrent=config.get("estimator", "grsn") == "grsn",
                             grsn_threshold=config.get("grsn_threshold", 1.0),
                             grsn_bias=config.get("grsn_bias"),
                             critic_widths=config.get("critic_widths", (256, 128))).to(env.device)
                model.load_state_dict(saved["model"])
                model = model.fused()
                model.actor.static_projection = not args.legacy_actor_projection
                spike_inputs = shim_spike_inputs(model)
                histories = None
                converter = shim_observation_converter(core)
            meter = SynapticMeter(model, spike_inputs, event_inputs) if args.energy else None
        widths = list(env.terrain.column_types) if env.terrain is not None else ["flat"]
        terrain_id = core.terrain_types if env.terrain is not None else torch.zeros_like(core.terrain_levels)
        episodes = CourseEpisodes(core.root_states[:, 0], terrain_id, core.terrain_levels, widths, args.goal_distance)
        stop = StopAndGo(core.commands, args.pause_at, args.pause_steps, args.stop_speed,
                         args.pause_trigger_distance) if args.pause_steps else None
        obstacle_offsets = None
        if args.pause_trigger_distance is not None:
            from graduation.sim.parkour import FIRST_OBSTACLE_X, SPAWN_X

            obstacle_offsets = torch.tensor(FIRST_OBSTACLE_X, device=env.device)[terrain_id] - SPAWN_X
        terminal = torch.zeros(count, device=env.device, dtype=torch.bool)
        active_pixels = 0
        held_depth = None
        source_steps = None
        frame_clock = []
        rssm_clock = []
        with torch.no_grad(), meter if meter else nullcontext():
            for step in range(args.steps):
                due = step % (period * args.frame_stride) == 0
                stopping = torch.zeros_like(terminal)
                policy_obs = obs
                if stop:
                    obstacle_x = core.env_origins[:, 0] + obstacle_offsets if obstacle_offsets is not None else None
                    stopping, command_override = stop.prepare(step, core.commands, core.root_states[:, 0], obstacle_x)
                    command_slice = slice(core.privileged_dim + 6, core.privileged_dim + 9)
                    policy_obs = obs.clone()
                    policy_obs[command_override, command_slice] = core.commands[command_override, :3] * core.commands_scale
                prop = body_observation(env, policy_obs)
                if method == "wmp":
                    if due:
                        held_depth = camera.depth(newest=False)[..., None]
                        source_steps = camera.source_steps.clone()
                        frame_clock.append({"control_step": step, "source_steps": source_steps.tolist()})
                    if step % teacher.period == 0:
                        rssm_clock.append({"control_step": step, "teacher_depth_source_steps": source_steps.tolist()})
                    action, _ = act_wmp(policy_obs, step, image_override=held_depth)
                elif method == "event-spiking-wmp":
                    if due:
                        event = camera.capture()
                        frame_clock.append({"control_step": step, "source_steps": camera.source_steps.tolist()})
                        if stop:
                            event = stop.events(event, stopping, args.mask_events_on_pause)
                        active_pixels += int(event.sum())
                        visual = model.encode(event)
                    memory_input = torch.zeros_like(state) if args.memory_control == "zero-state" else state
                    action, _, state = model(prop, visual, memory_input)
                elif method == "es-parkour":
                    proprio = torch.cat((prop[:, :8], prop[:, 9:]), dim=1)
                    if due:
                        event = camera.capture()
                        frame_clock.append({"control_step": step, "source_steps": camera.source_steps.tolist()})
                        if stop:
                            event = stop.events(event, stopping, args.mask_events_on_pause)
                        active_pixels += int(event.sum())
                        state = model.observe(event, proprio, state)
                    action, _ = model(proprio, state)
                else:
                    him_frame = converter(prop, core.commands)
                    if histories is None:
                        histories = him_frame[:, None].expand(-1, 6, -1).clone()
                        observation_probe = (prop.clone(), core.commands[:, :3].clone(), histories.clone())
                    else:
                        histories = torch.cat((him_frame[:, None], histories[:, :-1]), dim=1)
                        histories[terminal] = him_frame[terminal, None].expand(-1, 6, -1)
                    action = model(histories.flatten(1))
                if not torch.isfinite(action).all():
                    raise RuntimeError("Nonfinite visual benchmark action")
                target_speed = core.commands[:, 0].clone()
                obs, _, _, done, _, *_ = adapter.step(action)
                terminal = done
                if method == "wmp":
                    teacher.advance(action, done)
                elif method == "event-spiking-wmp":
                    visual = visual * (~done[:, None])
                    state = state * (~done[:, None])
                elif method == "es-parkour":
                    state = state * (~done[:, None])
                camera.reset(done)
                if held_depth is not None and done.any():
                    held_depth = held_depth.masked_fill(done[:, None, None, None], .5)
                    source_steps = source_steps.masked_fill(done, -1)
                episodes.advance(env.transition_x, target_speed - env.transition_speed, ~stopping,
                                 env.transition_failed, done, core.root_states[:, 0], terrain_id, core.terrain_levels)
                if stop:
                    stop.advance(step, env.transition_x, env.transition_planar_speed, done, action)
        episode_rows = episodes.finish()
        n_speed = sum(row["speed_samples"] for row in episode_rows)
        n_finished = sum(row["completed"] for row in episode_rows)
        n_success = sum(row["success"] for row in episode_rows)
        report = {"status": "passed", "protocol_version": 2, "method": method, "purpose": args.purpose, "seed": args.seed,
                  "terrain": args.terrain, "num_envs": count, "steps": args.steps,
                  "goal_distance_m": args.goal_distance, "checkpoint": str(args.checkpoint),
                  "event_threshold": args.threshold, "flow_mode": args.flow_mode,
                  "memory_control": args.memory_control,
                  "checkpoint_memory": saved["config"].get("memory", "grsn") if method == "event-spiking-wmp" else None,
                  "mask_events_on_pause": args.mask_events_on_pause,
                  "pause_trigger_distance_m": args.pause_trigger_distance,
                  "stop_speed_mps": args.stop_speed if stop else None,
                  "delay_frames": args.delay_frames, "frame_stride": args.frame_stride,
                  "camera_hz": 1 / (period * args.frame_stride * env.step_dt),
                  "blur_pixels": args.blur, "pause_at": args.pause_at,
                  "pause_steps": args.pause_steps, "episode_seconds": args.episode_seconds,
                  "mean_forward_progress_m": float(np.mean([row["distance_m"] for row in episode_rows])),
                  "tracking_rmse_mps": (sum(row["speed_sq_error"] for row in episode_rows) / n_speed) ** .5 if n_speed else None,
                  "successes": n_success, "finished_episodes": n_finished,
                  "success_rate": n_success / n_finished if n_finished else None,
                  "censored_episodes": len(episode_rows) - n_finished,
                  "timeout_episodes": sum(row["outcome"] == "timeout" for row in episode_rows),
                  "falls": sum(row["fall"] for row in episode_rows),
                  "active_event_pixels": active_pixels, "episodes": episode_rows}
        report["camera_clock"] = {
            "control_dt_s": env.step_dt,
            "native_depth_period_control_steps": period,
            "delivery_period_control_steps": period * args.frame_stride,
            "delay_unit": "delivered_samples",
            "invalid_source_step": -1,
            "teacher_depth_buffer_index": -2 if method == "wmp" else None,
            "student_depth_buffer_index": -1 if method in ("es-parkour", "event-spiking-wmp") else None,
            "rssm_period_control_steps": teacher.period if method == "wmp" else None,
            "teacher_depth_deliveries": frame_clock if method == "wmp" else [],
            "student_event_deliveries": frame_clock if method in ("es-parkour", "event-spiking-wmp") else [],
            "rssm_updates": rssm_clock,
        }
        if stop:
            records = stop.finish()
            outcomes = {(row["env"], row["episode"]): row["outcome"] for row in episode_rows}
            recovery = [0.] * count
            resumed = set()
            for record in records:
                index = record["env"]
                record["course_outcome"] = outcomes.get((index, record["episode"]))
                recovery[index] = record["recovery_distance_m"]
                if record["resume_step"] >= 0 and record["recovery_distance_m"] >= args.resume_distance:
                    resumed.add(index)
            report["stop_and_go"] = records
            report["stop_attempts"] = len(records)
            report["stop_state_resumed_envs"] = len({row["env"] for row in records if row["resume_step"] >= 0})
            report["post_pause_progress_m"] = recovery
            report["post_pause_resumed_envs"] = len(resumed)
            report["resume_distance_m"] = args.resume_distance
        if observation_probe is not None:
            first_prop, physical_commands, corrected = observation_probe
            legacy_frame = torch.cat((first_prop[:, 6:9], first_prop[:, :6], first_prop[:, 9:]), dim=1)
            legacy = legacy_frame[:, None].expand_as(corrected)
            with torch.no_grad():
                delta = model(corrected.flatten(1)) - model(legacy.flatten(1))
            report["shim_observation_probe"] = {
                "physical_commands": physical_commands.cpu().tolist(),
                "legacy_commands": legacy_frame[:, :3].cpu().tolist(),
                "corrected_commands": corrected[:, 0, :3].cpu().tolist(),
                "action_delta_max_abs": float(delta.abs().max()),
            }
        if meter:
            report["energy"] = meter.report(count * args.steps, 1 / env.step_dt)
        (args.out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print("VALIDATION", {key: val for key, val in report.items() if key not in ("episodes", "energy", "camera_clock", "stop_and_go")}, flush=True)
    finally:
        env.close()
