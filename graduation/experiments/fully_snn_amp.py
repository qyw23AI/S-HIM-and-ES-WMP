"""Migrate the authors' dedicated FullySNN A1 AMP path to Isaac Lab PhysX."""

import json


def run(args):
    import numpy as np
    import torch

    from rsl_rl.algorithms.amp_discriminator import AMPDiscriminator
    from rsl_rl.algorithms.amp_ppo import AMPPPO
    from rsl_rl.modules.actor_critic import ActorCritic
    from rsl_rl.utils.utils import Normalizer
    from graduation.methods.fully_snn.amp import make_amp_env

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env, adapter, motion = make_amp_env(args)
    try:
        obs, critic = adapter.reset()
        model = ActorCritic(42, 48, 12, actor_hidden_dims=[512, 256, 128],
                            critic_hidden_dims=[512, 256, 128]).to(args.device)
        model.actor.encoder.device = model.actor.snn.device = args.device
        discriminator = AMPDiscriminator(2 * motion.observation_dim, 2., [1024, 512], args.device, .3).to(args.device)
        normalizer = Normalizer(motion.observation_dim)
        algorithm = AMPPPO(model, discriminator, motion, normalizer, device=args.device,
                            amp_replay_buffer_size=args.replay, num_learning_epochs=5, num_mini_batches=4,
                            entropy_coef=.01, clip_param=.2, gamma=.99, lam=.95, desired_kl=.01,
                            schedule="adaptive", learning_rate=.001, max_grad_norm=1.)
        algorithm.init_storage(args.num_envs, args.steps, [42], [48], [12])
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            discriminator.load_state_dict(checkpoint["discriminator"])
            if not args.eval:
                algorithm.optimizer.load_state_dict(checkpoint["optimizer"])
        report = {"mode": "eval" if args.eval else "train", "rollout_steps": args.steps,
                  "dt": env.step_dt, "num_envs": args.num_envs, "obs_shape": list(obs.shape),
                  "critic_shape": list(critic.shape), "amp_obs_shape": list(adapter.get_amp_observations().shape),
                  "updates": [], "resets": 0, "seed": args.seed}
        amp_obs = adapter.get_amp_observations()
        for _ in range(1 if args.eval else args.iterations):
            with torch.inference_mode():
                for _ in range(args.steps):
                    action = model.act_inference(obs) if args.eval else algorithm.act(obs, critic, amp_obs)
                    if not torch.isfinite(action).all():
                        raise RuntimeError("Nonfinite FullySNN AMP action")
                    obs, critic, reward, done, info, ids, terminal = adapter.step(action)
                    report["resets"] += int(done.sum())
                    successor = adapter.get_amp_observations()
                    if not args.eval:
                        after = successor.clone()
                        after[ids] = terminal
                        reward, *_ = discriminator.predict_amp_reward(amp_obs, after, reward, normalizer=normalizer)
                        algorithm.process_env_step(reward, done, info, after)
                    amp_obs = successor
                if not args.eval:
                    algorithm.compute_returns(critic)
            if not args.eval:
                losses = list(map(float, algorithm.update()))
                if not np.isfinite(losses).all():
                    raise RuntimeError("Nonfinite AMP training loss")
                report["updates"].append(losses)
        args.out.mkdir(parents=True, exist_ok=True)
        if not args.eval:
            torch.save({"model": model.state_dict(), "discriminator": discriminator.state_dict(),
                        "optimizer": algorithm.optimizer.state_dict()}, args.out / "model.pt")
        report["status"] = "passed"
        (args.out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print("VALIDATION", report, flush=True)
    finally:
        env.close()
