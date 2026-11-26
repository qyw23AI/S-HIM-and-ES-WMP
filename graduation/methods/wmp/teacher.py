"""Frozen WMP camera, policy and RSSM interface."""
from pathlib import Path

import torch


from graduation.paths import WMP


class FrozenWMP:
    def __init__(self, checkpoint, env, adapter, out):
        from wmp_lab.runner import Runner

        path = Path(checkpoint)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        self.iteration = int(saved["iter"])
        train = saved["train_cfg"]
        train["runner"]["amp_num_preload_transitions"] = 64
        train["algorithm"]["amp_replay_buffer_size"] = 512
        dataset = WMP / "datasets/mocap_motions"
        train["runner"]["amp_motion_files"] = sorted(str(p) for p in dataset.glob("*.txt"))
        wm = saved["wm_config"]
        del saved
        runner = Runner(adapter, train, str(out), env.device, wm_batch=wm["batch_size"],
                        wm_length=wm["batch_length"], wm_updates=wm["train_steps_per_iter"],
                        wm_start=wm["train_start_steps"])
        runner.load(str(path), load_optimizer=False)
        self.checkpoint_verified = runner.round_trip_verified
        self.policy = runner.alg.actor_critic.eval().requires_grad_(False)
        self.world = runner._world_model.eval().requires_grad_(False)
        self.env = env
        self.offset = adapter.privileged_dim
        self.height_dim = adapter.height_dim
        self.period = adapter.cfg.depth.update_interval
        self.latent_size = runner.wm_feature_dim
        device, count = env.device, env.num_envs
        self.history = torch.zeros(count, 5, 42, device=device)
        self.action_history = torch.zeros(count, self.period, 12, device=device)
        self.first = torch.ones(count, device=device)
        self.feature = torch.zeros(count, self.latent_size, device=device)
        self.state = None

    @torch.no_grad()
    def act(self, obs, step, image_override=None):
        start = self.offset
        no_command = torch.cat((obs[:, start:start + 6], obs[:, start + 9:-self.height_dim]), dim=1)
        self.history = torch.cat((self.history[:, 1:], no_command[:, None]), dim=1)
        if step % self.period == 0:
            prop = obs[:, start:start + 33]
            image = self.env.core.depth_buffer[:, -2, ..., None] if image_override is None else image_override
            embedding = self.world.encoder({"prop": prop, "image": image, "is_first": self.first})
            self.state, _ = self.world.dynamics.obs_step(self.state, self.action_history.flatten(1),
                                                       embedding, self.first, sample=False)
            self.feature = self.world.dynamics.get_deter_feat(self.state)
            self.first.zero_()
        action = self.policy.act_inference(obs, self.history.flatten(1), self.feature)
        return action, self.feature

    def advance(self, executed, done):
        self.action_history = torch.cat((self.action_history[:, 1:], executed[:, None]), dim=1)
        self.action_history[done] = 0
        self.history[done] = 0
        self.feature = self.feature.masked_fill(done[:, None], 0)
        self.first[done] = 1
