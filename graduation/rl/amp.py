"""Adversarial Motion Priors (AMP) dataset, discriminator, and replay utilities for Unitree A1."""

import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch import nn

from graduation.paths import AMP_A1_MOTIONS, AMP_FOR_HARDWARE, WMP_A1_MOTIONS

UPSTREAM_AMP_DIR = AMP_FOR_HARDWARE / "datasets/mocap_motions"
AMP_MOTION_PRESETS = {
    "a1_diagonal": (WMP_A1_MOTIONS, ("trot1.txt", "trot2.txt"), "fl_fr_rl_rr"),
    "a1_diagonal_turn": (
        None,
        (
            WMP_A1_MOTIONS / "trot1.txt",
            WMP_A1_MOTIONS / "trot2.txt",
            UPSTREAM_AMP_DIR / "leftturn0.txt",
            UPSTREAM_AMP_DIR / "rightturn0.txt",
        ),
        "auto",
    ),
    "upstream_trot": (UPSTREAM_AMP_DIR, ("trot0.txt", "trot1.txt", "leftturn0.txt"), "fr_fl_rr_rl"),
    "upstream": (
        UPSTREAM_AMP_DIR,
        ("trot0.txt", "trot1.txt", "pace0.txt", "pace1.txt", "leftturn0.txt"),
        "fr_fl_rr_rl",
    ),
    "a1_trot": (
        AMP_A1_MOTIONS,
        ("trot0.txt", "trot1.txt", "trot2.txt", "trot_reverse0.txt", "left_turn0.txt", "left_turn1.txt", "right_turn0.txt"),
        "fl_fr_rl_rr",
    ),
    "a1_trot_fwd": (AMP_A1_MOTIONS, ("trot0.txt", "trot1.txt", "trot2.txt"), "fl_fr_rl_rr"),
    "a1_locomotion": (
        AMP_A1_MOTIONS,
        (
            "trot0.txt", "trot1.txt", "trot2.txt", "trot_reverse0.txt",
            "pace0.txt", "pace1.txt", "pace2.txt", "pace_reverse0.txt",
            "left_turn0.txt", "left_turn1.txt", "right_turn0.txt",
        ),
        "fl_fr_rl_rr",
    ),
}
# HIMLabEnv exposes joints in [FR, FL, RR, RL]; canonical AMP uses [FL, FR, RL, RR].
LEGACY_TO_CANONICAL_JOINTS = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)
AMP_OBS_DIM = 43


def resolve_amp_motion_files(preset="a1_diagonal", motion_files=None):
    """Resolve local third_party Unitree A1 mocap clips and source leg ordering."""
    if motion_files:
        paths = [Path(path) for path in motion_files]
        leg_order = "auto"
    else:
        if preset not in AMP_MOTION_PRESETS:
            raise ValueError(f"Unsupported AMP motion preset: {preset}")
        base_dir, names, leg_order = AMP_MOTION_PRESETS[preset]
        paths = [base_dir / name for name in names] if base_dir is not None else list(names)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing AMP motion files: {missing}")
    return paths, leg_order


def _canonical_hip_offsets(device, dtype):
    return torch.tensor(
        (
            (0.1805, 0.047, 0.0),
            (0.1805, -0.047, 0.0),
            (-0.1805, 0.047, 0.0),
            (-0.1805, -0.047, 0.0),
        ),
        device=device,
        dtype=dtype,
    )


def a1_foot_positions_base(joint_pos_canonical):
    """Compute Unitree A1 foot positions in the base frame from [FL, FR, RL, RR] joints."""
    offsets = _canonical_hip_offsets(joint_pos_canonical.device, joint_pos_canonical.dtype)
    signs = (1.0, -1.0, 1.0, -1.0)
    l_hip, l_thigh, l_calf = 0.08505, 0.2, 0.2
    legs = []
    for leg in range(4):
        hip = joint_pos_canonical[:, leg * 3]
        thigh = joint_pos_canonical[:, leg * 3 + 1]
        calf = joint_pos_canonical[:, leg * 3 + 2]
        off_x = -l_thigh * torch.sin(thigh) - l_calf * torch.sin(thigh + calf)
        radius = l_thigh * torch.cos(thigh) + l_calf * torch.cos(thigh + calf)
        off_y = signs[leg] * l_hip * torch.cos(hip) + radius * torch.sin(hip)
        off_z = signs[leg] * l_hip * torch.sin(hip) - radius * torch.cos(hip)
        legs.append(torch.stack((off_x, off_y, off_z), dim=-1) + offsets[leg])
    return torch.cat(legs, dim=-1)


def extract_him_amp_observations(core):
    """Extract noise-free 43-dim A1 AMP observations ordered [FL, FR, RL, RR]."""
    perm = torch.tensor(LEGACY_TO_CANONICAL_JOINTS, device=core.dof_pos.device, dtype=torch.long)
    joint_pos = core.dof_pos.index_select(1, perm)
    joint_vel = core.dof_vel.index_select(1, perm)
    foot_pos = a1_foot_positions_base(joint_pos)
    if hasattr(core, "_get_base_heights"):
        base_height = core._get_base_heights().unsqueeze(-1)
    else:
        base_height = core.root_states[:, 2:3]
    return torch.cat((joint_pos, foot_pos, core.base_lin_vel, core.base_ang_vel, joint_vel, base_height), dim=-1)


def attach_amp_terminal_hook(env):
    """Record noise-free pre-reset AMP observations right before env resets."""
    core = env.core
    if getattr(core, "_amp_terminal_hooked", False):
        return
    original = core.compute_termination_observations

    def wrapped(self, env_ids):
        self._amp_terminal_obs = extract_him_amp_observations(self)[env_ids].clone()
        return original(env_ids)

    core.compute_termination_observations = MethodType(wrapped, core)
    core._amp_terminal_obs = torch.empty(0, AMP_OBS_DIM, device=core.dof_pos.device, dtype=core.dof_pos.dtype)
    core._amp_terminal_hooked = True


def _interpolate_frames(frames, positions):
    max_idx = frames.shape[0] - 1
    low = positions.floor().long().clamp(0, max_idx)
    high = (low + 1).clamp(0, max_idx)
    blend = (positions - low.to(positions.dtype)).unsqueeze(-1)
    return (1.0 - blend) * frames[low] + blend * frames[high]


def _reorder_fr_fl_to_canonical(frames):
    reordered = frames.copy()
    perm = np.asarray(LEGACY_TO_CANONICAL_JOINTS, dtype=np.int64)
    reordered[:, 7:19] = frames[:, 7:19][:, perm]
    reordered[:, 19:31] = frames[:, 19:31][:, perm]
    reordered[:, 37:49] = frames[:, 37:49][:, perm]
    return reordered


class AmpMotionDataset:
    """Pure PyTorch/NumPy Unitree A1 mocap transition sampler matching 43-dim canonical AMP obs."""

    def __init__(self, motion_files, time_between_frames=0.02, preload_transitions=200000,
                 device="cpu", seed=0, leg_order="auto"):
        self.device = torch.device(device)
        self.time_between_frames = float(time_between_frames)
        self.trajectories = []
        self.frame_durations = []
        self.durations = []
        weights = []
        for path in motion_files:
            payload = json.loads(Path(path).read_text())
            frames = np.asarray(payload["Frames"], dtype=np.float32)
            if frames.ndim != 2 or frames.shape[1] < 49 or frames.shape[0] < 2:
                raise ValueError(f"Invalid AMP motion file shape in {path}: {frames.shape}")
            if not np.isfinite(frames).all():
                raise ValueError(f"Nonfinite values in AMP motion file: {path}")
            order = leg_order
            if order == "auto":
                # Front-right leg has negative base-frame y (~ -0.13); front-left has positive y (~ +0.13).
                order = "fr_fl_rr_rl" if float(frames[:, 20].mean()) < 0.0 else "fl_fr_rl_rr"
            if order == "fr_fl_rr_rl":
                frames = _reorder_fr_fl_to_canonical(frames)
            elif order != "fl_fr_rl_rr":
                raise ValueError(f"Unsupported leg_order: {order}")
            amp_obs = np.concatenate((frames[:, 7:49], frames[:, 2:3]), axis=1)
            dt = float(payload["FrameDuration"])
            if dt <= 0:
                raise ValueError(f"Nonpositive FrameDuration in {path}: {dt}")
            self.trajectories.append(torch.from_numpy(amp_obs).to(self.device))
            self.frame_durations.append(dt)
            self.durations.append((frames.shape[0] - 1) * dt)
            weights.append(float(payload.get("MotionWeight", 1.0)))
        total_w = sum(weights)
        self.weights = np.asarray(weights, dtype=np.float64) / total_w
        self.preloaded_s = None
        self.preloaded_s_next = None
        if preload_transitions > 0:
            self._preload(preload_transitions, seed)

    def _preload(self, count, seed):
        rng = np.random.default_rng(seed)
        traj_ids = rng.choice(len(self.trajectories), size=count, replace=True, p=self.weights)
        states = torch.empty((count, AMP_OBS_DIM), device=self.device, dtype=torch.float32)
        next_states = torch.empty((count, AMP_OBS_DIM), device=self.device, dtype=torch.float32)
        for tid in np.unique(traj_ids):
            mask_np = traj_ids == tid
            mask = torch.from_numpy(mask_np).to(self.device)
            n = int(mask_np.sum())
            max_start = max(self.durations[tid] - self.time_between_frames, 0.0)
            times = torch.from_numpy(rng.uniform(0.0, max_start, size=n).astype(np.float32)).to(self.device)
            dt = self.frame_durations[tid]
            frames = self.trajectories[tid]
            states[mask] = _interpolate_frames(frames, times / dt)
            next_states[mask] = _interpolate_frames(frames, (times + self.time_between_frames) / dt)
        self.preloaded_s = states
        self.preloaded_s_next = next_states

    def sample_minibatches(self, num_batches, batch_size):
        if self.preloaded_s is None:
            raise RuntimeError("AmpMotionDataset requires preload_transitions > 0")
        total = len(self.preloaded_s)
        for _ in range(num_batches):
            idx = torch.randint(0, total, (batch_size,), device=self.device)
            yield self.preloaded_s[idx], self.preloaded_s_next[idx]


class RunningMomentNormalizer(nn.Module):
    """Online channel-wise mean/variance normalizer stored on the target device."""

    def __init__(self, dim, epsilon=1e-4, clip=10.0):
        super().__init__()
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, batch):
        if batch.numel() == 0:
            return
        flat = batch.detach().reshape(-1, self.mean.shape[0]).float()
        batch_count = float(flat.shape[0])
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * (self.count * batch_count / total)
        self.mean.copy_(new_mean)
        self.var.copy_((m2 / total).clamp_min(1e-6))
        self.count.copy_(total)

    def normalize(self, x):
        scaled = (x - self.mean) * torch.rsqrt(self.var.clamp_min(1e-6) + self.epsilon)
        return scaled.clamp(-self.clip, self.clip) if self.clip else scaled


class AmpReplayBuffer:
    """Fixed-capacity GPU ring buffer of policy state transitions (s_t, s_{t+1})."""

    def __init__(self, obs_dim, buffer_size, device):
        self.buffer_size = int(buffer_size)
        self.device = torch.device(device)
        self.states = torch.zeros((self.buffer_size, obs_dim), device=self.device)
        self.next_states = torch.zeros((self.buffer_size, obs_dim), device=self.device)
        self.step = 0
        self.num_samples = 0

    @torch.no_grad()
    def insert(self, states, next_states):
        n = int(states.shape[0])
        if n == 0:
            return
        if n >= self.buffer_size:
            self.states.copy_(states[-self.buffer_size :])
            self.next_states.copy_(next_states[-self.buffer_size :])
            self.step = 0
            self.num_samples = self.buffer_size
            return
        end = self.step + n
        if end <= self.buffer_size:
            self.states[self.step : end] = states
            self.next_states[self.step : end] = next_states
        else:
            split = self.buffer_size - self.step
            self.states[self.step :] = states[:split]
            self.next_states[self.step :] = next_states[:split]
            self.states[: end - self.buffer_size] = states[split:]
            self.next_states[: end - self.buffer_size] = next_states[split:]
        self.step = end % self.buffer_size
        self.num_samples = min(self.num_samples + n, self.buffer_size)

    def sample_minibatches(self, num_batches, batch_size):
        if self.num_samples == 0:
            return
        for _ in range(num_batches):
            idx = torch.randint(0, self.num_samples, (batch_size,), device=self.device)
            yield self.states[idx], self.next_states[idx]


class AmpDiscriminator(nn.Module):
    """LSGAN transition discriminator producing step-scaled style reward for HIMLoco/S-HIM."""

    def __init__(
        self,
        obs_dim=AMP_OBS_DIM,
        hidden_dims=(1024, 512),
        amp_reward_coef=2.0,
        task_reward_lerp=0.0,
        dt=0.02,
        grad_pen_lambda=10.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.amp_reward_coef = float(amp_reward_coef)
        self.task_reward_lerp = float(task_reward_lerp)
        self.dt = float(dt)
        self.grad_pen_lambda = float(grad_pen_lambda)
        layers = []
        in_dim = 2 * obs_dim
        for width in hidden_dims:
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ELU())
            in_dim = width
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, state, next_state):
        return self.head(self.trunk(torch.cat((state, next_state), dim=-1)))

    def gradient_penalty(self, expert_state, expert_next_state):
        pair = torch.cat((expert_state, expert_next_state), dim=-1).detach().requires_grad_(True)
        score = self.head(self.trunk(pair))
        grad = torch.autograd.grad(
            outputs=score,
            inputs=pair,
            grad_outputs=torch.ones_like(score),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return self.grad_pen_lambda * grad.square().sum(dim=-1).mean()

    @torch.no_grad()
    def predict_reward(self, state, next_state, task_reward, normalizer=None):
        was_training = self.training
        self.eval()
        s = normalizer.normalize(state) if normalizer is not None else state
        ns = normalizer.normalize(next_state) if normalizer is not None else next_state
        logit = self.forward(s, ns).squeeze(-1)
        style_per_sec = self.amp_reward_coef * torch.clamp(1.0 - 0.25 * (logit - 1.0).square(), min=0.0)
        style_step = style_per_sec * self.dt
        if self.task_reward_lerp > 0.0:
            combined = (1.0 - self.task_reward_lerp) * style_step + self.task_reward_lerp * task_reward
        else:
            combined = task_reward + style_step
        if was_training:
            self.train()
        return combined, style_per_sec, logit

    def update_from_batches(self, policy_batches, expert_batches, optimizer, normalizer=None, max_grad_norm=1.0):
        self.train()
        losses, gpens, exp_scores, pol_scores = [], [], [], []
        for (pol_s_raw, pol_ns_raw), (exp_s_raw, exp_ns_raw) in zip(policy_batches, expert_batches):
            if normalizer is not None:
                pol_s = normalizer.normalize(pol_s_raw)
                pol_ns = normalizer.normalize(pol_ns_raw)
                exp_s = normalizer.normalize(exp_s_raw)
                exp_ns = normalizer.normalize(exp_ns_raw)
            else:
                pol_s, pol_ns, exp_s, exp_ns = pol_s_raw, pol_ns_raw, exp_s_raw, exp_ns_raw
            pol_d = self.forward(pol_s, pol_ns)
            exp_d = self.forward(exp_s, exp_ns)
            disc_loss = 0.5 * ((exp_d - 1.0).square().mean() + (pol_d + 1.0).square().mean())
            grad_pen = self.gradient_penalty(exp_s, exp_ns)
            total = disc_loss + grad_pen
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            if normalizer is not None:
                normalizer.update(pol_s_raw)
                normalizer.update(exp_s_raw)
            losses.append(float(disc_loss.detach()))
            gpens.append(float(grad_pen.detach()))
            exp_scores.append(float(exp_d.detach().mean()))
            pol_scores.append(float(pol_d.detach().mean()))
        if not losses:
            return {}
        return {
            "amp_loss": sum(losses) / len(losses),
            "amp_grad_pen": sum(gpens) / len(gpens),
            "amp_expert_d": sum(exp_scores) / len(exp_scores),
            "amp_policy_d": sum(pol_scores) / len(pol_scores),
        }
