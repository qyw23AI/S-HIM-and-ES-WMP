"""Synchronized camera sampling and controlled observation perturbations."""

from collections import deque
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from graduation.sensors.events import analytical_flow, event_frame


@dataclass
class CameraSample:
    depth: torch.Tensor
    step: torch.Tensor
    geometry: tuple | None = None

    def invalidate(self, done):
        self.depth[done] = .5
        self.step[done] = -1


class EventCamera:
    def __init__(self, env, threshold=0.08, delay_frames=0, blur=1, seed=1, flow_mode="random"):
        self.env = env
        self.threshold = threshold
        self.delay_frames = delay_frames
        self.blur = blur
        self.flow_mode = flow_mode
        self.frames = deque(maxlen=delay_frames + 1)
        self.previous = None
        self.source_steps = None
        self.generator = torch.Generator(device=env.device).manual_seed(seed + 140)

    def _sample(self, newest, geometry=False):
        index = -1 if newest else -2
        depth = self.env.core.depth_buffer[:, index].clone()
        if self.blur > 1:
            depth = F.avg_pool2d(depth[:, None], self.blur, stride=1, padding=self.blur // 2,
                                 count_include_pad=False)[:, 0]
        stamps = getattr(self.env, "depth_source_steps", None)
        step = stamps[:, index].clone() if stamps is not None else torch.full(
            (len(depth),), -1, dtype=torch.long, device=depth.device)
        optical = self.env.optical_sample if geometry else None
        self.frames.append(CameraSample(depth, step, optical))
        if len(self.frames) > self.delay_frames:
            sample = self.frames[0]
        else:
            sample = CameraSample(torch.full_like(depth, .5), torch.full_like(step, -1))
        self.source_steps = sample.step
        return sample

    def depth(self, newest=True):
        return self._sample(newest).depth

    def capture(self):
        core = self.env.core
        near, far = core.cfg.depth.near_clip, core.cfg.depth.far_clip
        analytic = self.flow_mode == "analytical"
        sample = self._sample(newest=True, geometry=analytic)
        depth_m = (sample.depth + .5) * (far - near) + near
        if not analytic:
            return event_frame(depth_m, threshold=self.threshold, generator=self.generator)

        previous, self.previous = self.previous, sample
        if previous is None or previous.geometry is None or sample.geometry is None:
            return depth_m.new_zeros((len(depth_m), 2, *depth_m.shape[-2:]))
        flow = analytical_flow(*sample.geometry, *previous.geometry, near, far)
        valid_pair = (previous.step >= 0) & (sample.step > previous.step)
        flow = torch.where(valid_pair[:, None, None, None], flow, 0.)
        return event_frame(depth_m, threshold=self.threshold, flow_pixels=flow)

    def reset(self, done):
        if done.any():
            for sample in self.frames:
                sample.invalidate(done)
            if self.previous is not None:
                self.previous.invalidate(done)
