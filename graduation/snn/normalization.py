"""Spiking normalization and inference fusion."""

import torch
from torch import nn


class CaReBN(nn.Module):
    """Confidence-weighted moving estimates and population-statistic recalibration."""

    def __init__(self, width, mode="care", threshold=1.0):
        super().__init__()
        self.mode = mode
        self.weight = nn.Parameter(torch.full((width,), threshold / 2))
        self.bias = nn.Parameter(torch.full((width,), threshold / 2 if mode == "care" else 0.0))
        for name in ("mean", "variance", "mean_drift", "var_drift", "cal_mean", "cal_var"):
            self.register_buffer(name, torch.ones(width) if name == "variance" else torch.zeros(width))
        self.count = 0

    def forward(self, current, statistics="fixed"):
        if self.mode == "none":
            return current
        if statistics == "calibrate" and self.mode != "care":
            statistics = "fixed"
        if statistics == "fixed":
            mean, var = self.mean, self.variance
        else:
            data = current.reshape(-1, current.shape[-1])
            var, mean = torch.var_mean(data, dim=0, unbiased=False)
            with torch.no_grad():
                if statistics == "calibrate":
                    n = data.shape[0]
                    ratio = n / (self.count + n)
                    delta = mean - self.cal_mean
                    self.cal_var.lerp_(var, ratio).add_(delta.square() * ratio * (1 - ratio))
                    self.cal_mean.lerp_(mean, ratio)
                    self.count += n
                elif self.mode == "tdbn":
                    self.mean.lerp_(mean, 0.1)
                    self.variance.lerp_(var, 0.1)
                else:
                    degrees = max(data.shape[0] - 1, 1)
                    for running, observed, drift, uncertainty in (
                        (self.mean, mean, self.mean_drift, var / degrees),
                        (self.variance, var, self.var_drift, 2 * var.square() / degrees),
                    ):
                        delta = observed - running
                        drift.lerp_(delta.square(), 0.2)
                        running.add_(delta * drift / (drift + uncertainty).clamp_min(1e-12))
            if statistics == "calibrate":
                return (current - self.cal_mean) * self.weight * torch.rsqrt(self.cal_var + 1e-5) + self.bias
        return (current - mean) * self.weight * torch.rsqrt(var + 1e-5) + self.bias

    @torch.no_grad()
    def commit(self):
        if self.count:
            self.mean.copy_(self.cal_mean)
            self.variance.copy_(self.cal_var)
            self.count = 0

    @torch.no_grad()
    def fuse(self, layer):
        scale = self.weight * torch.rsqrt(self.variance + 1e-5)
        if self.mode != "none":
            layer.weight.mul_(scale.reshape(-1, *([1] * (layer.weight.ndim - 1))))
            layer.bias.copy_(scale * (layer.bias - self.mean) + self.bias)
        self.mode = "none"
