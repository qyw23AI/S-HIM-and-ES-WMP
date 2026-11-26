"""Bipolar event-frame synthesis from depth and optical flow."""
import math

import torch
from torch.nn import functional as F


def image_gradient(log_intensity):
    sobel = log_intensity.new_tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))) / 8
    kernels = torch.stack((sobel, sobel.T)).unsqueeze(1)
    return F.conv2d(F.pad(log_intensity[:, None], (1, 1, 1, 1), mode="replicate"), kernels)


def event_frame(depth_m, threshold=0.08, displacement=1.0, flow_pixels=None, generator=None):
    """Apply ES-Parkour Eq. 4 and signed thresholding to a single [B,H,W] depth frame.

    Depth is mapped to positive pseudo-luminance. Explicit flow is in pixels
    per sampling interval, ordered x then y. The default direction follows EOLO.
    """
    if depth_m.ndim != 3 or threshold <= 0 or displacement < 0:
        raise ValueError("Expected [B,H,W] depth, positive threshold and nonnegative displacement")
    depth = torch.nan_to_num(depth_m, nan=2.0, posinf=2.0, neginf=0.02).clamp_min(0.02)
    gradient = image_gradient((0.05 + depth.reciprocal()).log())
    if flow_pixels is None:
        angle = torch.rand(depth.shape, device=depth.device, dtype=depth.dtype, generator=generator) * (2 * math.pi)
        flow_pixels = torch.stack((angle.cos(), angle.sin()), dim=1) * displacement
    if flow_pixels.shape != gradient.shape:
        raise ValueError("Optical flow must have shape [B,2,H,W]")
    change = -(gradient * flow_pixels).sum(dim=1)
    return torch.stack((change >= threshold, change <= -threshold), dim=1).to(depth.dtype)


def analytical_flow(depth, intrinsics, position, rotation, previous_depth, previous_intrinsics,
                    previous_position, previous_rotation, near, far):
    """Return forward image displacement on the current grid for a rigid scene.

    Depth is optical Z. Rotations map ROS optical coordinates to world coordinates.
    Pixels use the Isaac camera convention of half-integer centers.
    """
    batch, height, width = depth.shape
    y, x = torch.meshgrid(torch.arange(height, device=depth.device, dtype=depth.dtype) + .5,
                          torch.arange(width, device=depth.device, dtype=depth.dtype) + .5, indexing="ij")
    pixels = torch.stack((x, y, torch.ones_like(x))).reshape(3, -1).expand(batch, -1, -1)
    valid = torch.isfinite(depth) & (depth >= near) & (depth < far)
    z = torch.where(valid, depth, 0.).reshape(batch, 1, -1)
    current_points = torch.linalg.solve(intrinsics, pixels) * z
    inverse = previous_rotation.transpose(1, 2)
    previous_points = inverse @ (rotation @ current_points + (position - previous_position).unsqueeze(-1))
    projected = previous_intrinsics @ previous_points
    previous_z = projected[:, 2]
    previous_xy = projected[:, :2] / previous_z[:, None].clamp_min(1e-6)
    px, py = previous_xy.unbind(1)
    in_view = (px >= .5) & (px <= width - .5) & (py >= .5) & (py <= height - .5)
    grid = torch.stack((2 * px / width - 1, 2 * py / height - 1), dim=-1).reshape(batch, height, width, 2)
    sampled = F.grid_sample(previous_depth[:, None], grid, mode="nearest", align_corners=False).flatten(1)
    visible = torch.isfinite(sampled) & (sampled >= near) & (sampled < far)
    # Reject points hidden behind the preceding depth surface; allow a small rasterization margin.
    visible &= previous_z <= sampled + (.02 + .02 * sampled)
    valid = valid.flatten(1) & (previous_z >= near) & (previous_z < far) & in_view & visible
    stationary = ((position == previous_position).all(dim=1)
                  & (rotation == previous_rotation).all(dim=(1, 2))
                  & (intrinsics == previous_intrinsics).all(dim=(1, 2)))
    flow = pixels[:, :2] - previous_xy
    return torch.where((valid & ~stationary[:, None])[:, None], flow, 0.).reshape(batch, 2, height, width)
