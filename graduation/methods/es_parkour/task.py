"""ES-Parkour rewards and oracle heading on the shared course."""

from types import SimpleNamespace

import torch
from wmp_lab.core import Core


class ParkourCore(Core):
    """Task rewards and oracle heading from the course's upcoming waypoints."""

    def __init__(self, backend):
        super().__init__(backend)
        if self.terrain is not None and hasattr(self.terrain, "waypoints"):
            self.terrain.waypoints = torch.as_tensor(self.terrain.waypoints, device=self.device)

    def goal_direction(self):
        if self.terrain is not None and hasattr(self.terrain, "waypoints"):
            points = self.terrain.waypoints[self.terrain_levels, self.terrain_types]
            local_x = self.root_states[:, 0] - self.env_origins[:, 0]
            next_index = (points[:, :, 0] < local_x[:, None] + .25).sum(dim=1).clamp_max(points.shape[1] - 1)
            offsets = points[torch.arange(self.num_envs, device=self.device), next_index]
            displacement = self.env_origins[:, :2] + offsets - self.root_states[:, :2]
        else:
            displacement = torch.stack((torch.full_like(self.base_lin_vel[:, 0], 1.25),
                                        self.env_origins[:, 1] - self.root_states[:, 1]), dim=1)
        return displacement / displacement.norm(dim=1, keepdim=True).clamp_min(1e-5)

    def heading_error(self):
        direction = self.goal_direction()
        desired = torch.atan2(direction[:, 1], direction[:, 0])
        x, y, z, w = self.base_quat.unbind(1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        error = desired - yaw
        return torch.atan2(error.sin(), error.cos())

    def compute_reward(self):
        contact = self.sensor_forces[:, :, 2] > 1.
        self.contact_filt = contact | self.last_contacts
        self.last_contacts = contact
        super().compute_reward()

    def _reward_feet_edge(self):
        if self.terrain is None:
            return torch.zeros_like(self.base_lin_vel[:, 0])
        positions = self.rigid_body_states[:, self.feet_indices, :2]
        indices = ((positions + self.cfg.terrain.border_size) / self.cfg.terrain.horizontal_scale).round().long()
        x = indices[..., 0].clamp(0, self.x_edge_mask.shape[0] - 1)
        y = indices[..., 1].clamp(0, self.x_edge_mask.shape[1] - 1)
        return (self.x_edge_mask[x, y] & self.contact_filt).sum(dim=1) * (self.terrain_levels > 3)

    def _reward_tracking_goal_vel(self):
        speed = (self.goal_direction() * self.root_states[:, 7:9]).sum(dim=1)
        command = self.commands[:, 0].clamp_min(.05)
        return torch.minimum(speed, command) / command

    def _reward_tracking_yaw(self):
        return (-self.heading_error().abs()).exp()


def configure_rewards(legacy, reward_profile):
    legacy.core_factory = ParkourCore
    paper_scales = {
        "tracking_goal_vel": 1.5, "tracking_yaw": .5, "lin_vel_z": -1.,
        "ang_vel_xy": -.05, "orientation": -1., "dof_acc": -2.5e-7,
        "collision": -10., "action_rate": -.1, "delta_torques": -1e-7,
        "torques": -1.e-5, "hip_pos": -.5, "dof_error": -.04,
        "feet_stumble": -1., "feet_edge": -1.,
    }
    if reward_profile == "soft":
        paper_scales.update(collision=-1., feet_stumble=-.1, feet_edge=-.1)
    legacy.rewards.scales = SimpleNamespace(**paper_scales)
    legacy.rewards.reward_curriculum = False
    legacy.rewards.only_positive_rewards = reward_profile == "strict"
