"""ES-Parkour rewards and oracle heading on the shared course."""

import torch
from wmp_lab.core import Core


def course_waypoints(core):
    """Return the course waypoints as a tensor on the simulation device."""
    waypoints = getattr(core.terrain, "waypoints", None)
    return torch.as_tensor(waypoints, device=core.device) if waypoints is not None else None


def goal_direction(core, waypoints):
    """Point towards the next waypoint or along the lane on other terrain."""
    if waypoints is not None:
        points = waypoints[core.terrain_levels, core.terrain_types]
        local_x = core.root_states[:, 0] - core.env_origins[:, 0]
        next_index = (points[:, :, 0] < local_x[:, None] + 0.25).sum(dim=1).clamp_max(points.shape[1] - 1)
        offsets = points[torch.arange(core.num_envs, device=core.device), next_index]
        displacement = core.env_origins[:, :2] + offsets - core.root_states[:, :2]
    else:
        displacement = torch.stack(
            (torch.full_like(core.base_lin_vel[:, 0], 1.25), core.env_origins[:, 1] - core.root_states[:, 1]), dim=1,
        )
    return displacement / displacement.norm(dim=1, keepdim=True).clamp_min(1e-5)


def heading_error(core, waypoints):
    """Compute wrapped yaw error from the course's oracle heading."""
    direction = goal_direction(core, waypoints)
    desired = torch.atan2(direction[:, 1], direction[:, 0])
    x, y, z, w = core.base_quat.unbind(1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
    error = desired - yaw
    return torch.atan2(error.sin(), error.cos())


class ParkourCore(Core):
    """Task rewards and oracle heading from the course's upcoming waypoints."""

    def __init__(self, backend):
        super().__init__(backend)
        self._waypoints = course_waypoints(self)

    def goal_direction(self):
        return goal_direction(self, self._waypoints)

    def heading_error(self):
        return heading_error(self, self._waypoints)

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
