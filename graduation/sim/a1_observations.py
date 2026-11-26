"""Observation slices shared by the visual experiments."""


def body_observation(env, observation):
    core = env.core
    return observation[:, core.privileged_dim:-core.height_dim]


def shim_observation_converter(core):
    """Build the WMP-to-HIM observation transform from the two A1 configs."""
    from himloco_lab.config import asset_contract as him_asset_contract
    from himloco_lab.config import load_config
    from wmp_lab.config import asset_contract as wmp_asset_contract

    him, _ = load_config("a1", core.num_envs, 4)
    _, him_joints, _, _ = him_asset_contract(him)
    _, wmp_joints, _, _ = wmp_asset_contract(core.cfg)
    if him_joints != wmp_joints:
        raise ValueError("A1 policy and simulator joint orders differ")
    if him.control.action_scale != core.cfg.control.action_scale:
        raise ValueError("A1 policy and simulator action scales differ")
    source = core.cfg.normalization.obs_scales
    target = him.normalization.obs_scales
    defaults = [core.cfg.init_state.default_joint_angles[j] - him.init_state.default_joint_angles[j]
                for j in him_joints]
    return ShimObservation(source, target, defaults)


class ShimObservation:
    """Convert WMP's normalized A1 frame to the HIM policy convention."""

    def __init__(self, source, target, default_offsets):
        self.source = source
        self.target = target
        self.default_offsets = default_offsets

    def __call__(self, body, physical_commands):
        import torch

        target = self.target
        commands = physical_commands[:, :3] * body.new_tensor((target.lin_vel, target.lin_vel, target.ang_vel))
        gyro = body[:, :3] * (target.ang_vel / self.source.ang_vel)
        position = (body[:, 9:21] / self.source.dof_pos + body.new_tensor(self.default_offsets)) * target.dof_pos
        velocity = body[:, 21:33] * (target.dof_vel / self.source.dof_vel)
        return torch.cat((commands, gyro, body[:, 3:6], position, velocity, body[:, 33:45]), dim=1)
