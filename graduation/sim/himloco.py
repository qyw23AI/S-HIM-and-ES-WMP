"""Isaac Lab environment factory for the shared rough-terrain benchmark."""

from copy import deepcopy
from types import MethodType


def configure_progressive_command_curriculum(core, *, init_x, init_y=None, max_x=None, step=0.2, threshold=0.8):
    """Install a progressive velocity command curriculum on one HIMLoco core.

    Linear velocity ranges are symmetric in m/s. The initial lateral half-range
    defaults to ``min(init_x, 1.0)`` and an explicit zero disables lateral commands.
    The forward ceiling defaults to the core's configured command curriculum
    limit, and the lateral ceiling is the smaller of that limit and 1.0 m/s.

    After at least one episode length, a reset evaluates the tracking rewards
    accumulated from resets in the window. Each non-empty velocity group must
    exceed ``threshold`` of the maximum per-step linear tracking reward. A promotion
    adds ``step`` to the active half-ranges up to their ceilings. Only the first
    20% of environment IDs can sample forward speeds beyond 1.0 m/s.
    """
    import torch

    target_max_x = core.cfg.commands.max_curriculum if max_x is None else max_x
    max_y = min(1.0, target_max_x)
    init_y = min(init_x, 1.0) if init_y is None else init_y
    core.cfg.commands.curriculum = True
    core.cfg.commands.max_curriculum = target_max_x
    core.command_ranges["lin_vel_x"] = [-init_x, init_x]
    core.command_ranges["lin_vel_y"] = [-init_y, init_y] if init_y else [0.0, 0.0]

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return

        cur_x_lo, cur_x_hi = self.command_ranges["lin_vel_x"]
        cur_y_lo, cur_y_hi = self.command_ranges["lin_vel_y"]
        low_x_lo, low_x_hi = max(cur_x_lo, -1.0), min(cur_x_hi, 1.0)
        samples = self.commands.new_empty(len(env_ids))
        self.commands[env_ids, 0] = samples.uniform_(low_x_lo, low_x_hi)
        self.commands[env_ids, 1] = samples.uniform_(cur_y_lo, cur_y_hi)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = samples.uniform_(*self.command_ranges["heading"])
        else:
            self.commands[env_ids, 2] = samples.uniform_(*self.command_ranges["ang_vel_yaw"])

        high_vel_env_ids = env_ids[env_ids < int(self.num_envs * 0.2)]
        high_x = self.commands.new_empty(len(high_vel_env_ids)).uniform_(cur_x_lo, cur_x_hi)
        self.commands[high_vel_env_ids, 0] = high_x
        self.commands[high_vel_env_ids, 1] *= high_x.abs() < 1.0

        cutoff = min(0.2, 0.5 * max(abs(cur_x_lo), abs(cur_x_hi)))
        self.commands[env_ids, :2] *= (self.commands[env_ids, :2].norm(dim=1) > cutoff).unsqueeze(1)

    def _split_velocity_groups(self, env_ids):
        high_vel = env_ids < int(self.num_envs * 0.2)
        return env_ids[high_vel], env_ids[~high_vel]

    def _expand_command_ranges(self):
        ceilings = (("lin_vel_x", target_max_x), ("lin_vel_y", max_y if init_y > 0 else 0.0))
        for axis, ceiling in ceilings:
            if ceiling > 0.0:
                expanded = self.command_ranges[axis][1] + step
                half_range = ceiling if expanded >= ceiling else min(round(expanded, 6), ceiling)
                self.command_ranges[axis] = [-half_range, half_range]

    inside_original_reset = False

    def update_command_curriculum(self, env_ids):
        if inside_original_reset or len(env_ids) == 0:
            return

        tracking = self.episode_sums["tracking_lin_vel"]
        reward_threshold = threshold * self.reward_scales["tracking_lin_vel"]
        for group in _split_velocity_groups(self, env_ids):
            if len(group) and not (tracking[group].mean() / self.max_episode_length > reward_threshold):
                return
        _expand_command_ranges(self)

    core._resample_commands = MethodType(_resample_commands, core)
    core.update_command_curriculum = MethodType(update_command_curriculum, core)
    if hasattr(core, "reset_idx"):
        orig_reset_idx = core.reset_idx
        group_sums = [0.0, 0.0]
        group_counts = [0, 0]
        last_eval_step = int(getattr(core, "common_step_counter", 0))

        def reset_idx(self, env_ids):
            nonlocal inside_original_reset, last_eval_step

            if len(env_ids) and getattr(self, "init_done", True):
                tracking = self.episode_sums["tracking_lin_vel"]
                for index, group in enumerate(_split_velocity_groups(self, env_ids)):
                    if len(group):
                        group_sums[index] += tracking[group].sum().item()
                        group_counts[index] += len(group)

            # The legacy reset zeroes these rewards and may invoke its modulo-based update.
            was_inside_original_reset = inside_original_reset
            inside_original_reset = True
            try:
                orig_reset_idx(env_ids)
            finally:
                inside_original_reset = was_inside_original_reset

            current_step = int(getattr(self, "common_step_counter", 0))
            if current_step - last_eval_step < self.max_episode_length or not any(group_counts):
                return

            reward_threshold = threshold * self.reward_scales["tracking_lin_vel"]
            if all(
                (total / count) / self.max_episode_length > reward_threshold
                for total, count in zip(group_sums, group_counts)
                if count
            ):
                _expand_command_ranges(self)
            group_sums[:] = [0.0, 0.0]
            group_counts[:] = [0, 0]
            last_eval_step = current_step

        core.reset_idx = MethodType(reset_idx, core)

    if getattr(core, "commands", None) is not None:
        core._resample_commands(torch.arange(core.num_envs, device=core.device))


def himloco_policy_kwargs(train):
    """Select the native ANN constructor options from the shared policy section."""
    fields = ("init_noise_std", "actor_hidden_dims", "critic_hidden_dims", "activation")
    return {name: train["policy"][name] for name in fields}


def himloco_algorithm_kwargs(train):
    """Select the native HIMPPO options from the shared algorithm section."""
    fields = (
        "value_loss_coef", "use_clipped_value_loss", "clip_param", "entropy_coef", "num_learning_epochs",
        "num_mini_batches", "learning_rate", "schedule", "desired_kl", "gamma", "lam", "max_grad_norm",
    )
    return {name: train["algorithm"][name] for name in fields}


def make_blind_env(args):
    """Build the A1 scene with the requested terrain, rewards, and command curriculum."""
    from himloco_lab.adapter import LegacyRunnerAdapter
    from himloco_lab.env import HIMLabEnv
    from himloco_lab.env_cfg import make_env_cfg

    from graduation.legacy import class_to_dict, load_config
    from graduation.legacy.base.s_him_base_config import SHIMBaseCfg, SHIMBaseCfgPPO

    num_envs = getattr(args, "num_envs", None)
    task = getattr(args, "task", None)
    if task is None:
        legacy = deepcopy(SHIMBaseCfg())
        defaults = class_to_dict(SHIMBaseCfgPPO())
        if num_envs is not None:
            legacy.env.num_envs = num_envs
    else:
        legacy, defaults = load_config(task, num_envs)
    if getattr(args, "terrain", None) == "flat":
        legacy.terrain.mesh_type = "plane"
        legacy.terrain.curriculum = False

    cfg = make_env_cfg(legacy, 4, args.device)
    cfg.seed = getattr(args, "seed", defaults["seed"])
    env = HIMLabEnv(cfg)
    commands = legacy.commands
    init_x = getattr(args, "cmd_curriculum_init_x", None)
    if init_x is not None or getattr(commands, "progressive_curriculum", False):
        if init_x is None:
            init_x = commands.curriculum_init_x
        if init_x is None:
            init_x = max(map(abs, commands.ranges.lin_vel_x))
        init_y = getattr(args, "cmd_curriculum_init_y", None)
        configure_progressive_command_curriculum(
            env.core,
            init_x=init_x,
            init_y=init_y if init_y is not None else commands.curriculum_init_y,
            max_x=getattr(args, "cmd_curriculum_max_x", None),
            step=getattr(args, "cmd_curriculum_step", commands.curriculum_step),
            threshold=getattr(args, "cmd_curriculum_threshold", commands.curriculum_threshold),
        )
    return env, LegacyRunnerAdapter(env), defaults
