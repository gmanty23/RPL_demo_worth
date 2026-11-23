import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_inv


def mug_on_rack_success(
    env: ManagerBasedRLEnv,
    mug_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
    rack_cfg: SceneEntityCfg = SceneEntityCfg("mug_rack"),
    handle_local_offset: tuple[float, float, float] = (-0.048, 0.0, 0.060),
    peg_r_min: float = 0.015,
    peg_r_max: float = 0.095,
    peg_z_min: float = 0.090,
    peg_z_max: float = 0.335,
    min_mug_z: float = 0.040,
    max_lin_vel: float = 0.02,
    max_ang_vel: float = 0.25,
    hold_steps: int = 5,
) -> torch.Tensor:
    """True when mug handle sits in the rack peg zone, mug is off the table, and at rest."""
    mug = env.scene[mug_cfg.name]
    rack = env.scene[rack_cfg.name]

    mug_pos_w = mug.data.root_pos_w
    mug_quat_w = mug.data.root_quat_w
    rack_pos_w = rack.data.root_pos_w
    rack_quat_w = rack.data.root_quat_w

    # World position of mug handle center
    offset = torch.tensor(handle_local_offset, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    handle_pos_w = mug_pos_w + quat_apply(mug_quat_w, offset)

    # Express handle center in rack local frame
    handle_in_rack = quat_apply(quat_inv(rack_quat_w), handle_pos_w - rack_pos_w)
    r_xy = torch.linalg.norm(handle_in_rack[:, :2], dim=-1)
    z_rel = handle_in_rack[:, 2]

    mug_z = mug_pos_w[:, 2] - env.scene.env_origins[:, 2]
    lin_speed = torch.linalg.norm(mug.data.root_lin_vel_w, dim=-1)
    ang_speed = torch.linalg.norm(mug.data.root_ang_vel_w, dim=-1)

    on_peg = (
        (r_xy > peg_r_min)
        & (r_xy < peg_r_max)
        & (z_rel > peg_z_min)
        & (z_rel < peg_z_max)
        & (mug_z > min_mug_z)
    )
    at_rest = (lin_speed < max_lin_vel) & (ang_speed < max_ang_vel)
    cond = on_peg & at_rest

    if not hasattr(env, "_mor_success_steps"):
        env._mor_success_steps = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    env._mor_success_steps = torch.where(
        cond, env._mor_success_steps + 1, torch.zeros_like(env._mor_success_steps)
    )
    return env._mor_success_steps >= hold_steps


def mug_out_of_workspace(
    env: ManagerBasedRLEnv,
    mug_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
    rack_cfg: SceneEntityCfg = SceneEntityCfg("mug_rack"),
    min_z: float = -0.05,
    x_bounds: tuple[float, float] = (0.15, 0.90),
    y_bounds: tuple[float, float] = (-0.50, 0.50),
) -> torch.Tensor:
    """Failure check: mug or rack fell off the table or left workspace."""
    failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for cfg in (mug_cfg, rack_cfg):
        pos = env.scene[cfg.name].data.root_pos_w - env.scene.env_origins
        out = (
            (pos[:, 2] < min_z)
            | (pos[:, 0] < x_bounds[0])
            | (pos[:, 0] > x_bounds[1])
            | (pos[:, 1] < y_bounds[0])
            | (pos[:, 1] > y_bounds[1])
        )
        failed |= out
    return failed
