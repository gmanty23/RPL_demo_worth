import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def cup_in_cup_success(
    env: ManagerBasedRLEnv,
    cup_a_cfg: SceneEntityCfg = SceneEntityCfg("cup_a"),
    cup_b_cfg: SceneEntityCfg = SceneEntityCfg("cup_b"),
    xy_threshold: float = 0.020,     # horizontal alignment tolerance (2.0 cm)
    z_min: float = 0.016,            # nested base height lower bound
    z_max: float = 0.033,            # nested base height upper bound (below cup_b rim 9.76 cm)
    max_lin_vel: float = 0.02,       # m/s
    max_ang_vel: float = 0.50,       # rad/s
    hold_steps: int = 5,
) -> torch.Tensor:
    """True when cup_a is nested inside cup_b, aligned horizontally, and both are at rest."""
    cup_a = env.scene[cup_a_cfg.name]
    cup_b = env.scene[cup_b_cfg.name]

    pos_a = cup_a.data.root_pos_w - env.scene.env_origins
    pos_b = cup_b.data.root_pos_w - env.scene.env_origins

    xy_dist = torch.linalg.norm(pos_a[:, :2] - pos_b[:, :2], dim=-1)
    z_rel = pos_a[:, 2] - pos_b[:, 2]

    a_rest = (
        (torch.linalg.norm(cup_a.data.root_lin_vel_w, dim=-1) < max_lin_vel)
        & (torch.linalg.norm(cup_a.data.root_ang_vel_w, dim=-1) < max_ang_vel)
    )
    b_rest = (
        (torch.linalg.norm(cup_b.data.root_lin_vel_w, dim=-1) < max_lin_vel)
        & (torch.linalg.norm(cup_b.data.root_ang_vel_w, dim=-1) < max_ang_vel)
    )

    nested = (xy_dist < xy_threshold) & (z_rel > z_min) & (z_rel < z_max)
    cond = nested & a_rest & b_rest

    if not hasattr(env, "_cic_success_steps"):
        env._cic_success_steps = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    env._cic_success_steps = torch.where(
        cond, env._cic_success_steps + 1, torch.zeros_like(env._cic_success_steps)
    )
    return env._cic_success_steps >= hold_steps


def cups_out_of_workspace(
    env: ManagerBasedRLEnv,
    cup_a_cfg: SceneEntityCfg = SceneEntityCfg("cup_a"),
    cup_b_cfg: SceneEntityCfg = SceneEntityCfg("cup_b"),
    min_z: float = -0.05,
    x_bounds: tuple[float, float] = (0.15, 0.90),
    y_bounds: tuple[float, float] = (-0.50, 0.50),
) -> torch.Tensor:
    """Failure check: either cup fell off the table or left workspace."""
    failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for cfg in (cup_a_cfg, cup_b_cfg):
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
