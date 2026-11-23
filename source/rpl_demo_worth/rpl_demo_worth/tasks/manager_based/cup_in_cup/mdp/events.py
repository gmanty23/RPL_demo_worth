import math
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def reset_cups(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    cup_a_cfg: SceneEntityCfg = SceneEntityCfg("cup_a"),
    cup_b_cfg: SceneEntityCfg = SceneEntityCfg("cup_b"),
    x_range: tuple[float, float] = (0.40, 0.62),
    y_range: tuple[float, float] = (-0.18, 0.18),
    min_separation: float = 0.14,
):
    """Randomize cup_a (glass1) and cup_b (glass2) on the table with >= 14 cm separation."""
    cup_a = env.scene[cup_a_cfg.name]
    cup_b = env.scene[cup_b_cfg.name]
    n = len(env_ids)
    device = env.device

    b_xy = torch.empty((n, 2), device=device)
    a_xy = torch.empty((n, 2), device=device)

    for i in range(n):
        bx = torch.empty(1, device=device).uniform_(*x_range).item()
        by = torch.empty(1, device=device).uniform_(*y_range).item()
        b_xy[i, 0], b_xy[i, 1] = bx, by

        for _ in range(50):
            ax = torch.empty(1, device=device).uniform_(*x_range).item()
            ay = torch.empty(1, device=device).uniform_(*y_range).item()
            if math.hypot(ax - bx, ay - by) >= min_separation:
                a_xy[i, 0], a_xy[i, 1] = ax, ay
                break
        else:
            a_xy[i, 0] = min(max(bx + min_separation, x_range[0]), x_range[1])
            a_xy[i, 1] = by

    origins = env.scene.env_origins[env_ids]

    for asset, xy in [(cup_b, b_xy), (cup_a, a_xy)]:
        yaw = torch.empty(n, device=device).uniform_(-math.pi, math.pi)
        quat = torch.zeros((n, 4), device=device)
        quat[:, 0] = torch.cos(0.5 * yaw)
        quat[:, 3] = torch.sin(0.5 * yaw)

        pose = torch.zeros((n, 7), device=device)
        pose[:, :2] = origins[:, :2] + xy
        pose[:, 2] = origins[:, 2] + 0.002
        pose[:, 3:] = quat
        vel = torch.zeros((n, 6), device=device)

        asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(vel, env_ids=env_ids)

    if hasattr(env, "_cic_success_steps"):
        env._cic_success_steps[env_ids] = 0
