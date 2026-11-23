import math
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def reset_mug_and_rack(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mug_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
    rack_cfg: SceneEntityCfg = SceneEntityCfg("mug_rack"),
    x_range: tuple[float, float] = (0.40, 0.62),
    y_range: tuple[float, float] = (-0.20, 0.20),
    min_separation: float = 0.20,
):
    """Randomize mug and mug_rack on the table with >= 20 cm separation."""
    mug = env.scene[mug_cfg.name]
    rack = env.scene[rack_cfg.name]
    n = len(env_ids)
    device = env.device

    rack_xy = torch.empty((n, 2), device=device)
    mug_xy = torch.empty((n, 2), device=device)

    for i in range(n):
        rx = torch.empty(1, device=device).uniform_(*x_range).item()
        ry = torch.empty(1, device=device).uniform_(*y_range).item()
        rack_xy[i, 0], rack_xy[i, 1] = rx, ry

        for _ in range(50):
            mx = torch.empty(1, device=device).uniform_(*x_range).item()
            my = torch.empty(1, device=device).uniform_(*y_range).item()
            if math.hypot(mx - rx, my - ry) >= min_separation:
                mug_xy[i, 0], mug_xy[i, 1] = mx, my
                break
        else:
            mug_xy[i, 0] = min(max(rx - min_separation, x_range[0]), x_range[1])
            mug_xy[i, 1] = -ry if abs(ry) > 0.05 else 0.15

    origins = env.scene.env_origins[env_ids]

    for asset, xy in [(rack, rack_xy), (mug, mug_xy)]:
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

    if hasattr(env, "_mor_success_steps"):
        env._mor_success_steps[env_ids] = 0
