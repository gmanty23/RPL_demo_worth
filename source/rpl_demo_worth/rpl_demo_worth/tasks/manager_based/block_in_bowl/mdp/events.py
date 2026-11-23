import math
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def reset_block_and_bowl(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    x_range: tuple[float, float] = (0.40, 0.62),
    y_range: tuple[float, float] = (-0.18, 0.18),
    min_separation: float = 0.14,
):
    """Randomize block and bowl positions on the table with guaranteed non-overlap."""
    block = env.scene[block_cfg.name]
    bowl = env.scene[bowl_cfg.name]
    n = len(env_ids)
    device = env.device

    bowl_xy = torch.empty((n, 2), device=device)
    block_xy = torch.empty((n, 2), device=device)

    for i in range(n):
        bx = torch.empty(1, device=device).uniform_(*x_range).item()
        by = torch.empty(1, device=device).uniform_(*y_range).item()
        bowl_xy[i, 0], bowl_xy[i, 1] = bx, by

        for _ in range(50):
            kx = torch.empty(1, device=device).uniform_(*x_range).item()
            ky = torch.empty(1, device=device).uniform_(*y_range).item()
            if math.hypot(kx - bx, ky - by) >= min_separation:
                block_xy[i, 0], block_xy[i, 1] = kx, ky
                break
        else:
            block_xy[i, 0] = min(max(bx + min_separation, x_range[0]), x_range[1])
            block_xy[i, 1] = by

    origins = env.scene.env_origins[env_ids]

    for asset, xy in [(bowl, bowl_xy), (block, block_xy)]:
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

    # Reset per-env success hold counter
    if hasattr(env, "_bib_success_steps"):
        env._bib_success_steps[env_ids] = 0
