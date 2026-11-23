import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def block_in_bowl_success(
    env: ManagerBasedRLEnv,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    xy_threshold: float = 0.038,   # bowl inner radius ~4.2 cm
    rim_height: float = 0.050,     # bowl rim height = 5.50 cm
    max_lin_vel: float = 0.02,     # m/s
    max_ang_vel: float = 0.25,     # rad/s
    hold_steps: int = 5,
) -> torch.Tensor:
    """True when block center is inside bowl, below rim, and at rest for hold_steps."""
    block = env.scene[block_cfg.name]
    bowl = env.scene[bowl_cfg.name]

    block_pos = block.data.root_pos_w - env.scene.env_origins
    bowl_pos = bowl.data.root_pos_w - env.scene.env_origins

    xy_dist = torch.linalg.norm(block_pos[:, :2] - bowl_pos[:, :2], dim=-1)
    z_rel = block_pos[:, 2] - bowl_pos[:, 2]

    lin_speed = torch.linalg.norm(block.data.root_lin_vel_w, dim=-1)
    ang_speed = torch.linalg.norm(block.data.root_ang_vel_w, dim=-1)

    in_bowl = (xy_dist < xy_threshold) & (z_rel > -0.005) & (z_rel < rim_height)
    at_rest = (lin_speed < max_lin_vel) & (ang_speed < max_ang_vel)
    cond = in_bowl & at_rest

    if not hasattr(env, "_bib_success_steps"):
        env._bib_success_steps = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    env._bib_success_steps = torch.where(
        cond, env._bib_success_steps + 1, torch.zeros_like(env._bib_success_steps)
    )
    return env._bib_success_steps >= hold_steps


def object_out_of_workspace(
    env: ManagerBasedRLEnv,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    min_z: float = -0.05,
    x_bounds: tuple[float, float] = (0.15, 0.90),
    y_bounds: tuple[float, float] = (-0.50, 0.50),
) -> torch.Tensor:
    """Failure check: block or bowl fell off the table or left workspace."""
    failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for cfg in (block_cfg, bowl_cfg):
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
