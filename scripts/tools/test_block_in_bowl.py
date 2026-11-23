import argparse
import pathlib
import numpy as np
import torch
from PIL import Image

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="RPL-BlockInBowl-Render-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import rpl_demo_worth.tasks  # noqa: F401
from rpl_demo_worth.tasks.manager_based.block_in_bowl.mdp.terminations import (
    block_in_bowl_success,
    object_out_of_workspace,
)


def save_cam_views(env, out_dir: pathlib.Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for cam_name in ("table_cam", "wrist_cam"):
        cam = env.unwrapped.scene[cam_name]
        rgb = cam.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
        Image.fromarray(rgb).save(out_dir / f"{prefix}_{cam_name}_rgb.png")

        depth = cam.data.output["distance_to_image_plane"][0, :, :, 0].cpu().numpy()
        finite = np.isfinite(depth)
        d_vis = np.zeros_like(depth, dtype=np.uint8)
        if finite.any():
            d_min, d_max = np.percentile(depth[finite], [2, 98])
            d_norm = np.clip((depth - d_min) / max(d_max - d_min, 1e-5), 0.0, 1.0)
            d_vis[finite] = (d_norm[finite] * 255).astype(np.uint8)
        Image.fromarray(d_vis).save(out_dir / f"{prefix}_{cam_name}_depth.png")


def main():
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    zero_action = torch.zeros((1, env.action_space.shape[-1]), device=env.unwrapped.device)

    # 1. Settle after reset & save camera images
    for _ in range(15):
        env.step(zero_action)

    out_dir = pathlib.Path("outputs/cam_check")
    save_cam_views(env, out_dir, prefix="block_in_bowl")
    print(f"[CAM] Saved RGB & depth snapshots to {out_dir}/")

    init_succ = bool(block_in_bowl_success(env.unwrapped)[0].item())
    init_drop = bool(object_out_of_workspace(env.unwrapped)[0].item())
    print(f"[CHECK 1 - Reset]   success={init_succ} (expect False), dropped={init_drop} (expect False)")

    # 2. Place block inside bowl & step until settled
    bowl_pos = env.unwrapped.scene["bowl"].data.root_pos_w.clone()
    place_pose = torch.zeros((1, 7), device=env.unwrapped.device)
    place_pose[:, :2] = bowl_pos[:, :2]
    place_pose[:, 2] = bowl_pos[:, 2] + 0.025
    place_pose[:, 3] = 1.0
    env.unwrapped.scene["block"].write_root_pose_to_sim(place_pose)
    env.unwrapped.scene["block"].write_root_velocity_to_sim(torch.zeros((1, 6), device=env.unwrapped.device))

    got_success = False
    for step in range(40):
        _, _, terminated, _, _ = env.step(zero_action)
        if bool(terminated[0].item()) and bool(env.unwrapped.termination_manager.get_term("success")[0].item()):
            got_success = True
            print(f"  -> Triggered 'success' termination cleanly on step {step + 1}!")
            break
    print(f"[CHECK 2 - Placed]  success={got_success} (expect True)")

    # 3. Drop block off table
    drop_pose = torch.tensor([[0.5, 0.0, -0.10, 1.0, 0.0, 0.0, 0.0]], device=env.unwrapped.device)
    env.unwrapped.scene["block"].write_root_pose_to_sim(drop_pose)
    env.unwrapped.scene["block"].write_root_velocity_to_sim(torch.zeros((1, 6), device=env.unwrapped.device))
    env.unwrapped.sim.step()
    env.unwrapped.scene.update(dt=env.unwrapped.sim.get_physics_dt())
    got_drop = bool(object_out_of_workspace(env.unwrapped)[0].item())
    print(f"[CHECK 3 - Dropped] dropped={got_drop} (expect True)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
