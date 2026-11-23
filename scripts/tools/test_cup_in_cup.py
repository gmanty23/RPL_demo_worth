import argparse
import pathlib
import numpy as np
import torch
from PIL import Image

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="RPL-CupInCup-Render-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import rpl_demo_worth.tasks  # noqa: F401
from rpl_demo_worth.tasks.manager_based.cup_in_cup.mdp.terminations import (
    cup_in_cup_success,
    cups_out_of_workspace,
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
    save_cam_views(env, out_dir, prefix="cup_in_cup")
    print(f"[CAM] Saved RGB & depth snapshots to {out_dir}/")

    init_succ = bool(cup_in_cup_success(env.unwrapped)[0].item())
    init_drop = bool(cups_out_of_workspace(env.unwrapped)[0].item())
    print(f"[CHECK 1 - Reset]   success={init_succ} (expect False), dropped={init_drop} (expect False)")

    # 2. Physically drop cup_a into cup_b and step until settled
    cup_b_pos = env.unwrapped.scene["cup_b"].data.root_pos_w.clone()
    nest_pose = torch.zeros((1, 7), device=env.unwrapped.device)
    nest_pose[:, :2] = cup_b_pos[:, :2]
    nest_pose[:, 2] = cup_b_pos[:, 2] + 0.06  # drop cleanly into cup_b
    nest_pose[:, 3] = 1.0
    env.unwrapped.scene["cup_a"].write_root_pose_to_sim(nest_pose)
    env.unwrapped.scene["cup_a"].write_root_velocity_to_sim(torch.zeros((1, 6), device=env.unwrapped.device))

    got_success = False
    last_z_rel = 0.0
    for step in range(60):
        pos_a = env.unwrapped.scene["cup_a"].data.root_pos_w
        pos_b = env.unwrapped.scene["cup_b"].data.root_pos_w
        last_z_rel = float((pos_a[0, 2] - pos_b[0, 2]).item())
        _, _, terminated, _, _ = env.step(zero_action)
        if bool(terminated[0].item()) and bool(env.unwrapped.termination_manager.get_term("success")[0].item()):
            got_success = True
            print(f"  -> Triggered 'success' cleanly on step {step + 1} (settled z_rel = {last_z_rel*1000:.1f} mm)!")
            break
    if not got_success:
        pos_a = env.unwrapped.scene["cup_a"].data.root_pos_w
        pos_b = env.unwrapped.scene["cup_b"].data.root_pos_w
        xy_d = float(torch.linalg.norm(pos_a[0, :2] - pos_b[0, :2]).item()) * 1000.0
        z_r = float((pos_a[0, 2] - pos_b[0, 2]).item()) * 1000.0
        v_a = float(torch.linalg.norm(env.unwrapped.scene["cup_a"].data.root_lin_vel_w[0]).item())
        w_a = float(torch.linalg.norm(env.unwrapped.scene["cup_a"].data.root_ang_vel_w[0]).item())
        print(f"  -> Debug: xy_dist={xy_d:.1f} mm, z_rel={z_r:.1f} mm, lin_vel={v_a:.4f} m/s, ang_vel={w_a:.4f} rad/s")
    print(f"[CHECK 2 - Nested]  success={got_success} (expect True)")

    # 3. Drop cup_a off table
    drop_pose = torch.tensor([[0.5, 0.80, -0.10, 1.0, 0.0, 0.0, 0.0]], device=env.unwrapped.device)
    env.unwrapped.scene["cup_a"].write_root_pose_to_sim(drop_pose)
    env.unwrapped.scene["cup_a"].write_root_velocity_to_sim(torch.zeros((1, 6), device=env.unwrapped.device))
    env.unwrapped.sim.step()
    env.unwrapped.scene.update(dt=env.unwrapped.sim.get_physics_dt())
    got_drop = bool(cups_out_of_workspace(env.unwrapped)[0].item())
    print(f"[CHECK 3 - Dropped] dropped={got_drop} (expect True)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
