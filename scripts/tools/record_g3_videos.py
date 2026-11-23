import argparse
import os
import pathlib
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase G3: Record 20s 512x512 table_cam and wrist_cam videos.")
parser.add_argument("--task", type=str, required=True, help="Gym task ID (e.g., RPL-BlockInBowl-Render-v0)")
parser.add_argument("--out_dir", type=str, default="outputs/g3_media", help="Output directory")
parser.add_argument("--res", type=int, default=512, help="Square camera resolution for recording")
parser.add_argument("--fps", type=int, default=30, help="Video FPS")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import math
from isaaclab.utils.math import quat_apply, quat_mul
from isaaclab_tasks.utils import parse_env_cfg
import rpl_demo_worth.tasks  # noqa: F401


def save_mp4(frames: list[np.ndarray], path: pathlib.Path, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8) as w:
            for f in frames:
                w.append_data(f)
    except Exception:
        import cv2
        h, w_px = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_px, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()


def save_png(frame: np.ndarray, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        imageio.imwrite(str(path), frame)
    except Exception:
        import cv2
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def get_task_info(env, task_name: str):
    keys = set(env.unwrapped.scene.keys())
    if "BlockInBowl" in task_name:
        obj = "block" if "block" in keys else "box"
        return "block_in_bowl", obj, "bowl", torch.tensor([0.0, 0.0, 0.06]), False
    elif "MugOnRack" in task_name:
        return "mug_on_rack", "mug", "mug_rack", torch.tensor([0.095, 0.0, 0.16]), True
    elif "CupInCup" in task_name:
        return "cup_in_cup", "cup_a", "cup_b", torch.tensor([0.0, 0.0, 0.06]), False
    raise ValueError(f"Unknown task: {task_name}")


def to_rgb_uint8(cam_tensor: torch.Tensor) -> np.ndarray:
    img = cam_tensor[0, ..., :3].detach().cpu().numpy()
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
    return img


def main():
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
    # Override camera resolution for crisp presentation renders
    for cam_name in ("table_cam", "wrist_cam"):
        if hasattr(env_cfg.scene, cam_name):
            cam_cfg = getattr(env_cfg.scene, cam_name)
            cam_cfg.width = args_cli.res
            cam_cfg.height = args_cli.res

    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device
    zero_act = torch.zeros((1, env.action_space.shape[-1]), device=device)

    short_name, obj_name, target_name, goal_offset, hold_at_end = get_task_info(env, args_cli.task)
    goal_offset = goal_offset.to(device)

    env.reset()
    # Step a few frames to let scene settle and warm up renderer
    for _ in range(10):
        env.step(zero_act)

    start_pos = env.unwrapped.scene[obj_name].data.root_pos_w[0].clone()
    target_pos = env.unwrapped.scene[target_name].data.root_pos_w[0].clone()
    target_quat = env.unwrapped.scene[target_name].data.root_quat_w[0].clone()

    pose = torch.zeros((1, 7), device=device)
    pose[:, 3] = 1.0  # identity quaternion

    if hold_at_end:
        # MugOnRack: thread the mug's -X handle loop directly onto the upper rack peg
        peg_inner = torch.tensor([0.0299, 0.0404, 0.2310], device=device)
        peg_tip   = torch.tensor([0.0485, 0.0727, 0.2359], device=device)
        peg_mid   = 0.65 * peg_inner + 0.35 * peg_tip
        peg_dir   = peg_tip - peg_inner
        peg_dir   = peg_dir / torch.linalg.norm(peg_dir)

        # Rotate mug so its -X handle points inward along -peg_dir
        # Add +pi/2 so the handle hole (mug Y-axis) aligns with peg_dir
        peg_yaw = math.atan2(float(peg_dir[1].item()), float(peg_dir[0].item())) + math.pi / 2.0
        local_mug_quat = torch.tensor([math.cos(0.5 * peg_yaw), 0.0, 0.0, math.sin(0.5 * peg_yaw)], device=device)
        handle_local = torch.tensor([0.0421, 0.009, 0.0712], device=device)
        handle_in_rack = quat_apply(local_mug_quat.unsqueeze(0), handle_local.unsqueeze(0))[0]

        mug_goal_in_rack = peg_mid - handle_in_rack
        mug_pre_in_rack  = mug_goal_in_rack + 0.10 * peg_dir

        goal_pos = target_pos + quat_apply(target_quat.unsqueeze(0), mug_goal_in_rack.unsqueeze(0))[0]
        p2       = target_pos + quat_apply(target_quat.unsqueeze(0), mug_pre_in_rack.unsqueeze(0))[0]
        p1       = start_pos.clone(); p1[2] = 0.36
        pose[0, 3:] = quat_mul(target_quat.unsqueeze(0), local_mug_quat.unsqueeze(0))[0]
    else:
        goal_pos = target_pos + goal_offset
        lift_z = max(float(start_pos[2].item()), float(goal_pos[2].item())) + 0.14
        p1 = start_pos.clone(); p1[2] = lift_z
        p2 = goal_pos.clone();  p2[2] = lift_z
    zero_vel = torch.zeros((1, 6), device=device)

    # 600 frames @ 30 fps = 20.0 seconds
    total_frames = 600
    table_frames, wrist_frames = [], []

    print(f"[G3] Recording {total_frames} frames ({total_frames / args_cli.fps:.1f}s) for {args_cli.task}...")
    for t in range(total_frames):
        p2_high = p2.clone(); p2_high[2] = 0.36 if hold_at_end else float(p2[2].item())
        if 60 <= t < 140:
            # Lift straight up (above rack)
            alpha = (t - 60) / 80.0
            pos = (1.0 - alpha) * start_pos + alpha * p1
            pose[0, :3] = pos
            env.unwrapped.scene[obj_name].write_root_pose_to_sim(pose)
            env.unwrapped.scene[obj_name].write_root_velocity_to_sim(zero_vel)
        elif 140 <= t < 240:
            # Move horizontally above pre-approach point
            alpha = (t - 140) / 100.0
            pos = (1.0 - alpha) * p1 + alpha * p2_high
            pose[0, :3] = pos
            env.unwrapped.scene[obj_name].write_root_pose_to_sim(pose)
            env.unwrapped.scene[obj_name].write_root_velocity_to_sim(zero_vel)
        elif 240 <= t < 320:
            # Descend cleanly to pre-approach point (aligned with peg tip)
            alpha = (t - 240) / 80.0
            pos = (1.0 - alpha) * p2_high + alpha * p2
            pose[0, :3] = pos
            env.unwrapped.scene[obj_name].write_root_pose_to_sim(pose)
            env.unwrapped.scene[obj_name].write_root_velocity_to_sim(zero_vel)
        elif 320 <= t < 420:
            # Slide straight onto peg / lower into goal pose
            alpha = (t - 320) / 100.0
            pos = (1.0 - alpha) * p2 + alpha * goal_pos
            pose[0, :3] = pos
            env.unwrapped.scene[obj_name].write_root_pose_to_sim(pose)
            env.unwrapped.scene[obj_name].write_root_velocity_to_sim(zero_vel)
        # t >= 420: release mug to hang under gravity

        env.step(zero_act)

        t_rgb = to_rgb_uint8(env.unwrapped.scene["table_cam"].data.output["rgb"])
        w_rgb = to_rgb_uint8(env.unwrapped.scene["wrist_cam"].data.output["rgb"])
        table_frames.append(t_rgb)
        wrist_frames.append(w_rgb)

    out_dir = pathlib.Path(args_cli.out_dir)
    save_mp4(table_frames, out_dir / f"{short_name}_table_cam.mp4", args_cli.fps)
    save_mp4(wrist_frames, out_dir / f"{short_name}_wrist_cam.mp4", args_cli.fps)
    save_png(table_frames[0], out_dir / f"{short_name}_still_table.png")
    print(f"[G3] Saved videos & still to {out_dir}/")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
