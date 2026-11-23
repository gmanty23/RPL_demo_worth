import argparse
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase G1: Automated reset, cheat/null policy, and stability tests.")
parser.add_argument("--task", type=str, required=True, help="Gym task ID (e.g., RPL-BlockInBowl-Render-v0)")
parser.add_argument("--num_resets", type=int, default=50, help="Number of resets for overlap, null, and cheat tests")
parser.add_argument("--num_random_eps", type=int, default=10, help="Number of random-action stability episodes")
parser.add_argument("--random_steps", type=int, default=50, help="Steps per random-action episode")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import rpl_demo_worth.tasks  # noqa: F401
from rpl_demo_worth.tasks.manager_based.mug_on_rack.mdp.terminations import mug_on_rack_success


def get_task_objects(env, task_name: str):
    keys = set(env.unwrapped.scene.keys())
    if "BlockInBowl" in task_name:
        obj = "block" if "block" in keys else "box"
        return obj, "bowl", 0.08
    elif "MugOnRack" in task_name:
        return "mug", "mug_rack", 0.12
    elif "CupInCup" in task_name:
        return "cup_a", "cup_b", 0.08
    raise ValueError(f"Unknown task: {task_name}")


def run_cheat_policy(env, task_name: str, obj_name: str, target_name: str, zero_act: torch.Tensor) -> bool:
    device = env.unwrapped.device
    target_pos = env.unwrapped.scene[target_name].data.root_pos_w.clone()
    pose = torch.zeros((1, 7), device=device)
    pose[:, 3] = 1.0  # identity quat

    if "BlockInBowl" in task_name:
        pose[:, :2] = target_pos[:, :2]
        pose[:, 2] = target_pos[:, 2] + 0.06
        env.unwrapped.scene[obj_name].write_root_pose_to_sim(pose)
        env.unwrapped.scene[obj_name].write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
        for _ in range(60):
            _, _, term, _, _ = env.step(zero_act)
            if bool(term[0].item()) and bool(env.unwrapped.termination_manager.get_term("success")[0].item()):
                return True
        return False

    elif "CupInCup" in task_name:
        pose[:, :2] = target_pos[:, :2]
        pose[:, 2] = target_pos[:, 2] + 0.06
        env.unwrapped.scene[obj_name].write_root_pose_to_sim(pose)
        env.unwrapped.scene[obj_name].write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
        for _ in range(100):
            _, _, term, _, _ = env.step(zero_act)
            if bool(term[0].item()) and bool(env.unwrapped.termination_manager.get_term("success")[0].item()):
                return True
        p_a = env.unwrapped.scene[obj_name].data.root_pos_w[0]
        p_b = env.unwrapped.scene[target_name].data.root_pos_w[0]
        xy = float(torch.linalg.norm(p_a[:2] - p_b[:2]).item()) * 1000.0
        zr = float((p_a[2] - p_b[2]).item()) * 1000.0
        v = float(torch.linalg.norm(env.unwrapped.scene[obj_name].data.root_lin_vel_w[0]).item())
        w = float(torch.linalg.norm(env.unwrapped.scene[obj_name].data.root_ang_vel_w[0]).item())
        print(f"    [cheat miss] xy={xy:.1f}mm, z_rel={zr:.1f}mm, v={v:.3f}, w={w:.3f}")
        return False

    elif "MugOnRack" in task_name:
        # Place handle center in peg zone (kinematic hold check over 5 frames)
        pose[:, 0] = target_pos[:, 0] + 0.095
        pose[:, 1] = target_pos[:, 1]
        pose[:, 2] = target_pos[:, 2] + 0.16
        env.unwrapped.scene[obj_name].write_root_pose_to_sim(pose)
        env.unwrapped.scene[obj_name].write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
        env.unwrapped.scene.update(dt=0.0)
        for _ in range(6):
            if bool(mug_on_rack_success(env.unwrapped)[0].item()):
                return True
        return False

    return False


def main():
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device
    zero_act = torch.zeros((1, env.action_space.shape[-1]), device=device)

    obj_name, target_name, min_sep = get_task_objects(env, args_cli.task)

    overlaps = 0
    null_successes = 0
    cheat_successes = 0

    print(f"\n=== Running G1 Tests on {args_cli.task} ({args_cli.num_resets} resets) ===")
    for i in range(args_cli.num_resets):
        env.reset()

        # 1. Check initial spawn separation
        p_obj = env.unwrapped.scene[obj_name].data.root_pos_w[0]
        p_tgt = env.unwrapped.scene[target_name].data.root_pos_w[0]
        if float(torch.linalg.norm(p_obj[:2] - p_tgt[:2]).item()) < min_sep:
            overlaps += 1

        # 2. Null policy (15 steps doing nothing)
        null_hit = False
        for _ in range(15):
            _, _, term, _, _ = env.step(zero_act)
            if bool(term[0].item()) and bool(env.unwrapped.termination_manager.get_term("success")[0].item()):
                null_hit = True
                break
        if null_hit:
            null_successes += 1

        # 3. Cheat policy
        if run_cheat_policy(env, args_cli.task, obj_name, target_name, zero_act):
            cheat_successes += 1

    # 4. Random-action stability test
    explosions = 0
    fell_through = 0
    for ep in range(args_cli.num_random_eps):
        env.reset()
        for _ in range(args_cli.random_steps):
            rand_act = torch.empty((1, env.action_space.shape[-1]), device=device).uniform_(-1.0, 1.0)
            _, _, term, trunc, _ = env.step(rand_act)

            for name in (obj_name, target_name):
                pos = env.unwrapped.scene[name].data.root_pos_w[0] - env.unwrapped.scene.env_origins[0]
                vel = env.unwrapped.scene[name].data.root_lin_vel_w[0]
                if torch.isnan(pos).any() or torch.isinf(pos).any() or float(torch.linalg.norm(vel).item()) > 10.0:
                    explosions += 1
                    break
                on_table_xy = (0.15 < pos[0] < 0.85) and (-0.45 < pos[1] < 0.45)
                if on_table_xy and float(pos[2].item()) < -0.05:
                    fell_through += 1
                    break

            if bool(term[0].item()) or bool(trunc[0].item()):
                break

    print(f"--- G1 SUMMARY: {args_cli.task} ---")
    print(f"  Spawn overlaps : {overlaps}/{args_cli.num_resets} (expect 0)")
    print(f"  Null policy    : {null_successes}/{args_cli.num_resets} ({100.0*null_successes/args_cli.num_resets:.1f}%, expect ~0%)")
    print(f"  Cheat policy   : {cheat_successes}/{args_cli.num_resets} ({100.0*cheat_successes/args_cli.num_resets:.1f}%, expect ~100%)")
    print(f"  Random eps ({args_cli.num_random_eps}): explosions={explosions}, fell_through_table={fell_through} (expect 0, 0)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
