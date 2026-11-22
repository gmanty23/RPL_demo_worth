import argparse
import pathlib
import yaml
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase E visual comparison, drop test, and nesting test.")
parser.add_argument("--mode", choices=["compare", "drop", "nest", "hang"], required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationCfg, SimulationContext

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def setup_scene(sim: SimulationContext):
    # Ground plane (acts as the tabletop at Z = 0.0)
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # Dome light so textures and shapes are clearly visible
    cfg_light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))
    cfg_light.func("/World/Light", cfg_light)


def run_compare(sim: SimulationContext, names: list[str]):
    """Spawn Render USDs in front row (Y=0.0) and Physics USDs in back row (Y=0.25)."""
    spacing = 0.22
    for i, name in enumerate(names):
        x = (i - (len(names) - 1) / 2.0) * spacing
        r_usd = str((REPO_ROOT / "meshes/usd/render" / name / f"{name}.usd").resolve())
        p_usd = str((REPO_ROOT / "meshes/usd/physics" / name / f"{name}.usd").resolve())

        sim_utils.UsdFileCfg(usd_path=r_usd).func(f"/World/Render_{name}", sim_utils.UsdFileCfg(usd_path=r_usd), translation=(x, 0.0, 0.0))
        sim_utils.UsdFileCfg(usd_path=p_usd).func(f"/World/Physics_{name}", sim_utils.UsdFileCfg(usd_path=p_usd), translation=(x, 0.25, 0.0))

    sim.set_camera_view(eye=[0.0, -0.85, 0.45], target=[0.0, 0.12, 0.08])
    print("\n[COMPARE MODE] Front row = Render USDs | Back row = Physics USDs.")
    print("Inspect size, origin, orientation, and textures. Close the window or press Ctrl+C when done.\n")
    while simulation_app.is_running():
        sim.render()


def run_drop(sim: SimulationContext, names: list[str]):
    """Drop each Physics USD from 5 cm above the tabletop and check settling stability."""
    spacing = 0.25
    objects = {}
    init_xy = {}

    for i, name in enumerate(names):
        x = (i - (len(names) - 1) / 2.0) * spacing
        p_usd = str((REPO_ROOT / "meshes/usd/physics" / name / f"{name}.usd").resolve())
        cfg = RigidObjectCfg(
            prim_path=f"/World/{name}",
            spawn=sim_utils.UsdFileCfg(
                    usd_path=p_usd,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=2,
                        linear_damping=0.5,
                        angular_damping=0.5,
                        sleep_threshold=0.01,
                        stabilization_threshold=0.01,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002,
                        rest_offset=0.0,
                    ),
                ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x, 0.0, 0.05)),
        )
        objects[name] = RigidObject(cfg)
        init_xy[name] = torch.tensor([x, 0.0], device=sim.device)

    sim.set_camera_view(eye=[0.0, -0.95, 0.40], target=[0.0, 0.0, 0.08])
    sim.reset()
    for obj in objects.values():
        obj.reset()

    sim_dt = sim.get_physics_dt()
    max_lin = {n: 0.0 for n in names}
    max_ang = {n: 0.0 for n in names}

    print("\n[DROP TEST] Dropping all 6 Physics USDs from Z = 5.0 cm (300 steps)...")
    for step in range(600):
        sim.step()
        for name, obj in objects.items():
            obj.update(sim_dt)
            if step >= 480:  # Measure residual jitter during the final 1 second
                lin = float(torch.linalg.norm(obj.data.root_lin_vel_w[0]).item())
                ang = float(torch.linalg.norm(obj.data.root_ang_vel_w[0]).item())
                max_lin[name] = max(max_lin[name], lin)
                max_ang[name] = max(max_ang[name], ang)

    print(f"\n{'Object':10s} | {'Final Z (cm)':12s} | {'XY Drift (cm)':13s} | {'Max Lin (cm/s)':14s} | {'Max Ang (rad/s)':15s} | {'Status'}")
    print("-" * 90)
    for name, obj in objects.items():
        pos = obj.data.root_pos_w[0]
        z_cm = float(pos[2].item()) * 100.0
        drift_cm = float(torch.linalg.norm(pos[:2] - init_xy[name]).item()) * 100.0
        lin_cms = max_lin[name] * 100.0
        ang_rs = max_ang[name]
        ok = (-0.5 <= z_cm <= 2.0) and (lin_cms < 1.0) and (ang_rs < 0.2)
        status = "PASS" if ok else "CHECK"
        print(f"{name:10s} | {z_cm:12.2f} | {drift_cm:13.2f} | {lin_cms:14.3f} | {ang_rs:15.3f} | {status}")

    if not args_cli.headless:
        print("\nKeep inspecting in the GUI, or press Ctrl+C / close the window to exit.")
        while simulation_app.is_running():
            sim.step()


def run_nest(sim: SimulationContext):
    """Test (1) box dropping into bowl and (2) glass1 nesting into glass2."""
    def make_obj(name: str, pos: tuple[float, float, float]):
        p_usd = str((REPO_ROOT / "meshes/usd/physics" / name / f"{name}.usd").resolve())
        return RigidObject(
            RigidObjectCfg(
                prim_path=f"/World/{name}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=p_usd,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=2,
                        linear_damping=0.5,
                        angular_damping=0.5,
                        sleep_threshold=0.01,
                        stabilization_threshold=0.01,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002,
                        rest_offset=0.0,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            )
        )

    # Left pair (X = -0.15): bowl on table, box dropped from above into the opening
    bowl = make_obj("bowl", (-0.15, 0.0, 0.002))
    box = make_obj("box", (-0.15, 0.0, 0.12))

    # Right pair (X = +0.15): glass2 on table, glass1 dropped from above to nest inside
    glass2 = make_obj("glass2", (0.15, 0.0, 0.002))
    glass1 = make_obj("glass1", (0.15, 0.0, 0.14))

    objs = {"bowl": bowl, "box": box, "glass2": glass2, "glass1": glass1}

    sim.set_camera_view(eye=[0.0, -0.55, 0.30], target=[0.0, 0.0, 0.06])
    sim.reset()
    for o in objs.values():
        o.reset()

    sim_dt = sim.get_physics_dt()
    print("\n[NESTING TEST] Running 300 steps: (left) box -> bowl | (right) glass1 -> glass2 ...")
    for _ in range(600):
        sim.step()
        for o in objs.values():
            o.update(sim_dt)

    bowl_z = float(bowl.data.root_pos_w[0, 2].item()) * 100.0
    box_z = float(box.data.root_pos_w[0, 2].item()) * 100.0
    g2_z = float(glass2.data.root_pos_w[0, 2].item()) * 100.0
    g1_z = float(glass1.data.root_pos_w[0, 2].item()) * 100.0

    print("\n--- Nesting Results ---")
    print(f"Bowl Z = {bowl_z:.2f} cm | Box Z = {box_z:.2f} cm  (bowl rim ~5.5 cm -> box inside if < 4.0 cm)")
    print(f"Glass2 Z = {g2_z:.2f} cm | Glass1 Z = {g1_z:.2f} cm (glass2 rim ~9.8 cm -> nested if < 8.0 cm)")

    if not args_cli.headless:
        print("\nInspect the settled objects in the GUI, then close the window or press Ctrl+C to exit.")
        while simulation_app.is_running():
            sim.step()



def run_hang(sim: SimulationContext):
    """Spawn mug_rack and mug Physics USDs close up for interactive hanging test."""
    def make_obj(name: str, pos: tuple[float, float, float]):
        p_usd = str((REPO_ROOT / "meshes/usd/physics" / name / f"{name}.usd").resolve())
        return RigidObject(
            RigidObjectCfg(
                prim_path=f"/World/{name}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=p_usd,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=2,
                        linear_damping=0.5,
                        angular_damping=0.5,
                        sleep_threshold=0.01,
                        stabilization_threshold=0.01,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002,
                        rest_offset=0.0,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            )
        )

    rack = make_obj("mug_rack", (0.0, 0.0, 0.005))
    mug = make_obj("mug", (0.18, 0.0, 0.005))

    sim.set_camera_view(eye=[0.08, -0.55, 0.28], target=[0.05, 0.0, 0.14])
    sim.reset()
    rack.reset()
    mug.reset()

    print("\n[HANG TEST] Hold Shift + Left-Click on the mug in the viewport to drag and hang it on a peg.")
    print("Check that: (1) the mug handle hole is open, (2) the rack pegs are separate, and (3) it hangs cleanly.\n")
    while simulation_app.is_running():
        sim.step()


def main():
    cfg = yaml.safe_load((REPO_ROOT / "objects.yaml").read_text())
    names = list(cfg["objects"].keys())

    sim = SimulationContext(SimulationCfg(
        dt=1.0 / 120.0,
        device="cpu" if args_cli.mode == "hang" else "cuda:0",
        physx=sim_utils.PhysxCfg(
            enable_stabilization=True,
            bounce_threshold_velocity=0.2,
        ),
    ))
    setup_scene(sim)

    if args_cli.mode == "compare":
        run_compare(sim, names)
    elif args_cli.mode == "drop":
        run_drop(sim, names)
    elif args_cli.mode == "nest":
        run_nest(sim)
    elif args_cli.mode == "hang":
        run_hang(sim)


if __name__ == "__main__":
    main()
    simulation_app.close()
