import os
import pathlib
import shutil
import subprocess
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ISAACLAB_DIR = pathlib.Path(
    os.environ.get("ISAACLAB_PATH", str(pathlib.Path.home() / "repos" / "IsaacLab"))
).expanduser().resolve()
ISAACLAB_SH = ISAACLAB_DIR / "isaaclab.sh"
CONVERT_SCRIPT = ISAACLAB_DIR / "scripts" / "tools" / "convert_mesh.py"
COMPOUND_GLASS_SCRIPT = REPO_ROOT / "scripts" / "tools" / "build_glass_compound_usd.py"


def run_convert(in_path: pathlib.Path, out_path: pathlib.Path, approx: str, mass: float):
    if out_path.parent.exists():
        shutil.rmtree(out_path.parent)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(ISAACLAB_SH),
        "-p",
        str(CONVERT_SCRIPT),
        str(in_path),
        str(out_path),
        "--make-instanceable",
        "--collision-approximation",
        approx,
        "--mass",
        str(mass),
    ]
    print(f"\n>>> Converting {in_path.name} -> {out_path.relative_to(REPO_ROOT)} (approx={approx}, mass={mass} kg)")
    subprocess.run(cmd, check=True)


def main():
    if not ISAACLAB_SH.exists():
        raise FileNotFoundError(
            f"Could not find isaaclab.sh at {ISAACLAB_SH}. "
            "Set ISAACLAB_PATH=/path/to/IsaacLab and try again."
        )

    cfg = yaml.safe_load((REPO_ROOT / "objects.yaml").read_text())
    needs_compound_glasses = False

    for name, info in cfg["objects"].items():
        mass = float(info["mass"])
        vis_in = (REPO_ROOT / info["visual_mesh"]).resolve()
        coll_in = (REPO_ROOT / info["collision_mesh"]).resolve()

        render_out = (REPO_ROOT / "meshes" / "usd" / "render" / name / f"{name}.usd").resolve()
        physics_out = (REPO_ROOT / "meshes" / "usd" / "physics" / name / f"{name}.usd").resolve()

        # 1. Render USD (visual mesh + textures, no collision)
        run_convert(vis_in, render_out, approx="none", mass=mass)

        # 2. Physics USD
        phys_approx = info["usd_collision_approx"]
        if phys_approx == "compoundConvexHull":
            needs_compound_glasses = True
        else:
            run_convert(coll_in, physics_out, approx=phys_approx, mass=mass)

    if needs_compound_glasses:
        print("\n>>> Building compound convex-hull Physics USDs for glass1 & glass2...")
        subprocess.run([str(ISAACLAB_SH), "-p", str(COMPOUND_GLASS_SCRIPT)], check=True)

    print("\nAll 12 USD files converted successfully.")


if __name__ == "__main__":
    main()
