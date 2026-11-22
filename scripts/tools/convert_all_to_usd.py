import pathlib
import subprocess
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ISAACLAB_DIR = pathlib.Path.home() / "repos" / "IsaacLab"
ISAACLAB_SH = ISAACLAB_DIR / "isaaclab.sh"
CONVERT_SCRIPT = ISAACLAB_DIR / "scripts" / "tools" / "convert_mesh.py"


def run_convert(in_path: pathlib.Path, out_path: pathlib.Path, approx: str, mass: float):
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
    cfg = yaml.safe_load((REPO_ROOT / "objects.yaml").read_text())
    for name, info in cfg["objects"].items():
        mass = float(info["mass"])
        vis_in = (REPO_ROOT / info["visual_mesh"]).resolve()
        coll_in = (REPO_ROOT / info["collision_mesh"]).resolve()

        # Put each object in its own subfolder so Props/instanceable_meshes.usd is never overwritten
        render_out = (REPO_ROOT / "meshes" / "usd" / "render" / name / f"{name}.usd").resolve()
        physics_out = (REPO_ROOT / "meshes" / "usd" / "physics" / name / f"{name}.usd").resolve()

        # 1. Render USD (visual mesh + textures, no collision)
        run_convert(vis_in, render_out, approx="none", mass=mass)

        # 2. Physics USD (collision mesh + mass)
        phys_approx = "boundingCube" if info["collision_type"] == "box" else "convexDecomposition"
        run_convert(coll_in, physics_out, approx=phys_approx, mass=mass)

    print("\nAll 12 USD files converted successfully.")


if __name__ == "__main__":
    main()
