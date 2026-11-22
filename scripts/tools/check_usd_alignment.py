import pathlib
import yaml
import numpy as np
from isaaclab.app import AppLauncher

# Launch minimal headless Kit app so pxr / UsdPhysics modules are available
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def get_world_bbox(usd_path: pathlib.Path):
    stage = Usd.Stage.Open(str(usd_path))
    root_prim = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    bbox = bbox_cache.ComputeWorldBound(root_prim).ComputeAlignedRange()
    bmin = np.array(bbox.GetMin())
    bmax = np.array(bbox.GetMax())

    masses = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.MassAPI):
            m = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            if m is not None:
                masses.append(round(float(m), 4))
    return bmin, bmax, masses


def main():
    cfg = yaml.safe_load((REPO_ROOT / "objects.yaml").read_text())
    print(f"{'Object':10s} | {'Render Extents (cm)':22s} | {'Physics Extents (cm)':22s} | {'Max Diff':9s} | {'Mass (kg)'}")
    print("-" * 88)
    for name in cfg["objects"]:
        r_path = REPO_ROOT / "meshes" / "usd" / "render" / name / f"{name}.usd"
        p_path = REPO_ROOT / "meshes" / "usd" / "physics" / name / f"{name}.usd"

        r_min, r_max, _ = get_world_bbox(r_path)
        p_min, p_max, p_mass = get_world_bbox(p_path)

        r_ext = (r_max - r_min) * 100.0
        p_ext = (p_max - p_min) * 100.0
        max_diff_mm = max(np.max(np.abs(r_min - p_min)), np.max(np.abs(r_max - p_max))) * 1000.0

        print(
            f"{name:10s} | "
            f"[{r_ext[0]:5.2f}, {r_ext[1]:5.2f}, {r_ext[2]:5.2f}] | "
            f"[{p_ext[0]:5.2f}, {p_ext[1]:5.2f}, {p_ext[2]:5.2f}] | "
            f"{max_diff_mm:6.2f} mm | {p_mass}"
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
