import pathlib
import yaml
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def strip_rigid_body_and_mass(prim):
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            p.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if p.HasAPI(UsdPhysics.MassAPI):
            p.RemoveAPI(UsdPhysics.MassAPI)


def strip_all_collisions(prim):
    for p in Usd.PrimRange(prim):
        if p.IsInstanceable():
            p.SetInstanceable(False)
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            p.RemoveAPI(UsdPhysics.CollisionAPI)
        if p.HasAPI(UsdPhysics.MeshCollisionAPI):
            p.RemoveAPI(UsdPhysics.MeshCollisionAPI)


def build_twin(name: str, mass: float):
    phys_usd = (REPO_ROOT / f"meshes/usd/physics/{name}/{name}.usd").resolve()
    rend_usd = (REPO_ROOT / f"meshes/usd/render/{name}/{name}.usd").resolve()
    out_dir = REPO_ROOT / f"meshes/usd/render_sim/{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_usd = out_dir / f"{name}.usd"

    stage = Usd.Stage.CreateNew(str(out_usd))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr().Set(float(mass))

    # 1. Visuals from Render USD (strip duplicate RigidBody + strip all collisions)
    vis = UsdGeom.Xform.Define(stage, f"/{name}/visual")
    vis.GetPrim().GetReferences().AddReference(str(rend_usd))
    strip_rigid_body_and_mass(vis.GetPrim())
    strip_all_collisions(vis.GetPrim())

    # 2. Collision from Physics USD (strip duplicate RigidBody, hide from cameras)
    coll = UsdGeom.Xform.Define(stage, f"/{name}/collision")
    coll.GetPrim().GetReferences().AddReference(str(phys_usd))
    strip_rigid_body_and_mass(coll.GetPrim())
    UsdGeom.Imageable(coll.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.guide)

    stage.GetRootLayer().Save()
    print(f"[RENDER TWIN] Saved {out_usd.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    cfg = yaml.safe_load((REPO_ROOT / "objects.yaml").read_text())
    for name, info in cfg["objects"].items():
        build_twin(name, info["mass"])
    simulation_app.close()
