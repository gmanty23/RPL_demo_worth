import pathlib
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pxr import Gf, Usd, UsdGeom, UsdPhysics

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def parse_obj_groups(obj_path: pathlib.Path):
    verts = []
    groups = {}
    cur = "hull_0"
    with obj_path.open() as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith(("o ", "g ")):
                cur = line.split()[1].replace("-", "_").replace(".", "_")
                groups.setdefault(cur, [])
            elif line.startswith("f "):
                face = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                groups.setdefault(cur, []).append(face)
    return verts, {k: v for k, v in groups.items() if v}


def build_compound_usd(name: str, mass: float):
    obj_path = REPO_ROOT / f"meshes/collision/{name}_coll.obj"
    out_dir = REPO_ROOT / f"meshes/usd/physics/{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_usd = out_dir / f"{name}.usd"

    verts, groups = parse_obj_groups(obj_path)
    if name == "mug_rack":
        # Flatten underside scan noise so the collision base sits 100% planar on the table
        verts = [(x, y, 0.0 if z < 0.006 else z) for (x, y, z) in verts]

    stage = Usd.Stage.CreateNew(str(out_usd))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr().Set(float(mass))
    if name == "mug_rack":
        mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.04))

    for i, (gname, faces) in enumerate(groups.items()):
        # Remap vertices used by this hull
        used = sorted({idx for face in faces for idx in face})
        remap = {old: new for new, old in enumerate(used)}
        local_pts = [Gf.Vec3f(*verts[idx]) for idx in used]
        counts = [len(face) for face in faces]
        indices = [remap[idx] for face in faces for idx in face]

        mesh = UsdGeom.Mesh.Define(stage, f"/{name}/hull_{i:02d}")
        mesh.CreatePointsAttr(local_pts)
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")

    stage.GetRootLayer().Save()
    print(f"Built compound Physics USD: {out_usd} ({len(groups)} hulls, mass={mass} kg)")


if __name__ == "__main__":
    build_compound_usd("mug", 0.20)
    build_compound_usd("mug_rack", 1.50)
    simulation_app.close()
