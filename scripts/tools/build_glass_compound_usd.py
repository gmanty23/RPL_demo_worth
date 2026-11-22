import math
import pathlib
import numpy as np
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pxr import Gf, Usd, UsdGeom, UsdPhysics

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_obj_vertices(obj_path: pathlib.Path) -> np.ndarray:
    verts = []
    with obj_path.open() as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts, dtype=np.float64)


def add_convex_mesh(stage, prim_path: str, pts: np.ndarray, tris: list[tuple[int, int, int]]):
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts])
    mesh.CreateFaceVertexCountsAttr([3] * len(tris))
    mesh.CreateFaceVertexIndicesAttr([idx for tri in tris for idx in tri])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_coll = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_coll.CreateApproximationAttr().Set("convexHull")


def build_glass_usd(name: str, mass: float = 0.15, wall_t: float = 0.0025, bot_t: float = 0.005, n_seg: int = 16):
    vis_path = REPO_ROOT / f"meshes/visual/{name}/{name}.obj"
    out_dir = REPO_ROOT / f"meshes/usd/physics/{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_usd = out_dir / f"{name}.usd"

    verts = load_obj_vertices(vis_path)
    z = verts[:, 2]
    h = float(z.max() - z.min())
    r = np.hypot(verts[:, 0], verts[:, 1])

    r_bot = float(np.percentile(r[(z >= 0.05 * h) & (z <= 0.15 * h)], 95))
    r_top = float(np.percentile(r[(z >= 0.85 * h) & (z <= 0.98 * h)], 95))
    rb_in = max(r_bot - wall_t, 0.002)
    rt_in = max(r_top - wall_t, 0.002)

    stage = Usd.Stage.CreateNew(str(out_usd))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr().Set(mass)

    # 1. Flat convex bottom disk
    angles = np.linspace(0.0, 2.0 * math.pi, n_seg, endpoint=False)
    c, s = np.cos(angles), np.sin(angles)
    bot_ring = np.column_stack([r_bot * c, r_bot * s, np.zeros(n_seg)])
    top_ring = np.column_stack([r_bot * c, r_bot * s, np.full(n_seg, bot_t)])
    disk_pts = np.vstack([bot_ring, top_ring, [[0.0, 0.0, 0.0], [0.0, 0.0, bot_t]]])
    c_bot, c_top = 2 * n_seg, 2 * n_seg + 1
    disk_tris = []
    for i in range(n_seg):
        j = (i + 1) % n_seg
        disk_tris.append((c_bot, j, i))
        disk_tris.append((c_top, n_seg + i, n_seg + j))
        disk_tris.append((i, j, n_seg + j))
        disk_tris.append((i, n_seg + j, n_seg + i))
    add_convex_mesh(stage, f"/{name}/bottom_disk", disk_pts, disk_tris)

    # 2. 16 thin convex wall wedges
    wedge_tris = [
        (0, 3, 2), (0, 2, 1),  # bottom
        (4, 5, 6), (4, 6, 7),  # top
        (0, 1, 5), (0, 5, 4),  # outer
        (2, 3, 7), (2, 7, 6),  # inner
        (0, 4, 7), (0, 7, 3),  # side 0
        (1, 2, 6), (1, 6, 5),  # side 1
    ]
    for i in range(n_seg):
        a0 = 2.0 * math.pi * i / n_seg
        a1 = 2.0 * math.pi * (i + 1) / n_seg
        c0, s0 = math.cos(a0), math.sin(a0)
        c1, s1 = math.cos(a1), math.sin(a1)
        pts = np.array([
            [r_bot * c0, r_bot * s0, 0.0],
            [r_bot * c1, r_bot * s1, 0.0],
            [rb_in * c1, rb_in * s1, 0.0],
            [rb_in * c0, rb_in * s0, 0.0],
            [r_top * c0, r_top * s0, h],
            [r_top * c1, r_top * s1, h],
            [rt_in * c1, rt_in * s1, h],
            [rt_in * c0, rt_in * s0, h],
        ])
        add_convex_mesh(stage, f"/{name}/wall_{i:02d}", pts, wedge_tris)

    stage.GetRootLayer().Save()
    print(f"[COMPOUND USD] Saved {out_usd} (17 convex hulls, H={h*100:.2f} cm, R_bot={r_bot*100:.2f} cm, R_top={r_top*100:.2f} cm)")


if __name__ == "__main__":
    build_glass_usd("glass1", mass=0.15)
    build_glass_usd("glass2", mass=0.15)
    simulation_app.close()
