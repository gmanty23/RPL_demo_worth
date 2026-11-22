import pathlib
import yaml
import numpy as np
import trimesh
import coacd


def write_multi_hull_obj(parts, out_path: pathlib.Path):
    """Write a list of (vertices, faces) convex hulls as separate groups in one OBJ."""
    lines = ["# CoACD convex collision decomposition\n"]
    v_offset = 1
    for i, (verts, faces) in enumerate(parts):
        lines.append(f"o hull_{i}\n")
        for v in verts:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in faces:
            lines.append(f"f {int(f[0]) + v_offset} {int(f[1]) + v_offset} {int(f[2]) + v_offset}\n")
        v_offset += len(verts)
    out_path.write_text("".join(lines))


def main():
    cfg = yaml.safe_load(pathlib.Path("objects.yaml").read_text())
    for name, info in cfg["objects"].items():
        vis_path = pathlib.Path(info["visual_mesh"])
        out_path = pathlib.Path(info["collision_mesh"])
        out_path.parent.mkdir(parents=True, exist_ok=True)

        m = trimesh.load(vis_path, force="mesh", skip_materials=True)
        ctype = info["collision_type"]

        if ctype == "box":
            # Exact axis-aligned box matching the visual mesh bounds
            center = 0.5 * (m.bounds[0] + m.bounds[1])
            box_mesh = trimesh.creation.box(extents=m.extents)
            box_mesh.apply_translation(center)
            write_multi_hull_obj([(box_mesh.vertices, box_mesh.faces)], out_path)
            print(f"[BOX]   {name:10s} -> {out_path} (1 primitive box)")
        elif ctype == "coacd":
            max_hulls = int(info.get("max_convex_hulls", 12))
            print(f"[COACD] {name:10s} running (max_convex_hull={max_hulls})...")
            cmesh = coacd.Mesh(m.vertices, m.faces)
            parts = coacd.run_coacd(
                cmesh,
                threshold=0.05,
                max_convex_hull=max_hulls,
                preprocess_mode="auto",
            )
            write_multi_hull_obj(parts, out_path)
            print(f"[COACD] {name:10s} -> {out_path} ({len(parts)} convex pieces)")
        else:
            raise ValueError(f"Unknown collision_type: {ctype}")

        # Quick alignment check between visual and generated collision mesh
        coll_m = trimesh.load(out_path, force="mesh", skip_materials=True)
        diff = np.max(np.abs(m.bounds - coll_m.bounds))
        print(f"        max bounds diff vs visual: {diff*1000:.2f} mm")


if __name__ == "__main__":
    main()
