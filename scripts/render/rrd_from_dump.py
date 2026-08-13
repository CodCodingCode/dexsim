"""Turn a scene dump (npz) into a Rerun ``.rrd`` you can open in the viewer.

Runs in the SEPARATE rerun venv (.venv-rerun): Isaac Sim pins numpy<2 while
rerun-sdk>=0.36 needs numpy>=2, so this stage deliberately imports NO Isaac /
dexsim code -- just numpy and rerun. The dump is produced by the warm server's
``rerun`` job (see render.py / render_server.py).

  .venv-rerun/bin/python scripts/render/rrd_from_dump.py \
      --dump logs/piano_scene.npz --out results/piano_scene.rrd
"""
from __future__ import annotations

import argparse

import numpy as np
import rerun as rr


def smooth_normals(vertices, triangles):
    """Area-weighted vertex normals so Rerun shades curved hand meshes cleanly."""
    vertices = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(triangles, dtype=np.int64)
    normals = np.zeros_like(vertices)
    face = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                    vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
    return normals


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", required=True, help="npz written by the rerun job")
    p.add_argument("--out", required=True, help="destination .rrd")
    p.add_argument("--app_id", default="dexsim_piano")
    a = p.parse_args()

    d = np.load(a.dump, allow_pickle=False)
    links = [str(s) for s in d["__links__"]]
    n_frames = int(d["__meta__"][0])

    # explicit stream (not the global one) so the final flush is guaranteed --
    # a half-written .rrd opens as an empty viewer and looks like a scene bug.
    rec = rr.RecordingStream(a.app_id)
    rec.save(a.out)
    # Isaac/USD is right-handed Z-up; without this Rerun assumes its own default
    # and the whole rig shows up lying on its side.
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    for key in links:                            # geometry is rigid -> log once
        for part in range(int(d[f"{key}|parts"])):
            visual = f"{key}/visual_{part}"
            verts, tris = d[f"{visual}|verts"], d[f"{visual}|tris"]
            rec.log(visual, rr.Mesh3D(vertex_positions=verts,
                                      triangle_indices=tris,
                                      vertex_normals=smooth_normals(verts, tris),
                                      albedo_factor=d[f"{visual}|color"].tolist()),
                    static=True)

    for i in range(n_frames):
        rec.set_time("frame", sequence=i)
        for key in links:
            pos = d[f"{key}|pos"][i]
            w, x, y, z = (float(v) for v in d[f"{key}|quat"][i])   # Isaac: wxyz
            rec.log(key, rr.Transform3D(translation=pos,
                                        quaternion=rr.Quaternion(xyzw=[x, y, z, w])))

    rec.flush()
    tris = sum(len(d[f"{key}/visual_{part}|tris"])
               for key in links for part in range(int(d[f"{key}|parts"])))
    print(f"[rrd] {a.out}  links={len(links)} frames={n_frames} tris={tris}")


if __name__ == "__main__":
    main()
