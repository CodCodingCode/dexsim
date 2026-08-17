"""Turn a scene dump (npz) into a Rerun ``.rrd`` you can open in the viewer.

Runs in the SEPARATE rerun venv (.venv-rerun): Isaac Sim pins numpy<2 while
rerun-sdk>=0.36 needs numpy>=2, so this stage deliberately imports NO Isaac /
dexsim code -- just numpy and rerun. The dump is produced by the warm server's
``rerun`` job (see render.py / render_server.py).

  .venv-rerun/bin/python scripts/render/rrd_from_dump.py \
      --dump logs/piano_scene.npz --out results/piano_scene.rrd

The recording carries a baked-in blueprint (3D view, collapsed side panels) and
recording properties (source dump, frame/link/triangle counts, git commit, plus
any ``--prop key=value``), and every link's pose track is written as one
``send_columns`` batch rather than a per-frame log loop.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb


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


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=Path(__file__).resolve().parent,
                              capture_output=True, text=True, timeout=5,
                              check=True).stdout.strip()
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", required=True, help="npz written by the rerun job")
    p.add_argument("--out", required=True, help="destination .rrd")
    p.add_argument("--app_id", default="dexsim_piano")
    p.add_argument("--name", help="recording name shown in the viewer (default: --out stem)")
    p.add_argument("--prop", action="append", default=[], metavar="KEY=VALUE",
                   help="extra recording property, repeatable (e.g. --prop reward_mode=rp1m)")
    a = p.parse_args()

    d = np.load(a.dump, allow_pickle=False)
    links = [str(s) for s in d["__links__"]]
    n_frames = int(d["__meta__"][0])
    control_dt = float(d["__meta__"][1])
    # Fail BEFORE rec.save() so a bad dump can't truncate an existing .rrd.
    stale = [k for k in links if f"{k}|parts" not in d]
    if stale:
        raise SystemExit(f"{a.dump}: pre-`|parts` dump format (e.g. {stale[0]!r}); "
                         "re-export it with the current render server")

    # explicit stream (not the global one) so the final flush is guaranteed --
    # a half-written .rrd opens as an empty viewer and looks like a scene bug.
    rec = rr.RecordingStream(a.app_id)
    rec.save(a.out)
    rec.send_blueprint(rrb.Blueprint(
        rrb.Spatial3DView(origin="/world", name="Piano scene"),
        collapse_panels=True))
    rec.send_recording_name(a.name or Path(a.out).stem)
    # Isaac/USD is right-handed Z-up; without this Rerun assumes its own default
    # and the whole rig shows up lying on its side.
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    tris = 0
    for key in links:                            # geometry is rigid -> log once
        for part in range(int(d[f"{key}|parts"])):
            visual = f"{key}/visual_{part}"
            verts, faces = d[f"{visual}|verts"], d[f"{visual}|tris"]
            tris += len(faces)
            rec.log(visual, rr.Mesh3D(vertex_positions=verts,
                                      triangle_indices=faces,
                                      vertex_normals=smooth_normals(verts, faces),
                                      albedo_factor=d[f"{visual}|color"].tolist()),
                    static=True)

    frames = np.arange(n_frames)
    indexes = [rr.TimeColumn("frame", sequence=frames)]
    if control_dt > 0.0:
        indexes.append(rr.TimeColumn("sim_time", duration=frames * control_dt))
    for key in links:
        quat = np.asarray(d[f"{key}|quat"], dtype=np.float32)      # Isaac: wxyz
        rec.send_columns(key, indexes=indexes,
                         columns=rr.Transform3D.columns(
                             translation=d[f"{key}|pos"],
                             quaternion=quat[:, [1, 2, 3, 0]]))    # rerun: xyzw

    if "__goal__" in d and control_dt > 0.0:   # falling-note roll above the keys
        # numpy-only helper shared with the rollout converter; resolve source/
        # by repo layout so this also works without env.sh's PYTHONPATH.
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source"))
        from dexsim.visualization.rerun_rollout import note_roll_frames

        roll = note_roll_frames(d["__goal__"].astype(bool), d["__key_xyz__"],
                                d["__key_black__"].astype(bool), control_dt)
        rec.log("world/note_roll/upcoming",
                rr.Boxes3D.from_fields(colors=[130, 175, 255, 130], fill_mode="solid"), static=True)
        rec.log("world/note_roll/active",
                rr.Boxes3D.from_fields(colors=[40, 230, 80, 210], fill_mode="solid"), static=True)
        for kind, (centers, halves, counts) in roll.items():
            rec.send_columns(f"world/note_roll/{kind}", indexes=indexes,
                             columns=rr.Boxes3D.columns(
                                 centers=np.asarray(centers, np.float32).reshape(-1, 3),
                                 half_sizes=np.asarray(halves, np.float32).reshape(-1, 3))
                             .partition(counts))

    extra = dict(kv.split("=", 1) for kv in a.prop)
    commit = git_commit()
    rec.send_property("dexsim", rr.AnyValues(
        source=str(a.dump), frames=n_frames, links=len(links), tris=tris,
        control_dt=control_dt,
        **({"git_commit": commit} if commit else {}), **extra))

    rec.flush()
    print(f"[rrd] {a.out}  links={len(links)} frames={n_frames} tris={tris}")


if __name__ == "__main__":
    main()
