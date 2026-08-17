"""Verify the exported Shadow Hand URDF reproduces Isaac's kinematics.

The URDF written by ``scripts/tools/export_hand_urdf.py`` is what PyRoki solves
IK against, so if its frames drift from the USD the IK answers are quietly wrong
in the sim. This runs forward kinematics through yourdfpy at the joint values
the simulator ACTUALLY achieved and compares every body's world pose against
PhysX.

Runs in the pyroki venv; the warm render server supplies the reference (the
thin client is stdlib-only, so no Isaac import happens here):

    .venv-pyroki/bin/python scripts/smoke/check_urdf_fk.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yourdfpy

REPO = Path(__file__).resolve().parents[2]
REF_JSON = REPO / "logs" / "urdf_fk_ref.json"

# A deliberately non-trivial pose: bend every finger chain and shift the rail,
# so a frame error anywhere in the tree shows up rather than cancelling at rest.
PROBE = {"railJoint": 0.05, "robot0_FFJ2": 0.6, "robot0_FFJ1": 0.5,
         "robot0_MFJ2": 0.4, "robot0_RFJ2": 0.3, "robot0_LFJ2": 0.35,
         "robot0_LFJ4": 0.2, "robot0_THJ4": 0.5, "robot0_THJ3": 0.4,
         "robot0_THJ0": -0.5}


def quat_matrix(q):
    """(w, x, y, z) -> 3x3 rotation."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def fetch_reference(out: Path) -> dict:
    joints = ",".join(f"{k}={v}" for k, v in PROBE.items())
    cmd = [sys.executable, str(REPO / "scripts" / "render" / "render.py"), "query",
           "--kind", "fk", "--left_joints", joints, "--right_joints", joints,
           "--out", str(out)]
    subprocess.run(cmd, cwd=REPO, check=True, capture_output=True, text=True)
    return json.loads(out.read_text())


def check(side: str, ref: dict, tol: float) -> tuple[float, float, int]:
    urdf_path = REPO / "assets" / "urdf" / f"shadow_hand_{side}" / f"shadow_hand_{side}.urdf"
    urdf = yourdfpy.URDF.load(str(urdf_path), load_meshes=False, build_collision_scene_graph=False)
    hand = ref[side]

    achieved = dict(zip(hand["joint_names"], hand["joint_pos"]))
    urdf.update_cfg({n: achieved[n] for n in urdf.actuated_joint_names if n in achieved})

    root_R = quat_matrix(hand["root_quat"])
    root_t = np.asarray(hand["root_pos"], float)

    pos_err, rot_err, n = [], [], 0
    for name, body in hand["bodies"].items():
        if name not in urdf.link_map:
            continue
        local = urdf.get_transform(name, urdf.base_link)      # root -> link
        world_R = root_R @ local[:3, :3]
        world_t = root_t + root_R @ local[:3, 3]
        pos_err.append(np.linalg.norm(world_t - np.asarray(body["pos"], float)))
        # geodesic angle between the two orientations
        rel = world_R.T @ quat_matrix(body["quat"])
        rot_err.append(np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1))))
        n += 1

    max_pos, max_rot = max(pos_err), max(rot_err)
    status = "PASS" if max_pos < tol else "FAIL"
    print(f"  [{status}] {side:5s} {n} bodies  max pos err {max_pos * 1000:.3f} mm  "
          f"max rot err {max_rot:.3f} deg")
    return max_pos, max_rot, n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tol_m", type=float, default=1.0e-3, help="max position error (m)")
    p.add_argument("--ref", type=Path, default=REF_JSON)
    p.add_argument("--reuse", action="store_true", help="reuse a previous reference json")
    a = p.parse_args()

    if a.reuse and a.ref.exists():
        ref = json.loads(a.ref.read_text())
    else:
        print("querying the warm render server for PhysX ground truth ...")
        ref = fetch_reference(a.ref)

    print("URDF forward kinematics vs Isaac:")
    worst = max(check(side, ref, a.tol_m)[0] for side in ("left", "right"))
    if worst >= a.tol_m:
        raise SystemExit(f"\nFAIL: URDF disagrees with the sim by {worst * 1000:.2f} mm")
    print(f"\nALL CHECKS PASSED (worst {worst * 1000:.3f} mm)")


if __name__ == "__main__":
    main()
