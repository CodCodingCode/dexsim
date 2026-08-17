"""Export a Shadow Hand USD articulation to URDF + OBJ meshes.

PyRoki / yourdfpy / viser speak URDF; our embodiment is authored in USD. Rather
than pointing the IK tools at *some other* Shadow Hand model (whose kinematics
would silently disagree with the sim), this converts the ACTUAL asset the
piano task loads, so IK solved on the URDF is valid in Isaac.

Reads the USD with plain ``pxr`` -- no ``AppLauncher``, no Isaac boot, ~1 s.

    source env.sh
    python scripts/tools/export_hand_urdf.py            # both hands
    python scripts/tools/export_hand_urdf.py --hand left

Output (gitignored, regenerate at will):
    assets/urdf/shadow_hand_<side>/shadow_hand_<side>.urdf
    assets/urdf/shadow_hand_<side>/meshes/<link>_<i>.obj

Frame conversion. USD gives each joint its frame twice -- (localPos0, localRot0)
in the PARENT body and (localPos1, localRot1) in the CHILD body. URDF instead
makes the child link frame coincide with the joint frame and expresses the axis
there. Keeping the USD child frame (so the meshes need no rebaking) gives

    origin = (localPos0,  localRot0 * conj(localRot1))
    axis_child = R(localRot1) * axis_joint

which is exact as long as localPos1 == 0 (the joint sits at the child's origin);
that holds for every joint in this asset and is asserted below.

Units. These assets declare ``metersPerUnit = 0.01`` but are authored in METERS
(the forearm measures 0.256 -- the real Shadow Hand's length -- and Isaac loads
them unscaled). Honouring that metadata would shrink the hand 100x, so the
export defaults to ``--scale 1.0``. The guard against getting this wrong is
``scripts/smoke/check_urdf_fk.py``, which compares this URDF's forward
kinematics against PhysX and fails on sub-millimetre drift.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

from dexsim.render.rerun_export import _gather, _material_color

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "assets" / "urdf"
SOURCES = {"left": REPO / "assets" / "shadow_hand_left_slider.usda",
           "right": REPO / "assets" / "shadow_hand_right_slider.usda"}

# USD authors revolute limits in degrees and prismatic limits in stage units.
_AXIS_VEC = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
_JOINT_TYPE = {"PhysicsRevoluteJoint": "revolute",
               "PhysicsPrismaticJoint": "prismatic",
               "PhysicsFixedJoint": "fixed"}


def _quat(q) -> np.ndarray:
    """Gf.Quat* -> (w, x, y, z) float64."""
    return np.array([q.GetReal(), *q.GetImaginary()], dtype=np.float64)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                     w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                     w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                     w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1])


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def _quat_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _rpy(q: np.ndarray) -> tuple[float, float, float]:
    """Quaternion -> URDF fixed-axis roll/pitch/yaw (X then Y then Z)."""
    m = _quat_matrix(q)
    pitch = np.arcsin(np.clip(-m[2, 0], -1.0, 1.0))
    if np.cos(pitch) > 1.0e-7:
        return (float(np.arctan2(m[2, 1], m[2, 2])), float(pitch),
                float(np.arctan2(m[1, 0], m[0, 0])))
    return (float(np.arctan2(-m[1, 2], m[1, 1])), float(pitch), 0.0)   # gimbal lock


def _rigid_bodies(stage) -> list:
    return [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]


def _write_obj(path: Path, verts: np.ndarray, tris: np.ndarray) -> None:
    lines = [f"v {x:.6g} {y:.6g} {z:.6g}" for x, y, z in verts]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in tris]
    path.write_text("\n".join(lines) + "\n")


def export(side: str, out_root: Path, scale: float = 1.0) -> Path:
    src = SOURCES[side]
    stage = Usd.Stage.Open(str(src))
    declared = UsdGeom.GetStageMetersPerUnit(stage)
    if abs(declared - scale) > 1.0e-9:
        print(f"[urdf] note: {src.name} declares metersPerUnit={declared}; "
              f"exporting at scale={scale} (see module docstring)")
    out_dir = out_root / f"shadow_hand_{side}"
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    links = _rigid_bodies(stage)
    link_names = [p.GetName() for p in links]
    prim_set = set(links)
    xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    # ---- geometry: one OBJ per (link, material group), in the link's frame ---
    visuals: dict[str, list[tuple[str, tuple]]] = {}
    tips: dict[str, np.ndarray] = {}
    for prim in links:
        inv = xf_cache.GetLocalToWorldTransform(prim).GetInverse()
        parts = _gather(prim, prim_set, xf_cache, inv) or []
        entries = []
        for i, (verts, tris, color) in enumerate(parts):
            rel = f"meshes/{prim.GetName()}_{i}.obj"
            _write_obj(out_dir / rel, np.asarray(verts, np.float64) * scale, tris)
            entries.append((rel, color or _material_color(prim) or (200, 200, 205, 255)))
        visuals[prim.GetName()] = entries
        # A distal link's ORIGIN sits at the base of the fingertip segment, ~30 mm
        # short of the pad that actually touches a key -- IK aimed at the origin
        # reports a key as unreachable when the finger is really on it. Emit an
        # explicit "<link>_tip" frame at the mesh's far end so solvers can target
        # the contact point directly.
        if prim.GetName().endswith("distal") and parts:
            pts = np.vstack([np.asarray(v, np.float64) for v, _, _ in parts]) * scale
            axis = int(np.argmax(np.abs(pts).max(axis=0)))
            far = pts[pts[:, axis] >= np.quantile(pts[:, axis], 0.95)]
            tips[prim.GetName()] = far.mean(axis=0)

    # ---- joints -----------------------------------------------------------
    joints, children = [], set()
    for prim in stage.Traverse():
        kind = _JOINT_TYPE.get(str(prim.GetTypeName()))
        if kind is None:
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if not body1:
            continue
        child = str(body1[0]).rsplit("/", 1)[-1]
        if not body0:                       # world attachment -> URDF root link
            continue
        parent = str(body0[0]).rsplit("/", 1)[-1]
        if parent not in link_names or child not in link_names:
            continue

        pos1 = np.asarray(joint.GetLocalPos1Attr().Get() or (0, 0, 0), np.float64)
        if np.linalg.norm(pos1) > 1.0e-6:
            raise SystemExit(f"{prim.GetName()}: localPos1={pos1} is non-zero; URDF "
                             "cannot offset a joint from its child link origin")
        rot0 = _quat(joint.GetLocalRot0Attr().Get())
        rot1 = _quat(joint.GetLocalRot1Attr().Get())
        origin_xyz = np.asarray(joint.GetLocalPos0Attr().Get() or (0, 0, 0), np.float64) * scale
        origin_q = _quat_mul(rot0, _quat_conj(rot1))

        axis_name = str(prim.GetAttribute("physics:axis").Get() or "X")
        axis = _quat_matrix(rot1) @ np.asarray(_AXIS_VEC[axis_name], np.float64)
        lo = prim.GetAttribute("physics:lowerLimit").Get()
        hi = prim.GetAttribute("physics:upperLimit").Get()
        if kind == "revolute" and lo is not None:
            lo, hi = np.deg2rad(lo), np.deg2rad(hi)       # USD authors degrees
        elif kind == "prismatic" and lo is not None:
            lo, hi = lo * scale, hi * scale
        joints.append(dict(name=prim.GetName(), kind=kind, parent=parent, child=child,
                           xyz=origin_xyz, rpy=_rpy(origin_q), axis=axis,
                           lower=lo, upper=hi))
        children.add(child)

    roots = [n for n in link_names if n not in children]
    if len(roots) != 1:
        raise SystemExit(f"expected exactly one root link, found {roots}")

    # ---- URDF -------------------------------------------------------------
    def _vec(v):
        return " ".join(f"{float(x):.9g}" for x in v)

    xml = ['<?xml version="1.0"?>',
           f'<!-- generated by scripts/tools/export_hand_urdf.py from {src.name} -->',
           f'<robot name="shadow_hand_{side}">']
    for name in link_names:
        xml.append(f'  <link name="{name}">')
        # Kinematics-only model: a token inertial keeps strict URDF parsers happy.
        xml.append('    <inertial><mass value="0.01"/>'
                   '<inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>'
                   '</inertial>')
        for i, (rel, color) in enumerate(visuals.get(name, [])):
            rgba = " ".join(f"{c / 255.0:.4f}" for c in color)
            xml += [f'    <visual name="{name}_{i}">',
                    f'      <geometry><mesh filename="{rel}"/></geometry>',
                    f'      <material name="{name}_{i}_mat"><color rgba="{rgba}"/></material>',
                    '    </visual>']
        xml.append('  </link>')
    for j in joints:
        xml += [f'  <joint name="{j["name"]}" type="{j["kind"]}">',
                f'    <parent link="{j["parent"]}"/>',
                f'    <child link="{j["child"]}"/>',
                f'    <origin xyz="{_vec(j["xyz"])}" rpy="{_vec(j["rpy"])}"/>']
        if j["kind"] != "fixed":
            xml.append(f'    <axis xyz="{_vec(j["axis"])}"/>')
            lo = 0.0 if j["lower"] is None else j["lower"]
            hi = 0.0 if j["upper"] is None else j["upper"]
            xml.append(f'    <limit lower="{lo:.9g}" upper="{hi:.9g}" '
                       'effort="10" velocity="10"/>')
        xml.append('  </joint>')
    for parent, offset in sorted(tips.items()):
        name = f"{parent}_tip"
        xml += [f'  <link name="{name}">',
                '    <inertial><mass value="1e-6"/>'
                '<inertia ixx="1e-9" ixy="0" ixz="0" iyy="1e-9" iyz="0" izz="1e-9"/>'
                '</inertial>',
                '  </link>',
                f'  <joint name="{name}_fixed" type="fixed">',
                f'    <parent link="{parent}"/>',
                f'    <child link="{name}"/>',
                f'    <origin xyz="{_vec(offset)}" rpy="0 0 0"/>',
                '  </joint>']
    xml.append('</robot>')

    urdf_path = out_dir / f"shadow_hand_{side}.urdf"
    urdf_path.write_text("\n".join(xml) + "\n")
    movable = [j["name"] for j in joints if j["kind"] != "fixed"]
    print(f"[urdf] {urdf_path}  links={len(link_names)} joints={len(joints)} "
          f"dof={len(movable)} root={roots[0]}")
    return urdf_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hand", choices=("left", "right", "both"), default="both")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--scale", type=float, default=1.0,
                   help="stage units -> meters; 1.0 because the asset's declared "
                        "metersPerUnit=0.01 does not match its authored geometry")
    a = p.parse_args()
    for side in (("left", "right") if a.hand == "both" else (a.hand,)):
        export(side, a.out, a.scale)


if __name__ == "__main__":
    main()
