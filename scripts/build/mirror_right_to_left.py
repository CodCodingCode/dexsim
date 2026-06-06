"""Make a TRUE LEFT Shadow Hand by mirroring the NVIDIA RIGHT hand across the
YZ plane (negate X). One source for both hands -> they match exactly.

Reflection R = diag(-1,1,1). For a consistent global reflection:
  - every prim's LOCAL transform L -> R L R   (translation: x->-x ; rotation: conjugated)
  - every Mesh: points.x *= -1, reverse per-face winding (+ face-varying normals)
  - every joint: localPos.x*=-1 ; localRot quat (w,x,y,z)->(w,x,-y,-z)
  - axis tokens + limits UNCHANGED -> the kept axis under the conjugated frame
    comes out as -R*a, the correct mirrored (pseudovector) axis, so a positive
    joint command still flexes the finger INWARD (verified below).
Inertia principal axes / CoM mirrored if authored.

Output: assets/shadow_hand_left_mirror.usd  (bodies already robot0_* -> drop-in
for build_combined_usd.py, no rename needed).

  python scripts/build/mirror_right_to_left.py --headless
"""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app

import json
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Vt
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

SRC = f"{ISAAC_NUCLEUS_DIR}/Robots/ShadowHand/shadow_hand.usd"
OUT = "assets/shadow_hand_left_mirror.usd"

R3 = Gf.Matrix3d(-1, 0, 0, 0, 1, 0, 0, 0, 1)
R4 = Gf.Matrix4d(R3, Gf.Vec3d(0, 0, 0))


def refl_quat(q):
    """Conjugate a unit quaternion by R=diag(-1,1,1): (w,x,y,z)->(w,x,-y,-z)."""
    w = q.GetReal(); im = q.GetImaginary()
    return type(q)(w, im[0], -im[1], -im[2])


# 1. flatten the source so all referenced/instanced geometry becomes local/editable
src_stage = Usd.Stage.Open(SRC)
flat = src_stage.Flatten()
st = Usd.Stage.Open(flat)

n_xform = n_mesh = n_joint = 0

for prim in st.Traverse():
    # ---- joints: handle via localPos/localRot only (NOT as xforms) ----
    if prim.IsA(UsdPhysics.Joint):
        j = UsdPhysics.Joint(prim)
        for getp in (j.GetLocalPos0Attr, j.GetLocalPos1Attr):
            attr = getp()
            v = attr.Get()
            if v is not None:
                attr.Set(Gf.Vec3f(-v[0], v[1], v[2]))
        for getr in (j.GetLocalRot0Attr, j.GetLocalRot1Attr):
            attr = getr()
            v = attr.Get()
            if v is not None:
                attr.Set(refl_quat(v))
        n_joint += 1
        continue

    # ---- every other Xformable: reflect its local transform L -> R L R ----
    if prim.IsA(UsdGeom.Xformable):
        xf = UsdGeom.Xformable(prim)
        L = xf.GetLocalTransformation()      # local-to-parent (Gf.Matrix4d)
        Lp = R4 * L * R4
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(Lp)
        n_xform += 1

    # ---- meshes: reflect points + flip winding (+ face-varying normals) ----
    if prim.IsA(UsdGeom.Mesh):
        m = UsdGeom.Mesh(prim)
        pts = m.GetPointsAttr().Get()
        if pts:
            m.GetPointsAttr().Set(Vt.Vec3fArray(
                [Gf.Vec3f(-q[0], q[1], q[2]) for q in pts]))
        fvc = m.GetFaceVertexCountsAttr().Get()
        fvi = m.GetFaceVertexIndicesAttr().Get()
        if fvc and fvi:
            new_idx = []
            face_perm = []          # index permutation, to reorder face-varying data
            off = 0
            for c in fvc:
                run = list(range(off, off + c))
                run_rev = run[::-1]
                face_perm.extend(run_rev)
                new_idx.extend([fvi[k] for k in run_rev])
                off += c
            m.GetFaceVertexIndicesAttr().Set(Vt.IntArray(new_idx))
            # face-varying normals follow the same per-face reversal + x-negate
            na = m.GetNormalsAttr()
            nv = na.Get()
            if nv and m.GetNormalsInterpolation() == UsdGeom.Tokens.faceVarying \
                    and len(nv) == len(face_perm):
                na.Set(Vt.Vec3fArray(
                    [Gf.Vec3f(-nv[k][0], nv[k][1], nv[k][2]) for k in face_perm]))
        else:
            # vertex/varying normals (no face reorder needed): just negate x
            na = m.GetNormalsAttr(); nv = na.Get()
            if nv:
                na.Set(Vt.Vec3fArray([Gf.Vec3f(-q[0], q[1], q[2]) for q in nv]))
        # refresh extent
        npts = m.GetPointsAttr().Get()
        if npts:
            ext = UsdGeom.PointBased.ComputeExtent(npts)
            if ext:
                m.GetExtentAttr().Set(ext)
        n_mesh += 1

    # ---- inertia: mirror CoM + principal axes if authored ----
    if prim.HasAPI(UsdPhysics.MassAPI):
        mass = UsdPhysics.MassAPI(prim)
        com = mass.GetCenterOfMassAttr().Get()
        if com is not None:
            mass.GetCenterOfMassAttr().Set(Gf.Vec3f(-com[0], com[1], com[2]))
        pax = mass.GetPrincipalAxesAttr().Get()
        if pax is not None:
            mass.GetPrincipalAxesAttr().Set(refl_quat(pax))

st.Export(OUT)
print(f"[mirror] xforms={n_xform} meshes={n_mesh} joints={n_joint} -> {OUT}", flush=True)

# ============================ VERIFY ============================
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import SimulationContext, SimulationCfg

sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device=a.device))
hand = Articulation(ArticulationCfg(
    prim_path="/World/H",
    spawn=sim_utils.UsdFileCfg(
        usd_path=OUT,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0.5)),
    actuators={"f": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=3.0, damping=0.1)}))
sim.reset()


def bid(want):
    for i, n in enumerate(hand.body_names):
        if want in n.lower():
            return i
    return None


def chirality():
    P = hand.data.body_pos_w[0, bid("palm")].cpu().numpy()
    T = hand.data.body_pos_w[0, bid("thdistal")].cpu().numpy()
    I = hand.data.body_pos_w[0, bid("ffdistal")].cpu().numpy()
    Lp = hand.data.body_pos_w[0, bid("lfdistal")].cpu().numpy()
    return float(np.dot(np.cross(I - P, Lp - P), T - P))


# settle at zero pose
for _ in range(30):
    hand.set_joint_position_target(hand.data.default_joint_pos); hand.write_data_to_sim()
    sim.step(); hand.update(1 / 120.0)
s0 = chirality()
ff0 = hand.data.body_pos_w[0, bid("ffdistal")].cpu().numpy()
palm = hand.data.body_pos_w[0, bid("palm")].cpu().numpy()

# command all joints toward +0.6 rad and see which way the index fingertip goes
tgt = hand.data.default_joint_pos.clone() + 0.6
maxv = 0.0
for _ in range(60):
    hand.set_joint_position_target(tgt); hand.write_data_to_sim()
    sim.step(); hand.update(1 / 120.0)
    maxv = max(maxv, float(hand.data.joint_vel[0].abs().max()))
ff1 = hand.data.body_pos_w[0, bid("ffdistal")].cpu().numpy()
# inward flex => fingertip moves toward the palm centroid
d_to_palm = float(np.linalg.norm(ff1 - palm) - np.linalg.norm(ff0 - palm))

res = {
    "n_bodies": hand.num_bodies, "n_joints": hand.num_joints,
    "signed_volume": round(s0, 5),
    "chirality": "LEFT" if s0 < 0 else "RIGHT",
    "fingertip_delta_toward_palm": round(d_to_palm, 4),
    "fingers_curl_inward": bool(d_to_palm < 0),
    "max_vel": round(maxv, 3), "stable": bool(maxv < 8),
}
print("\n================ MIRRORED LEFT HAND ================")
print(json.dumps(res, indent=2), flush=True)
with open("logs/mirror_verify.json", "w") as f:
    json.dump(res, f, indent=2)
app.close()
