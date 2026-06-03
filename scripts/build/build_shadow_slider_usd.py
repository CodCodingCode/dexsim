"""Build a hands-only RP1M embodiment for Isaac: Shadow hand on a 2-DoF slider.

This is the RoboPianist/RP1M arm replacement — instead of bonding the Shadow hand
to a UR10e, we mount it on a 2-DoF prismatic carriage (lateral Y + vertical Z),
which is exactly RP1M's "forearm (tx, ty)" sliding base. The resulting
articulation's action space matches RP1M's 39-d layout (24 hand joints + 2 slider
+ shared sustain), so RP1M data and the offline SFT policy drop straight in, and
there is NO arm/embodiment gap. (The UR10e is added later as a separate layer.)

Mechanism: deactivate the hand's native world-anchor fixed joint, then add a D6
joint world -> robot0_forearm with two FREE translational axes (Y lateral, Z
vertical), all rotations + transX locked. PhysX exposes the two free axes as
prismatic articulation DoFs; we add position drives on both.

  python scripts/build_shadow_slider_usd.py --src assets/shadow_hand_left_r0.usd \
      --out assets/shadow_slider.usd --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Mount Shadow hand on a 2-DoF slider.")
parser.add_argument("--src", default="assets/shadow_hand_left_r0.usd")
parser.add_argument("--out", default="assets/shadow_slider.usd")
parser.add_argument("--forearm", default=None, help="hand root body (default: auto)")
parser.add_argument("--slide-y", default="-0.6,0.6", help="lateral travel (m)")
parser.add_argument("--slide-z", default="-0.05,0.10", help="vertical travel (m)")
parser.add_argument("--inspect", action="store_true", help="print the prim tree and exit")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import shutil
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf, Gf

FOREARM_CANDIDATES = ["robot0_forearm", "robot0_wrist", "robot0_palm", "forearm", "palm"]


def _find_link(stage, candidates):
    for cand in candidates:
        for prim in stage.Traverse():
            if prim.GetName() == cand and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return prim.GetPath().pathString
    # fallback: any rigid body whose name contains a candidate
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            for cand in candidates:
                if cand in prim.GetName():
                    return prim.GetPath().pathString
    return None


def main():
    src = Path(args.src); out = Path(args.out)
    shutil.copy(src, out)
    stage = Usd.Stage.Open(str(out))

    if args.inspect:
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
                print(("JOINT " if prim.IsA(UsdPhysics.Joint) else "BODY  ")
                      + prim.GetPath().pathString)
        return

    forearm = args.forearm or _find_link(stage, FOREARM_CANDIDATES)
    if not forearm:
        raise RuntimeError(f"forearm link not found; candidates={FOREARM_CANDIDATES}")
    print(f"[slider] forearm root = {forearm}")

    # 1) deactivate the hand's native world-anchor joint(s) (body0=world pin)
    killed = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            b0 = prim.GetRelationship("physics:body0")
            b1 = prim.GetRelationship("physics:body1")
            t0 = b0.GetTargets() if b0 else []
            t1 = b1.GetTargets() if b1 else []
            # a world anchor has one empty side and binds the forearm/base
            anchors_forearm = any(forearm in str(t) for t in list(t0) + list(t1))
            if (not t0 or not t1) and (anchors_forearm or prim.GetName() in ("rootJoint", "root_joint")):
                prim.SetActive(False); killed.append(prim.GetPath().pathString)
    print(f"[slider] deactivated world-anchor joint(s): {killed}")

    # 2) add a D6 joint world -> forearm with 2 free translational axes
    sy = [float(v) for v in args.slide_y.split(",")]
    sz = [float(v) for v in args.slide_z.split(",")]
    jpath = "/" + stage.GetDefaultPrim().GetName() + "/slider_joint" \
        if stage.GetDefaultPrim() else "/slider_joint"
    joint = UsdPhysics.Joint.Define(stage, jpath)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(forearm)])   # body0 empty = world
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    # lock everything, then free transY (lateral) and transZ (vertical)
    for axis in ("transX", "rotX", "rotY", "rotZ"):
        la = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
        la.CreateLowAttr().Set(1.0); la.CreateHighAttr().Set(-1.0)   # low>high => locked
    for axis, rng in (("transY", sy), ("transZ", sz)):
        la = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
        la.CreateLowAttr().Set(rng[0]); la.CreateHighAttr().Set(rng[1])
        drv = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
        drv.CreateTypeAttr().Set("force")
        drv.CreateStiffnessAttr().Set(4000.0)     # stiff slider (gross positioning)
        drv.CreateDampingAttr().Set(200.0)
        drv.CreateMaxForceAttr().Set(2000.0)
    print(f"[slider] added D6 slider: transY{sy} transZ{sz} (others locked)")

    # 3) articulation root: the source hand USD ALREADY has one (on its default
    # prim). Adding another (e.g. on the forearm) -> "Nested articulation roots
    # are not allowed". So reuse the existing root and add nothing; the D6 slider
    # joint world->forearm is incorporated as the articulation's 2-DoF floating base.
    roots = [p.GetPath().pathString for p in stage.Traverse()
             if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    print(f"[slider] existing articulation root(s): {roots} (reusing, adding none)")
    if not roots:
        fp = stage.GetPrimAtPath(forearm)
        UsdPhysics.ArticulationRootAPI.Apply(fp)
        PhysxSchema.PhysxArticulationAPI.Apply(fp)
        print(f"[slider] no root found; applied one on {forearm}")

    stage.GetRootLayer().Save()
    print(f"[OK] wrote hands-only slider asset -> {out}")


main()
simulation_app.close()
