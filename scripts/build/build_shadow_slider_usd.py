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

    # 2) build a FIXED-BASE prismatic carriage so the 2 slide axes become real
    # articulation DoFs (a single D6 world->forearm becomes a FLOATING BASE instead
    # -> PhysX reports 0 slider DoF, which is the bug). Chain:
    #   world --fixed--> slider_base --prismatic Y--> carriage --prismatic Z--> forearm
    sy = [float(v) for v in args.slide_y.split(",")]
    sz = [float(v) for v in args.slide_z.split(",")]
    root_prim = stage.GetDefaultPrim()
    base_path = root_prim.GetPath().pathString
    fa_prim = stage.GetPrimAtPath(forearm)
    # world transform of the forearm: place the dummy carriage bodies coincident with
    # it so all joint frames are at 0 and slider pos 0 = the hand's native pose.
    fa_xf = UsdGeom.Xformable(fa_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    fa_pos = fa_xf.ExtractTranslation()

    def _dummy_body(name):
        p = f"{base_path}/{name}"
        xf = UsdGeom.Xform.Define(stage, p)
        xf.AddTranslateOp().Set(Gf.Vec3d(fa_pos[0], fa_pos[1], fa_pos[2]))
        prim = xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        m = UsdPhysics.MassAPI.Apply(prim)
        m.CreateMassAttr().Set(0.01)            # tiny carriage mass
        return p

    slider_base = _dummy_body("slider_base")
    carriage = _dummy_body("slider_carriage")

    # world -> slider_base : FIXED (anchors the articulation = fixed base)
    fj = UsdPhysics.FixedJoint.Define(stage, f"{base_path}/slider_anchor")
    fj.CreateBody1Rel().SetTargets([Sdf.Path(slider_base)])     # body0 empty = world

    def _prismatic(name, b0, b1, axis, rng):
        j = UsdPhysics.PrismaticJoint.Define(stage, f"{base_path}/{name}")
        if b0:
            j.CreateBody0Rel().SetTargets([Sdf.Path(b0)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(b1)])
        j.CreateAxisAttr().Set(axis)
        j.CreateLowerLimitAttr().Set(rng[0])
        j.CreateUpperLimitAttr().Set(rng[1])
        j.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
        j.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
        drv.CreateTypeAttr().Set("force")
        drv.CreateStiffnessAttr().Set(8000.0)      # stiff, fast slider (mm placement)
        drv.CreateDampingAttr().Set(400.0)
        drv.CreateMaxForceAttr().Set(4000.0)
        return j

    _prismatic("slider_y", slider_base, carriage, "Y", sy)   # lateral along keyboard
    _prismatic("slider_z", carriage, forearm, "Z", sz)       # vertical (press)
    print(f"[slider] built fixed-base prismatic carriage: Y{sy} Z{sz} "
          f"(slider_base@{tuple(round(v,3) for v in fa_pos)})")

    # 3) articulation root: reuse the hand's existing root on the default prim; the
    # fixed anchor makes it a FIXED-base articulation, so the 2 prismatics + 24 hand
    # joints become 26 articulated DoFs (the smoke test asserts this).
    roots = [p.GetPath().pathString for p in stage.Traverse()
             if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    print(f"[slider] existing articulation root(s): {roots}")
    if not roots:
        UsdPhysics.ArticulationRootAPI.Apply(root_prim)
        PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
        print(f"[slider] applied articulation root on {base_path}")

    stage.GetRootLayer().Save()
    print(f"[OK] wrote hands-only slider asset -> {out}")


main()
simulation_app.close()
