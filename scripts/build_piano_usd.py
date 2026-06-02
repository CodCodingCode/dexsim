"""Generate an 88-key piano USD: each key is a sprung hinge the hands press.

Layout: 52 white + 36 black keys (A0..C8) arranged along +Y. Each key is a rigid
body hinged to a static base by a revolute joint about Y at the key's back edge;
a joint drive (stiffness/damping, target 0) acts as the return spring. Pressing
the front of a key rotates it down; the env reads the joint angle to decide
whether the key "sounds". Keys are named ``key_<i>`` for i in 0..87 (i = MIDI-21),
so the MIDI schedule maps straight onto prim/joint names.

  python scripts/build_piano_usd.py                 # -> assets/piano88.usd
  python scripts/build_piano_usd.py --inspect       # print layout, no write
"""

from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Build an 88-key piano USD.")
parser.add_argument("--inspect", action="store_true")
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf

from dexsim import ASSETS_DIR
from dexsim.piano import NUM_KEYS, PIANO_MIN_MIDI, key_is_black

OUT = args.out or str(ASSETS_DIR / "piano88.usd")

# dimensions (meters)
WHITE_W, WHITE_L, WHITE_H = 0.0225, 0.145, 0.012
BLACK_W, BLACK_L, BLACK_H = 0.011, 0.090, 0.010
WHITE_PITCH = 0.0235          # center-to-center spacing of white keys
BLACK_RAISE = 0.009           # black key sits this much above white tops
KEY_RANGE = 0.18              # max press rotation is small; drive holds it up
SPRING_STIFFNESS = 8.0        # return-spring strength (N*m/rad-ish)
SPRING_DAMPING = 0.5

# semitone offsets (within octave, from A) that are white, and their order
def _layout():
    """Return list of (key_index, is_black, y_center, z_top) for all 88 keys."""
    keys = []
    white_count = 0
    for i in range(NUM_KEYS):
        black = key_is_black(i)
        if not black:
            y = white_count * WHITE_PITCH
            keys.append((i, False, y, WHITE_H))
            white_count += 1
        else:
            # black key sits between the previous and next white key centers
            y = (white_count - 0.5) * WHITE_PITCH
            keys.append((i, True, y, WHITE_H + BLACK_RAISE))
    return keys


def main():
    layout = _layout()
    n_white = sum(1 for k in layout if not k[1])
    n_black = NUM_KEYS - n_white
    if args.inspect:
        print(f"88-key layout: {n_white} white + {n_black} black, "
              f"white pitch {WHITE_PITCH} m")
        for i, black, y, z in layout[:8] + layout[-4:]:
            print(f"  key_{i:02d} midi={i+PIANO_MIN_MIDI} "
                  f"{'BLACK' if black else 'white'} y={y:.4f} z_top={z:.4f}")
        print(f"  keyboard width ~ {n_white * WHITE_PITCH:.3f} m")
        simulation_app.close()
        return

    stage = Usd.Stage.CreateNew(OUT)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Piano")
    stage.SetDefaultPrim(root.GetPrim())
    # make the whole keyboard ONE articulation (kinematic base + 88 hinged keys)
    # so Isaac Lab can read all 88 key-joint angles via articulation.data.joint_pos.
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    physx_art = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    physx_art.CreateSolverPositionIterationCountAttr(16)
    physx_art.CreateSolverVelocityIterationCountAttr(1)

    # static base/frame the keys hinge to
    base = UsdGeom.Cube.Define(stage, "/Piano/base")
    base.GetSizeAttr().Set(1.0)
    base_x = UsdGeom.Xformable(base.GetPrim())
    span_y = sum(1 for k in layout if not k[1]) * WHITE_PITCH
    base_x.AddTranslateOp().Set(Gf.Vec3d(-WHITE_L / 2 - 0.02, span_y / 2, -0.03))
    base_x.AddScaleOp().Set(Gf.Vec3f(0.08, span_y + 0.05, 0.05))
    UsdPhysics.CollisionAPI.Apply(base.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    # Fix the base to the world with a fixed joint -> fixed-base articulation.
    # (A kinematic body can't live inside a PhysX articulation, so we pin it.)
    world_fix = UsdPhysics.FixedJoint.Define(stage, "/Piano/base_to_world")
    world_fix.CreateBody1Rel().SetTargets([Sdf.Path("/Piano/base")])

    for i, black, y, z_top in layout:
        w, l, h = (BLACK_W, BLACK_L, BLACK_H) if black else (WHITE_W, WHITE_L, WHITE_H)
        # key body
        kpath = f"/Piano/key_{i}"
        key = UsdGeom.Cube.Define(stage, kpath)
        key.GetSizeAttr().Set(1.0)
        kx = UsdGeom.Xformable(key.GetPrim())
        # front of key at x=0, extends back to +l; hinge at back (x=+l)
        cx = (l / 2.0) - WHITE_L / 2.0
        cz = z_top - h / 2.0
        kx.AddTranslateOp().Set(Gf.Vec3d(cx, y, cz))
        kx.AddScaleOp().Set(Gf.Vec3f(l, w, h))
        UsdPhysics.CollisionAPI.Apply(key.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(key.GetPrim())
        UsdPhysics.MassAPI.Apply(key.GetPrim()).CreateMassAttr(0.05 if not black else 0.03)
        # colour
        key.GetDisplayColorAttr().Set([(0.05, 0.05, 0.05) if black else (0.96, 0.96, 0.94)])

        # revolute joint: base -> key, axis Y, located at the back edge
        jpath = f"/Piano/joint_{i}"
        joint = UsdPhysics.RevoluteJoint.Define(stage, jpath)
        joint.CreateBody0Rel().SetTargets([Sdf.Path("/Piano/base")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(kpath)])
        joint.CreateAxisAttr("Y")
        # anchor at back of the key (local frames)
        back_x = (WHITE_L / 2.0)  # world-ish back position relative to key center handled by local pose
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(l / 2.0, 0.0, 0.0))
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(back_x + 0.02 - WHITE_L / 2.0, y - span_y / 2.0, 0.03 + cz))
        joint.CreateLowerLimitAttr(-12.0)   # degrees; small downward press
        joint.CreateUpperLimitAttr(0.5)
        # return-spring drive (angular)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateStiffnessAttr(SPRING_STIFFNESS)
        drive.CreateDampingAttr(SPRING_DAMPING)
        drive.CreateMaxForceAttr(50.0)

    stage.GetRootLayer().Save()
    print(f"[build_piano_usd] wrote {NUM_KEYS}-key piano -> {OUT}")
    print(f"  white-key span ~ {span_y:.3f} m; keys named key_0..key_{NUM_KEYS-1}")
    print("  press detection: read joint_<i> angle; < -threshold => sounding")


main()
simulation_app.close()
