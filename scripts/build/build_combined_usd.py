"""Compose UR10e + Shadow Hand into ONE articulation USD.

This is the genuinely fiddly first-hour bit: both the UR10e and the Shadow Hand
ship as *separate* articulations, each with its own articulation root. To drive
them as a single robot in Isaac Lab they must become one articulation tree --
one root, with a fixed joint bonding the hand base to the arm's tool flange.

What this script does:
  1. boots Isaac Sim headless so omniverse asset paths resolve,
  2. references the UR10e and the Shadow Hand into a fresh stage,
  3. disables the Shadow Hand's articulation root (the arm root wins),
  4. adds a fixed joint:  ur10e/<flange_link>  ->  shadow/<hand_root_link>,
  5. positions the hand at the flange with a configurable mount transform,
  6. writes assets/ur10e_shadow.usd.

Run ``--inspect`` first to print both source hierarchies and the auto-detected
flange / hand-root link names, then re-run without it to build.

Usage:
  python scripts/build_combined_usd.py --inspect
  python scripts/build_combined_usd.py
  python scripts/build_combined_usd.py --flange-link wrist_3_link --hand-root robot0_forearm
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compose UR10e + Shadow Hand into one USD.")
parser.add_argument("--inspect", action="store_true",
                    help="Print source hierarchies + auto-detected frames, then exit.")
parser.add_argument("--flange-link", default=None,
                    help="UR10e link the hand mounts to (default: auto-detect wrist_3_link/tool0).")
parser.add_argument("--hand-root", default=None,
                    help="Shadow Hand root body to bond (default: auto-detect forearm/wrist).")
parser.add_argument("--mount-xyz", default="0,0,0.0",
                    help="Hand offset from flange, meters 'x,y,z'.")
parser.add_argument("--mount-rpy", default="0,0,0",
                    help="Hand orientation offset from flange, radians 'r,p,y'.")
parser.add_argument("--hand-usd", default=None,
                    help="Hand USD to bond (default: stock RIGHT shadow_hand_instanceable.usd). "
                         "Pass a LEFT-hand USD here to build the left combined.")
parser.add_argument("--out", default=None,
                    help="Output combined USD path (default: assets/ur10e_shadow.usd).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Force headless for asset baking.
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- everything below runs with omni / pxr available -----------------------
import math
import os

from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from dexsim.assets import COMBINED_USD_PATH

UR10E_USD = f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10e/ur10e.usd"
# RIGHT hand by default; --hand-usd overrides (e.g. a LEFT-hand USD). Local paths
# are abspath'd so the reference resolves regardless of the output USD's dir.
SHADOW_USD = os.path.abspath(args.hand_usd) if args.hand_usd else \
    f"{ISAAC_NUCLEUS_DIR}/Robots/ShadowHand/shadow_hand_instanceable.usd"
OUT_PATH = os.path.abspath(args.out) if args.out else COMBINED_USD_PATH

# preferred link names, in priority order, when nothing is passed on the CLI
FLANGE_CANDIDATES = ["tool0", "wrist_3_link", "flange", "ee_link"]
HANDROOT_CANDIDATES = ["robot0_forearm", "robot0_wrist", "robot0_palm", "forearm", "palm"]


def _rpy_to_quat(r, p, y):
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    yq = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return Gf.Quatf(w, Gf.Vec3f(x, yq, z))


def _print_tree(stage, title):
    print(f"\n===== {title} =====")
    rigid, joints = [], []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid.append(prim.GetPath().pathString)
        if prim.IsA(UsdPhysics.Joint):
            joints.append(prim.GetPath().pathString)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            print(f"  [articulation root] {prim.GetPath()}")
    print(f"  rigid bodies ({len(rigid)}):")
    for p in rigid:
        print(f"    {p}")
    print(f"  joints ({len(joints)})")


def _find_link(stage, candidates):
    """Return the first rigid-body prim whose leaf name matches a candidate."""
    by_leaf = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            by_leaf.setdefault(prim.GetName(), prim.GetPath().pathString)
    for c in candidates:
        if c in by_leaf:
            return c, by_leaf[c]
    # fall back to a substring match
    for c in candidates:
        for leaf, path in by_leaf.items():
            if c in leaf:
                return leaf, path
    return None, None


def main():
    # open the two sources just to introspect frame names
    ur_stage = Usd.Stage.Open(UR10E_USD)
    sh_stage = Usd.Stage.Open(SHADOW_USD)
    if ur_stage is None or sh_stage is None:
        raise RuntimeError(f"Could not open source USDs:\n  {UR10E_USD}\n  {SHADOW_USD}")

    flange_leaf, flange_path = _find_link(ur_stage, FLANGE_CANDIDATES)
    handroot_leaf, handroot_path = _find_link(sh_stage, HANDROOT_CANDIDATES)
    flange_leaf = args.flange_link or flange_leaf
    handroot_leaf = args.hand_root or handroot_leaf

    if args.inspect:
        _print_tree(ur_stage, "UR10e")
        _print_tree(sh_stage, "Shadow Hand")
        print("\n----- auto-detected bonding frames -----")
        print(f"  UR10e flange link : {flange_leaf}")
        print(f"  Shadow root link  : {handroot_leaf}")
        print("\nRe-run without --inspect to build, or override with "
              "--flange-link / --hand-root.")
        simulation_app.close()
        return

    if not flange_leaf or not handroot_leaf:
        raise RuntimeError(
            f"Could not resolve bonding frames (flange={flange_leaf}, "
            f"hand_root={handroot_leaf}). Run --inspect and pass them explicitly."
        )

    mount_xyz = [float(v) for v in args.mount_xyz.split(",")]
    mount_rpy = [float(v) for v in args.mount_rpy.split(",")]

    # ---- build the composed stage ----
    stage = Usd.Stage.CreateNew(OUT_PATH)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/ur10e_shadow")
    stage.SetDefaultPrim(root.GetPrim())
    # single articulation root for the whole robot
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    # reference the arm
    arm = UsdGeom.Xform.Define(stage, "/ur10e_shadow/ur10e")
    arm.GetPrim().GetReferences().AddReference(UR10E_USD)

    # reference the hand under its own xform, offset to the flange
    hand = UsdGeom.Xform.Define(stage, "/ur10e_shadow/shadow")
    hand.GetPrim().GetReferences().AddReference(SHADOW_USD)
    hand_xform = UsdGeom.Xformable(hand.GetPrim())
    hand_xform.ClearXformOpOrder()
    # actual placement (coincide forearm with the flange) is computed below,
    # once we can resolve both link prims' world transforms.

    # The combined robot must be ONE articulation with exactly one root. Both
    # referenced sub-assets (UR10e and Shadow) carry their own articulation root,
    # so strip the root API from every DESCENDANT and keep only the top
    # /ur10e_shadow root we applied above.
    from pxr import PhysxSchema
    removed = []
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if path == "/ur10e_shadow":
            continue
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            removed.append(path)
    print(f"  stripped {len(removed)} descendant articulation root(s): {removed}")

    # CRITICAL: the Shadow Hand ships its OWN world-anchor fixed joint
    # (shadow/joints/rootJoint -> body0=world) that pins the hand's mount to the
    # world ORIGIN. Left in place it fights the flange_to_hand bond we add below:
    # the hand is yanked to (0,0,0) and drags the whole arm down to the floor
    # (wrist ends up at z~=-0.01, nowhere near the keyboard). The hand must be
    # held ONLY by the arm's wrist, so deactivate every world-anchored fixed
    # joint under the /shadow subtree. (Deactivating a referenced prim prunes it
    # from the composed stage, so PhysX never sees it.)
    killed = []
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not path.startswith("/ur10e_shadow/shadow"):
            continue
        if prim.IsA(UsdPhysics.Joint):
            b0 = prim.GetRelationship("physics:body0")
            if not (b0 and b0.GetTargets()):          # empty body0 == anchored to world
                prim.SetActive(False)
                killed.append(path)
    print(f"  deactivated {len(killed)} hand->world anchor joint(s): {killed}")

    # resolve the absolute prim paths of the two bonding links in the new stage
    def _resolve(under, leaf):
        # match the RIGID BODY link (not a same-named joint prim) under `under`
        for prim in stage.Traverse():
            p = prim.GetPath().pathString
            if (p.startswith(under) and prim.GetName() == leaf
                    and prim.HasAPI(UsdPhysics.RigidBodyAPI)):
                return p
        return None

    arm_link = _resolve("/ur10e_shadow/ur10e", flange_leaf)
    hand_link = _resolve("/ur10e_shadow/shadow", handroot_leaf)
    if not arm_link or not hand_link:
        raise RuntimeError(
            f"Could not locate bonding links in composed stage: "
            f"arm({flange_leaf})={arm_link}, hand({handroot_leaf})={hand_link}"
        )

    # ---- place the hand so robot0_forearm is COINCIDENT with the flange ----
    # The two sub-assets reference at unrelated world poses; bonding them with a
    # zero-offset fixed joint while they're far apart makes the constraint snap
    # them together violently on step 1 (joints -> tens of rad). Instead, move
    # the hand root so the forearm lands exactly on the flange first, then the
    # zero-offset fixed joint starts satisfied. An optional mount_xyz/rpy in the
    # FLANGE frame fine-tunes palm placement.
    cache = UsdGeom.XformCache()
    wrist_prim = stage.GetPrimAtPath(arm_link)
    forearm_prim = stage.GetPrimAtPath(hand_link)
    hand_root_prim = stage.GetPrimAtPath("/ur10e_shadow/shadow")
    T_wrist = cache.GetLocalToWorldTransform(wrist_prim)
    T_forearm = cache.GetLocalToWorldTransform(forearm_prim)
    T_hand_root = cache.GetLocalToWorldTransform(hand_root_prim)
    # optional fine-tune offset expressed in the flange frame
    T_offset = Gf.Matrix4d().SetRotate(_rpy_to_quat(*mount_rpy)) \
        * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*mount_xyz))
    # T_hand_new = T_hand_root * inv(T_forearm) * T_offset * T_wrist
    T_hand_new = T_hand_root * T_forearm.GetInverse() * T_offset * T_wrist
    hand_xform.ClearXformOpOrder()
    hand_xform.AddTransformOp().Set(T_hand_new)

    # fixed joint flange -> hand root. ENCODE the mount offset in the joint's body0 (flange)
    # local frame so the rotation/translation SURVIVES -- with zero local frames the joint
    # snapped the hand flat onto the flange and discarded T_offset (so --mount-rpy was a no-op).
    # The initial placement (T_hand_new, which already applied T_offset) then satisfies this
    # joint at t=0, so no snap.
    joint = UsdPhysics.FixedJoint.Define(stage, "/ur10e_shadow/flange_to_hand")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(arm_link)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(hand_link)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in mount_xyz]))
    joint.CreateLocalRot0Attr().Set(_rpy_to_quat(*mount_rpy))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # FIXED-base articulation: the UR10e ships its OWN world-anchor fixed joint
    # (ur10e/root_joint -> body0=world, body1=base_link) and we only stripped the
    # articulation-root *API*, not that joint -- so it survives and already pins
    # the base AND honours init_state.pos (same as the stand-alone UR10E_CFG).
    # Adding a second base->world joint here just over-constrains the base (two
    # identical world anchors -> "disjointed body transforms" snap warnings), so
    # we keep the native one and add nothing. If a future asset lacks it, re-add
    # a base_to_world fixed joint as a fallback.
    ur_rootjoint = None
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if (p.startswith("/ur10e_shadow/ur10e") and prim.IsA(UsdPhysics.Joint)
                and prim.GetName() in ("root_joint", "rootJoint")):
            b0 = prim.GetRelationship("physics:body0")
            if not (b0 and b0.GetTargets()):
                ur_rootjoint = p
                break
    if ur_rootjoint:
        print(f"  arm fixed-base via native world anchor: {ur_rootjoint} (no extra joint added)")
    else:
        base_link = _resolve("/ur10e_shadow/ur10e", "base_link") or \
                    _resolve("/ur10e_shadow/ur10e", "base")
        if base_link:
            world_fix = UsdPhysics.FixedJoint.Define(stage, "/ur10e_shadow/base_to_world")
            world_fix.CreateBody1Rel().SetTargets([Sdf.Path(base_link)])
            print(f"  no native anchor; bolted arm base to world: {base_link}")
        else:
            print("  WARNING: base_link not found and no native anchor; arm will float!")

    stage.GetRootLayer().Save()
    print(f"\n[OK] wrote combined articulation -> {OUT_PATH}")
    print(f"     hand source: {SHADOW_USD}")
    print(f"     fixed joint: {arm_link}  ->  {hand_link}")
    print(f"     mount xyz={mount_xyz} rpy={mount_rpy}")


main()
simulation_app.close()
