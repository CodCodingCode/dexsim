"""Wrap the left/right Shadow Hands in a world-anchored Y-axis rail.

The stock hand's ``rootJoint`` is a world-fixed joint.  Overriding that prim as
a prismatic joint gives the hand exactly one base DoF without touching the
physical wrist/finger chain:

    world --[rootJoint: prismatic Y]--> hand_mount --> forearm --> fingers

Run from the repository root in an Isaac/USD Python environment.  The output
assets are referenced by ``PIANO_SHADOW_HAND_{LEFT,RIGHT}_CFG``.
"""
from pathlib import Path
import os

from pxr import Usd, UsdGeom, UsdPhysics


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
RAIL_LIMIT_M = 0.12


def build(source: Path, output: Path) -> None:
    stage = Usd.Stage.CreateNew(str(output))
    root = UsdGeom.Xform.Define(stage, "/shadow_hand").GetPrim()
    stage.SetDefaultPrim(root)
    reference = os.path.relpath(source.resolve(), output.resolve().parent)
    root.GetReferences().AddReference(reference, "/shadow_hand")

    # Disable the stock world->hand fixed joint. A fixed rail anchor plus an
    # internal anchor->mount prismatic joint is required for PhysX to expose the
    # slider as an articulation DoF (a direct world->mount prismatic root is not).
    stage.OverridePrim("/shadow_hand/joints/rootJoint").SetActive(False)
    anchor = UsdGeom.Xform.Define(stage, "/shadow_hand/rail_anchor").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(anchor)
    fixed = UsdPhysics.FixedJoint.Define(stage, "/shadow_hand/joints/railAnchorJoint")
    fixed.CreateBody1Rel().SetTargets([anchor.GetPath()])

    rail = UsdPhysics.PrismaticJoint.Define(stage, "/shadow_hand/joints/railJoint")
    rail.CreateBody0Rel().SetTargets([anchor.GetPath()])
    rail.CreateBody1Rel().SetTargets(["/shadow_hand/robot0_hand_mount"])
    # The piano mounts the hand fingers-down (180deg about (1,1,0)/sqrt2): local
    # +X (the knuckle row) maps to world Y, the keyboard-parallel rail direction.
    rail.CreateAxisAttr(UsdPhysics.Tokens.x)
    rail.CreateLowerLimitAttr(-RAIL_LIMIT_M)
    rail.CreateUpperLimitAttr(RAIL_LIMIT_M)
    stage.GetRootLayer().Save()
    print(f"[slider] {output} <- {source}  Y=[{-RAIL_LIMIT_M}, {RAIL_LIMIT_M}] m")


if __name__ == "__main__":
    build(ASSETS / "shadow_hand_left.usd", ASSETS / "shadow_hand_left_slider.usda")
    build(ASSETS / "nvidia_shadow_right" / "shadow_hand_right.usd",
          ASSETS / "shadow_hand_right_slider.usda")
