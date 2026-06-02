"""Dump every physics joint (body0/body1 targets + local poses) and articulation
root for the source UR10e USD and our composed ur10e_shadow USD, so we can see
how fixed-base anchoring is authored and why the base lands at the origin."""

from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); args.headless = True
app = AppLauncher(args).app

from pxr import Usd, UsdPhysics, PhysxSchema  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402
from dexsim.assets import COMBINED_USD_PATH  # noqa: E402

UR10E_USD = f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10e/ur10e.usd"


def rel(prim, name):
    r = prim.GetRelationship(name)
    return [p.pathString for p in r.GetTargets()] if r else []


def attr(prim, name):
    a = prim.GetAttribute(name)
    return a.Get() if a and a.HasAuthoredValue() else None


def dump(path, title):
    print(f"\n############## {title} ##############\n  {path}")
    stage = Usd.Stage.Open(path)
    if stage is None:
        print("   <could not open>"); return
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            print(f"  [ARTICULATION ROOT] {p}")
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        p = prim.GetPath().pathString
        jtype = prim.GetTypeName()
        b0 = rel(prim, "physics:body0")
        b1 = rel(prim, "physics:body1")
        lp0 = attr(prim, "physics:localPos0")
        lp1 = attr(prim, "physics:localPos1")
        excl = attr(prim, "physics:excludeFromArticulation")
        if jtype == "PhysicsFixedJoint" or "root" in p.lower() or "world" in p.lower():
            print(f"  [{jtype}] {p}")
            print(f"        body0={b0} body1={b1}")
            print(f"        localPos0={lp0} localPos1={lp1} exclude={excl}")


dump(UR10E_USD, "SOURCE ur10e.usd")
dump(COMBINED_USD_PATH, "COMBINED ur10e_shadow.usd")
app.close()
