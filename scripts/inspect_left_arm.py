"""Diagnose why the left hand doesn't join the arm articulation: print each hand
joint's body0/body1 targets and whether those body prims actually exist + carry
RigidBodyAPI. A broken/missing target = the rename didn't fix up the joint ref."""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app
from pxr import Usd, UsdPhysics
from dexsim import ASSETS_DIR
st = Usd.Stage.Open(str(ASSETS_DIR / "ur10e_shadow_left.usd"))

def exists_rb(path):
    pr = st.GetPrimAtPath(path)
    return pr.IsValid(), (pr.IsValid() and pr.HasAPI(UsdPhysics.RigidBodyAPI))

print("=== hand joints under /ur10e_shadow_left/shadow ===")
n_ok = n_bad = 0
for pr in st.Traverse():
    pp = pr.GetPath().pathString
    if not (pp.startswith("/ur10e_shadow_left/shadow") and pr.IsA(UsdPhysics.Joint)):
        continue
    b0 = pr.GetRelationship("physics:body0"); b1 = pr.GetRelationship("physics:body1")
    t0 = b0.GetTargets()[0].pathString if (b0 and b0.GetTargets()) else "(empty)"
    t1 = b1.GetTargets()[0].pathString if (b1 and b1.GetTargets()) else "(empty)"
    e0 = exists_rb(t0) if t0 != "(empty)" else (False, False)
    e1 = exists_rb(t1) if t1 != "(empty)" else (False, False)
    ok = (t0 == "(empty)" or e0[1]) and e1[1]
    n_ok += ok; n_bad += (not ok)
    if n_ok + n_bad <= 8 or not ok:
        print(f"  {pr.GetName():16s} b0={t0.split('/')[-1]:18s}({e0[1]}) "
              f"b1={t1.split('/')[-1]:18s}({e1[1]})  {'OK' if ok else 'BROKEN'}")
print(f"\njoints OK={n_ok} BROKEN={n_bad}")
print("=== flange_to_hand ===")
fj = st.GetPrimAtPath("/ur10e_shadow_left/flange_to_hand")
for rel in ("physics:body0", "physics:body1"):
    r = fj.GetRelationship(rel); t = r.GetTargets()[0].pathString if (r and r.GetTargets()) else "(empty)"
    print(f"  {rel}={t}  exists_rb={exists_rb(t) if t!='(empty)' else '-'}")
app.close()
