"""All-in-one (single Isaac boot): rename the left hand lh_*->robot0_* robustly
(iterative, deepest-first), bond it to a UR10e, and stability-test the result.
Writes assets/ur10e_shadow_left.usd.

  python scripts/build_left_arm.py --headless [--mount-rpy 0,0,3.14159]
"""
from __future__ import annotations
import argparse, math, shutil
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--mount-rpy", default="0,0,0")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from dexsim import ASSETS_DIR

SRC = str(ASSETS_DIR / "shadow_hand_left.usd")
HAND = str(ASSETS_DIR / "shadow_hand_left_r0.usd")
UR10E_USD = f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10e/ur10e.usd"
OUT = str(ASSETS_DIR / "ur10e_shadow_left.usd")
rpy = [float(v) for v in a.mount_rpy.split(",")]

# ---- 1. iterative rename lh_* -> robot0_* (deepest first) ----
# FLATTEN first: the MJCF-imported prims are brought in via composition, so they
# aren't local specs in the root layer and namespace edits fail ("Object does not
# exist"). Flatten bakes everything into one local layer we CAN rename.
_src = Usd.Stage.Open(SRC)
_src.Flatten().Export(HAND)
stage = Usd.Stage.Open(HAND)
while True:
    lh = [pr.GetPath() for pr in stage.Traverse() if pr.GetName().startswith("lh_")]
    if not lh:
        break
    deepest = max(lh, key=lambda p: len(p.pathString))
    new = "robot0_" + deepest.name[3:]
    e = Sdf.BatchNamespaceEdit(); e.Add(deepest, deepest.GetParentPath().AppendChild(new))
    if not stage.GetRootLayer().Apply(e):
        print(f"  RENAME FAIL at {deepest}"); break
# CRITICAL: renaming body PRIMS does NOT fix up joints' body0/body1 relationship
# TARGETS -- they still point to the old /lh_* paths, so PhysX can't traverse
# into the hand (only the 6 arm joints end up in the articulation). Rewrite every
# joint target /lh_ -> /robot0_ to match the renamed bodies.
n_fix = 0
for pr in stage.Traverse():
    if not pr.IsA(UsdPhysics.Joint):
        continue
    for rel_name in ("physics:body0", "physics:body1"):
        rel = pr.GetRelationship(rel_name)
        if rel and rel.GetTargets():
            newt = [Sdf.Path(t.pathString.replace("/lh_", "/robot0_")) for t in rel.GetTargets()]
            if newt != list(rel.GetTargets()):
                rel.SetTargets(newt); n_fix += 1
print(f"[left_arm] fixed {n_fix} joint body-target refs lh_->robot0_")
stage.GetRootLayer().Save()
n_r0 = sum(pr.GetName().startswith("robot0_") for pr in stage.Traverse())
print(f"[left_arm] renamed -> {n_r0} robot0_* prims; remaining lh_: "
      f"{sum(pr.GetName().startswith('lh_') for pr in stage.Traverse())}")

# ---- 2. bond UR10e + left hand ----
def rpy_qf(r,p_,y):
    cr,sr=math.cos(r/2),math.sin(r/2); cp,sp=math.cos(p_/2),math.sin(p_/2); cy,sy=math.cos(y/2),math.sin(y/2)
    return Gf.Quatf(cr*cp*cy+sr*sp*sy, Gf.Vec3f(sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy))

st = Usd.Stage.CreateNew(OUT)
UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
root = UsdGeom.Xform.Define(st, "/ur10e_shadow_left"); st.SetDefaultPrim(root.GetPrim())
UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
UsdGeom.Xform.Define(st, "/ur10e_shadow_left/ur10e").GetPrim().GetReferences().AddReference(UR10E_USD)
hand = UsdGeom.Xform.Define(st, "/ur10e_shadow_left/shadow")
hand.GetPrim().GetReferences().AddReference(HAND)
UsdGeom.Xformable(hand.GetPrim()).ClearXformOpOrder()
for pr in st.Traverse():
    if pr.GetPath().pathString == "/ur10e_shadow_left": continue
    if pr.HasAPI(UsdPhysics.ArticulationRootAPI):
        pr.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if pr.HasAPI(PhysxSchema.PhysxArticulationAPI): pr.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
def resolve(under, leaf):
    for pr in st.Traverse():
        pp = pr.GetPath().pathString
        if pp.startswith(under) and pr.GetName()==leaf and pr.HasAPI(UsdPhysics.RigidBodyAPI): return pp
    return None
arm_link = resolve("/ur10e_shadow_left/ur10e", "wrist_3_link")
hand_link = resolve("/ur10e_shadow_left/shadow", "robot0_forearm")
print(f"[left_arm] bonding {arm_link} -> {hand_link}")
assert arm_link and hand_link

# diagnostics + PRECISE world-anchor removal: only deactivate a joint that
# anchors the hand ROOT to world (body0 empty AND body1 == hand_link). Never
# touch the hand's internal finger joints (MJCF joints can have empty body0).
hand_joints = [pr for pr in st.Traverse()
               if pr.GetPath().pathString.startswith("/ur10e_shadow_left/shadow")
               and pr.IsA(UsdPhysics.Joint)]
killed = 0
for pr in hand_joints:
    b0 = pr.GetRelationship("physics:body0"); b1 = pr.GetRelationship("physics:body1")
    b0_empty = not (b0 and b0.GetTargets())
    b1_root = bool(b1 and b1.GetTargets() and b1.GetTargets()[0].pathString.endswith("robot0_forearm"))
    if b0_empty and b1_root:
        pr.SetActive(False); killed += 1
print(f"[left_arm] hand joints in stage: {len(hand_joints)}; world-anchors removed: {killed}")
cache = UsdGeom.XformCache()
Tw = cache.GetLocalToWorldTransform(st.GetPrimAtPath(arm_link))
Tf = cache.GetLocalToWorldTransform(st.GetPrimAtPath(hand_link))
Th = cache.GetLocalToWorldTransform(st.GetPrimAtPath("/ur10e_shadow_left/shadow"))
Tnew = Th * Tf.GetInverse() * Gf.Matrix4d().SetRotate(rpy_qf(*rpy)) * Tw
UsdGeom.Xformable(hand.GetPrim()).AddTransformOp().Set(Tnew)
j = UsdPhysics.FixedJoint.Define(st, "/ur10e_shadow_left/flange_to_hand")
j.CreateBody0Rel().SetTargets([Sdf.Path(arm_link)]); j.CreateBody1Rel().SetTargets([Sdf.Path(hand_link)])
st.GetRootLayer().Save()
print(f"[left_arm] wrote {OUT}")

# ---- 3. stability test ----
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import SimulationContext, SimulationCfg
sim = SimulationContext(SimulationCfg(dt=1/120.0, device=a.device))
sim_utils.GroundPlaneCfg().func("/g", sim_utils.GroundPlaneCfg())
robot = Articulation(ArticulationCfg(prim_path="/World/R",
    spawn=sim_utils.UsdFileCfg(usd_path=OUT,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False, solver_position_iteration_count=16)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,0.75),
        joint_pos={"shoulder_lift_joint":-0.9,"elbow_joint":1.6,"wrist_1_joint":-1.2,"wrist_2_joint":-1.57,"robot0_.*":0.0}),
    actuators={"arm": ImplicitActuatorCfg(joint_names_expr=["shoulder.*","elbow.*","wrist.*"], stiffness=800.0, damping=40.0),
               "hand": ImplicitActuatorCfg(joint_names_expr=["robot0_.*"], stiffness=3.0, damping=0.1)}))
sim.reset()
ft = [b for b in robot.data.body_names if "distal" in b]
print(f"[left_arm] spawned {robot.num_joints} joints, {robot.num_bodies} bodies; fingertips={ft}")
maxv=0.0
for _ in range(40):
    robot.set_joint_position_target(robot.data.default_joint_pos); robot.write_data_to_sim(); sim.step(); robot.update(1/120.0)
    maxv=max(maxv, float(robot.data.joint_vel[0].abs().max()))
print(f"[left_arm] 40-step max|vel|={maxv:.3f} ({'STABLE' if maxv<5 else 'UNSTABLE'})")
app.close()
