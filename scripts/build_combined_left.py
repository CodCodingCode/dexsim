"""Bond the imported LEFT hand onto a UR10e -> assets/ur10e_shadow_left.usd.

Same proven recipe as build_combined_usd.py (single articulation root, deactivate
hand->world anchors, place the hand's forearm COINCIDENT with the flange), but the
hand is the MuJoCo-derived left hand (shadow_hand_left_r0.usd, robot0_* names).
--mount-rpy lets you spin the hand about the flange so the palm faces the keys.

  python scripts/build_combined_left.py --headless                 # build
  python scripts/build_combined_left.py --mount-rpy 0,0,3.14159 --headless
"""
from __future__ import annotations
import argparse, math
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--mount-rpy", default="0,0,0")
p.add_argument("--flange-link", default="wrist_3_link")
p.add_argument("--hand-root", default="robot0_forearm")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from dexsim import ASSETS_DIR

UR10E_USD = f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10e/ur10e.usd"
LEFT_HAND_USD = str(ASSETS_DIR / "shadow_hand_left_r0.usd")
OUT = str(ASSETS_DIR / "ur10e_shadow_left.usd")
rpy = [float(v) for v in a.mount_rpy.split(",")]

def rpy_qf(r, p_, y):
    cr,sr=math.cos(r/2),math.sin(r/2); cp,sp=math.cos(p_/2),math.sin(p_/2); cy,sy=math.cos(y/2),math.sin(y/2)
    return Gf.Quatf(cr*cp*cy+sr*sp*sy, Gf.Vec3f(sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy))

stage = Usd.Stage.CreateNew(OUT)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
root = UsdGeom.Xform.Define(stage, "/ur10e_shadow_left")
stage.SetDefaultPrim(root.GetPrim()); UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
UsdGeom.Xform.Define(stage, "/ur10e_shadow_left/ur10e").GetPrim().GetReferences().AddReference(UR10E_USD)
hand = UsdGeom.Xform.Define(stage, "/ur10e_shadow_left/shadow")
hand.GetPrim().GetReferences().AddReference(LEFT_HAND_USD)
UsdGeom.Xformable(hand.GetPrim()).ClearXformOpOrder()

# one articulation root
for prim in stage.Traverse():
    if prim.GetPath().pathString == "/ur10e_shadow_left": continue
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI): prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
# deactivate hand->world anchors
for prim in stage.Traverse():
    pth = prim.GetPath().pathString
    if pth.startswith("/ur10e_shadow_left/shadow") and prim.IsA(UsdPhysics.Joint):
        b0 = prim.GetRelationship("physics:body0")
        if not (b0 and b0.GetTargets()): prim.SetActive(False)

def resolve(under, leaf):
    for prim in stage.Traverse():
        pp = prim.GetPath().pathString
        if pp.startswith(under) and prim.GetName()==leaf and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return pp
    return None
arm_link = resolve("/ur10e_shadow_left/ur10e", a.flange_link)
hand_link = resolve("/ur10e_shadow_left/shadow", a.hand_root)
print(f"  arm_link={arm_link}  hand_link={hand_link}")
assert arm_link and hand_link, "could not resolve bonding links"

# coincide forearm with flange (+ optional rpy)
cache = UsdGeom.XformCache()
Tw = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(arm_link))
Tf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(hand_link))
Th = cache.GetLocalToWorldTransform(stage.GetPrimAtPath("/ur10e_shadow_left/shadow"))
Toff = Gf.Matrix4d().SetRotate(rpy_qf(*rpy))
Tnew = Th * Tf.GetInverse() * Toff * Tw
UsdGeom.Xformable(hand.GetPrim()).AddTransformOp().Set(Tnew)

joint = UsdPhysics.FixedJoint.Define(stage, "/ur10e_shadow_left/flange_to_hand")
joint.CreateBody0Rel().SetTargets([Sdf.Path(arm_link)])
joint.CreateBody1Rel().SetTargets([Sdf.Path(hand_link)])
stage.GetRootLayer().Save()
print(f"[build_left] wrote {OUT}  (flange {arm_link} -> hand {hand_link}, rpy={rpy})")

# stability check
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
print(f"[build_left] spawned: {robot.num_joints} joints, {robot.num_bodies} bodies")
maxv=0.0
for _ in range(40):
    robot.set_joint_position_target(robot.data.default_joint_pos); robot.write_data_to_sim(); sim.step(); robot.update(1/120.0)
    maxv=max(maxv, float(robot.data.joint_vel[0].abs().max()))
print(f"[build_left] 40-step max|vel|={maxv:.3f} ({'STABLE' if maxv<5 else 'UNSTABLE'})")
app.close()
