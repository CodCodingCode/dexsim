"""Make a LEFT Shadow Hand by mirroring the right one (Isaac ships only the
right). A right hand reflected across its sagittal plane IS a left hand. We
reference the right instanceable USD under a reflection (scale Y = -1) and save
``assets/shadow_hand_left_instanceable.usd``. Keeps all robot0_* names so it's a
drop-in for the env. Then spawn it to check PhysX is stable under the mirror.

  python scripts/build_left_hand.py --headless          # build + stability check
"""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--test", action="store_true", default=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app

from pxr import Usd, UsdGeom, Gf
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from dexsim import ASSETS_DIR

RIGHT = f"{ISAAC_NUCLEUS_DIR}/Robots/ShadowHand/shadow_hand_instanceable.usd"
OUT = str(ASSETS_DIR / "shadow_hand_left_instanceable.usd")

# --- build the mirrored stage ---
stage = Usd.Stage.CreateNew(OUT)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
root = UsdGeom.Xform.Define(stage, "/shadow_hand_left")
stage.SetDefaultPrim(root.GetPrim())
# reflect across the XZ plane (negate Y) -> right hand becomes left hand
xf = UsdGeom.Xformable(root.GetPrim())
xf.ClearXformOpOrder()
xf.AddScaleOp().Set(Gf.Vec3f(1.0, -1.0, 1.0))
root.GetPrim().GetReferences().AddReference(RIGHT)
stage.GetRootLayer().Save()
print(f"[build_left_hand] wrote mirrored left hand -> {OUT}")

# --- stability / appearance check ---
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import SimulationContext, SimulationCfg

sim = SimulationContext(SimulationCfg(dt=1/120.0, device=a.device))
sim_utils.GroundPlaneCfg().func("/g", sim_utils.GroundPlaneCfg())
cfg = ArticulationCfg(
    prim_path="/World/LeftHand",
    spawn=sim_utils.UsdFileCfg(usd_path=OUT,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0.5)),
    actuators={"f": ImplicitActuatorCfg(joint_names_expr=[".*"], effort_limit=0.9,
                                        stiffness=3.0, damping=0.1)},
)
hand = Articulation(cfg)
sim.reset()
import torch
print(f"[build_left_hand] spawned: {hand.num_joints} joints, {hand.num_bodies} bodies")
q = hand.data.default_joint_pos.clone()
maxv = 0.0
for s in range(40):
    hand.set_joint_position_target(q); hand.write_data_to_sim(); sim.step(); hand.update(1/120.0)
    maxv = max(maxv, float(hand.data.joint_vel[0].abs().max()))
print(f"[build_left_hand] 40-step max|vel| = {maxv:.3f}  "
      f"({'STABLE' if maxv < 5 else 'UNSTABLE -> mirror needs PhysX-safe handling'})")
app.close()
