"""Rename the imported left hand's prims lh_* -> robot0_* so it shares the right
hand's naming convention (env uses robot0_*distal fingertips + robot0_.* actuators).
Uses an Sdf namespace batch edit, which fixes up joint body0/body1 targets.
Writes assets/shadow_hand_left_r0.usd and verifies it spawns with the new names.

  python scripts/rename_left_hand.py --headless
"""
from __future__ import annotations
import argparse, shutil
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app

from pxr import Usd, Sdf
from dexsim import ASSETS_DIR
SRC = str(ASSETS_DIR / "shadow_hand_left.usd")
DST = str(ASSETS_DIR / "shadow_hand_left_r0.usd")
shutil.copy(SRC, DST)

stage = Usd.Stage.Open(DST)
layer = stage.GetRootLayer()

# collect prims to rename, DEEPEST first so parent renames don't invalidate
# child paths within a single batch (batch edit also orders, but this is safe).
targets = []
for prim in stage.Traverse():
    if prim.GetName().startswith("lh_"):
        targets.append(prim.GetPath())
targets.sort(key=lambda p: len(p.pathString), reverse=True)

edit = Sdf.BatchNamespaceEdit()
for path in targets:
    new_name = "robot0_" + path.name[3:]
    edit.Add(path, path.GetParentPath().AppendChild(new_name))

ok = layer.CanApply(edit)
print(f"[rename] {len(targets)} prims lh_*->robot0_*  CanApply={ok}")
applied = layer.Apply(edit)
print(f"[rename] applied={applied}")
stage.GetRootLayer().Save()

# verify spawn + names
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import SimulationContext, SimulationCfg
sim = SimulationContext(SimulationCfg(dt=1/120.0, device=a.device))
sim_utils.GroundPlaneCfg().func("/g", sim_utils.GroundPlaneCfg())
hand = Articulation(ArticulationCfg(prim_path="/World/H",
    spawn=sim_utils.UsdFileCfg(usd_path=DST,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,0.5)),
    actuators={"f": ImplicitActuatorCfg(joint_names_expr=["robot0_.*"], stiffness=3.0, damping=0.1)}))
sim.reset()
ft = [b for b in hand.data.body_names if "distal" in b]
print(f"[rename] spawned {hand.num_joints} joints, {hand.num_bodies} bodies")
print(f"[rename] fingertips now: {ft}")
print(f"[rename] robot0_ joints matched by actuator: {sum('robot0_' in n for n in hand.data.joint_names)}")
print(f"[rename] OK -> {DST}")
app.close()
