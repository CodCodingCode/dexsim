"""Decisive one-boot diagnostic for the combined arm explosion.

Prints, BEFORE stepping physics:
  * base_link world z   (did init_state.pos apply, or is the base pinned at 0?)
  * distance(wrist_3_link, robot0_forearm) at spawn (is the mount coincident, or
    is the fixed joint about to snap them together?)
Then steps a few times reporting max arm joint velocity.
"""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app

import numpy as np, torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext, SimulationCfg
from dexsim.assets import UR10E_SHADOW_CFG

sim = SimulationContext(SimulationCfg(dt=1/120.0, device=a.device))
sim_utils.GroundPlaneCfg().func("/g", sim_utils.GroundPlaneCfg())
cfg = UR10E_SHADOW_CFG.replace(prim_path="/World/Robot")
cfg.init_state.pos = (0.0, 0.0, 0.75)
robot = Articulation(cfg)
sim.reset()
robot.update(0.0)

names = robot.data.body_names
pos = robot.data.body_pos_w[0].cpu().numpy()
def gp(sub):
    for i, n in enumerate(names):
        if n == sub: return pos[i]
    for i, n in enumerate(names):
        if sub in n: return pos[i]
    return None

base = gp("base_link"); wrist = gp("wrist_3_link"); fore = gp("robot0_forearm"); palm = gp("robot0_palm")
print("\n=== MOUNT DIAGNOSTIC (pre-step) ===")
print(f"  base_link world : {None if base is None else base.round(3)}  (want z~0.75)")
print(f"  wrist_3_link    : {None if wrist is None else wrist.round(3)}")
print(f"  robot0_forearm  : {None if fore is None else fore.round(3)}")
print(f"  robot0_palm     : {None if palm is None else palm.round(3)}")
if wrist is not None and fore is not None:
    print(f"  >>> dist(wrist, forearm) = {np.linalg.norm(wrist-fore):.4f} m  "
          f"(want ~0; large => fixed joint will SNAP -> explosion)")

arm_i = [i for i, n in enumerate(robot.data.joint_names) if "joint" in n and "robot0" not in n]
q = robot.data.default_joint_pos.clone()
for s in range(20):
    robot.set_joint_position_target(q); robot.write_data_to_sim(); sim.step(); robot.update(sim.get_physics_dt())
    if s in (0, 1, 5, 19):
        v = robot.data.joint_vel[0].cpu().numpy()
        print(f"  step {s:2d} max|arm vel| = {max(abs(float(v[i])) for i in arm_i):.2f}")
app.close()
