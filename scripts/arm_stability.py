"""Isolate the combined UR10e+Shadow arm: spawn ONE, fixed base, hold a pose,
and report joint pos/vel + applied actuator gains over time. No piano, no second
arm -- pure stability check of the combined articulation.

  python scripts/arm_stability.py --headless
"""

from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=120)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); args.headless = True
app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext, SimulationCfg
from dexsim.assets import UR10E_SHADOW_CFG

sim = SimulationContext(SimulationCfg(dt=1/120.0, device=args.device))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

cfg = UR10E_SHADOW_CFG.replace(prim_path="/World/Robot")
cfg.init_state.pos = (0.0, 0.0, 0.75)
cfg.init_state.joint_pos = {
    "shoulder_pan_joint": 0.0, "shoulder_lift_joint": -1.05, "elbow_joint": 1.30,
    "wrist_1_joint": -1.85, "wrist_2_joint": -1.57, "wrist_3_joint": 0.0, "robot0_.*": 0.0,
}
robot = Articulation(cfg)
sim.reset()

arm_idx = [i for i, n in enumerate(robot.data.joint_names) if "joint" in n and "robot0" not in n]
arm_names = [robot.data.joint_names[i] for i in arm_idx]
print("arm joints:", arm_names)
# applied gains
stiff = robot.data.joint_stiffness[0].cpu().numpy()
damp = robot.data.joint_damping[0].cpu().numpy()
print("applied stiffness (arm):", [round(float(stiff[i]),1) for i in arm_idx])
print("applied damping  (arm):", [round(float(damp[i]),1) for i in arm_idx])

target = robot.data.default_joint_pos.clone()
for s in range(args.steps):
    robot.set_joint_position_target(target)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.get_physics_dt())
    if s in (1, 10, 30, 60, args.steps - 1):
        jp = robot.data.joint_pos[0].cpu().numpy()
        jv = robot.data.joint_vel[0].cpu().numpy()
        maxv = max(abs(float(jv[i])) for i in arm_idx)
        pos = [round(float(jp[i]), 2) for i in arm_idx]
        print(f"  step {s:3d}: arm pos={pos}  max|vel|={maxv:.2f}")
app.close()
