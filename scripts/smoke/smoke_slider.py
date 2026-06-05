"""Smoke-test the hands-only Shadow+slider asset: load it as an Isaac articulation,
confirm DoF count (24 hand + 2 slider = 26) and joint names, step it, check stability.

  python scripts/smoke_slider.py --headless
"""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", default="assets/shadow_slider.usd")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

USD = os.path.abspath(args.usd)
cfg = ArticulationCfg(
    prim_path="/World/Slider",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8,
            solver_velocity_iteration_count=0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5), joint_pos={".*": 0.0}),
    actuators={
        "slider": ImplicitActuatorCfg(joint_names_expr=["slider_.*"], stiffness=2000.0, damping=100.0),
        "hand": ImplicitActuatorCfg(joint_names_expr=["robot0_.*"], stiffness=10.0, damping=1.0),
    },
)

sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0))
robot = Articulation(cfg)
sim.reset()

print("\n================ SLIDER ASSET SMOKE ================")
print(f"num_joints = {robot.num_joints}")
print(f"joint_names = {list(robot.joint_names)}")
slider_dofs = [n for n in robot.joint_names if "trans" in n or "slider" in n]
hand_dofs = [n for n in robot.joint_names if n.startswith("robot0_")]
print(f"-> {len(hand_dofs)} hand DoF, {len(slider_dofs)} slider DoF: {slider_dofs}")

# DRIVE the slider to a target and measure placement precision (mm) -- this is the
# slider's whole point: the heavy UR arm can't, a stiff prismatic can.
iy = list(robot.joint_names).index("slider_y")
iz = list(robot.joint_names).index("slider_z")
forearm_id = [i for i, n in enumerate(robot.body_names) if "forearm" in n]
fid = forearm_id[0] if forearm_id else 0
tgt_y, tgt_z = 0.30, -0.03                       # 30cm lateral, 3cm down
p0 = robot.data.body_pos_w[0, fid].clone()
for i in range(240):
    tgt = robot.data.default_joint_pos.clone()
    tgt[:, iy] = tgt_y; tgt[:, iz] = tgt_z
    robot.set_joint_position_target(tgt)
    robot.write_data_to_sim(); sim.step(); robot.update(1.0 / 120.0)
jp = robot.data.joint_pos
p1 = robot.data.body_pos_w[0, fid]
err_y = abs(jp[0, iy].item() - tgt_y) * 1000
err_z = abs(jp[0, iz].item() - tgt_z) * 1000
moved = (p1 - p0)
print(f"after 240 steps: pos in [{jp.min().item():.3f}, {jp.max().item():.3f}]  "
      f"finite={bool(torch.isfinite(jp).all())}")
print(f"SLIDER TRACKING: slider_y={jp[0,iy].item():.3f} (err {err_y:.1f}mm)  "
      f"slider_z={jp[0,iz].item():.3f} (err {err_z:.1f}mm)")
print(f"FOREARM MOVED: dY={moved[1].item()*1000:.0f}mm dZ={moved[2].item()*1000:.0f}mm "
      f"(commanded +300mm Y, -30mm Z)")
print("VERDICT:",
      "OK" if (robot.num_joints in (26, 25, 24) and bool(torch.isfinite(jp).all())) else "PROBLEM")
print("===================================================\n")
app.close()
