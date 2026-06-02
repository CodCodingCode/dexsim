"""Fast arm-pose sweep: boot Isaac ONCE, spawn one arm + the piano, then try
many candidate joint poses by re-writing joint state (no reboot per pose). For
each pose, report where the fingertips land vs the keyboard, ranked by how close
they get to a target key point. ~0.5s per pose after a single boot.

  python scripts/tune_pose.py --headless --side left
"""

from __future__ import annotations
import argparse, itertools
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--side", choices=["left", "right"], default="left")
parser.add_argument("--settle", type=int, default=45)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); args.headless = True
app = AppLauncher(args).app

import numpy as np, torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext, SimulationCfg
from dexsim.assets import UR10E_SHADOW_CFG, PIANO_CFG
from dexsim.tasks.piano.piano_env_cfg import PianoEnvCfg

cfg = PianoEnvCfg()
base = cfg.left_base_pos if args.side == "left" else cfg.right_base_pos
# target key point: this hand's half of the keyboard, just at key height
ty = -0.40 if args.side == "left" else 0.40
target = np.array([0.352, ty, 0.74])

sim = SimulationContext(SimulationCfg(dt=1/120.0, device=args.device))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))
piano = Articulation(PIANO_CFG.replace(prim_path="/World/Piano").replace(
    init_state=PIANO_CFG.init_state.replace(pos=cfg.piano_pos)))
robot = Articulation(UR10E_SHADOW_CFG.replace(prim_path="/World/Robot").replace(
    init_state=UR10E_SHADOW_CFG.init_state.replace(pos=base)))
sim.reset()

jn = robot.data.joint_names
arm_i = {n: i for i, n in enumerate(jn)}
tip_i = [i for i, n in enumerate(robot.data.body_names) if "distal" in n]

def set_pose(pan, lift, elb, w1, w2=-1.57, w3=0.0):
    q = robot.data.default_joint_pos.clone()
    for name, val in {"shoulder_pan_joint": pan, "shoulder_lift_joint": lift,
                      "elbow_joint": elb, "wrist_1_joint": w1,
                      "wrist_2_joint": w2, "wrist_3_joint": w3}.items():
        q[:, arm_i[name]] = val
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    for _ in range(args.settle):
        robot.set_joint_position_target(q); robot.write_data_to_sim(); sim.step()
        robot.update(sim.get_physics_dt())
    tips = robot.data.body_pos_w[0, tip_i].cpu().numpy()
    return tips.mean(0), tips

pan0 = -0.5 if args.side == "left" else 0.5
results = []
# elevated base (z~1.05) -> arm reaches forward AND down onto the keys
for lift, elb, w1 in itertools.product([-1.7, -1.3, -0.9, -0.5],
                                       [1.2, 1.6, 2.0, 2.4],
                                       [-2.2, -1.7, -1.2, -0.7]):
    c, tips = set_pose(pan0, lift, elb, w1)
    d = float(np.linalg.norm(c - target))
    results.append((d, lift, elb, w1, c))

results.sort()
print(f"\n=== {args.side} arm pose sweep (target {target}, base {base}) ===")
print(f"{'dist':>6} {'lift':>6} {'elbow':>6} {'wr1':>6}   fingertip_mean(x,y,z)")
for d, lift, elb, w1, c in results[:10]:
    print(f"{d:6.3f} {lift:6.2f} {elb:6.2f} {w1:6.2f}   ({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})")
print(f"\nBEST: lift={results[0][1]} elbow={results[0][2]} wrist_1={results[0][3]} "
      f"(fingertips {results[0][4].round(3)}, {results[0][0]:.3f} m from keys)")
app.close()
