"""Grid-search UR10e joint angles for a NEAR-SIDE, ELBOW-UP, FINGERS-DOWN seed:
palm over the keys (~0.60, -0.30, 0.87 for the left hand) with the flange (wrist_3)
ON or above the near side (flange_x >= palm_x) and ABOVE the palm (flange_z > palm_z)
so the hand points straight down and the forearm never crosses past the keyboard.
Reports the best few configs. Writes logs/seed_search.json.
"""
from __future__ import annotations
import argparse, json, itertools
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args([]); a.headless = True
app = AppLauncher(a).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext, SimulationCfg
from dexsim.tasks.piano.piano_env_cfg import PianoEnvCfg

cfg = PianoEnvCfg()
sim = SimulationContext(SimulationCfg(dt=1/120.0, device="cuda"))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
left = Articulation(cfg.left_robot_cfg.replace(prim_path="/World/L"))
sim.reset()
names = left.data.joint_names; body_names = left.body_names
def bid(s):
    for i, n in enumerate(body_names):
        if s in n: return i
fl_id = bid("wrist_3_link"); pa_id = bid("robot0_palm"); fa_id = bid("forearm_link")

TARGET = torch.tensor([0.60, -0.30, 0.87])
def setq(pan, lift, elb, w1):
    q = left.data.default_joint_pos.clone()
    for jn, v in {"shoulder_pan_joint":pan,"shoulder_lift_joint":lift,"elbow_joint":elb,
                  "wrist_1_joint":w1,"wrist_2_joint":-1.570,"wrist_3_joint":0.0}.items():
        q[0, names.index(jn)] = v
    left.write_joint_state_to_sim(q, torch.zeros_like(q)); left.write_data_to_sim()

import math
results = []
PI = math.pi
pans = [-0.275, -0.275+PI, -0.275-PI, 0.0, PI]      # incl. the +-pi flip (Problem 1)
lifts = [-2.8,-2.4,-2.0,-1.6,-1.2,-0.8,-0.4]
elbs  = [-2.6,-2.0,-1.4,-0.8, 0.8,1.4,2.0,2.6]      # BOTH elbow branches (Problem 2)
w1s   = [-3.0,-2.6,-2.2,-1.8,-1.4,-1.0,-0.6,-0.2,0.2]
for pan, lift, elb, w1 in itertools.product(pans, lifts, elbs, w1s):
    setq(pan, lift, elb, w1)
    sim.step()
    pa = left.data.body_pos_w[0, pa_id].cpu(); fl = left.data.body_pos_w[0, fl_id].cpu(); fa = left.data.body_pos_w[0, fa_id].cpu()
    dist = float(torch.linalg.norm(pa - TARGET))
    ee_down = float(torch.linalg.norm((fl[:2] - pa[:2])))    # horiz flange-palm offset; small = EE straight DOWN
    down = bool(fl[2] > pa[2] + 0.04)             # flange above palm -> hand points down
    fa_above = bool(fa[2] > fl[2] + 0.02)         # forearm comes DOWN to the EE (elbow above wrist)
    if dist < 0.13 and ee_down < 0.10 and down and fa_above:
        results.append({"pan":round(pan,3),"lift":lift,"elb":elb,"w1":w1,"dist":round(dist,3),
                        "ee_down_off":round(ee_down,3),
                        "palm":[round(float(v),3) for v in pa.tolist()],
                        "flange":[round(float(v),3) for v in fl.tolist()]})
results.sort(key=lambda r: r["dist"] + r["ee_down_off"])
out = {"target": TARGET.tolist(), "n_found": len(results), "best": results[:8]}
json.dump(out, open("logs/seed_search.json","w"), indent=2)
print(json.dumps(out, indent=2), flush=True)
app.close()
