"""Verify the RECORDED rollout actually puts the palms over the keys (near side).
Loads an npz of joint trajectories, drives the robots to several PLAYING frames,
and reports each palm's world X vs the keyboard X-range. Palm X ~0.55 (in the key
range) = hands over the keys; palm X ~0.07 (past the far edge) = still reaching over.
Writes logs/palm_check.json (teardown-safe).
"""
from __future__ import annotations
import argparse, json
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--rollout", default="logs/rollout_front2.npz")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app

import numpy as np, torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext, SimulationCfg
from dexsim.tasks.piano.piano_env_cfg import PianoEnvCfg

data = np.load(a.rollout, allow_pickle=True)
L, R = data["left"], data["right"]
cfg = PianoEnvCfg()
sim = SimulationContext(SimulationCfg(dt=1/120.0, device="cuda"))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
piano = Articulation(cfg.piano_cfg.replace(prim_path="/World/P"))
left = Articulation(cfg.left_robot_cfg.replace(prim_path="/World/L"))
right = Articulation(cfg.right_robot_cfg.replace(prim_path="/World/R"))
sim.reset()
lpalm = left.find_bodies("robot0_palm")[0][0]; rpalm = right.find_bodies("robot0_palm")[0][0]

def drive(art, q):
    qt = torch.tensor(q, dtype=torch.float32, device="cuda").unsqueeze(0)
    art.write_joint_state_to_sim(qt, torch.zeros_like(qt)); art.write_data_to_sim()

n = L.shape[0]
frames = [int(n*f) for f in (0.3, 0.5, 0.7)]
kb = piano.data.body_pos_w[0]
kbx = (round(float(kb[:,0].min()),3), round(float(kb[:,0].max()),3))
rows = []
for t in frames:
    drive(left, L[t]); drive(right, R[t])
    for _ in range(3): sim.step()
    lp = left.data.body_pos_w[0, lpalm].cpu(); rp = right.data.body_pos_w[0, rpalm].cpu()
    rows.append({"frame": t,
                 "left_palm": [round(float(v),3) for v in lp.tolist()],
                 "right_palm": [round(float(v),3) for v in rp.tolist()]})
res = {"keyboard_x_range": kbx, "frames": rows,
       "verdict": "over keys" if all(kbx[0]-0.1 <= r["left_palm"][0] <= kbx[1]+0.15 for r in rows) else "NOT over keys (still reaching past)"}
json.dump(res, open("logs/palm_check.json","w"), indent=2)
print(json.dumps(res, indent=2), flush=True)
app.close()
