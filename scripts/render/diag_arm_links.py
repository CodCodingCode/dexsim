"""Report the UR10e link world positions at candidate ready poses so we can pick an
ELBOW-UP seed (elbow/forearm ABOVE the wrist) that puts the wrist over the keys.
Tests a list of candidate joint poses; for each, prints upper_arm / forearm / wrist /
palm Z and an elbow-up verdict. Writes logs/arm_links.json (teardown-safe).
"""
from __future__ import annotations
import argparse, json
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

names = left.data.joint_names
body_names = left.body_names
print("BODY NAMES:", body_names, flush=True)
def bid(substr):
    for i, n in enumerate(body_names):
        if substr in n: return i
    return None
ids = {k: bid(k) for k in ["upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3", "robot0_palm"]}

ARM = ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"]
def jset(pose):
    q = left.data.default_joint_pos.clone()
    for jn, v in pose.items():
        if jn in names: q[0, names.index(jn)] = v
    left.write_joint_state_to_sim(q, torch.zeros_like(q)); left.write_data_to_sim()

# candidate poses (shoulder_pan kept ~current; vary lift/elbow/wrist_1 for elbow-up branch)
cands = {
  "current":   dict(shoulder_pan_joint=-0.275, shoulder_lift_joint=-0.525, elbow_joint=1.150, wrist_1_joint=-1.275, wrist_2_joint=-1.570, wrist_3_joint=0.0),
  "elbowup_A": dict(shoulder_pan_joint=-0.275, shoulder_lift_joint=-1.95, elbow_joint=-1.55, wrist_1_joint=-1.20, wrist_2_joint=-1.570, wrist_3_joint=0.0),
  "elbowup_B": dict(shoulder_pan_joint=-0.275, shoulder_lift_joint=-1.60, elbow_joint=-1.90, wrist_1_joint=-0.80, wrist_2_joint=-1.570, wrist_3_joint=0.0),
  "elbowup_C": dict(shoulder_pan_joint=-0.275, shoulder_lift_joint=-2.30, elbow_joint=-1.30, wrist_1_joint=-1.10, wrist_2_joint=-1.570, wrist_3_joint=0.0),
}
res = {"body_names": list(body_names), "candidates": {}}
for nm, pose in cands.items():
    jset(pose)
    for _ in range(3): sim.step()
    rec = {}
    for k, i in ids.items():
        if i is not None:
            pz = left.data.body_pos_w[0, i].cpu().tolist()
            rec[k] = [round(float(v),3) for v in pz]
    fa_z = rec.get("forearm",[0,0,0])[2]; w_z = rec.get("wrist_3",[0,0,0])[2]; palm = rec.get("robot0_palm",[0,0,0])
    rec["elbow_up"] = bool(fa_z > w_z + 0.02)
    rec["palm"] = palm
    res["candidates"][nm] = rec
json.dump(res, open("logs/arm_links.json","w"), indent=2)
print(json.dumps(res["candidates"], indent=2), flush=True)
app.close()
