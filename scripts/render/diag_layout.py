"""Dump the scene geometry so we can reposition the piano: spawn the piano + both
robots at their real cfg placement and write to JSON the keyboard's world bbox/center,
the robot base positions, and the ready-pose palm positions. Used to compute the
180-deg + move-back reposition so the arms play from the player side.
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
piano = Articulation(cfg.piano_cfg.replace(prim_path="/World/P"))
left = Articulation(cfg.left_robot_cfg.replace(prim_path="/World/L"))
right = Articulation(cfg.right_robot_cfg.replace(prim_path="/World/R"))
sim.reset()
for _ in range(10):
    sim.step()

kb = piano.data.body_pos_w[0].cpu()                       # (Nbodies,3) all key/body positions
lpalm = left.data.body_pos_w[0, left.find_bodies("robot0_palm")[0][0]].cpu()
rpalm = right.data.body_pos_w[0, right.find_bodies("robot0_palm")[0][0]].cpu()
def bbox(t):
    return {"min": [round(float(t[:,i].min()),3) for i in range(3)],
            "max": [round(float(t[:,i].max()),3) for i in range(3)],
            "center": [round(float(t[:,i].mean()),3) for i in range(3)]}
res = {
    "piano_pos_cfg": list(cfg.piano_pos),
    "keyboard_bodies_bbox": bbox(kb),
    "left_base": [round(v,3) for v in cfg.left_robot_cfg.init_state.pos],
    "right_base": [round(v,3) for v in cfg.right_robot_cfg.init_state.pos],
    "left_palm_ready": [round(float(v),3) for v in lpalm.tolist()],
    "right_palm_ready": [round(float(v),3) for v in rpalm.tolist()],
    "note": "piano local +X = toward player (press side); +Y = keys low->high; robots reach +X",
}
with open("logs/layout_result.json", "w") as f:
    json.dump(res, f, indent=2)
print("[diag] wrote logs/layout_result.json", flush=True)
print(json.dumps(res, indent=2), flush=True)
app.close()
