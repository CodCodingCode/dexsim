"""Spawn the UR10e+Shadow at its ready pose and MEASURE which way the fingers
point: print each fingertip's position minus the palm position. Negative z =
tips below palm = fingers point DOWN (correct for piano). Positive z = inverted.
Also prints palm vs keyboard-top heights so we know if the hand is above the keys.
"""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args([]); a.headless = True
app = AppLauncher(a).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext, SimulationCfg
from dexsim.tasks.piano.piano_env_cfg import PianoEnvCfg
from dexsim.piano import FINGERTIP_BODIES

cfg = PianoEnvCfg()
sim = SimulationContext(SimulationCfg(dt=1/120.0, device="cuda"))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
left = Articulation(cfg.left_robot_cfg.replace(prim_path="/World/L"))
piano = Articulation(cfg.piano_cfg.replace(prim_path="/World/P"))
sim.reset()
for _ in range(10):
    sim.step()

names = left.data.joint_names
palm_id = left.find_bodies("robot0_palm")[0][0]
palm_p = left.data.body_pos_w[0, palm_id].cpu()
palm_q = left.data.body_quat_w[0, palm_id].cpu()
print(f"\nREADY-POSE arm joints:")
for n in [x for x in names if "robot0_" not in x]:
    print(f"   {n:<20}{float(left.data.joint_pos[0, names.index(n)]):+.3f}")
print(f"\npalm world pos  : {[round(v,3) for v in palm_p.tolist()]}")
print(f"palm world quat : {[round(v,3) for v in palm_q.tolist()]}")
kb_z = float(piano.data.body_pos_w[0, :, 2].max().cpu())
print(f"keyboard top z ~: {kb_z:.3f}   palm z: {float(palm_p[2]):.3f}   (palm above keys by {float(palm_p[2])-kb_z:+.3f} m)")
print(f"\nfingertip - palm  (z<0 => fingers point DOWN = correct):")
zs = []
for fb in FINGERTIP_BODIES:
    fid = left.find_bodies(fb)[0][0]
    fp = left.data.body_pos_w[0, fid].cpu()
    d = (fp - palm_p)
    zs.append(float(d[2]))
    print(f"   {fb:<22} dx={float(d[0]):+.3f} dy={float(d[1]):+.3f} dz={float(d[2]):+.3f}")
avgz = sum(zs)/len(zs)
verdict = 'FINGERS DOWN (ok)' if avgz < -0.02 else 'INVERTED / not pointing down (BAD)'
print(f"\n>>> mean fingertip dz = {avgz:+.3f}  ->  {verdict}")

# write results to a file BEFORE app.close() (whose teardown often hangs and eats
# buffered stdout), so the measurement always survives.
import json
res = {
    "palm_pos": [round(v, 4) for v in palm_p.tolist()],
    "palm_quat": [round(v, 4) for v in palm_q.tolist()],
    "keyboard_top_z": round(kb_z, 4),
    "palm_above_keys": round(float(palm_p[2]) - kb_z, 4),
    "fingertip_minus_palm": {fb: [round(float((left.data.body_pos_w[0, left.find_bodies(fb)[0][0]].cpu() - palm_p)[k]), 4) for k in range(3)] for fb in FINGERTIP_BODIES},
    "mean_fingertip_dz": round(avgz, 4),
    "verdict": verdict,
}
with open("logs/orient_result.json", "w") as f:
    json.dump(res, f, indent=2)
print("[diag] wrote logs/orient_result.json", flush=True)
app.close()
