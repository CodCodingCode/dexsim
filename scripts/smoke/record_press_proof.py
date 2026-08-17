"""Record the scripted key-press cycle from the REAL env to a rollout npz.

Same motion as live_press_demo.py (left middle finger, rail +0.04, curl 0.8 --
sounds key 33). Records left/right/keys joint trajectories at 20 Hz so the
render server's `rerun --rollout` job can replay it into a mesh .rrd: the key
angles in the file are PhysX contact truth from the env.

  source env51.sh && python -u scripts/smoke/record_press_proof.py --headless
"""
from __future__ import annotations

import argparse
import math
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--seconds", type=float, default=8.0)
p.add_argument("--period", type=float, default=2.0)
p.add_argument("--out", default="logs/press_proof.npz")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.headless = True
app = AppLauncher(a).app

import numpy as np
import torch
import gymnasium as gym
import dexsim.tasks  # noqa: F401
from dexsim.tasks.piano import PianoEnvCfg
from dexsim.assets import KEY_SOUND_ANGLE

RELAX = {
    "railJoint": 0.04,
    **{f"robot0_{f}J3": 0.0 for f in ("FF", "MF", "RF", "LF")},
    **{f"robot0_{f}J2": 0.35 for f in ("FF", "MF", "RF", "LF")},
    **{f"robot0_{f}J1": 0.30 for f in ("FF", "MF", "RF", "LF")},
    **{f"robot0_{f}J0": 0.25 for f in ("FF", "MF", "RF", "LF")},
    "robot0_LFJ4": 0.05,
    "robot0_THJ4": 0.3, "robot0_THJ3": 0.3, "robot0_THJ2": 0.0,
    "robot0_THJ1": 0.0, "robot0_THJ0": -0.3,
}
CURL = 0.8

cfg = PianoEnvCfg()
cfg.scene.num_envs = 1
env = gym.make("Dexsim-Piano-Bimanual-v0", cfg=cfg, render_mode=None)
env.reset()
le = env.unwrapped
names = list(le.left_robot.data.joint_names)
sim_dt = le.sim.get_physics_dt()
record_every = max(1, int(round(cfg.control_dt / sim_dt)))     # 20 Hz frames

base = le.left_robot.data.default_joint_pos.clone()
for jn, v in RELAX.items():
    base[0, names.index(jn)] = v
mf = [names.index(f"robot0_MFJ{j}") for j in (2, 1, 0)]
right_home = le.right_robot.data.default_joint_pos.clone()

L, R, K = [], [], []
t = 0.0
n_steps = int(a.seconds / sim_dt)
for i in range(n_steps):
    phase = 0.5 - 0.5 * math.cos(2.0 * math.pi * t / a.period)
    target = base.clone()
    target[0, mf[0]] += CURL * phase
    target[0, mf[1]] += CURL * 0.6 * phase
    target[0, mf[2]] += CURL * 0.4 * phase
    le.left_robot.set_joint_position_target(target)
    le.right_robot.set_joint_position_target(right_home)
    le.left_robot.write_data_to_sim()
    le.right_robot.write_data_to_sim()
    le.sim.step()
    for art in (le.left_robot, le.right_robot, le.piano):
        art.update(sim_dt)
    if i % record_every == 0:
        L.append(le.left_robot.data.joint_pos[0].cpu().numpy().copy())
        R.append(le.right_robot.data.joint_pos[0].cpu().numpy().copy())
        K.append(le.piano.data.joint_pos[0].cpu().numpy().copy())
    t += sim_dt

K_arr = np.array(K)
np.savez(a.out, left=np.array(L), right=np.array(R), keys=K_arr,
         control_dt=cfg.control_dt)
deep = float(np.abs(K_arr).max())
frames_sounding = int((K_arr <= KEY_SOUND_ANGLE).any(axis=1).sum())
print(f"[record] {len(L)} frames -> {a.out}")
print(f"[record] deepest key angle {deep:.4f} rad (threshold {abs(KEY_SOUND_ANGLE)}); "
      f"{frames_sounding}/{len(L)} frames have a SOUNDING key")
env.close()
app.close()
