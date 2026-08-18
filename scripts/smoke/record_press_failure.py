"""Record the A/B that explains the ladder plateau: the SAME finger press at
rail 0.00 (key barely moves -- the failing case the RL rung lived in) vs rail
+0.04 (key sounds -- the validated press). Key angles are PhysX truth from the
env; replay to a mesh .rrd with the render server's `rerun --rollout` job.

  source env51.sh && python -u scripts/smoke/record_press_failure.py --headless
"""
from __future__ import annotations

import argparse
import math
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--seconds_per_phase", type=float, default=8.0)
p.add_argument("--period", type=float, default=2.0)
p.add_argument("--out", default="logs/press_failure_ab.npz")
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

cfg = PianoEnvCfg()
cfg.scene.num_envs = 1
env = gym.make("Dexsim-Piano-Bimanual-v0", cfg=cfg, render_mode=None)
env.reset()
le = env.unwrapped
names = list(le.left_robot.data.joint_names)
sim_dt = le.sim.get_physics_dt()
rec_every = max(1, int(round(cfg.control_dt / sim_dt)))

mf = {j: names.index(f"robot0_MFJ{j}") for j in (2, 1, 0)}
rail = names.index("railJoint")
base = le.left_robot.data.default_joint_pos.clone()
right_home = le.right_robot.data.default_joint_pos.clone()
CURL = 0.9

L, R, K = [], [], []
phase_marks = []
for phase_i, rail_pos in enumerate((0.0, 0.04)):
    env.reset()
    phase_marks.append(len(L))
    t = 0.0
    n = int(a.seconds_per_phase / sim_dt)
    for i in range(n):
        c = CURL * (0.5 - 0.5 * math.cos(2.0 * math.pi * t / a.period))
        target = base.clone()
        target[0, rail] = rail_pos
        target[0, mf[2]] += c
        target[0, mf[1]] += c * 0.6
        target[0, mf[0]] += c * 0.4
        le.left_robot.set_joint_position_target(target)
        le.right_robot.set_joint_position_target(right_home)
        le.left_robot.write_data_to_sim()
        le.right_robot.write_data_to_sim()
        le.sim.step()
        for art in (le.left_robot, le.right_robot, le.piano):
            art.update(sim_dt)
        if i % rec_every == 0:
            L.append(le.left_robot.data.joint_pos[0].cpu().numpy().copy())
            R.append(le.right_robot.data.joint_pos[0].cpu().numpy().copy())
            K.append(le.piano.data.joint_pos[0].cpu().numpy().copy())
        t += sim_dt

K_arr = np.array(K)
half = phase_marks[1]
np.savez(a.out, left=np.array(L), right=np.array(R), keys=K_arr,
         control_dt=cfg.control_dt)
for label, seg in (("rail 0.00 (FAILING)", K_arr[:half]), ("rail 0.04 (WORKING)", K_arr[half:])):
    deep = float(np.abs(seg).max())
    snd = int((seg <= KEY_SOUND_ANGLE).any(axis=1).sum())
    print(f"[ab] {label}: deepest key {deep:.4f} rad, sounding frames {snd}/{len(seg)}")
print(f"[ab] {len(L)} frames -> {a.out} (phase B starts at frame {half})")
env.close()
app.close()
