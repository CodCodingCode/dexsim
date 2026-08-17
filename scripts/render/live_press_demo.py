"""Livestream the REAL env pressing a piano key, on a loop.

Builds the actual PianoEnv (working contacts, key springs, actuator gains) with
Isaac's WebRTC livestream enabled, then cycles the measured winning press from
scripts/smoke/press_env_test.py: left middle finger, rail +0.04, curl 0.8 --
which depresses key 33 to ~0.021 rad, past the 0.012 sounding threshold.

  source env51.sh && PUBLIC_IP=<tailscale-ip> \
      python -u scripts/render/live_press_demo.py --livestream 1
"""
from __future__ import annotations

import argparse
import math
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--period", type=float, default=2.0, help="seconds per press cycle")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.headless = True
app = AppLauncher(a).app

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
CURL = 0.8   # extra rad on the middle finger at full press

cfg = PianoEnvCfg()
cfg.scene.num_envs = 1
env = gym.make("Dexsim-Piano-Bimanual-v0", cfg=cfg, render_mode=None)
env.reset()
le = env.unwrapped
names = list(le.left_robot.data.joint_names)
sim_dt = le.sim.get_physics_dt()

base = le.left_robot.data.default_joint_pos.clone()
for jn, v in RELAX.items():
    base[0, names.index(jn)] = v
mf = [names.index(f"robot0_MFJ{j}") for j in (2, 1, 0)]
right_home = le.right_robot.data.default_joint_pos.clone()

print("[press_demo] cycling: left MIDDLE finger presses key 33 every "
      f"{a.period:.1f}s (sound threshold {abs(KEY_SOUND_ANGLE)} rad)", flush=True)

t = 0.0
step = 0
was_sounding = False
while app.is_running():
    # smooth press/release: 0 -> CURL -> 0 over one period
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
    app.update()
    t += sim_dt
    step += 1
    kq = le.piano.data.joint_pos[0]
    sounding = bool((kq <= KEY_SOUND_ANGLE).any())
    if sounding != was_sounding:
        mx = float(kq.abs().max()); which = int(kq.abs().argmax())
        state = "DOWN" if sounding else "up"
        print(f"[press_demo] t={t:7.1f}s key {which} {state} (|angle|={mx:.4f})", flush=True)
        was_sounding = sounding

env.close()
app.close()
