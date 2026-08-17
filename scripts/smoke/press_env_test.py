"""Find a scripted finger press that SOUNDS a key in the real env (PhysX truth).

Sweeps a handful of simple pressing postures (relaxed hand, one finger curled
hard, various rail offsets), settles each, and reports per-attempt key travel.
The goal is one attempt with |key angle| >= |KEY_SOUND_ANGLE| -- proof the
actuator fix + collisions + key springs make presses physically possible.

  source env51.sh && python -u scripts/smoke/press_env_test.py --headless
"""
from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.headless = True
app = AppLauncher(a).app

import torch
import gymnasium as gym
import dexsim.tasks  # noqa: F401
from dexsim.tasks.piano import PianoEnvCfg
from dexsim.assets import KEY_SOUND_ANGLE

# Relaxed base posture: fingers gently curved, nothing self-colliding.
RELAX = {
    "railJoint": 0.0,
    **{f"robot0_{f}J3": 0.0 for f in ("FF", "MF", "RF", "LF")},
    **{f"robot0_{f}J2": 0.35 for f in ("FF", "MF", "RF", "LF")},
    **{f"robot0_{f}J1": 0.30 for f in ("FF", "MF", "RF", "LF")},
    **{f"robot0_{f}J0": 0.25 for f in ("FF", "MF", "RF", "LF")},
    "robot0_LFJ4": 0.05,
    "robot0_THJ4": 0.3, "robot0_THJ3": 0.3, "robot0_THJ2": 0.0,
    "robot0_THJ1": 0.0, "robot0_THJ0": -0.3,
}

def press_with(finger: str, rail: float, curl: float) -> dict:
    """RELAX + one finger curled down by `curl` extra rad, rail offset."""
    d = dict(RELAX)
    d["railJoint"] = rail
    d[f"robot0_{finger}J2"] = RELAX[f"robot0_{finger}J2"] + curl
    d[f"robot0_{finger}J1"] = RELAX[f"robot0_{finger}J1"] + curl * 0.6
    d[f"robot0_{finger}J0"] = RELAX[f"robot0_{finger}J0"] + curl * 0.4
    return d

ATTEMPTS = [
    ("MF rail=0.00 curl=0.6", press_with("MF", 0.00, 0.6)),
    ("MF rail=0.00 curl=0.9", press_with("MF", 0.00, 0.9)),
    ("FF rail=-0.04 curl=0.8", press_with("FF", -0.04, 0.8)),
    ("MF rail=+0.04 curl=0.8", press_with("MF", 0.04, 0.8)),
    ("RF rail=0.00 curl=0.8", press_with("RF", 0.00, 0.8)),
    ("MF rail=-0.06 curl=0.8", press_with("MF", -0.06, 0.8)),
]

cfg = PianoEnvCfg()
cfg.scene.num_envs = 1
env = gym.make("Dexsim-Piano-Bimanual-v0", cfg=cfg, render_mode=None)
le = env.unwrapped
names = list(le.left_robot.data.joint_names)
sim_dt = le.sim.get_physics_dt()
thr = abs(KEY_SOUND_ANGLE)
best = (None, 0.0)

for label, pose in ATTEMPTS:
    env.reset()
    target = le.left_robot.data.default_joint_pos.clone()
    for jn, v in pose.items():
        target[0, names.index(jn)] = v
    right_home = le.right_robot.data.default_joint_pos.clone()
    for _ in range(a.steps):
        le.left_robot.set_joint_position_target(target)
        le.right_robot.set_joint_position_target(right_home)
        le.left_robot.write_data_to_sim()
        le.right_robot.write_data_to_sim()
        le.sim.step()
        for art in (le.left_robot, le.right_robot, le.piano):
            art.update(sim_dt)
    kq = le.piano.data.joint_pos[0]
    mx = float(kq.abs().max())
    which = int(kq.abs().argmax())
    hit = "SOUNDS" if mx >= thr else ""
    print(f"[press] {label:26s} max_travel={mx:.5f} rad (key {which:2d}) "
          f"{'<'+'='*3+' '+hit if hit else ''}", flush=True)
    if mx > best[1]:
        best = (label, mx)

print(f"[press] best: {best[0]} at {best[1]:.5f} rad (need {thr})")
print("===== PRESS " + ("REGISTERED =====" if best[1] >= thr else "DID NOT SOUND ====="))
env.close()
app.close()
