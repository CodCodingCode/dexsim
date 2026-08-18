"""Ceiling check: what recall does a PERFECT-KNOWLEDGE scripted presser get?

Plays the one-key song with the geometrically CORRECT hand+finger: the target
key is read from the song, the finger is whichever non-thumb fingertip sits
nearest that key in world Y, and the rail is swept so the tip lands on the key.
(The old version hardcoded left-MF for key 33 -- but key 33 lives at world
y=-0.14, the RIGHT hand's lane; the left hand cannot reach it at any rail.)

No learning -- this measures the maximum any policy could plausibly score
under the current key/strike physics.

  source env51.sh && python -u scripts/smoke/scripted_ceiling.py --headless
"""
from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=900)
p.add_argument("--lead", type=int, default=3, help="start pressing this many steps early")
p.add_argument("--strike_vel", type=float, default=0.25)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.headless = True
app = AppLauncher(a).app

import torch
import gymnasium as gym
import dexsim.tasks  # noqa: F401
from dexsim.tasks.piano import PianoEnvCfg

cfg = PianoEnvCfg()
cfg.scene.num_envs = 1
cfg.midi_path = "data/midi/one_key.mid"
cfg.fold_to_reach = False
cfg.key_strike_vel = a.strike_vel

env = gym.make("Dexsim-Piano-Bimanual-v0", cfg=cfg, render_mode=None)
env.reset()
le = env.unwrapped
sim_dt = le.sim.get_physics_dt()
sub = max(1, int(round(cfg.control_dt / sim_dt)))

# ---- pick target key from the song, then the nearest non-thumb fingertip ----
TARGET_KEY = int(le.goal_padded[0].sum(0).argmax())
key_y = float(le.piano.data.body_pos_w[0, le.key_body_ids[TARGET_KEY], 1])
tips = le._fingertips_world()[0]                       # (10,3): L th,ff,mf,rf,lf then R
FINGER_NAMES = ["TH", "FF", "MF", "RF", "LF"] * 2
cand = [i for i in range(10) if FINGER_NAMES[i] != "TH"]
fi = min(cand, key=lambda i: abs(float(tips[i, 1]) - key_y))
use_left = fi < 5
rob = le.left_robot if use_left else le.right_robot
oth = le.right_robot if use_left else le.left_robot
fpfx = FINGER_NAMES[fi]
names = list(rob.data.joint_names)
fj = {j: names.index(f"robot0_{fpfx}J{j}") for j in (2, 1, 0)}
rail_id = names.index("railJoint")
rail_pred = key_y - float(tips[fi, 1])                 # rail moves the hand in world Y
print(f"[pick] key {TARGET_KEY} at y={key_y:+.3f} -> {'left' if use_left else 'right'} "
      f"{fpfx} (tip y={float(tips[fi,1]):+.3f}), predicted rail {rail_pred:+.3f}", flush=True)

base = rob.data.default_joint_pos.clone()
oth_home = oth.data.default_joint_pos.clone()

def make_press(rail, curl=0.8):
    t = base.clone()
    t[0, rail_id] = rail
    t[0, fj[2]] += curl
    t[0, fj[1]] += curl * 0.6
    t[0, fj[0]] += curl * 0.4
    return t

def make_hover(rail):
    t = base.clone()
    t[0, rail_id] = rail
    return t

def run_hold(target, n):
    for _ in range(n):
        rob.set_joint_position_target(target)
        oth.set_joint_position_target(oth_home)
        rob.write_data_to_sim(); oth.write_data_to_sim()
        for _ in range(sub):
            le.sim.step()
        for art in (le.left_robot, le.right_robot, le.piano):
            art.update(sim_dt)

# ---- phase 1: sweep (rail, curl) around the predicted rail offset ----
best, best_travel = (max(-0.12, min(0.12, rail_pred)), 0.9), 0.0
sweep = sorted({round(max(-0.12, min(0.12, rail_pred + d)), 3)
                for d in (-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03)})
for rail in sweep:
    for curl in (0.7, 0.9, 1.1):
        env.reset()
        run_hold(make_hover(rail), 10)          # aim first so the curl is a clean strike
        run_hold(make_press(rail, curl), 30)
        kq = le.piano.data.joint_pos[0]
        kt = float(kq[TARGET_KEY].abs())
        deep = int(kq.abs().argmax())
        if kt > 0.001 or float(kq.abs().max()) > 0.005:
            print(f"[aim] rail={rail:+.3f} curl={curl:.1f}: key{TARGET_KEY}={kt:.4f} "
                  f"(deepest key {deep} at {float(kq.abs().max()):.4f})", flush=True)
        if kt > best_travel:
            best_travel, best = kt, (rail, curl)
print(f"[aim] best (rail, curl)={best} key{TARGET_KEY} travel {best_travel:.4f}", flush=True)
press = make_press(*best)
base_aimed = make_hover(best[0])                 # stay aimed between presses
env.reset()
run_hold(base_aimed, 20)                         # settle aimed before the song starts

L = int(a.lead)
tp = fp = fn = 0
for t in range(a.steps):
    # goal lookahead: press if the goal is on now or within the next L steps
    idx = (le.song_step[0:1].unsqueeze(1)
           + torch.arange(L + 1, device=le.device).unsqueeze(0)).clamp(
               max=le.goal_padded.shape[1] - 1)
    up = le.goal_padded[le.song_id[0:1].unsqueeze(1), idx]      # (1,L+1,88)
    want = bool(up.sum() > 0)
    target = press if want else base_aimed
    rob.set_joint_position_target(target)
    oth.set_joint_position_target(oth_home)
    rob.write_data_to_sim()
    oth.write_data_to_sim()
    for _ in range(sub):
        le.sim.step()
    for art in (le.left_robot, le.right_robot, le.piano):
        art.update(sim_dt)
    le._key_pressed_fraction()                                  # updates key_sounding
    if t < 100:
        _a = float(le.piano.data.joint_pos[0, TARGET_KEY])
        _v = float(le.piano.data.joint_vel[0, TARGET_KEY])
        _g = bool(le._goal_now()[0, TARGET_KEY] > 0.5)
        _snd = bool(le.key_sounding[0, TARGET_KEY])
        _cj = float(rob.data.joint_pos[0, fj[2]])
        print(f"[dbg] t={t:3d} want={int(want)} goal={int(_g)} {fpfx}J2={_cj:+.2f} "
              f"key{TARGET_KEY} angle={_a:+.5f} vel={_v:+.3f} sounding={int(_snd)}", flush=True)
    goal = le._goal_now()[0] > 0.5
    snd = le.key_sounding[0]
    tp += int((goal & snd).sum()); fp += int((~goal & snd).sum()); fn += int((goal & ~snd).sum())
    le.song_step[:] = torch.minimum(le.song_step + 1,
                                    le.song_lens[le.song_id] - 1)

rec = tp / max(1, tp + fn)
prec = tp / max(1, tp + fp)
f1 = 2 * rec * prec / max(1e-9, rec + prec)
print(f"[ceiling] scripted perfect-knowledge presser over {a.steps} steps (lead={L}):")
print(f"[ceiling] recall={rec:.3f} precision={prec:.3f} F1={f1:.3f}")
env.close()
app.close()
