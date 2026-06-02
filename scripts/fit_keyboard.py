"""Fit the keyboard to the hand: shift the piano so a cluster of keys sits
directly under the (well-placed) LEFT hand's fingertips, ~1.5cm below them.
Prints the new piano_pos and the MIDI note under each finger -> the easy song."""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--hand", default="left")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app
import torch, numpy as np, gymnasium as gym
import dexsim.tasks  # noqa
from dexsim.tasks.piano import PianoEnvCfg

cfg = PianoEnvCfg(); cfg.scene.num_envs = 1
env = gym.make("Dexsim-Piano-Bimanual-v0", cfg=cfg, render_mode=None); le = env.unwrapped
env.reset()
for _ in range(100): env.step(torch.zeros(1, cfg.action_space, device=le.device))

robot = le.left_robot if a.hand == "left" else le.right_robot
bn = robot.data.body_names; bp = robot.data.body_pos_w[0].cpu().numpy()
tips = {f: bp[bn.index(f"robot0_{f}distal")] for f in ("ff","mf","rf","lf","th")}
mid = tips["mf"]                                   # anchor on the middle finger

# current key world positions (top centers) + index
kp = le.piano.data.body_pos_w[0].cpu().numpy(); kn = le.piano.data.body_names
kw = {int(n.split('_')[1]): kp[i] for i, n in enumerate(kn) if n.startswith("key_")}
key_top_z = max(v[2] for v in kw.values())
# white key currently nearest the middle finger (xy)
whites = [i for i in range(88) if i not in (1,4,6,9,11) and ((i+21)%12 in (0,2,4,5,7,9,11))]
def nearest_white(xy):
    return min(whites, key=lambda i: np.hypot(kw[i][0]-xy[0], kw[i][1]-xy[1]))
k0 = nearest_white(mid[:2]); P0 = kw[k0]
shift = np.array([mid[0]-P0[0], mid[1]-P0[1], (mid[2]-0.015) - key_top_z])
new_pos = np.array(cfg.piano_pos) + shift
print(f"[fit] anchor middle finger {mid.round(3)} on white key {k0}")
print(f"[fit] SHIFT = {shift.round(3)}   NEW piano_pos = {tuple(new_pos.round(3))}")

# after the shift, which key is under each finger?
print("[fit] keys under fingers after shift:")
song = []
for f, t in tips.items():
    # shifted key world = old + shift; nearest white to the finger
    best = min(whites, key=lambda i: np.hypot((kw[i][0]+shift[0])-t[0], (kw[i][1]+shift[1])-t[1]))
    midi = best + 21; song.append(midi)
    gap = np.hypot((kw[best][0]+shift[0])-t[0], (kw[best][1]+shift[1])-t[1])
    print(f"   {f}: white key {best} (MIDI {midi})  horiz_gap={gap*1000:.0f}mm")
uniq = sorted(set(song))
print(f"[fit] EASY SONG keys (unique, MIDI): {uniq}")
env.close(); app.close()
