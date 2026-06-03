"""Geometry diagnostic for the right-arm reach problem.

Loads the env (fixed piano), reads the WORLD positions of the keys each hand must
play, the arm base positions, and the fingertip positions at the ready pose, then
reports the reach gap. No rollout -- just one settle step -- so it's fast.

  python scripts/diag_reach.py --headless
"""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym
import dexsim.tasks  # noqa
from dexsim.tasks.piano import PianoEnvCfg

cfg = PianoEnvCfg()
cfg.scene.num_envs = 1
cfg.use_reference = False   # reset to the READY POSE (not the stale reference) so
#                             this actually tests the ready-pose change
env = gym.make("Dexsim-Piano-Bimanual-v0", cfg=cfg, render_mode=None)
le = env.unwrapped
env.reset()
for _ in range(5):
    le.sim.step(); le.scene.update(le.physics_dt)

origin = le.scene.env_origins[0].cpu().numpy()
key_top = le._key_top_world()[0].cpu().numpy() - origin       # (88,3) rel to env origin
tips = le._fingertips_world()[0].cpu().numpy() - origin       # (10,3)
lbase = np.array(cfg.left_base_pos)
rbase = np.array(cfg.right_base_pos)

# which keys does the song use, split by hand assignment
from dexsim.piano import load_song, plan_fingering
song = load_song(cfg.midi_path, control_dt=cfg.control_dt)
plan = plan_fingering(song.key_activation)
fk, fa = plan.finger_key, plan.finger_active
left_keys = np.unique(fk[:, :5][fa[:, :5]])
right_keys = np.unique(fk[:, 5:][fa[:, 5:]])

def yrange(keys):
    ys = key_top[keys, 1]
    return ys.min(), ys.max(), ys.mean()

print("\n================ REACH DIAGNOSTIC (fixed piano) ================")
print(f"key X (forward, all):   min={key_top[:,1].min():+.3f} .. used below")
print(f"key top Z (height):     mean={key_top[:,2].mean():.3f}")
print(f"full keyboard Y span:   {key_top[:,1].min():+.3f} .. {key_top[:,1].max():+.3f}")
print()
lmin, lmax, lmean = yrange(left_keys)
rmin, rmax, rmean = yrange(right_keys)
print(f"LEFT-hand keys  ({left_keys.min()}..{left_keys.max()}): Y {lmin:+.3f}..{lmax:+.3f} (mean {lmean:+.3f})")
print(f"RIGHT-hand keys ({right_keys.min()}..{right_keys.max()}): Y {rmin:+.3f}..{rmax:+.3f} (mean {rmean:+.3f})")
print()
print(f"LEFT  base Y={lbase[1]:+.3f}   (XZ {lbase[0]:+.3f},{lbase[2]:.3f})")
print(f"RIGHT base Y={rbase[1]:+.3f}   (XZ {rbase[0]:+.3f},{rbase[2]:.3f})")
print()
print(f"LEFT  fingertips Y at ready: {tips[:5,1].min():+.3f}..{tips[:5,1].max():+.3f}  X {tips[:5,0].mean():+.3f}  Z {tips[:5,2].mean():.3f}")
print(f"RIGHT fingertips Y at ready: {tips[5:,1].min():+.3f}..{tips[5:,1].max():+.3f}  X {tips[5:,0].mean():+.3f}  Z {tips[5:,2].mean():.3f}")
print()
print(f">>> RIGHT reach gap: hand sits at Y~{tips[5:,1].mean():+.3f}, needs to cover Y {rmin:+.3f}..{rmax:+.3f}")
print(f">>> i.e. shift right base by ~{rmean - tips[5:,1].mean():+.3f} m in Y to center it on its keys")
print("===============================================================\n")
app.close()
