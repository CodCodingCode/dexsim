"""Print actual spawned world positions (arms, hands, keyboard) so placement can
be tuned from real numbers, not guesses."""

from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); args.headless = True
app = AppLauncher(args).app

import torch, numpy as np
import gymnasium as gym
import dexsim.tasks  # noqa
from dexsim.tasks.piano import PianoEnvCfg

cfg = PianoEnvCfg(); cfg.scene.num_envs = 1
env = gym.make("Dexsim-Piano-Bimanual-v0", cfg=cfg, render_mode=None); le = env.unwrapped
env.reset()
for _ in range(60): env.step(torch.zeros(1, cfg.action_space, device=le.device))

def span(art, names_filter=None):
    p = art.data.body_pos_w[0].cpu().numpy()
    names = art.data.body_names
    return p, names

# diagnostic: are the arm joints actually holding the commanded ready pose?
for tag, art in (("LEFT", le.left_robot), ("RIGHT", le.right_robot)):
    jn = art.data.joint_names
    jp = art.data.joint_pos[0].cpu().numpy()
    jd = art.data.default_joint_pos[0].cpu().numpy()
    arm = [(n, jp[i], jd[i]) for i, n in enumerate(jn) if "joint" in n and "robot0" not in n]
    print(f"\n-- {tag} ARM joints (actual vs default) --")
    for n, a, d in arm:
        print(f"     {n:22s} actual={a:+.3f}  default={d:+.3f}")

for tag, art in (("LEFT", le.left_robot), ("RIGHT", le.right_robot), ("PIANO", le.piano)):
    p, names = span(art)
    print(f"\n== {tag} ({len(names)} bodies) ==")
    print(f"   x[{p[:,0].min():.3f},{p[:,0].max():.3f}] "
          f"y[{p[:,1].min():.3f},{p[:,1].max():.3f}] "
          f"z[{p[:,2].min():.3f},{p[:,2].max():.3f}]")
    # show palm + fingertips for robots
    for i, n in enumerate(names):
        if any(k in n for k in ("palm", "distal", "base_link", "forearm")):
            print(f"     {n:24s} ({p[i,0]:.3f},{p[i,1]:.3f},{p[i,2]:.3f})")

# keyboard key extent specifically
kp = le.piano.data.body_pos_w[0].cpu().numpy()
kn = le.piano.data.body_names
keys = [(n, kp[i]) for i, n in enumerate(kn) if n.startswith("key_")]
if keys:
    ks = np.array([k[1] for k in keys])
    print(f"\n== KEYS: {len(keys)} keys, x[{ks[:,0].min():.3f},{ks[:,0].max():.3f}] "
          f"y[{ks[:,1].min():.3f},{ks[:,1].max():.3f}] z[{ks[:,2].min():.3f},{ks[:,2].max():.3f}]")
env.close(); app.close()
