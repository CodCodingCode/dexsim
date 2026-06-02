"""Print FULL prim paths (not just body names) for the wrist link, hand palm,
and a fingertip of each spawned robot, plus their world positions and the key
extents. This is what we need to rigidly parent a wrist camera to the hand.

  python scripts/inspect_links.py
"""

from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from pxr import UsdGeom  # noqa: E402
import omni.usd  # noqa: E402
import dexsim.tasks  # noqa: F401,E402
from dexsim.tasks.piano import PianoEnvCfg  # noqa: E402

cfg = PianoEnvCfg()
cfg.scene.num_envs = 1
env = gym.make("Dexsim-Piano-Bimanual-v0", cfg=cfg, render_mode=None)
le = env.unwrapped
env.reset()
for _ in range(60):
    env.step(torch.zeros(1, cfg.action_space, device=le.device))

stage = omni.usd.get_context().get_stage()

WANT = ("wrist_3_link", "robot0_palm", "robot0_forearm",
        "robot0_thdistal", "robot0_ffdistal")

for root in ("/World/envs/env_0/LeftRobot", "/World/envs/env_0/RightRobot"):
    print(f"\n===== prims under {root} matching wrist/palm/fingertip =====")
    rootprim = stage.GetPrimAtPath(root)
    for prim in stage.Traverse():
        name = prim.GetName()
        path = prim.GetPath().pathString
        if path.startswith(root) and name in WANT:
            print(f"   {name:18s} {path}")

# world positions from the articulation data
for tag, art in (("LEFT", le.left_robot), ("RIGHT", le.right_robot)):
    p = art.data.body_pos_w[0].cpu().numpy()
    names = art.data.body_names
    print(f"\n-- {tag} body world positions (key bodies) --")
    for i, n in enumerate(names):
        if any(k in n for k in ("wrist_3", "palm", "forearm", "thdistal", "ffdistal", "mfdistal")):
            print(f"     {n:24s} ({p[i,0]:+.3f},{p[i,1]:+.3f},{p[i,2]:+.3f})")

kp = le.piano.data.body_pos_w[0].cpu().numpy()
kn = le.piano.data.body_names
keys = [(n, kp[i]) for i, n in enumerate(kn) if n.startswith("key_")]
if keys:
    import numpy as np
    ks = np.array([k[1] for k in keys])
    print(f"\n== KEYS: {len(keys)}  x[{ks[:,0].min():.3f},{ks[:,0].max():.3f}] "
          f"y[{ks[:,1].min():.3f},{ks[:,1].max():.3f}] z[{ks[:,2].min():.3f},{ks[:,2].max():.3f}]")

env.close()
app.close()
