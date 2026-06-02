"""Map each fingertip to the key directly under it + the vertical gap, so we can
(a) raise the keyboard until fingers rest on keys and (b) build an easy song that
only uses the keys under the resting fingers (no hand movement needed)."""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app
import torch, numpy as np, gymnasium as gym
import dexsim.tasks  # noqa
from dexsim.tasks.piano import PianoEnvCfg

cfg = PianoEnvCfg(); cfg.scene.num_envs = 1
env = gym.make("Dexsim-Piano-Bimanual-v0", cfg=cfg, render_mode=None); le = env.unwrapped
env.reset()
for _ in range(80): env.step(torch.zeros(1, cfg.action_space, device=le.device))

# key world positions (top surface)
kp = le.piano.data.body_pos_w[0].cpu().numpy(); kn = le.piano.data.body_names
key_idx = {int(n.split('_')[1]): kp[i] for i, n in enumerate(kn) if n.startswith("key_")}
keys = np.array([key_idx[i] for i in range(88)])         # (88,3) centers
key_top_z = keys[:, 2].max()

def report(tag, robot):
    bn = robot.data.body_names; bp = robot.data.body_pos_w[0].cpu().numpy()
    print(f"\n== {tag} hand ==  (key top z ~= {key_top_z:.3f})")
    for f in ("ff","mf","rf","lf","th"):
        try: ti = bn.index(f"robot0_{f}distal")
        except ValueError: continue
        ft = bp[ti]
        d = np.linalg.norm(keys[:, :2] - ft[:2], axis=1)   # horizontal distance to each key
        nk = int(d.argmin()); midi = nk + 21
        print(f"   {f}distal at ({ft[0]:.3f},{ft[1]:.3f},{ft[2]:.3f})  "
              f"nearest key={nk} (MIDI {midi})  horiz_gap={d[nk]*1000:.0f}mm  "
              f"vert_gap={(ft[2]-key_top_z)*1000:+.0f}mm")

report("LEFT", le.left_robot)
report("RIGHT", le.right_robot)
env.close(); app.close()
