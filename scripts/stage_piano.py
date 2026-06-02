"""Stage the bimanual piano scene and render a snapshot (NO training).

Spawns the two UR10e+Shadow arms + the 88-key piano with your song loaded,
lets physics settle, and saves an RGB image so you can eyeball the layout
(arm reach, hand-over-keys, base placement) before committing to training.

  python scripts/stage_piano.py --headless --midi data/midi/song.mid --out logs/stage.png
"""

from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--midi", default="data/midi/song.mid")
parser.add_argument("--out", default="logs/stage_piano.png")
parser.add_argument("--settle", type=int, default=60, help="physics steps to settle")
parser.add_argument("--eye", default="1.4,0.0,1.6", help="camera eye x,y,z")
parser.add_argument("--target", default="0.4,0.0,0.78", help="camera look-at x,y,z")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True   # needed for offscreen RGB

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import numpy as np
import gymnasium as gym

import dexsim.tasks  # noqa: F401
from dexsim.tasks.piano import PianoEnvCfg


def main():
    cfg = PianoEnvCfg()
    cfg.scene.num_envs = 1
    cfg.midi_path = args.midi

    env = gym.make("Dexsim-Piano-Bimanual-v0", cfg=cfg, render_mode="rgb_array")
    le = env.unwrapped

    eye = tuple(float(v) for v in args.eye.split(","))
    tgt = tuple(float(v) for v in args.target.split(","))
    le.sim.set_camera_view(eye=eye, target=tgt)

    env.reset()
    import torch
    zero = torch.zeros(1, cfg.action_space, device=le.device)
    for _ in range(args.settle):
        env.step(zero)

    frame = env.render()
    if frame is None:
        print("[stage_piano] render() returned None (renderer gave no frame).")
    else:
        frame = np.asarray(frame)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        try:
            from PIL import Image
            Image.fromarray(frame[..., :3].astype("uint8")).save(args.out)
        except Exception:
            np.save(args.out + ".npy", frame)
        print(f"[stage_piano] saved snapshot {frame.shape} -> {args.out}")

    # report the embodiment + song so the RL format is explicit
    print("\n===== RL FORMAT (Dexsim-Piano-Bimanual-v0) =====")
    print(f"  song            : {args.midi}  ({le.song_len} steps @ {1/cfg.control_dt:.0f}Hz)")
    print(f"  left arm DOFs   : {le.left_robot.num_joints}")
    print(f"  right arm DOFs  : {le.right_robot.num_joints}")
    print(f"  piano key joints: {le.piano.num_joints}")
    print(f"  ACTION space    : {cfg.action_space}  (60 = 2 arms x 30 joint targets)")
    print(f"  OBS space       : {cfg.observation_space}  (arms pos+vel + key angles + goal lookahead)")
    print(f"  REWARD          : + sound goal keys, - wrong keys, - energy")
    print("================================================\n")
    env.close()


main()
simulation_app.close()
