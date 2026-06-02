"""Integration smoke test for the bimanual piano env: construct it, reset, step
random actions, and print obs/reward/done shapes. No training, no policy --
this just proves the whole env (2 arms + 88-key piano + MIDI goal) builds and
steps on GPU with the right tensor shapes.

  python scripts/piano_env_smoke.py --headless --num_envs 4
"""

from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=40)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

import dexsim.tasks  # noqa: F401
from dexsim.tasks.piano import PianoEnvCfg

TASK = "Dexsim-Piano-Bimanual-v0"


def main():
    cfg = PianoEnvCfg()
    cfg.scene.num_envs = args.num_envs
    print(f"[piano_smoke] action_space={cfg.action_space} obs_space={cfg.observation_space}")

    env = gym.make(TASK, cfg=cfg, render_mode=None)
    obs, _ = env.reset()
    print("[piano_smoke] reset OK")
    print(f"  obs['policy'] shape: {tuple(obs['policy'].shape)}")

    le = env.unwrapped
    print(f"  left_robot DOFs : {le.left_robot.num_joints}")
    print(f"  right_robot DOFs: {le.right_robot.num_joints}")
    print(f"  piano key joints: {le.piano.num_joints}")

    rew_sum = torch.zeros(args.num_envs, device=le.device)
    for i in range(args.steps):
        action = 2.0 * torch.rand(args.num_envs, cfg.action_space, device=le.device) - 1.0
        obs, rew, term, trunc, _ = env.step(action)
        rew_sum += rew
        if i == 0:
            print(f"  step0: rew shape {tuple(rew.shape)}, "
                  f"term {tuple(term.shape)}, trunc {tuple(trunc.shape)}")

    print(f"[piano_smoke] stepped {args.steps} steps. mean return = {rew_sum.mean().item():.3f}")
    print("===== PIANO ENV SMOKE OK =====")
    env.close()


main()
simulation_app.close()
