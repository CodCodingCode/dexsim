"""Roll out / visualize a trained reorientation policy.

Loads the latest (or a given) rsl_rl checkpoint for the reorientation task and
runs it. Use --headless --video to dump an mp4 on a server, or run with a
display to watch live.

Usage:
  python scripts/play_rl.py --num_envs 16
  python scripts/play_rl.py --headless --video --video_length 600
  python scripts/play_rl.py --checkpoint logs/rsl_rl/.../model_1500.pt
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained Shadow reorientation policy.")
parser.add_argument("--task", default="Dexsim-Reorient-Cube-Shadow-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--checkpoint", default=None, help="path to a .pt checkpoint")
parser.add_argument("--video", action="store_true", help="record a video")
parser.add_argument("--video_length", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import dexsim.tasks  # noqa: F401


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def run(env_cfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    env_cfg.scene.num_envs = args.num_envs

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)

    if args.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, "videos"),
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env)

    ckpt = args.checkpoint or get_checkpoint_path(
        os.path.join("logs", "rsl_rl", agent_cfg.experiment_name), ".*", "model_.*.pt"
    )
    print(f"[play_rl] loading checkpoint: {ckpt}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()
    steps = args.video_length if args.video else 1000
    for _ in range(steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
    env.close()


run()
simulation_app.close()
