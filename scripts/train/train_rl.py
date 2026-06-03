"""Train the in-hand cube reorientation policy (RL from scratch, no dataset).

Thin wrapper over Isaac Lab's rsl_rl PPO runner, defaulting to dexsim's
``Dexsim-Reorient-Cube-Shadow-v0`` task (Shadow Hand on our own config). This is
the turnkey "just hit run" path -- domain randomization and reward come from the
upstream Shadow reorientation env.

Usage:
  python scripts/train_rl.py --headless --num_envs 8192
  python scripts/train_rl.py --task Isaac-Repose-Cube-Shadow-Direct-v0 --headless
  python scripts/train_rl.py --headless --max_iterations 2000 --save_interval 100
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RL train: Shadow Hand reorientation.")
parser.add_argument("--task", default="Dexsim-Reorient-Cube-Shadow-v0")
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--save_interval", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()

# hand the remaining (non-CLI) args to Hydra: it re-parses sys.argv, so strip
# everything argparse already consumed or it errors on --task/--headless/etc.
import sys
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import dexsim.tasks  # noqa: F401  -> registers Dexsim-* envs

import os


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def run(env_cfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    # CLI overrides
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    agent_cfg.seed = args.seed
    agent_cfg.save_interval = args.save_interval
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations

    log_root = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    os.makedirs(log_root, exist_ok=True)
    log_dir = os.path.join(log_root, f"seed{args.seed}")

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    print(f"[train_rl] task={args.task}  num_envs={args.num_envs}  log_dir={log_dir}")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


run()
simulation_app.close()
