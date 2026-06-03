"""Train the bimanual piano policy (rsl_rl PPO).

  python scripts/train_piano.py --headless --num_envs 1024 --midi data/midi/twinkle.mid
  python scripts/train_piano.py --headless --num_envs 2048 --max_iterations 5000
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train bimanual piano policy.")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--midi", default=None, help="path to the song .mid (default: cfg's)")
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--bc_init", default=None, help="BC warm-start checkpoint (scripts/bc_pretrain.py)")
parser.add_argument("--freeze_hands", action="store_true", help="curriculum phase 1: drive arms only (hands frozen)")
parser.add_argument("--freeze_arms", action="store_true", help="fixed-hands mode: drive fingers only (arms held)")
parser.add_argument("--reference", default=None, help="explicit q_ref .npz (e.g. an RP1M reference); overrides the default per-song file")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.io import dump_yaml

import dexsim.tasks  # noqa: F401
from dexsim.tasks.piano import PianoEnvCfg
from dexsim.tasks.piano.agents.rsl_rl_ppo_cfg import PianoPPORunnerCfg

TASK = "Dexsim-Piano-Bimanual-v0"


def main():
    env_cfg = PianoEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    if args.midi:
        env_cfg.midi_path = args.midi
    env_cfg.freeze_hands = args.freeze_hands
    if args.freeze_arms:
        env_cfg.freeze_arms = True
    if args.reference:
        env_cfg.reference_path = args.reference

    agent_cfg = PianoPPORunnerCfg()
    agent_cfg.seed = args.seed
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations

    log_dir = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, f"seed{args.seed}")
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    # log-parameterize the action-noise std so it can't go negative (the scalar
    # std drifted < 0 and crashed PPO at iter 117 with "normal expects std>=0").
    agent_dict = agent_cfg.to_dict()
    agent_dict.setdefault("policy", {})["noise_std_type"] = "log"
    runner = OnPolicyRunner(env, agent_dict, log_dir=log_dir, device=agent_cfg.device)
    if args.bc_init:
        import torch
        ckpt = torch.load(args.bc_init, map_location=agent_cfg.device)
        # rsl_rl stores the actor-critic on alg.policy (older code used .actor_critic)
        net = getattr(runner.alg, "policy", None) or runner.alg.actor_critic
        net.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"[train_piano] warm-started actor-critic from {args.bc_init}")
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    print(f"[train_piano] task={TASK} num_envs={args.num_envs} "
          f"song={env_cfg.midi_path} log_dir={log_dir}")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


main()
simulation_app.close()
