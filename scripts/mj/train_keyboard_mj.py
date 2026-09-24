"""Train the bimanual TYPING policy on the MacBook keyboard scene (rsl_rl PPO).

Same recipe as train_piano_mj.py (same PPO hyper-parameters, CPU sim, policy
on the GPU when there is one). Metrics to watch: play/accuracy (correct /
all registered keystrokes), play/done_frac (text completed), play/cps
(characters per second at episode end).

  .venv/bin/python scripts/mj/train_keyboard_mj.py --num_envs 64 --workers 8 --device cpu
  .venv/bin/python scripts/mj/train_keyboard_mj.py --text "hello world" --right_only
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))

parser = argparse.ArgumentParser(description="Train the typing policy (MuJoCo).")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--workers", type=int, default=1, help=">1: shard envs across processes")
parser.add_argument("--threads", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=2000)
parser.add_argument("--save_interval", type=int, default=50)
parser.add_argument("--num_steps_per_env", type=int, default=32)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", default="cuda")
parser.add_argument("--text", default=None, help="fixed text (default: random corpus sentence)")
parser.add_argument("--corpus_file", default=None, help="one sentence per line")
parser.add_argument("--max_chars", type=int, default=None)
parser.add_argument("--time_per_char", type=float, default=None, help="episode budget s/char")
parser.add_argument("--right_only", action="store_true", help="freeze the left hand")
parser.add_argument("--left_only", action="store_true", help="freeze the right hand")
parser.add_argument("--key_reward", type=float, default=None)
parser.add_argument("--stray_penalty", type=float, default=None)
parser.add_argument("--hold_penalty", type=float, default=None)
parser.add_argument("--reach_weight", type=float, default=None)
parser.add_argument("--press_weight", type=float, default=None)
parser.add_argument("--time_penalty", type=float, default=None)
parser.add_argument("--hand_action_scale", type=float, default=None)
parser.add_argument("--gantry_xy_action_scale", type=float, default=None)
parser.add_argument("--gantry_z_action_scale", type=float, default=None)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--entropy_coef", type=float, default=None)
parser.add_argument("--init_noise", type=float, default=None)
parser.add_argument("--desired_kl", type=float, default=None)
parser.add_argument("--resume_from", default=None)
parser.add_argument("--logger", default="tensorboard", choices=["tensorboard", "wandb"])
parser.add_argument("--wandb_project", default="dexsim-keyboard-mj")
parser.add_argument("--tag", default=None)
args = parser.parse_args()

import torch  # noqa: E402

from dexsim.tasks.keyboard_mj import (KeyboardMjEnvCfg, KeyboardMjVecEnv,   # noqa: E402
                                      KeyboardMjSubprocVecEnv)
from dexsim.tasks.piano_mj.vec_env import make_rsl_rl_env  # noqa: E402
from dexsim.tasks.piano_mj.ppo_cfg import piano_ppo_cfg  # noqa: E402


def build_env_cfg() -> KeyboardMjEnvCfg:
    cfg = KeyboardMjEnvCfg()
    cfg.seed = args.seed
    if args.text:
        cfg.text = args.text
    if args.corpus_file:
        cfg.corpus = [l.rstrip("\n") for l in open(args.corpus_file) if l.strip()]
    if args.right_only:
        cfg.freeze_left_hand = True
    if args.left_only:
        cfg.freeze_right_hand = True
    for name in ("max_chars", "key_reward", "stray_penalty", "hold_penalty", "reach_weight",
                 "press_weight", "time_penalty", "hand_action_scale",
                 "gantry_xy_action_scale", "gantry_z_action_scale"):
        v = getattr(args, name)
        if v is not None:
            setattr(cfg, name, v)
    if args.time_per_char is not None:
        cfg.time_per_char_s = args.time_per_char
    cfg.__post_init__()
    return cfg


def build_train_cfg() -> dict:
    return piano_ppo_cfg(
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        num_steps_per_env=args.num_steps_per_env,
        save_interval=args.save_interval,
        logger=args.logger,
        wandb_project=args.wandb_project,
        run_name=args.tag or "",
        learning_rate=args.lr,
        entropy_coef=args.entropy_coef,
        desired_kl=args.desired_kl,
        init_std=args.init_noise,
    )


def main():
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu"
    if device != args.device:
        print(f"[train] CUDA unavailable -> policy on {device}")
    env_cfg = build_env_cfg()
    if args.workers > 1:
        venv = KeyboardMjSubprocVecEnv(env_cfg, num_envs=args.num_envs, workers=args.workers)
    else:
        venv = KeyboardMjVecEnv(env_cfg, num_envs=args.num_envs, threads=args.threads)
    env = make_rsl_rl_env(venv, device=device)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{stamp}" + (f"_{args.tag}" if args.tag else "")
    log_dir = ROOT / "logs" / "keyboard_mj" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] {args.num_envs} envs ({'%d workers' % args.workers if args.workers > 1 else 'threaded'}), "
          f"policy on {device}, obs {env_cfg.observation_space} act {env_cfg.action_space}, "
          f"logs -> {log_dir}")

    from rsl_rl.runners import OnPolicyRunner
    runner = OnPolicyRunner(env, build_train_cfg(), log_dir=str(log_dir), device=device)
    if args.resume_from:
        print(f"[train] resuming from {args.resume_from}")
        runner.load(args.resume_from)
    runner.learn(num_learning_iterations=args.max_iterations)
    runner.save(os.path.join(str(log_dir), "model_final.pt"))
    venv.close()
    print(f"[train] done -> {log_dir}")


if __name__ == "__main__":
    main()
