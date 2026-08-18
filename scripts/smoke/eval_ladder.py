"""Deterministic eval of a ladder checkpoint: same env cfg as training, mean
action (no exploration noise), reports recall/precision/F1 over N episodes.

  source env51.sh && python -u scripts/smoke/eval_ladder.py --headless \
      --checkpoint logs/rsl_rl/piano_bimanual/ladder1_onekey/model_XXXX.pt
"""
from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--steps", type=int, default=900, help="song is 889 steps")
p.add_argument("--midi", default="data/midi/one_key.mid")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.headless = True
app = AppLauncher(a).app

import torch
import gymnasium as gym
import dexsim.tasks  # noqa: F401
from dexsim.tasks.piano import PianoEnvCfg
from dexsim.tasks.piano.agents.rsl_rl_ppo_cfg import PianoPPORunnerCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# mirror the ladder1 training cfg exactly
cfg = PianoEnvCfg()
cfg.scene.num_envs = 64
cfg.midi_path = a.midi
cfg.fold_to_reach = False
cfg.solo_left_middle = True
cfg.key_strike_vel = 0.25
cfg.f1_weight = 2.0

env = gym.make("Dexsim-Piano-Bimanual-v0", cfg=cfg, render_mode=None)
wrapped = RslRlVecEnvWrapper(env)
le = env.unwrapped

ad = PianoPPORunnerCfg().to_dict()
runner = OnPolicyRunner(wrapped, ad, log_dir=None, device=le.device)
runner.load(a.checkpoint)
policy = runner.get_inference_policy(device=le.device)   # deterministic mean action

_o = wrapped.get_observations()
obs = _o[0] if isinstance(_o, tuple) else _o

tp = fp = fn = 0
for _ in range(a.steps):
    with torch.inference_mode():
        act = policy(obs)
        r = wrapped.step(act)
        obs = r[0]
    goal = le._goal_now()                          # (E,88)
    snd = le.key_sounding                          # (E,88) bool
    g = goal > 0.5
    tp += int((g & snd).sum()); fp += int((~g & snd).sum()); fn += int((g & ~snd).sum())

rec = tp / max(1, tp + fn)
prec = tp / max(1, tp + fp)
f1 = 2 * rec * prec / max(1e-9, rec + prec)
print(f"[eval] {a.steps} steps x 64 envs, checkpoint={a.checkpoint}")
print(f"[eval] DETERMINISTIC recall={rec:.3f} precision={prec:.3f} F1={f1:.3f}")
print(f"[eval] (training-time metrics with noise were ~0.15 recall)")
env.close()
app.close()
