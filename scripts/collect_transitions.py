"""Data firehose: roll out a policy (or random actions) across many Isaac envs and
log (obs, action, reward, next_obs, done) transitions to disk. This is the prior
dataset for an off-policy / RLPD-style DroQ trainer -- the "shit ton of PPO runs,
saved, then fed to SAC" idea. Each transition is a permanent fact about the env,
so even a half-trained / exploratory policy produces useful data (coverage > skill).

  # exploratory data from a checkpoint:
  python scripts/collect_transitions.py --headless --num_envs 256 --steps 400 \
         --checkpoint results/easy_song_policy.pt --out data/transitions/easy
  # pure-exploration random data (no checkpoint):
  python scripts/collect_transitions.py --headless --num_envs 256 --steps 400 --random
"""
from __future__ import annotations
import argparse, os
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--steps", type=int, default=400, help="control steps to collect")
p.add_argument("--checkpoint", default=None)
p.add_argument("--midi", default="data/midi/easy.mid")
p.add_argument("--random", action="store_true", help="uniform random actions (exploration prior)")
p.add_argument("--action_noise", type=float, default=0.1, help="extra Gaussian noise on policy actions for coverage")
p.add_argument("--out", default="data/transitions/easy")
p.add_argument("--shard_steps", type=int, default=100, help="flush a shard every N steps")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app

import numpy as np, torch, gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import dexsim.tasks  # noqa
from dexsim.tasks.piano import PianoEnvCfg
from dexsim.tasks.piano.agents.rsl_rl_ppo_cfg import PianoPPORunnerCfg

cfg = PianoEnvCfg(); cfg.scene.num_envs = a.num_envs; cfg.midi_path = a.midi
env = gym.make("Dexsim-Piano-Bimanual-v0", cfg=cfg, render_mode=None)
le = env.unwrapped
wrapped = RslRlVecEnvWrapper(env)
act_dim = env.action_space.shape[-1] if hasattr(env, "action_space") else le.cfg.action_space

policy = None
if not a.random:
    ckpt = a.checkpoint or "results/easy_song_policy.pt"
    _ad = PianoPPORunnerCfg().to_dict(); _ad.setdefault("policy", {})["noise_std_type"] = "log"
    runner = OnPolicyRunner(wrapped, _ad, log_dir=None, device=le.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=le.device)
    print(f"[collect] policy from {ckpt} (+{a.action_noise} action noise)")
else:
    print("[collect] RANDOM uniform actions")

os.makedirs(a.out, exist_ok=True)
obs, _ = wrapped.get_observations()
buf = {k: [] for k in ("obs", "act", "rew", "next_obs", "done")}
shard, total = 0, 0


def flush():
    global shard, buf
    if not buf["obs"]:
        return
    path = os.path.join(a.out, f"shard_{shard:04d}.npz")
    np.savez_compressed(
        path,
        obs=np.concatenate(buf["obs"]).astype(np.float16),
        act=np.concatenate(buf["act"]).astype(np.float16),
        rew=np.concatenate(buf["rew"]).astype(np.float32),
        next_obs=np.concatenate(buf["next_obs"]).astype(np.float16),
        done=np.concatenate(buf["done"]).astype(np.bool_),
    )
    print(f"[collect] wrote {path}  ({sum(x.shape[0] for x in buf['obs'])} transitions)", flush=True)
    shard += 1
    buf = {k: [] for k in buf}


for t in range(a.steps):
    o = obs.detach()
    with torch.inference_mode():
        if a.random:
            act = torch.empty((a.num_envs, act_dim), device=le.device).uniform_(-1, 1)
        else:
            act = policy(obs)
            if a.action_noise > 0:
                act = (act + a.action_noise * torch.randn_like(act)).clamp(-1, 1)
        next_obs, rew, done, _ = wrapped.step(act)
    buf["obs"].append(o.cpu().numpy())
    buf["act"].append(act.detach().cpu().numpy())
    buf["rew"].append(rew.detach().cpu().numpy())
    buf["next_obs"].append(next_obs.detach().cpu().numpy())
    buf["done"].append(done.detach().cpu().numpy())
    obs = next_obs
    total += a.num_envs
    if (t + 1) % a.shard_steps == 0:
        flush()
flush()
print(f"[collect] DONE: {total} transitions across {shard} shards -> {a.out}")
print(f"[collect] obs_dim={obs.shape[-1]} act_dim={act_dim}")
app.close()
