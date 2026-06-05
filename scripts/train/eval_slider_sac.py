"""Eval a skrl-SAC slider checkpoint: deterministic rollout, report key-press F1."""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--midi", default="data/midi/twinkle.mid")
parser.add_argument("--songs_npz", default=None)
parser.add_argument("--max_songs", type=int, default=0)
parser.add_argument("--song_offset", type=int, default=0)
parser.add_argument("--steps", type=int, default=480)
parser.add_argument("--goal_lookahead", type=int, default=5)
parser.add_argument("--slider_stiffness", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
import torch.nn as nn
import gymnasium as gym
from skrl.models.torch import Model, GaussianMixin
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import dexsim.tasks  # noqa: F401
from dexsim.tasks.piano import PianoEnvCfg

cfg = PianoEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.use_slider = True
cfg.slider_stiffness = args.slider_stiffness
cfg.goal_lookahead = args.goal_lookahead
cfg.obs_goal_sdf = False
cfg.__post_init__()
if args.songs_npz:
    cfg.songs_npz = args.songs_npz; cfg.max_songs = args.max_songs; cfg.song_offset = args.song_offset
else:
    cfg.midi_path = args.midi
cfg.__post_init__()

env = gym.make("Dexsim-Piano-Bimanual-v0", cfg=cfg, render_mode=None)
env = SkrlVecEnvWrapper(env)
device = env.device


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space=observation_space,
                       action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True,
                               min_log_std=-5.0, max_log_std=2.0)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role=""):
        return torch.tanh(self.net(inputs.get("observations"))), {"log_std": self.log_std_parameter}


policy = Policy(env.observation_space, env.action_space, device).to(device)
ckpt = torch.load(args.checkpoint, map_location=device)
sd = ckpt["policy"] if isinstance(ckpt, dict) and "policy" in ckpt else ckpt
try:
    policy.load_state_dict(sd)
except Exception:
    policy.migrate(state_dict=sd)   # skrl checkpoint wrapper
policy.eval()

obs, _ = env.reset()
tp = fp = fn = 0
uenv = env.unwrapped
for t in range(args.steps):
    with torch.no_grad():
        o = obs["policy"] if isinstance(obs, dict) else obs
        mean, _ = policy.compute({"observations": o})   # deterministic mean action
    obs, rew, term, trunc, info = env.step(mean)
    snd = uenv.key_sounding
    goal = uenv._goal_now().bool()
    tp += int((snd & goal).sum()); fp += int((snd & ~goal).sum()); fn += int((~snd & goal).sum())

eps = 1e-9
rec = tp/(tp+fn+eps); prec = tp/(tp+fp+eps); f1 = 2*rec*prec/(rec+prec+eps)
print(f"\n===== SAC SLIDER EVAL =====")
print(f"F1={f1:.3f}  recall={rec:.3f}  precision={prec:.3f}  (tp={tp} fp={fp} fn={fn})")
print(f"===========================\n")
env.close()
app.close()
