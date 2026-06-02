"""Phase 1 of filming: roll out the trained policy on the song and record the
arm + piano joint trajectories to an npz (fast, no rendering). Phase 2
(render_rollout.py) replays this with a camera to make the video.

  python scripts/record_rollout.py --headless --checkpoint results/easy_song_policy.pt
"""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="results/easy_song_policy.pt")
p.add_argument("--midi", default="data/midi/easy.mid")
p.add_argument("--out", default="logs/rollout.npz")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app

import numpy as np, torch, gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import dexsim.tasks  # noqa
from dexsim.tasks.piano import PianoEnvCfg
from dexsim.tasks.piano.agents.rsl_rl_ppo_cfg import PianoPPORunnerCfg

cfg = PianoEnvCfg(); cfg.scene.num_envs = 1; cfg.midi_path = a.midi
env = gym.make("Dexsim-Piano-Bimanual-v0", cfg=cfg, render_mode=None)
le = env.unwrapped
wrapped = RslRlVecEnvWrapper(env)
_ad = PianoPPORunnerCfg().to_dict(); _ad.setdefault("policy", {})["noise_std_type"] = "log"
runner = OnPolicyRunner(wrapped, _ad, log_dir=None, device=le.device)
runner.load(a.checkpoint)
policy = runner.get_inference_policy(device=le.device)

obs, _ = wrapped.get_observations()
L, R, K = [], [], []
for _ in range(le.song_len):
    with torch.inference_mode():
        obs, _, _, _ = wrapped.step(policy(obs))
    L.append(le.left_robot.data.joint_pos[0].cpu().numpy())
    R.append(le.right_robot.data.joint_pos[0].cpu().numpy())
    K.append(le.piano.data.joint_pos[0].cpu().numpy())
np.savez(a.out, left=np.array(L), right=np.array(R), keys=np.array(K),
         joint_names_left=le.left_robot.data.joint_names,
         piano_pos=np.array(cfg.piano_pos),
         left_base=np.array(cfg.left_base_pos), right_base=np.array(cfg.right_base_pos),
         control_dt=cfg.control_dt)
print(f"[record_rollout] saved {len(L)} frames -> {a.out}")
app.close()
