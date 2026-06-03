"""Phase 1 of filming: roll out the policy (or the pure IK reference with --zero)
on the song and record arm + piano joint trajectories AND the per-step goal vs
sounding notes, to an npz. Phase 2 (render_rollout.py) replays it with a camera
and a note-comparison UI overlay.

  # pure reference arms (no policy needed) -- "are the arms positioned right?"
  python scripts/record_rollout.py --headless --zero --midi data/midi/song.mid
  # a trained policy
  python scripts/record_rollout.py --headless --checkpoint <model.pt>
"""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default=None)
p.add_argument("--zero", action="store_true", help="zero residual = pure IK reference")
p.add_argument("--midi", default="data/midi/song.mid")
p.add_argument("--out", default="logs/rollout.npz")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app

import numpy as np, torch, gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import dexsim.tasks  # noqa
from dexsim.tasks.piano import PianoEnvCfg
from dexsim.tasks.piano.agents.rsl_rl_ppo_cfg import PianoPPORunnerCfg

cfg = PianoEnvCfg(); cfg.scene.num_envs = 1; cfg.midi_path = a.midi
env = gym.make("Dexsim-Piano-Bimanual-v0", cfg=cfg, render_mode=None)
le = env.unwrapped
wrapped = RslRlVecEnvWrapper(env)

if a.zero:
    def policy(obs):                              # pure reference (zero residual)
        return torch.zeros((le.num_envs, le.cfg.action_space), device=le.device)
    print("[record_rollout] ZERO residual (pure IK reference)")
else:
    from rsl_rl.runners import OnPolicyRunner
    _ad = PianoPPORunnerCfg().to_dict(); _ad.setdefault("policy", {})["noise_std_type"] = "log"
    runner = OnPolicyRunner(wrapped, _ad, log_dir=None, device=le.device)
    runner.load(a.checkpoint)
    policy = runner.get_inference_policy(device=le.device)

obs, _ = wrapped.get_observations()
L, R, K, GOAL, SOUND = [], [], [], [], []
for _ in range(le.song_len):
    goal_t = le._goal_now()[0].cpu().numpy().copy()      # which keys SHOULD sound now
    with torch.inference_mode():
        obs, _, _, _ = wrapped.step(policy(obs))
    L.append(le.left_robot.data.joint_pos[0].cpu().numpy())
    R.append(le.right_robot.data.joint_pos[0].cpu().numpy())
    K.append(le.piano.data.joint_pos[0].cpu().numpy())
    GOAL.append(goal_t)
    SOUND.append(le.key_sounding[0].cpu().numpy().copy())  # which keys the model SOUNDS
np.savez(a.out, left=np.array(L), right=np.array(R), keys=np.array(K),
         goal=np.array(GOAL).astype(np.uint8), sound=np.array(SOUND).astype(np.uint8),
         joint_names_left=le.left_robot.data.joint_names,
         piano_pos=np.array(cfg.piano_pos),
         left_base=np.array(cfg.left_base_pos), right_base=np.array(cfg.right_base_pos),
         control_dt=cfg.control_dt)
# quick text summary so we know the rollout is sane before rendering
g = np.array(GOAL); s = np.array(SOUND)
tp = (g.astype(bool) & s.astype(bool)).sum()
print(f"[record_rollout] saved {len(L)} frames -> {a.out}  "
      f"goal_notes={int(g.sum())} sounded={int(s.sum())} correct={int(tp)}")
app.close()
