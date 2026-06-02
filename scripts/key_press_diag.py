"""Why do no keys sound? Roll out the twinkle reference and, for the keys that
SHOULD be pressed each step, report how far they're actually depressed vs the
KEY_SOUND_ANGLE threshold. Tells us: are fingers contacting keys at all, and if
so do they press past the sound threshold?"""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--midi", default="data/midi/twinkle.mid")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app
import torch, gymnasium as gym, numpy as np
import dexsim.tasks  # noqa
from dexsim.tasks.piano import PianoEnvCfg
from dexsim.assets import KEY_SOUND_ANGLE

cfg = PianoEnvCfg(); cfg.scene.num_envs = 4; cfg.midi_path = a.midi
env = gym.make("Dexsim-Piano-Bimanual-v0", cfg=cfg, render_mode=None); le = env.unwrapped
env.reset()
print(f"KEY_SOUND_ANGLE = {KEY_SOUND_ANGLE} rad (key must rotate past this to sound)")
zero = torch.zeros(4, cfg.action_space, device=le.device)
worst_press = 0.0          # most-depressed any goal key got (most negative)
contact_steps = 0; goal_steps = 0; sounded_steps = 0
for t in range(le.song_len):
    env.step(zero)
    key_ang = le.piano.data.joint_pos[0]                  # (88,) negative = pressed
    goal = le.goal_padded[le.song_step[0]].bool()         # which keys should sound
    if goal.any():
        goal_steps += 1
        ga = key_ang[goal]                                # angles of goal keys
        mn = float(ga.min())                              # most pressed goal key
        worst_press = min(worst_press, mn)
        if mn < -0.005: contact_steps += 1                # any depression at all
        if (ga <= KEY_SOUND_ANGLE).any(): sounded_steps += 1
    if t in (20, 60, 120, 240) and goal.any():
        gi = torch.where(goal)[0].tolist()
        print(f"  step {t}: goal keys {gi[:4]} angles={[round(float(key_ang[i]),4) for i in gi[:4]]}  "
              f"(threshold {KEY_SOUND_ANGLE})")
print(f"\n=== over {goal_steps} steps with goal notes ===")
print(f"  steps where a goal key was depressed at all (<-0.005): {contact_steps} "
      f"({100*contact_steps/max(goal_steps,1):.0f}%)")
print(f"  steps where a goal key actually SOUNDED (<= {KEY_SOUND_ANGLE}): {sounded_steps} "
      f"({100*sounded_steps/max(goal_steps,1):.0f}%)")
print(f"  deepest any goal key was pressed: {worst_press:.4f} rad "
      f"(needs <= {KEY_SOUND_ANGLE} to sound)")
env.close(); app.close()
