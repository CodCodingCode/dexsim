"""Roll out a trained piano policy over the song, optionally record video, and
export what the hands ACTUALLY played back to a MIDI file (so you can hear it).

  python scripts/play_piano.py --num_envs 1 --checkpoint logs/rsl_rl/piano_bimanual/seed0/model_2000.pt
  python scripts/play_piano.py --num_envs 1 --video --export_midi played.mid
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained piano policy.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--midi", default=None)
parser.add_argument("--video", action="store_true")
parser.add_argument("--export_midi", default="logs/played.mid",
                    help="write the keys the policy actually sounded to this .mid")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import dexsim.tasks  # noqa: F401
from dexsim.tasks.piano import PianoEnvCfg
from dexsim.tasks.piano.agents.rsl_rl_ppo_cfg import PianoPPORunnerCfg
from dexsim.piano import PIANO_MIN_MIDI
from dexsim.assets import KEY_SOUND_ANGLE

TASK = "Dexsim-Piano-Bimanual-v0"


def export_played_midi(sounded_per_step, control_dt, path):
    """sounded_per_step: list of (88,) bool arrays. Write note on/off to MIDI."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    active = {}  # key -> start time
    for t, snd in enumerate(sounded_per_step):
        now = t * control_dt
        for k in range(88):
            if snd[k] and k not in active:
                active[k] = now
            elif not snd[k] and k in active:
                inst.notes.append(pretty_midi.Note(
                    velocity=90, pitch=k + PIANO_MIN_MIDI, start=active.pop(k), end=now))
    end = len(sounded_per_step) * control_dt
    for k, s in active.items():
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=k + PIANO_MIN_MIDI, start=s, end=end))
    pm.instruments.append(inst)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pm.write(path)
    print(f"[play_piano] exported what it played -> {path} ({len(inst.notes)} notes)")


def main():
    cfg = PianoEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.midi:
        cfg.midi_path = args.midi
    agent_cfg = PianoPPORunnerCfg()

    env = gym.make(TASK, cfg=cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        env = gym.wrappers.RecordVideo(
            env, video_folder="logs/rsl_rl/piano_bimanual/videos",
            step_trigger=lambda s: s == 0, video_length=cfg_len(env), disable_logger=True)
    env = RslRlVecEnvWrapper(env)

    ckpt = args.checkpoint or get_checkpoint_path("logs/rsl_rl/piano_bimanual", ".*", "model_.*.pt")
    print(f"[play_piano] checkpoint: {ckpt}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    le = env.unwrapped
    obs, _ = env.get_observations()
    sounded = []
    for _ in range(le.song_len):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        snd = (le.piano.data.joint_pos[0] <= KEY_SOUND_ANGLE).cpu().numpy()
        sounded.append(snd)
    export_played_midi(sounded, cfg.control_dt, args.export_midi)
    env.close()


def cfg_len(env):
    return env.unwrapped.song_len if hasattr(env.unwrapped, "song_len") else 600


main()
simulation_app.close()
