"""Roll out a trained (or zero-action) piano policy in MuJoCo and record it.

MuJoCo twin of scripts/train/play_piano.py -- no Isaac boot, no warm render
server needed: MuJoCo renders offscreen in-process at interactive speed.

  # zero-action baseline (ready pose + rail servo only):
  .venv/bin/python scripts/mj/play_piano_mj.py --zero --video results/mj_zero.mp4

  # trained checkpoint:
  .venv/bin/python scripts/mj/play_piano_mj.py \
      --checkpoint logs/piano_mj/<run>/model_final.pt --video results/mj_play.mp4

Prints the offline F1/precision/recall over the rollout and (optionally)
exports the keys that actually sounded as a MIDI file (--export_midi).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
os.environ.setdefault("MUJOCO_GL", "egl")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None, help="model_*.pt from train_piano_mj")
parser.add_argument("--zero", action="store_true", help="zero-action baseline")
parser.add_argument("--midi", default=None)
parser.add_argument("--songs_npz", default=None)
parser.add_argument("--steps", type=int, default=None, help="cap rollout steps (default: song length)")
parser.add_argument("--video", default=None, help="write an .mp4 here")
parser.add_argument("--camera", default="main", choices=["main", "front", "top"])
parser.add_argument("--size", default="1280x720")
parser.add_argument("--device", default="cpu")
parser.add_argument("--freeze_arms", action="store_true")
parser.add_argument("--rail_follow", action="store_true")
parser.add_argument("--mute_right", action="store_true")
parser.add_argument("--export_midi", default=None, help="write the SOUNDED keys as .mid")
parser.add_argument("--rollout_npz", default=None, help="dump qpos trajectory + metrics")
args = parser.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dexsim.tasks.piano_mj import PianoMjEnvCfg, PianoMjVecEnv, make_rsl_rl_env  # noqa: E402


def load_policy(env):
    """Inference policy from a train_piano_mj checkpoint (rsl_rl runner)."""
    from rsl_rl.runners import OnPolicyRunner
    from dexsim.tasks.piano_mj.ppo_cfg import piano_ppo_cfg  # same model shape

    runner = OnPolicyRunner(env, piano_ppo_cfg(), log_dir=None, device=args.device)
    runner.load(args.checkpoint, map_location=args.device)
    return runner.get_inference_policy(device=args.device)


def main():
    if not args.zero and not args.checkpoint:
        parser.error("pass --checkpoint <model.pt> or --zero")

    cfg = PianoMjEnvCfg()
    if args.midi:
        cfg.midi_path = args.midi
    if args.songs_npz:
        cfg.songs_npz = args.songs_npz
    cfg.freeze_arms = args.freeze_arms
    cfg.rail_follow = args.rail_follow
    cfg.mute_right_hand = args.mute_right
    cfg.__post_init__()

    venv = PianoMjVecEnv(cfg, num_envs=1, threads=1)
    env = venv.envs[0]
    rl_env = make_rsl_rl_env(venv, device=args.device)
    policy = None if args.zero else load_policy(rl_env)

    steps = args.steps or int(venv.bank.song_lens[0])
    fps = 1.0 / cfg.control_dt

    renderer = None
    frames = []
    if args.video:
        import mujoco
        w, h = (int(x) for x in args.size.split("x"))
        renderer = mujoco.Renderer(env.model, height=h, width=w)

    obs_td = rl_env.get_observations()
    qpos_hist, goal_hist, sound_hist, logs_hist = [], [], [], []
    for t in range(steps):
        if policy is None:
            actions = torch.zeros(1, cfg.action_space)
        else:
            with torch.no_grad():
                actions = policy(obs_td.to(args.device))
        goal_now = env._goal_now().copy()      # the goal THIS step is scored on
        obs_td, _, dones, extras = rl_env.step(actions)
        logs_hist.append(extras["log"])
        qpos_hist.append(env.data.qpos.copy())
        goal_hist.append(goal_now)
        sound_hist.append(env.key_sounding.copy())
        if renderer is not None:
            renderer.update_scene(env.data, camera=args.camera)
            frames.append(renderer.render())
        if bool(dones[0]):
            break

    # --- offline metrics over the rollout -----------------------------------
    goal = np.stack(goal_hist).astype(bool)              # (T, 88)
    sounded = np.stack(sound_hist)                       # (T, 88)
    has = goal.any(axis=1)
    tp = (sounded & goal).sum()
    rec = tp / max(goal.sum(), 1)
    prec = tp / max(sounded.sum(), 1)
    f1 = 2 * rec * prec / (rec + prec + 1e-9)
    mean = lambda k: float(np.mean([d[k] for d in logs_hist]))
    print(f"[play] {len(goal_hist)} steps | F1 {f1:.3f}  recall {rec:.3f}  "
          f"precision {prec:.3f}  (goal steps: {int(has.sum())})")
    print(f"[play] mean step logs: reward/total {mean('reward/total'):.3f}  "
          f"play/F1 {mean('play/F1'):.3f}  keys_sounding {mean('play/keys_sounding'):.2f}")

    if args.video:
        import imageio.v2 as imageio
        out = Path(args.video)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(out, frames, fps=int(round(fps)), quality=8)
        print(f"[play] video -> {out} ({len(frames)} frames @ {fps:.0f}fps)")

    if args.rollout_npz:
        out = Path(args.rollout_npz)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, qpos=np.stack(qpos_hist), goal=goal,
                            sounded=sounded, control_dt=cfg.control_dt)
        print(f"[play] rollout -> {out}")

    if args.export_midi:
        import pretty_midi
        from dexsim.piano.midi import PIANO_MIN_MIDI
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        T = sounded.shape[0]
        for k in range(88):
            on = None
            for t in range(T):
                if sounded[t, k] and on is None:
                    on = t
                elif not sounded[t, k] and on is not None:
                    inst.notes.append(pretty_midi.Note(
                        velocity=90, pitch=k + PIANO_MIN_MIDI,
                        start=on * cfg.control_dt, end=t * cfg.control_dt))
                    on = None
            if on is not None:
                inst.notes.append(pretty_midi.Note(
                    velocity=90, pitch=k + PIANO_MIN_MIDI,
                    start=on * cfg.control_dt, end=T * cfg.control_dt))
        pm.instruments.append(inst)
        Path(args.export_midi).parent.mkdir(parents=True, exist_ok=True)
        pm.write(args.export_midi)
        print(f"[play] sounded-notes MIDI -> {args.export_midi} "
              f"({len(inst.notes)} notes)")
    venv.close()


if __name__ == "__main__":
    main()
