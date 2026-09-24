"""Roll out a trained typing policy (or zero actions) on the keyboard scene,
print what the keyboard registered, optionally render a video.

  .venv/bin/python scripts/mj/play_keyboard_mj.py --checkpoint logs/keyboard_mj/<run>/model_final.pt \\
      --text "hello world" --video logs/typing.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--zero", action="store_true", help="zero actions (hold pose G)")
parser.add_argument("--text", default="hello world")
parser.add_argument("--right_only", action="store_true")
parser.add_argument("--left_only", action="store_true")
parser.add_argument("--time_per_char", type=float, default=None)
parser.add_argument("--deterministic", action="store_true", default=True)
parser.add_argument("--video", default="")
parser.add_argument("--camera", default="close")
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--size", default="1280x720")
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

import torch  # noqa: E402
import mujoco  # noqa: E402

from dexsim.mjcf import keyboard as kb  # noqa: E402
from dexsim.tasks.keyboard_mj import KeyboardMjEnvCfg, KeyboardMjVecEnv  # noqa: E402
from dexsim.tasks.piano_mj.vec_env import make_rsl_rl_env  # noqa: E402
from dexsim.tasks.piano_mj.ppo_cfg import piano_ppo_cfg  # noqa: E402


def main():
    if not args.zero and not args.checkpoint:
        parser.error("pass --checkpoint <model.pt> or --zero")
    cfg = KeyboardMjEnvCfg()
    cfg.text = args.text
    cfg.freeze_left_hand = args.right_only
    cfg.freeze_right_hand = args.left_only
    if args.time_per_char is not None:
        cfg.time_per_char_s = args.time_per_char
    cfg.__post_init__()

    venv = KeyboardMjVecEnv(cfg, num_envs=1, threads=1)
    env = venv.envs[0]
    rl_env = make_rsl_rl_env(venv, device=args.device)
    policy = None
    if not args.zero:
        from rsl_rl.runners import OnPolicyRunner
        runner = OnPolicyRunner(rl_env, piano_ppo_cfg(
            obs_groups={"actor": ["policy"], "critic": ["policy"]}),
            log_dir=None, device=args.device)
        runner.load(args.checkpoint, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)

    renderer, frames = None, []
    if args.video:
        w, h = (int(x) for x in args.size.split("x"))
        renderer = mujoco.Renderer(env.model, height=h, width=w)
        every = max(1, int(round(1.0 / (cfg.control_dt * args.fps))))

    obs = rl_env.get_observations()
    keylog = []
    n_reg = 0
    for t in range(env.max_episode_length):
        if policy is None:
            act = torch.zeros(1, cfg.action_space)
        else:
            with torch.no_grad():
                act = policy(obs.to(args.device))
        obs, rew, dones, extras = rl_env.step(act)
        # rsl-rl auto-resets on done; read the env's keylog before that happens
        for k in env._new_presses:
            keylog.append(kb.KEY_NAMES[k])
        if renderer is not None and t % every == 0:
            renderer.update_scene(env.data, camera=args.camera)
            frames.append(renderer.render().copy())
        if bool(dones[0]):
            done_step = t + 1
            break
    else:
        done_step = env.max_episode_length
    secs = done_step * cfg.control_dt
    typed = "".join(kb.key_to_char(k) or f"<{k}>" for k in keylog)
    print(f"[play] target : {args.text!r}")
    print(f"[play] typed  : {typed!r}")
    print(f"[play] {len(keylog)} keystrokes in {secs:.1f}s "
          f"({len(keylog) / max(secs, 1e-6):.2f} keys/s); "
          f"last logs: " + ", ".join(f"{k}={v:.3f}" for k, v in venv._last_logs.items()
                                     if k.startswith("play/")))
    if renderer is not None and frames:
        import imageio.v2 as imageio
        out = Path(args.video)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(out, frames, fps=args.fps, codec="libx264", quality=8,
                         macro_block_size=1)
        print(f"[play] video -> {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
