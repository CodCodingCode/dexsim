"""Smoke test for the MuJoCo piano env (no training).

Checks, in order:
  1. the composed scene compiles (piano + two rail-mounted Shadow Hands);
  2. the 🔒 locked ready pose puts every fingertip a few cm ABOVE the keys;
  3. a scripted finger strike actually SOUNDS a key (crosses the sound angle
     with enough velocity to trip the hammer gate) -- the mechanic every
     reward depends on;
  4. the env API: reset/step shapes, reward finiteness, song termination;
  5. (optional, --render) an offscreen render of the scene to PNG.

Run:  .venv/bin/python scripts/mj/smoke_piano_mj.py [--midi data/midi/song.mid] [--render]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source"))

import mujoco
import numpy as np

from dexsim.mjcf import KEY_SOUND_ANGLE
from dexsim.tasks.piano_mj import PianoMjEnv, PianoMjEnvCfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi", default=None)
    parser.add_argument("--render", action="store_true",
                        help="offscreen-render the ready pose to logs/mj_smoke.png")
    args = parser.parse_args()

    cfg = PianoMjEnvCfg()
    if args.midi:
        cfg.midi_path = args.midi

    t0 = time.time()
    env = PianoMjEnv(cfg)
    m, d = env.model, env.data
    print(f"[1] scene compiled in {time.time() - t0:.1f}s: nq={m.nq} nv={m.nv} "
          f"nu={m.nu} nbody={m.nbody}")
    assert m.nu == cfg.action_space, (m.nu, cfg.action_space)

    # --- [2] locked ready pose geometry ------------------------------------
    obs = env.reset()
    tips = env._fingertips_world()
    key_top_z = env._key_top_world()[:, 2].max()
    above = tips[:, 2] - key_top_z
    print(f"[2] ready pose: fingertip heights above key tops (cm): "
          f"{np.round(above * 100, 1)}")
    assert (above > 0.0).all(), "a fingertip starts below the key tops!"
    assert (above < 0.15).all(), "a fingertip is way off the keys -- mount broken?"

    # --- [3] scripted press: MCP-flex the RIGHT middle finger onto its key --
    # a piano press is a MODEST MCP flexion (+ light distal curl), not a fist:
    # a full curl arcs the tip up into the palm and off the key.
    tip = tips[7]                                          # R middle fingertip
    key_xy = env._key_top_world()[:, :2]
    key = int(np.argmin(np.linalg.norm(key_xy - tip[:2], axis=1)))
    print(f"    striking key {key} under the right middle fingertip {np.round(tip, 3)}")
    aid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
           for i in range(m.nu)}
    mf0, mf2 = aid["R_robot0_A_MFJ0"], aid["R_robot0_A_MFJ2"]
    wr0 = aid["R_robot0_A_WRJ0"]
    min_angle, sounded = 0.0, False
    for t in range(40):                                    # 2 s @ 20 Hz
        press = t >= 10
        # MCP-heavy press with light distal curl + a touch of wrist flex --
        # the pianist press for this layout (a full fist curl drags the tip
        # off the key instead of into it).
        d.ctrl[mf2] = env.ready_ctrl[mf2] + (0.80 if press else 0.0)
        d.ctrl[mf0] = env.ready_ctrl[mf0] + (0.40 if press else 0.0)
        d.ctrl[wr0] = env.ready_ctrl[wr0] + (0.20 if press else 0.0)
        for _ in range(cfg.decimation):
            mujoco.mj_step(m, d)
            env._update_strike_latch()     # substep hammer gate (as in env.step)
        frac = env._key_pressed_fraction()
        angle = float(d.qpos[env.key_qadr].min())
        min_angle = min(min_angle, angle)
        if frac.max() >= 0.5:
            sounded = True
    print(f"[3] deepest key angle {min_angle:.4f} rad (sound angle "
          f"{KEY_SOUND_ANGLE}) -> {'SOUNDED ✓' if sounded else 'no sound ✗'}")
    assert min_angle < KEY_SOUND_ANGLE, "finger strike never crossed the sound angle"
    assert sounded, "key crossed the angle but the velocity gate never latched"

    # --- [4] env API + a short random rollout -------------------------------
    obs = env.reset()
    assert obs.shape == (cfg.observation_space,), obs.shape
    rng = np.random.default_rng(0)
    t0 = time.time()
    steps, done_seen = 0, False
    for _ in range(100):
        a = rng.uniform(-1, 1, cfg.action_space)
        obs, reward, terminated, truncated, logs = env.step(a)
        assert np.isfinite(obs).all() and np.isfinite(reward)
        steps += 1
        if terminated or truncated:
            done_seen = True
            env.reset()
    dt = (time.time() - t0) / steps
    print(f"[4] 100 random steps OK: {dt * 1000:.1f} ms/control-step "
          f"({1 / dt:.0f} steps/s single env), reward finite, "
          f"episode end seen={done_seen}")
    print(f"    sample logs: " + ", ".join(f"{k}={v:.3f}" for k, v in
                                           list(logs.items())[:6]))

    # --- [5] optional render -------------------------------------------------
    if args.render:
        env.reset()
        try:
            renderer = mujoco.Renderer(m, height=720, width=1280)
            renderer.update_scene(d, camera="main")
            img = renderer.render()
            out = Path("logs/mj_smoke.png")
            out.parent.mkdir(exist_ok=True)
            import imageio.v2 as imageio
            imageio.imwrite(out, img)
            print(f"[5] rendered {out}")
        except Exception as e:  # EGL/GL may be unavailable on headless boxes
            print(f"[5] render skipped ({type(e).__name__}: {e})")

    print("SMOKE OK")


if __name__ == "__main__":
    main()
