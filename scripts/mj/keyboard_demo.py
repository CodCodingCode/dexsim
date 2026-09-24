"""Scripted typing demo on the MacBook keyboard scene -- watch the two Shadow
Hands type a string, and check what the keyboard actually registered.

No policy, no IK: each keystroke picks the touch-typing finger for the key,
extends that finger a little (raises the others), servo-aligns the fingertip
over the keycap with the hand's X/Y gantry (closed loop on the measured
fingertip site), lowers the hand on the Z gantry until the key's slide joint
crosses the actuation depth, holds briefly, and lifts. Shifted characters
hold the opposite hand's little finger on ``lshift``/``rshift`` meanwhile.

Usage (from the repo root, after ``source env.sh``):

  python scripts/mj/keyboard_demo.py --text "hello world"
  python scripts/mj/keyboard_demo.py --text "Hi, Nathan!" --video logs/kb.mp4
  python scripts/mj/keyboard_demo.py --xml            # also dump the scene XML
                                                      # for `python -m mujoco.viewer`
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source"))

from dexsim.mjcf import keyboard as kb                                    # noqa: E402
from dexsim.mjcf import keyboard_scene as ks                              # noqa: E402

CONTROL_HZ = 100
LONG_FINGERS = ("ff", "mf", "rf", "lf")
# finger posture deltas on the MCP (J2) and PIP+DIP tendon (J0) actuators,
# relative to the 🔒 pose G ready ctrl. Measured from pose G: CURLING a
# finger drops its tip (+0.2/+0.2 -> -14 mm, 12 mm toward the typist),
# EXTENDING raises it (-0.35/-0.40 -> +49 mm). So the striking finger curls a
# little and the idle ones straighten up out of the way, like a typist.
ACTIVE_DELTA = (+0.20, +0.20)
RAISED_DELTA = (-0.35, -0.40)
XY_GAIN = 0.7                       # fraction of the tip error applied per step
ALIGN_TOL = 0.0012                  # m
LOWER_SPEED = 0.10                  # m/s on the Z gantry ctrl
LIFT_SPEED = 0.15
RAISE_Z = 0.030                     # gantry lift for posture changes + travel
MAX_LOWER = 0.075                   # m below RAISE_Z before giving up
HOLD_STEPS = 6                      # control steps to hold a registered key
# key debounce: register at the actuation depth, release only when the cap
# has come back up past this (scissor switches have similar hysteresis)
RELEASE_DEPTH = 0.0002


class Typist:
    """Owns the model/data, the per-hand gantry setpoints and the keylog."""

    def __init__(self, cfg: ks.KeyboardSceneCfg, renderer=None, camera="close",
                 fps=30):
        self.cfg = cfg
        self.model = m = ks.compile_keyboard_scene(cfg)
        self.data = d = mujoco.MjData(m)
        d.qpos[:] = ks.ready_qpos(m, cfg)
        self.ready = ks.ready_ctrl(m, d.qpos.copy())
        d.ctrl[:] = self.ready
        mujoco.mj_forward(m, d)
        self.substeps = int(round(1.0 / (CONTROL_HZ * cfg.sim_dt)))
        self.key_sites = ks.key_site_ids(m)
        self.key_qadr = ks.key_qpos_adr(m)
        self.tips = ks.fingertip_site_ids(m)
        aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        self.gantry = {h: {t: aid(f"{h}_A_gantry_{t}") for t in "xyz"} for h in "LR"}
        jq = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
        self.gantry_q = {h: {t: jq(f"{h}_gantry_{t}") for t in "xyz"} for h in "LR"}
        self.finger_acts = {
            (h, f): (aid(f"{h}_robot0_A_{f.upper()}J2"), aid(f"{h}_robot0_A_{f.upper()}J0"))
            for h in "LR" for f in LONG_FINGERS}
        self.lower_z = {h: 0.0 for h in "LR"}   # current Z ctrl per hand
        self.registered = np.zeros(kb.NUM_KEYS, dtype=bool)
        self.keylog: list[tuple[str, bool]] = []     # (key, shift held)
        self.t = 0
        self.renderer, self.camera, self.fps = renderer, camera, fps
        self.frames: list[np.ndarray] = []
        self._frame_every = max(1, int(round(CONTROL_HZ / fps)))

    # --- low level -----------------------------------------------------------
    def step(self):
        """One control step: integrate, update keylog, maybe grab a frame."""
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        q = self.data.qpos[self.key_qadr]
        now = (q <= -kb.KEY_ACTUATION_DEPTH) | (self.registered & (q <= -RELEASE_DEPTH))
        shift = now[kb.KEY_INDEX["lshift"]] or now[kb.KEY_INDEX["rshift"]]
        for i in np.nonzero(now & ~self.registered)[0]:
            if kb.KEY_NAMES[i] not in ("lshift", "rshift"):
                self.keylog.append((kb.KEY_NAMES[i], bool(shift)))
        self.registered = now
        if self.renderer is not None and self.t % self._frame_every == 0:
            self.renderer.update_scene(self.data, camera=self.camera)
            self.frames.append(self.renderer.render().copy())
        self.t += 1

    def tip(self, hand, finger):
        return self.data.site_xpos[self.tips[(hand, finger)]].copy()

    def key_top(self, key):
        return self.data.site_xpos[self.key_sites[kb.KEY_INDEX[key]]].copy()

    def press_point(self, key, tip_xy):
        """Where to strike ``key``: its centre, except on wide caps (space,
        shift, return ...) where a typist hits the end nearest the finger --
        the point of the cap closest to the tip, 6 mm inside its edge."""
        c = self.key_top(key)
        cap = kb.layout()[kb.KEY_INDEX[key]]
        half = max(cap.width / 2.0 - 0.006, 0.0)
        c[1] = float(np.clip(tip_xy[1], c[1] - half, c[1] + half))
        return c

    def posture(self, hand, active):
        """Extend ``active`` (None = all at ready), raise the other long fingers."""
        for f in LONG_FINGERS:
            j2, j0 = self.finger_acts[(hand, f)]
            d2, d0 = (0.0, 0.0) if active is None else (
                ACTIVE_DELTA if f == active else RAISED_DELTA)
            self.data.ctrl[j2] = self.ready[j2] + d2
            self.data.ctrl[j0] = self.ready[j0] + d0

    def set_z(self, hand, z):
        lo, hi = self.cfg.gantry_z_range
        self.lower_z[hand] = float(np.clip(z, lo, hi))
        self.data.ctrl[self.gantry[hand]["z"]] = self.lower_z[hand]

    # --- motions -------------------------------------------------------------
    def _xy_servo(self, hand, finger, key):
        """One XY update: the tip rides rigidly on the carriage, so the
        setpoint is simply carriage position + tip error (no integrator)."""
        tip = self.tip(hand, finger)[:2]
        err = self.press_point(key, tip)[:2] - tip
        lim = self.cfg.gantry_xy_limit
        for t, e in zip("xy", err):
            q = self.data.qpos[self.gantry_q[hand][t]]
            self.data.ctrl[self.gantry[hand][t]] = np.clip(q + XY_GAIN * e, -lim, lim)
        return float(np.linalg.norm(err))

    def align(self, hand, finger, key, max_steps=150):
        """Servo the X/Y gantry until the fingertip is over the keycap."""
        for _ in range(max_steps):
            if self._xy_servo(hand, finger, key) < ALIGN_TOL:
                return True
            self.step()
        return False

    def lower_until_registered(self, hand, key):
        idx = kb.KEY_INDEX[key]
        start = self.lower_z[hand]
        while start - self.lower_z[hand] < MAX_LOWER:
            if self.data.qpos[self.key_qadr[idx]] <= -kb.KEY_ACTUATION_DEPTH:
                # freeze the setpoint just below the current carriage height
                self.set_z(hand, self.lower_z[hand] - 0.0005)
                return True
            self.set_z(hand, self.lower_z[hand] - LOWER_SPEED / CONTROL_HZ)
            self._xy_servo(hand, self._finger_for(key), key)   # keep centred
            self.step()
        return False

    def lift(self, hand, to=RAISE_Z, settle=5):
        while self.lower_z[hand] < to - 1e-6:
            self.set_z(hand, min(to, self.lower_z[hand] + LIFT_SPEED / CONTROL_HZ))
            self.step()
        for _ in range(settle):
            self.step()

    def descend(self, hand, to=0.0):
        while self.lower_z[hand] > to + 1e-6:
            self.set_z(hand, max(to, self.lower_z[hand] - LOWER_SPEED / CONTROL_HZ))
            self.step()

    def _finger_for(self, key):
        return self._active.get(key, kb.TOUCH_TYPING_FINGER[key][1])

    def press(self, hand, finger, key, hold=True):
        self._active = {key: finger}
        # posture changes and travel happen RAISED: a finger curling at hover
        # height sweeps the row in front of the target on its way down
        self.lift(hand, RAISE_Z, settle=0)
        # thumb strike (space): all four long fingers up, thumb stays at pose G
        self.posture(hand, finger if finger in LONG_FINGERS else "none")
        for _ in range(15):                   # let the posture settle
            self.step()
        ok = self.align(hand, finger, key)
        hit = self.lower_until_registered(hand, key)
        if hit and hold:
            for _ in range(HOLD_STEPS):
                self.step()
        return ok, hit

    def home(self, hand):
        """Back to the pose G hover over the home row."""
        self.lift(hand, RAISE_Z)
        self.posture(hand, None)
        for t in "xy":
            self.data.ctrl[self.gantry[hand][t]] = 0.0
        for _ in range(25):
            self.step()
        self.descend(hand, 0.0)

    def type_text(self, text, verbose=True):
        for key, shift in kb.text_to_keys(text):
            hand, finger = kb.TOUCH_TYPING_FINGER[key]
            other = "R" if hand == "L" else "L"
            shift_key = "rshift" if other == "R" else "lshift"
            if shift:
                self.press(other, "lf", shift_key, hold=False)
            ok, hit = self.press(hand, finger, key)
            # back to the home hover after every stroke, so a hand never parks
            # over the other one (the right thumb on space is the usual case)
            self.home(hand)
            if shift:
                self.home(other)
            if verbose:
                ch = kb.key_to_char(key, shift) or key
                print(f"  {hand}-{finger:>2} -> {ch!r:6} aligned={ok!s:5} "
                      f"registered={hit!s:5}  t={self.t / CONTROL_HZ:5.2f}s")
        self.home("L")
        self.home("R")

    def typed_string(self):
        return "".join(kb.key_to_char(k, s) or f"<{k}>" for k, s in self.keylog)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="hello world")
    ap.add_argument("--video", default="", help="write an mp4 of the run")
    ap.add_argument("--png", default="logs/keyboard_demo.png",
                    help="still of the scene at the end ('' to skip)")
    ap.add_argument("--camera", default="close",
                    choices=["main", "front", "top", "close", "keys"])
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--xml", action="store_true",
                    help="also dump assets/mj/keyboard_scene.xml")
    args = ap.parse_args()

    cfg = ks.KeyboardSceneCfg()
    if args.xml:
        print(f"[demo] wrote {ks.save_keyboard_scene_xml(cfg)}")
    typist = Typist(cfg, camera=args.camera, fps=args.fps)
    renderer = None
    if args.video or args.png:
        w, h = (int(x) for x in args.size.split("x"))
        renderer = mujoco.Renderer(typist.model, height=h, width=w)
        if args.video:
            typist.renderer = renderer

    print(f"[demo] typing {args.text!r} with the touch-typing finger map")
    t0 = time.time()
    typist.type_text(args.text)
    wall = time.time() - t0
    typed = typist.typed_string()
    target = args.text
    print(f"[demo] keyboard registered: {typed!r}")
    print(f"[demo] {'MATCH' if typed == target else 'MISMATCH'} "
          f"({sum(a == b for a, b in zip(typed, target))}/{len(target)} chars, "
          f"{max(0, len(typed) - len(target))} stray); sim {typist.t / CONTROL_HZ:.1f}s "
          f"in {wall:.1f}s wall")

    if args.png:
        out = Path(args.png)
        out.parent.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio
        renderer.update_scene(typist.data, camera=args.camera)
        imageio.imwrite(out, renderer.render())
        print(f"[demo] still -> {out}")
    if args.video:
        out = Path(args.video)
        out.parent.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio
        imageio.mimwrite(out, typist.frames, fps=args.fps, codec="libx264",
                         quality=8, macro_block_size=1)
        print(f"[demo] video -> {out} ({len(typist.frames)} frames)")


if __name__ == "__main__":
    main()
