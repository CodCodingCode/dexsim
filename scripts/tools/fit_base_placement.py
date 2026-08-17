"""Find where the hand bases must sit for the fingertips to actually press keys.

The placement in ``piano_env_cfg.py`` is 🔒 locked and this script does NOT write
it unless you pass ``--write``. What it does is measure: for a grid of (dx, dz)
offsets applied to both hand bases, how close can each finger get to pressing
the key under it?

The measurement that matters is the *press miss*: the residual between the
fingertip pad and a point ``PRESS_DEPTH`` below the key's top face. Zero means
the finger can put the key down; tens of millimetres means it cannot reach,
which is the state the locked placement was in (fingertips ~40 mm above the
keys, and behind their near edge).

    .venv-pyroki/bin/python scripts/tools/fit_base_placement.py
    .venv-pyroki/bin/python scripts/tools/fit_base_placement.py --write
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source"))

from dexsim.kinematics.piano_hands import (          # noqa: E402
    FINGERTIPS, PianoHands, key_to_name)
from dexsim.piano import geometry                    # noqa: E402

CFG = REPO / "source" / "dexsim" / "tasks" / "piano" / "piano_env_cfg.py"
FINGERS = ("thumb", "index", "middle", "ring", "little")


def assign_keys(hands: PianoHands, side: str) -> dict:
    """Nearest key (by lateral position) under each fingertip at the ready pose."""
    hand = hands.hands[side]
    keys = np.arange(len(geometry.KEY_IS_BLACK))
    white = keys[~geometry.KEY_IS_BLACK]              # press white keys: unambiguous
    tops = np.stack([hands.key_world(int(k)) for k in white])
    out = {}
    for finger in FINGERS:
        tip = hands._tip_world(hand, finger, hand.rest)
        out[finger] = int(white[int(np.argmin(np.abs(tops[:, 1] - tip[1])))])
    return out


def evaluate(hands: PianoHands, dx: float, dz: float, fingers=FINGERS) -> dict:
    """Max press miss (mm) per hand for a candidate base offset."""
    result = {}
    for side, hand in hands.hands.items():
        hand.base_pos = hands.poses[f"{side}_base_pos"] + np.array([dx, 0.0, dz])
        hand.cfg = hand.rest.copy()
        assign = assign_keys(hands, side)
        misses = []
        for finger in fingers:
            hands.release(side)
            misses.append(hands.press(side, finger, assign[finger] + 21).miss_mm)
        result[side] = {"max_miss": max(misses), "keys": assign,
                        "per_finger": dict(zip(fingers, misses))}
        hands.release(side)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dx", default="0.00,0.02,0.04,0.06,0.08",
                   help="forward (toward the keys) offsets to try, metres")
    p.add_argument("--dz", default="0.00,-0.02,-0.03,-0.04,-0.05,-0.06",
                   help="vertical offsets to try, metres")
    p.add_argument("--margin", type=float, default=0.008,
                   help="extra lowering past the first zero-miss solution (m)")
    p.add_argument("--key_span", type=float, default=0.35,
                   help="fraction of the key length that counts as a landing spot; "
                        "0.35 aims near the middle instead of the front lip")
    p.add_argument("--write", action="store_true",
                   help="write the winning offset into piano_env_cfg.py")
    a = p.parse_args()

    dxs = [float(v) for v in a.dx.split(",")]
    dzs = [float(v) for v in a.dz.split(",")]
    hands = PianoHands(key_span=a.key_span)
    base = {s: hands.poses[f"{s}_base_pos"].copy() for s in ("left", "right")}

    print("max press miss (mm), worst of both hands -- 0 means the key goes down")
    print("      dz:  " + "  ".join(f"{z * 1000:+6.0f}" for z in dzs))
    grid = {}
    for dx in dxs:
        row = []
        for dz in dzs:
            for side in ("left", "right"):
                hands.poses[f"{side}_base_pos"] = base[side]
            res = evaluate(hands, dx, dz, fingers=("index", "middle"))
            worst = max(r["max_miss"] for r in res.values())
            grid[(dx, dz)] = worst
            row.append(worst)
        print(f"dx {dx * 1000:+4.0f}:  " + "  ".join(f"{v:6.1f}" for v in row))

    # Pick the shallowest offset that reaches, then sink it by --margin so the
    # press is not marginal, and prefer the smallest dx that still works.
    ok = [(dx, dz) for (dx, dz), miss in grid.items() if miss < 1.0]
    if not ok:
        best = min(grid, key=grid.get)
        raise SystemExit(f"\nnothing reached the keys; closest was dx={best[0]:+.3f} "
                         f"dz={best[1]:+.3f} at {grid[best]:.1f} mm")
    dx = min(d for d, _ in ok)
    dz = max(z for d, z in ok if d == dx) - a.margin

    for side in ("left", "right"):
        hands.poses[f"{side}_base_pos"] = base[side]
    final = evaluate(hands, dx, dz)
    print(f"\nchosen offset: dx={dx * 1000:+.0f} mm (toward keys), "
          f"dz={dz * 1000:+.0f} mm (down), margin {a.margin * 1000:.0f} mm")
    for side, res in final.items():
        pos = base[side] + np.array([dx, 0.0, dz])
        print(f"  {side:5s} base -> ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
        for finger, miss in res["per_finger"].items():
            print(f"      {finger:7s} on {key_to_name(res['keys'][finger]):4s}: "
                  f"{miss:5.1f} mm")

    if not a.write:
        print("\n(dry run -- pass --write to update piano_env_cfg.py)")
        return

    src = CFG.read_text()
    for side in ("left", "right"):
        pos = base[side] + np.array([dx, 0.0, dz])
        new = "(" + ", ".join(f"{v:.4f}" for v in pos) + ")"
        src, n = re.subn(rf"({side}_base_pos\s*=\s*)\([^)]*\)", rf"\g<1>{new}", src)
        if n != 1:
            raise SystemExit(f"expected one {side}_base_pos assignment, found {n}")
    CFG.write_text(src)
    print(f"\n[write] updated {CFG}")


if __name__ == "__main__":
    main()
