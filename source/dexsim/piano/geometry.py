"""Keyboard geometry — the single source of truth for where each key *is*.

Mirrors the layout authored by ``scripts/build_piano_usd.py`` so that the
fingering planner, the IK reference builder, and the env all agree on where a
finger must go to press key ``i``.

Two coordinate views:
  * **local** — the piano's own frame (the build script's frame): +Y runs along
    the keyboard low->high, +X points toward the player, +Z is up. Key 0 (A0)
    sits at y=0; the white-key span is ~1.22 m.
  * **world** — at runtime we don't trust the analytic local->world transform;
    instead we read each key body's *measured* world position from the
    articulation (``data.body_pos_w``) and add only the small local offset from a
    key's body-center to its playable top surface. That keeps us correct under
    env-origin cloning and any piano pose.

The press target for key ``i`` is the center of its top surface: forward of the
back hinge, so pressing it rotates the key down past ``KEY_SOUND_ANGLE``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .midi import NUM_KEYS, PIANO_MIN_MIDI, key_is_black

# --- must match scripts/build_piano_usd.py exactly -------------------------
WHITE_W, WHITE_L, WHITE_H = 0.0225, 0.145, 0.012
BLACK_W, BLACK_L, BLACK_H = 0.011, 0.090, 0.010
WHITE_PITCH = 0.0235          # center-to-center spacing of white keys
BLACK_RAISE = 0.009           # black key top sits this much above white tops
# ---------------------------------------------------------------------------

# A finger "rests" this far above a key's top before pressing; the press goal
# dips below the surface so the spring is compressed past the sounding angle.
HOVER_CLEARANCE = 0.030       # m above the key top for idle fingers. Was 0.010:
#   1cm didn't lift idle fingers clear, so they sounded ~5 wrong keys every step
#   (reference precision capped at 0.077). 3cm lifts them off the keys.
PRESS_DEPTH = 0.008           # m below key top for the "pressed" pose. LIGHT press:
#   15mm drove the flat hand deep into the keyboard -> each finger struck ~2 neighbor
#   keys (~22 ring, precision capped 0.03). 8mm still crosses the (light, velocity-
#   gated) sound threshold for the target but barely touches neighbors. (0.040 was
#   way too deep -> unreachable target.)


@dataclass(frozen=True)
class KeyGeom:
    """Per-key geometry in the piano-local frame."""

    index: int          # 0..87 (== MIDI - 21)
    is_black: bool
    y: float            # lateral position along the keyboard (local +Y)
    z_top: float        # top-surface height (local +Z)
    half_height: float  # body half-thickness (center -> top)
    x_center: float     # key body-center X in local frame


def layout() -> list[KeyGeom]:
    """All 88 keys, in index order, matching the USD build."""
    keys: list[KeyGeom] = []
    white_count = 0
    for i in range(NUM_KEYS):
        black = key_is_black(i)
        if not black:
            y = white_count * WHITE_PITCH
            keys.append(KeyGeom(i, False, y, WHITE_H, WHITE_H / 2.0,
                                (WHITE_L / 2.0) - WHITE_L / 2.0))  # == 0.0
            white_count += 1
        else:
            y = (white_count - 0.5) * WHITE_PITCH
            keys.append(KeyGeom(i, True, y, WHITE_H + BLACK_RAISE, BLACK_H / 2.0,
                                (BLACK_L / 2.0) - WHITE_L / 2.0))
    return keys


_LAYOUT = layout()

# Convenience arrays (index-aligned, length 88).
KEY_Y = np.array([k.y for k in _LAYOUT], dtype=np.float32)               # local Y
KEY_Z_TOP = np.array([k.z_top for k in _LAYOUT], dtype=np.float32)        # local Z top
KEY_HALF_H = np.array([k.half_height for k in _LAYOUT], dtype=np.float32)
KEY_IS_BLACK = np.array([k.is_black for k in _LAYOUT], dtype=bool)
KEYBOARD_SPAN_Y = float(sum(1 for k in _LAYOUT if not k.is_black) * WHITE_PITCH)


def center_to_top_offset() -> np.ndarray:
    """(88, 3) local offset from each key's *body center* to its top-surface
    center. X/Y are 0 (we press the center); Z is the half-height. Add this to a
    measured key-center world position to get the top-surface world point."""
    off = np.zeros((NUM_KEYS, 3), dtype=np.float32)
    off[:, 2] = KEY_HALF_H
    return off


def key_local_top_positions() -> np.ndarray:
    """(88, 3) nominal top-surface positions in the piano-local frame.

    Used by the fingering planner (which only needs relative key spacing) and by
    sim-free tests. X uses the body center (0 for white, slightly back for black).
    """
    pos = np.zeros((NUM_KEYS, 3), dtype=np.float32)
    pos[:, 0] = np.array([k.x_center for k in _LAYOUT], dtype=np.float32)
    pos[:, 1] = KEY_Y
    pos[:, 2] = KEY_Z_TOP
    return pos
