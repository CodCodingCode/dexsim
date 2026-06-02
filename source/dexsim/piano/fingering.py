"""Automatic fingering: assign each active note to a specific finger per step.

RoboPianist needs human fingering labels (the PIG dataset) and shows that
*without* a fingering signal the policy never learns (F1 = 0). PianoMime gets the
same signal from human video. We have neither, so we synthesize a fingering with
a simple, robust heuristic — which is exactly the structure-injection that makes
the high-DoF search tractable.

Heuristic (good enough to bootstrap; not claimed optimal):
  * Split the active notes at a pitch boundary: lower notes -> left hand, upper
    -> right hand, balancing the count so neither hand is asked for >5 keys.
  * Within a hand, sort the assigned keys by pitch and map them to fingers so the
    thumbs meet in the middle (natural piano fingering):
        left  (low->high pitch): little, ring, middle, index, THUMB
        right (low->high pitch): THUMB, index, middle, ring, little
  * Idle fingers hover over a per-finger "home" key so they stay spread and ready.

Output is per control step:
  * ``finger_key``    (T, 10) int  -- target key index per finger, -1 if idle.
  * ``finger_active`` (T, 10) bool -- whether the finger is assigned a note now.
Finger order is [L_thumb, L_index, L_middle, L_ring, L_little, R_thumb, ...].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .midi import NUM_KEYS
from . import geometry as geom

NUM_FINGERS = 10
FINGERS_PER_HAND = 5
# global finger indices
L_THUMB, L_INDEX, L_MIDDLE, L_RING, L_LITTLE = 0, 1, 2, 3, 4
R_THUMB, R_INDEX, R_MIDDLE, R_RING, R_LITTLE = 5, 6, 7, 8, 9

# Shadow fingertip body names, in the per-hand finger order [thumb, index,
# middle, ring, little]. Same names on both hands (each is its own articulation).
FINGERTIP_BODIES = [
    "robot0_thdistal", "robot0_ffdistal", "robot0_mfdistal",
    "robot0_rfdistal", "robot0_lfdistal",
]

# low->high pitch -> finger slot, per hand (thumbs toward the split in the middle)
_LEFT_ORDER = [L_LITTLE, L_RING, L_MIDDLE, L_INDEX, L_THUMB]
_RIGHT_ORDER = [R_THUMB, R_INDEX, R_MIDDLE, R_RING, R_LITTLE]


@dataclass
class FingeringPlan:
    finger_key: np.ndarray     # (T, 10) int, -1 = idle
    finger_active: np.ndarray  # (T, 10) bool
    home_key: np.ndarray       # (10,) int, the idle "home" key per finger

    @property
    def num_steps(self) -> int:
        return int(self.finger_key.shape[0])


def _home_keys() -> np.ndarray:
    """Per-finger idle home key: spread the 10 fingers across the keyboard, thumbs
    near the middle. Used so idle fingers hover sensibly instead of flailing."""
    # left hand owns the lower half, right hand the upper half
    lo, hi = 0, NUM_KEYS - 1
    mid = NUM_KEYS // 2
    home = np.zeros(NUM_FINGERS, dtype=np.int64)
    # left: little(low) .. thumb(just below mid)
    left_cols = np.linspace(lo + 6, mid - 4, FINGERS_PER_HAND).round().astype(int)
    for slot, key in zip(_LEFT_ORDER, left_cols):
        home[slot] = key
    # right: thumb(just above mid) .. little(high)
    right_cols = np.linspace(mid + 4, hi - 6, FINGERS_PER_HAND).round().astype(int)
    for slot, key in zip(_RIGHT_ORDER, right_cols):
        home[slot] = key
    return home


def _assign_hand(keys_sorted: list[int], order: list[int],
                 finger_key: np.ndarray, finger_active: np.ndarray, t: int) -> None:
    """Map up to 5 pitch-sorted keys onto a hand's finger slots (in `order`)."""
    n = len(keys_sorted)
    if n == 0:
        return
    if n <= FINGERS_PER_HAND:
        # contiguous block of fingers, anchored so the lowest key takes the
        # lowest finger in `order` (keeps thumbs toward the split)
        chosen = order[:n]
        for finger, key in zip(chosen, keys_sorted):
            finger_key[t, finger] = key
            finger_active[t, finger] = True
    else:
        # too many simultaneous notes for one hand: take 5 spanning the range
        idx = np.linspace(0, n - 1, FINGERS_PER_HAND).round().astype(int)
        for finger, j in zip(order, idx):
            finger_key[t, finger] = keys_sorted[j]
            finger_active[t, finger] = True


def plan_fingering(key_activation: np.ndarray) -> FingeringPlan:
    """Assign fingers for every control step.

    Args:
        key_activation (T, 88) bool -- which keys should sound each step.
    """
    T = key_activation.shape[0]
    finger_key = np.full((T, NUM_FINGERS), -1, dtype=np.int64)
    finger_active = np.zeros((T, NUM_FINGERS), dtype=bool)
    home = _home_keys()

    for t in range(T):
        active = np.nonzero(key_activation[t])[0]
        if active.size == 0:
            continue
        active = np.sort(active)
        # balance the hand split: aim for <=5 per hand. Default split at the
        # keyboard middle, then rebalance if one hand is overloaded.
        if active.size <= FINGERS_PER_HAND:
            # all on one hand if they cluster, else split at the median gap
            split = _balanced_split(active)
        else:
            split = _balanced_split(active)
        left = [int(k) for k in active if k < split]
        right = [int(k) for k in active if k >= split]
        # if a hand is overloaded but the other is empty/light, shift the split
        left, right = _rebalance(left, right)
        _assign_hand(left, _LEFT_ORDER, finger_key, finger_active, t)
        _assign_hand(right, _RIGHT_ORDER, finger_key, finger_active, t)

    return FingeringPlan(finger_key=finger_key, finger_active=finger_active, home_key=home)


def _balanced_split(active: np.ndarray) -> int:
    """Pick a pitch boundary (key index) splitting notes into left/right hands.
    Default: the keyboard midpoint; if all notes sit on one side, split at their
    own median so a lone melody still uses the nearer hand."""
    mid = NUM_KEYS // 2
    if active.max() < mid or active.min() >= mid:
        # all notes on one side of center -> split at their median
        return int(np.median(active)) + 1
    return mid


def _rebalance(left: list[int], right: list[int]) -> tuple[list[int], list[int]]:
    """If one hand has >5 notes and the other has room, move the boundary notes."""
    while len(left) > FINGERS_PER_HAND and len(right) < FINGERS_PER_HAND:
        right.insert(0, left.pop())          # highest left note -> right hand
    while len(right) > FINGERS_PER_HAND and len(left) < FINGERS_PER_HAND:
        left.append(right.pop(0))            # lowest right note -> left hand
    return left, right


def finger_targets_local(plan: FingeringPlan) -> np.ndarray:
    """(T, 10, 3) target positions in the piano-local frame.

    Active fingers target their assigned key's top surface (slightly pressed so
    the IK reference actually sounds the key); idle fingers hover above home.
    """
    key_top = geom.key_local_top_positions()                  # (88, 3)
    T = plan.num_steps
    out = np.zeros((T, NUM_FINGERS, 3), dtype=np.float32)
    for f in range(NUM_FINGERS):
        keys = plan.finger_key[:, f]
        active = plan.finger_active[:, f]
        src = np.where(active, keys, plan.home_key[f])        # (T,)
        out[:, f, :] = key_top[src]
        # press depth for active, hover for idle
        out[:, f, 2] += np.where(active, -geom.PRESS_DEPTH, geom.HOVER_CLEARANCE)
    return out
