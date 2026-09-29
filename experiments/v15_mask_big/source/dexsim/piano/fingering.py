"""Automatic fingering: assign each active note to a specific finger per step.

RoboPianist needs human fingering labels (the PIG dataset) and shows that
*without* a fingering signal the policy never learns (F1 = 0). PianoMime gets the
same signal from human video. We have neither, so we synthesize the fingering —
which is exactly the structure-injection that makes the high-DoF search tractable.

Two synthesizers, selected by ``plan_fingering(..., method=...)``:

  * ``"heuristic"`` (default) — the hand-rolled rule below; cheap and stable.
  * ``"ot"`` — **optimal transport**, the RP1M trick: at each step assign the
    active keys to fingers by *minimum total movement cost* (a linear-sum /
    Jonker-Volgenant assignment over the distance from each finger's current
    position to each candidate key, plus hand-side and black-key penalties).
    This drops the dependence on human fingering labels entirely — fingering is
    discovered purely from geometry, exactly as RP1M does for ~2k songs. See
    :func:`plan_fingering_ot`.

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

# Optimal-transport (RP1M-style) assignment uses a min-cost matching solver.
try:  # SciPy is in the venv; guard so the heuristic path works without it.
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:  # pragma: no cover
    _lsa = None

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


def plan_fingering(key_activation: np.ndarray, method: str = "heuristic",
                   swap_hands: bool = False, **ot_kwargs) -> FingeringPlan:
    """Assign fingers for every control step.

    Args:
        key_activation (T, 88) bool -- which keys should sound each step.
        method: ``"heuristic"`` (default, the rule below) or ``"ot"`` (RP1M-style
            optimal-transport assignment, see :func:`plan_fingering_ot`).
        swap_hands: GEOMETRY GUARDRAIL. The split sends low-pitch keys to the LEFT
            finger group (0-4) and high-pitch to the RIGHT (5-9), which assumes low
            pitch is physically on the left robot's side. If the piano is mounted so
            that low-pitch keys actually sit on the RIGHT robot's side (e.g. a 180deg
            flip), pass ``swap_hands=True`` so low-pitch keys go to the RIGHT group
            and high-pitch to the LEFT -- i.e. each hand is assigned the keys in ITS
            OWN physical vicinity instead of reaching across the body. The caller
            (PianoEnv) derives this from the actual key/base world-Y geometry, so it
            is not hardcoded. Without it the two arms cross into the middle and jam.
        **ot_kwargs: forwarded to :func:`plan_fingering_ot` when ``method="ot"``.
    """
    if method == "ot":
        return plan_fingering_ot(key_activation, **ot_kwargs)
    if method == "hand":
        return plan_fingering_hand(key_activation, swap_hands=swap_hands, **ot_kwargs)
    if method == "traj":
        return plan_fingering_traj(key_activation, swap_hands=swap_hands, **ot_kwargs)
    if method == "seq":
        from .fingering_seq import plan_fingering_seq
        return plan_fingering_seq(key_activation, swap_hands=swap_hands, **ot_kwargs)
    if method != "heuristic":
        raise ValueError(f"unknown fingering method: {method!r}")
    T = key_activation.shape[0]
    finger_key = np.full((T, NUM_FINGERS), -1, dtype=np.int64)
    finger_active = np.zeros((T, NUM_FINGERS), dtype=bool)
    home = _home_keys()
    # low-pitch -> low_order group, high-pitch -> high_order group. swap_hands flips
    # which physical hand each spatial half is routed to (see docstring).
    low_order, high_order = (_RIGHT_ORDER, _LEFT_ORDER) if swap_hands else (_LEFT_ORDER, _RIGHT_ORDER)
    if swap_hands:
        # keep idle "home" keys with their hand: left group now owns the upper half.
        home = np.concatenate([home[FINGERS_PER_HAND:], home[:FINGERS_PER_HAND]])

    for t in range(T):
        active = np.nonzero(key_activation[t])[0]
        if active.size == 0:
            continue
        active = np.sort(active)
        # balance the hand split: aim for <=5 per hand. Default split at the
        # keyboard middle, then rebalance if one hand is overloaded.
        split = _balanced_split(active)
        low = [int(k) for k in active if k < split]    # spatial low-pitch half
        high = [int(k) for k in active if k >= split]  # spatial high-pitch half
        # if a hand is overloaded but the other is empty/light, shift the split
        low, high = _rebalance(low, high)
        _assign_hand(low, low_order, finger_key, finger_active, t)
        _assign_hand(high, high_order, finger_key, finger_active, t)

    return FingeringPlan(finger_key=finger_key, finger_active=finger_active, home_key=home)


def _balanced_split(active: np.ndarray) -> int:
    """Pitch boundary (key index) splitting notes into left/right hands.

    ALWAYS the keyboard middle (~spatial center Y=0). The old code split a
    one-sided cluster at its OWN median, which handed half of a left-side cluster
    to the RIGHT hand -> a cross-body reach the right arm physically can't make
    (110mm+ fingertip error). Splitting strictly at the middle keeps each hand in
    its own reachable half: a left-clustered passage now goes entirely to the
    left hand (all keys < mid), and vice-versa."""
    return NUM_KEYS // 2


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


# ---------------------------------------------------------------------------
# Optimal-transport fingering (the RP1M trick — no human labels needed)
# ---------------------------------------------------------------------------

# Per-hand finger order matching FINGERTIP_BODIES / the global finger indices:
#   left  = [0..4]  = [thumb, index, middle, ring, little]
#   right = [5..9]  = [thumb, index, middle, ring, little]
_LEFT_FINGERS = (L_THUMB, L_INDEX, L_MIDDLE, L_RING, L_LITTLE)
_RIGHT_FINGERS = (R_THUMB, R_INDEX, R_MIDDLE, R_RING, R_LITTLE)
_THUMBS = (L_THUMB, R_THUMB)


def window_home_keys(left_window: tuple[int, int],
                     right_window: tuple[int, int]) -> np.ndarray:
    """(10,) idle home keys spreading each hand's five fingers evenly over its
    reachable key window (inclusive), little fingers outward, thumbs inward:
    left [lf..th] low->high over left_window, right [th..lf] over right_window."""
    home = np.zeros(NUM_FINGERS, dtype=np.int64)
    l0, l1 = left_window
    r0, r1 = right_window
    for slot, key in zip(_LEFT_ORDER, np.linspace(l0, l1, FINGERS_PER_HAND).round().astype(int)):
        home[slot] = key
    for slot, key in zip(_RIGHT_ORDER, np.linspace(r0, r1, FINGERS_PER_HAND).round().astype(int)):
        home[slot] = key
    return home


# default fingertip offsets from the palm along the keyboard (m), per finger
# [L th,ff,mf,rf,lf, R th,ff,mf,rf,lf]; measured on the Menagerie hands at the
# ready pose (thumb inward toward the keyboard middle, little finger outward).
# The env passes the live measurement; this is the fallback.
DEFAULT_FINGER_OFFSETS = np.array([+0.052, +0.022, 0.0, -0.022, -0.044,
                                   -0.052, -0.022, 0.0, +0.022, +0.044])


def _fit_hand(keys, fingers, off, key_y, is_black, black_thumb_weight,
              balance_weight, usage, sticky_weight=0.0, key_owner=None):
    """Best (worst-finger error, total error, hand centre, assignment) for one
    chord. The hand centre is free: search it over +/-6 cm around the chord
    midpoint and keep the assignment with the smallest worst-finger error. A
    fixed midpoint picked ff+lf for a 10 cm chord (1.7 cm off each) when th+rf
    fits it to 0.65 cm."""
    c0 = key_y[keys].mean()
    best = None
    thumb_rows = np.isin(fingers, list(_THUMBS))
    for dc in np.arange(-0.06, 0.0601, 0.005):
        c = c0 + dc
        cost = np.abs(key_y[keys][None, :] - (c + off[fingers])[:, None])   # (5, K)
        cost = cost + (black_thumb_weight * is_black[keys][None, :]) * thumb_rows[:, None]
        if balance_weight > 0.0:
            share = usage[fingers] / max(usage.sum(), 1.0)
            cost = cost + balance_weight * share[:, None]
        if sticky_weight > 0.0 and key_owner is not None:
            # Charge a finger for taking a key that another finger already plays.
            # Measured on the 0.673 policy: 6 of 16 goal keys were handed to two
            # different fingers, and the split was not benign -- key 28 sounded
            # on 2 of 4 middle-finger notes and 0 of 18 little-finger notes, key
            # 40 on 5 of 6 middle-finger notes and 0 of 4 thumb notes. Those two
            # keys alone are 22 of the song's 55 missed notes. One key wants one
            # motor pattern; splitting it starves the finger that sees it less.
            owners = np.array([key_owner.get(int(k), -1) for k in keys])
            mine = (owners[None, :] == fingers[:, None])
            cost = cost + sticky_weight * (~mine & (owners[None, :] >= 0))
        r_, c_ = _lsa(cost)
        worst = float(cost[r_, c_].max()); total = float(cost[r_, c_].sum())
        if best is None or (worst, total) < (best[0], best[1]):
            best = (worst, total, c, r_, c_)
    return best


def plan_fingering_hand(key_activation: np.ndarray, *, finger_offsets=None,
                        swap_hands: bool = False,
                        black_thumb_weight: float = 0.03,
                        pedal_release: bool = True,
                        hand_span: float = 0.15,
                        stagger_steps: int = 2,
                        exclude_fingers=(),
                        balance_weight: float = 0.0,
                        hand_tol: float = 0.0,
                        sticky_weight: float = 0.0,
                        split_key: int | None = None) -> FingeringPlan:
    """Hand-relative assignment for a hand that MOVES (rail servo / free hand).

    Absolute-key planners (heuristic, ot) assume fixed finger homes, so once
    the song is not folded into a fixed window they hand a single chord to
    fingers 30 cm apart. Here, per step and per hand: place the hand at the
    centroid of that hand's notes, then match notes to fingers by the cost
        |key_y[k] - (centroid + finger_offset[f])|
    (linear-sum assignment), i.e. whichever finger naturally sits over the key
    once the hand is centred. Consistent with the env's finger-offset rail
    servo, which centres the hand so the ASSIGNED finger lands on its key.
    Hands split at the keyboard middle with overflow rebalancing, as in the
    heuristic planner. Idle homes are the finger offsets around each hand's
    base position (used only for the hover targets).
    """
    if _lsa is None:  # pragma: no cover
        raise RuntimeError("plan_fingering_hand needs scipy (linear_sum_assignment).")
    off = (DEFAULT_FINGER_OFFSETS if finger_offsets is None
           else np.asarray(finger_offsets, dtype=np.float64))
    T = key_activation.shape[0]
    is_black = geom.KEY_IS_BLACK
    finger_key = np.full((T, NUM_FINGERS), -1, dtype=np.int64)
    finger_active = np.zeros((T, NUM_FINGERS), dtype=bool)
    home = _home_keys()
    low_group, high_group = ((_RIGHT_FINGERS, _LEFT_FINGERS) if swap_hands
                             else (_LEFT_FINGERS, _RIGHT_FINGERS))
    act = key_activation.astype(bool)
    onset = act & ~np.concatenate([np.zeros((1, act.shape[1]), bool), act[:-1]])
    # PEDAL-AWARE: each hand's fingers serve only its NEWEST onset group; older
    # notes still sounding are carried by the sustain pedal (finger_active
    # False for them, so the servo / targets / shaping all follow the new
    # note). ARPEGGIATE: an onset group wider than hand_span is split in
    # time -- the half nearer the hand's previous position plays at t, the
    # other half `stagger_steps` later (the onset reward window absorbs it).
    key_y = geom.key_local_top_positions()[:, 1]
    # Running share of notes each finger has taken. With balance_weight > 0 an
    # over-used finger costs more, so work spreads instead of collapsing onto
    # the two centre fingers (measured: middle+ring took 62% of this song).
    # The term is in metres so it trades directly against finger-to-key error:
    # balance_weight is the extra "distance" a finger pays at 100% usage share.
    usage = np.zeros(NUM_FINGERS, dtype=np.float64)
    key_owner: dict[int, int] = {}    # key -> the finger that has played it
    cur_group = [None, None]          # per hand: np.array of keys the fingers hold
    hand_y = [None, None]             # per hand: last centre y
    pending = [[], []]                # per hand: (t_due, keys) staggered halves
    for t in range(T):
        active = np.sort(np.nonzero(act[t])[0])
        # hand split boundary: keys >= split go to the right hand. cfg.hand_split_key
        # (default 44 = keyboard middle). 40 on nettspend: key 40 is assigned to
        # the left thumb (cannot reach it) but is in fact hit by the RIGHT index
        # finger 11/12 times in the 0.750 model -- give it to the hand that plays it.
        split = int(split_key) if split_key is not None else (_balanced_split(active) if active.size else NUM_KEYS // 2)
        low = [int(k) for k in active if k < split]
        high = [int(k) for k in active if k >= split]
        low, high = _rebalance(low, high)
        for hi, (keys_all, fingers) in enumerate(((low, low_group), (high, high_group))):
            keys_all = np.array(keys_all, dtype=int)
            fingers = np.array([f for f in fingers
                                if int(f) not in {int(x) for x in exclude_fingers}]
                               or list(fingers))
            if pedal_release:
                new = np.array([k for k in keys_all if onset[t, k]], dtype=int)
                # staggered halves that are now due
                due = [k for (td, ks) in pending[hi] if td <= t for k in ks]
                pending[hi] = [(td, ks) for (td, ks) in pending[hi] if td > t]
                if new.size:
                    too_wide = (new.size >= 2
                                and key_y[new].max() - key_y[new].min() > hand_span)
                    # A chord can be far narrower than hand_span and still be
                    # unplayable in one position: the fingers' natural spacing
                    # has to match the chord's. Measured on nettspend, every
                    # left-little-finger note is in a 2-note chord, and the ones
                    # it misses need a palm spread of 2.7-6.3 cm that no single
                    # position satisfies -- the servo splits the difference and
                    # BOTH fingers land half of it off their key. Stagger those
                    # too, so each note gets the hand to itself.
                    if not too_wide and hand_tol > 0.0 and new.size >= 2:
                        fit = _fit_hand(new, fingers, off, key_y, is_black,
                                        black_thumb_weight, balance_weight, usage)
                        too_wide = fit[0] > hand_tol
                    if too_wide:
                        mid = 0.5 * (key_y[new].max() + key_y[new].min())
                        lo_half = new[key_y[new] <= mid]; hi_half = new[key_y[new] > mid]
                        ref = hand_y[hi] if hand_y[hi] is not None else key_y[new].mean()
                        first, later = ((lo_half, hi_half) if abs(key_y[lo_half].mean() - ref)
                                        <= abs(key_y[hi_half].mean() - ref) else (hi_half, lo_half))
                        pending[hi].append((t + stagger_steps, [int(k) for k in later]))
                        new = first
                    group = np.array(sorted(set(new.tolist()) | set(due)), dtype=int)
                elif due:
                    group = np.array(sorted(due), dtype=int)
                else:
                    group = cur_group[hi]
                # keep only keys still active
                group = (np.array([k for k in group if act[t, k]], dtype=int)
                         if group is not None else np.array([], dtype=int))
                cur_group[hi] = group if group.size else None
                keys = group
            else:
                keys = keys_all
            if keys.size == 0:
                continue
            if len(keys) > FINGERS_PER_HAND:            # >5 notes: keep 5 spanning the range
                idx = np.linspace(0, len(keys) - 1, FINGERS_PER_HAND).round().astype(int)
                keys = keys[idx]
            _, _, c, rows, cols = _fit_hand(
                keys, fingers, off, key_y, is_black, black_thumb_weight,
                balance_weight, usage, sticky_weight, key_owner)
            hand_y[hi] = c
            for r, cidx in zip(rows, cols):
                f = int(fingers[r]); k = int(keys[cidx])
                finger_key[t, f] = k
                finger_active[t, f] = True
                usage[f] += 1.0
                key_owner.setdefault(k, f)
    return FingeringPlan(finger_key=finger_key, finger_active=finger_active, home_key=home)


def plan_fingering_ot(
    key_activation: np.ndarray,
    *,
    side_weight: float = 2.5,
    black_thumb_weight: float = 0.03,
    home_pull_weight: float = 0.15,
    smooth: bool = True,
    home_keys: np.ndarray | None = None,
) -> FingeringPlan:
    """RP1M-style optimal-transport fingering.

    ``home_keys`` (10,) overrides the idle home key per finger. The default
    spreads the fingers over the whole keyboard (thumbs at the middle), which
    for a song folded into two narrow hand windows leaves only ONE finger per
    hand anywhere near the notes -- the matching then re-uses that finger for
    every key. Pass windowed homes (see :func:`window_home_keys`) so all five
    fingers of a hand start over its window and get distinct keys.

    At each control step the active keys are assigned to fingers by **minimum
    total movement cost** — a linear-sum (Jonker-Volgenant) assignment, exactly
    the matching RP1M solves online to auto-finger ~2k pieces without any human
    labels. The cost from finger ``f`` to key ``k`` is

        ||current_pos[f] - key_pos[k]||                       (move it the least)
      + side_weight * (how far k is into the *wrong* keyboard half for f)
      + black_thumb_weight * [k is a black key and f is a thumb]

    Idle fingers are pulled gently back toward their spread "home" so they stay
    ready (``home_pull_weight``). ``current_pos`` carries across steps (when
    ``smooth=True``) so fingers prefer to stay put — giving temporally coherent,
    low-motion fingerings, just like the online RP1M solver.

    Returns the same :class:`FingeringPlan` as the heuristic planner, so it is a
    drop-in replacement for the IK reference / reward-shaping signal.
    """
    if _lsa is None:  # pragma: no cover
        raise RuntimeError("plan_fingering_ot needs scipy (scipy.optimize). "
                           "Install scipy or use method='heuristic'.")
    T = key_activation.shape[0]
    key_pos = geom.key_local_top_positions()                  # (88, 3)
    mid_y = float(key_pos[NUM_KEYS // 2, 1])                   # keyboard centre (local Y)
    home = (_home_keys() if home_keys is None
            else np.asarray(home_keys, dtype=np.int64).copy())   # (10,) home key idx
    home_pos = key_pos[home]                                   # (10, 3)
    is_black = geom.KEY_IS_BLACK                               # (88,)

    finger_key = np.full((T, NUM_FINGERS), -1, dtype=np.int64)
    finger_active = np.zeros((T, NUM_FINGERS), dtype=bool)
    cur = home_pos.copy()                                      # (10, 3) live finger pos
    n_dropped = 0

    for t in range(T):
        active = np.nonzero(key_activation[t])[0]
        if active.size == 0:
            if smooth:                                        # drift idle hands home
                cur += home_pull_weight * (home_pos - cur)
            continue

        kp = key_pos[active]                                  # (K, 3)
        K = active.size
        # base move cost: every finger -> every active key
        cost = np.linalg.norm(cur[:, None, :] - kp[None, :, :], axis=2)  # (10, K)
        # hand-side penalty: left fingers pay for keys above mid, right below.
        dy = kp[:, 1] - mid_y                                 # (K,) +ve = right half
        left_pen = np.maximum(dy, 0.0)[None, :]               # left fingers cross up
        right_pen = np.maximum(-dy, 0.0)[None, :]             # right fingers cross down
        side = np.zeros((NUM_FINGERS, K), dtype=np.float64)
        side[list(_LEFT_FINGERS), :] = left_pen
        side[list(_RIGHT_FINGERS), :] = right_pen
        cost = cost + side_weight * side
        # black-key-on-thumb penalty (thumbs are short / awkward on black keys)
        black = is_black[active][None, :].astype(np.float64)  # (1, K)
        cost[list(_THUMBS), :] += black_thumb_weight * black[0]

        if K <= NUM_FINGERS:
            # pad with idle columns so all 10 fingers get a column; idle cost is a
            # gentle pull home, so unneeded fingers return to a ready spread.
            n_idle = NUM_FINGERS - K
            idle = home_pull_weight * np.linalg.norm(cur - home_pos, axis=1)  # (10,)
            pad = np.repeat(idle[:, None], n_idle, axis=1)    # (10, n_idle)
            full = np.concatenate([cost, pad], axis=1)        # (10, 10)
            rows, cols = _lsa(full)
            for f, c in zip(rows, cols):
                if c < K:                                     # assigned a real key
                    k = int(active[c])
                    finger_key[t, f] = k
                    finger_active[t, f] = True
                    cur[f] = key_pos[k]
                elif smooth:                                  # idle -> drift home
                    cur[f] += home_pull_weight * (home_pos[f] - cur[f])
        else:
            # more simultaneous notes than fingers: cover the cheapest 10, drop rest.
            rows, cols = _lsa(cost)                            # 10 finger<->key pairs
            for f, c in zip(rows, cols):
                k = int(active[c])
                finger_key[t, f] = k
                finger_active[t, f] = True
                cur[f] = key_pos[k]
            n_dropped += K - NUM_FINGERS

    if n_dropped:
        print(f"[fingering:ot] {n_dropped} note-instances exceeded 10 fingers "
              f"and were dropped (>10-key polyphony).")
    return FingeringPlan(finger_key=finger_key, finger_active=finger_active,
                         home_key=home)


def plan_fingering_traj(key_activation: np.ndarray, *, finger_offsets=None,
                        swap_hands: bool = False, black_thumb_weight: float = 0.03,
                        exclude_fingers=(L_THUMB, R_THUMB),
                        hand_speed_m_s: float = 1.5, control_dt: float = 0.05,
                        repeat_penalty: float = 0.0) -> FingeringPlan:
    """Trajectory-tracking assignment: score fingers from where the hand ACTUALLY
    is, not from a pose re-centred on this step's notes.

    `plan_fingering_hand` re-centres the hand on the notes before assigning, so
    for a single note the centroid equals that note and the cost collapses to
    each finger's own offset from the palm. The two centre fingers then win
    every time by construction -- measured on nettspend, middle + ring took 62%
    of the workload while the little fingers took 14%.

    Here each hand carries its position across steps. A note is scored against
    where each finger currently sits, so a note left of the hand naturally goes
    to a left-side finger. After assigning, the hand moves toward the position
    that puts the assigned fingers on their keys, rate-limited to
    ``hand_speed_m_s * control_dt`` -- the same constraint the rail servo has,
    so the plan and the servo agree instead of fighting.

    ``exclude_fingers`` never receive notes. The thumbs are excluded by default:
    measured on this embodiment, a thumb at its joint limit reaches only the key
    top (+0.0 cm) and cannot depress a key at all, so any note assigned to one
    is unplayable.
    """
    if _lsa is None:  # pragma: no cover
        raise RuntimeError("plan_fingering_traj needs scipy (linear_sum_assignment).")
    off = (DEFAULT_FINGER_OFFSETS if finger_offsets is None
           else np.asarray(finger_offsets, dtype=np.float64))
    T = key_activation.shape[0]
    key_y = geom.key_local_top_positions()[:, 1]
    is_black = geom.KEY_IS_BLACK
    finger_key = np.full((T, NUM_FINGERS), -1, dtype=np.int64)
    finger_active = np.zeros((T, NUM_FINGERS), dtype=bool)
    home = _home_keys()
    low_group, high_group = ((_RIGHT_FINGERS, _LEFT_FINGERS) if swap_hands
                             else (_LEFT_FINGERS, _RIGHT_FINGERS))
    banned = set(int(f) for f in exclude_fingers)
    max_step = float(hand_speed_m_s) * float(control_dt)
    # each hand starts centred on its own home keys
    hand_y = {}
    for keys_group in (low_group, high_group):
        g = tuple(int(f) for f in keys_group)
        hand_y[g] = float(np.mean([key_y[home[f]] - off[f] for f in g]))
    last_key = {f: None for f in range(NUM_FINGERS)}
    last_step = {f: -10 ** 6 for f in range(NUM_FINGERS)}

    for t in range(T):
        active = np.sort(np.nonzero(key_activation[t])[0])
        if active.size == 0:
            continue
        split = _balanced_split(active)
        low = [int(k) for k in active if k < split]
        high = [int(k) for k in active if k >= split]
        low, high = _rebalance(low, high)
        for keys, fingers in ((low, low_group), (high, high_group)):
            g = tuple(int(f) for f in fingers)
            usable = [f for f in g if f not in banned]
            if not keys or not usable:
                continue
            keys_a = np.array(keys)
            if len(keys_a) > len(usable):
                idx = np.linspace(0, len(keys_a) - 1, len(usable)).round().astype(int)
                keys_a = keys_a[idx]
            fa = np.array(usable)
            # cost from where each finger IS right now
            finger_pos = hand_y[g] + off[fa]
            cost = np.abs(key_y[keys_a][None, :] - finger_pos[:, None])
            thumb_rows = np.isin(fa, list(_THUMBS))
            if thumb_rows.any():
                cost[thumb_rows] += black_thumb_weight * is_black[keys_a][None, :]
            if repeat_penalty > 0.0:
                for r, f in enumerate(fa):
                    for c, k in enumerate(keys_a):
                        if last_key[int(f)] is not None and last_key[int(f)] != int(k) \
                           and t - last_step[int(f)] <= 2:
                            cost[r, c] += repeat_penalty
            rows, cols = _lsa(cost)
            assigned = []
            for r, c in zip(rows, cols):
                f = int(fa[r]); k = int(keys_a[c])
                finger_key[t, f] = k
                finger_active[t, f] = True
                last_key[f] = k
                last_step[f] = t
                assigned.append((f, k))
            if assigned:
                # where the palm would need to be for these fingers to be on
                # their keys; move toward it at the rail's real speed limit
                target = float(np.mean([key_y[k] - off[f] for f, k in assigned]))
                delta = np.clip(target - hand_y[g], -max_step, max_step)
                hand_y[g] = hand_y[g] + delta
    return FingeringPlan(finger_key=finger_key, finger_active=finger_active, home_key=home)
