"""Sequence fingering planner ("seq", 2026-09-27).

The per-step planners (``hand``, ``traj``) decide each control step on its
own: re-centre the hand on this step's notes, pick the finger whose resting
offset is nearest. On a moving hand that collapses to ONE finger playing
everything while the rail carries it note to note (measured on nettspend:
the right ring finger took 74% of the hand's onsets, both thumbs 0, and the
right palm travelled 12 m in a 64 s song; 44/45 alternations were one finger
with the rail shuffling 1.2 cm per note, and the 44+56 octave was one finger
and a 16 cm sprint each way).

This planner chooses the PALM POSITION of both hands over the whole song
with a dynamic program (Viterbi over a 1 cm palm grid per hand), then assigns
fingers by their offset from that palm. The four rules it encodes:

  1. The hand stays put unless it has to move: every palm move pays
     ``move_cost`` plus ``travel_weight`` per metre.
  2. Adjacent notes get adjacent fingers: with the palm fixed, the finger
     over the key takes it; re-using a finger on a different key within
     ``switch_window`` steps pays ``switch_penalty``; a finger holding a key
     keeps it.
  3. Chords use the spread of the hand: notes are matched to fingers in
     keyboard order (monotone), so a 16 cm pair goes thumb + little with the
     palm between them.
  4. A hand is never given notes it cannot span: which hand takes each note is
     part of the same optimisation (every split of the sorted onset group is
     scored against both palms), a note farther than ``reach_max`` from every
     finger is dropped at ``drop_cost`` and re-issued ``stagger_steps`` later
     (the onset reward window absorbs it), and the palms may not cross.

Pedal-aware like ``plan_fingering_hand``: fingers serve the newest onset
group of their hand; older notes still sounding ride the sustain pedal.
Costs are in metres of finger-to-key error so they trade directly.
"""

from __future__ import annotations

import itertools

import numpy as np

from .midi import NUM_KEYS
from . import geometry as geom
from .fingering import (FingeringPlan, DEFAULT_FINGER_OFFSETS, NUM_FINGERS,
                        FINGERS_PER_HAND, _LEFT_FINGERS, _RIGHT_FINGERS, _THUMBS,
                        _home_keys, _lsa)

_INF = 1e9


def _hand_cost_grid(keys, off_sorted, P, key_y, is_black, thumb_mask,
                    reach_tol, reach_max, drop_cost, black_thumb_weight,
                    thumb_reach_max=None, allowed=None):
    """Best cost of putting sorted ``keys`` (K<=5) on one hand at every palm in
    ``P``, matching notes to fingers in keyboard order and allowing a note to be
    dropped at ``drop_cost``. Returns (G,) costs. Small DP over (note, finger)."""
    K = len(keys)
    F = len(off_sorted)
    if K == 0:
        return np.zeros(len(P))
    # per-note, per-finger stretch cost at every palm: (G, K, F)
    d = np.abs(key_y[keys][None, :, None] - (P[:, None, None] + off_sorted[None, None, :]))
    c = np.maximum(d - reach_tol, 0.0)
    rmax = np.where(thumb_mask, thumb_reach_max if thumb_reach_max is not None else reach_max, reach_max)
    c = np.where(d > rmax[None, None, :], _INF, c)
    if allowed is not None:                     # v15: measured (finger, key) feasibility, (F, K)
        c = np.where(~np.asarray(allowed, bool).T[None, :, :], _INF, c)
    c = c + black_thumb_weight * (is_black[keys][None, :, None] & thumb_mask[None, None, :])
    # f[i][j] = min cost for notes i.. using fingers j..
    f = [[None] * (F + 1) for _ in range(K + 1)]
    for j in range(F + 1):
        f[K][j] = np.zeros(len(P))
    for i in range(K - 1, -1, -1):
        f[i][F] = np.full(len(P), drop_cost * (K - i))
        for j in range(F - 1, -1, -1):
            f[i][j] = np.minimum.reduce([
                f[i + 1][j + 1] + c[:, i, j],      # note i on finger j
                f[i][j + 1],                       # skip finger j
                f[i + 1][j] + drop_cost,           # drop note i
            ])
    return f[0][0]


def _hand_assign(keys, fingers_sorted, off_sorted, p, key_y, is_black,
                 reach_tol, reach_max, drop_cost, black_thumb_weight,
                 last_key, last_step, t, hold, switch_penalty, switch_window,
                 thumb_reach_max=None, allowed=None):
    """Finger assignment for one hand at a known palm ``p`` (output pass).
    Linear-sum assignment with continuity terms; notes out of reach are
    dropped. Returns list of (finger, key)."""
    K = len(keys)
    if K == 0:
        return []
    F = len(fingers_sorted)
    d = np.abs(key_y[keys][None, :] - (p + off_sorted)[:, None])           # (F, K)
    cost = np.maximum(d - reach_tol, 0.0)
    thumb = np.isin(fingers_sorted, list(_THUMBS))
    rmax = np.where(thumb, thumb_reach_max if thumb_reach_max is not None else reach_max, reach_max)
    cost = np.where(d > rmax[:, None], drop_cost * 4.0, cost)             # never beats dropping
    if allowed is not None:                                                # v15 feasibility mask (F, K)
        cost = np.where(~np.asarray(allowed, bool), drop_cost * 4.0, cost)
    cost = cost + black_thumb_weight * (thumb[:, None] & is_black[keys][None, :])
    for r, f in enumerate(fingers_sorted):
        for cidx, k in enumerate(keys):
            if hold.get(int(k)) == int(f):
                cost[r, cidx] -= 0.05                                      # keep a held key
            elif (last_key[int(f)] is not None and last_key[int(f)] != int(k)
                  and t - last_step[int(f)] <= switch_window):
                cost[r, cidx] += switch_penalty
    # pad with drop columns so every note may be dropped instead of stretched
    full = np.concatenate([cost, np.full((F, K), drop_cost)], axis=1)
    rows, cols = _lsa(full)
    out = []
    for r, cidx in zip(rows, cols):
        if cidx < K and full[r, cidx] < drop_cost:
            out.append((int(fingers_sorted[r]), int(keys[cidx])))
    return out


def plan_fingering_seq(key_activation: np.ndarray, *, finger_offsets=None,
                       swap_hands: bool = False, black_thumb_weight: float = 0.03,
                       stagger_steps: int = 4, base_offset: float = 0.30,
                       rail_limit: float = 0.32, grid: float = 0.01,
                       reach_tol: float = 0.010, reach_max: float = 0.030,
                       thumb_reach_max: float = 0.045,
                       travel_weight: float = 0.5, move_cost: float = 0.01,
                       drop_cost: float = 0.15, min_hand_sep: float = 0.20,
                       switch_penalty: float = 0.01, switch_window: int = 10,
                       finger_key_mask=None, exclude_fingers=(),
                       verbose: bool = True) -> FingeringPlan:
    if _lsa is None:  # pragma: no cover
        raise RuntimeError("plan_fingering_seq needs scipy (linear_sum_assignment).")
    off = (DEFAULT_FINGER_OFFSETS if finger_offsets is None
           else np.asarray(finger_offsets, dtype=np.float64))
    act = key_activation.astype(bool)
    T = act.shape[0]
    key_y = geom.key_local_top_positions()[:, 1]
    is_black = np.asarray(geom.KEY_IS_BLACK, dtype=bool)
    low_group, high_group = ((_RIGHT_FINGERS, _LEFT_FINGERS) if swap_hands
                             else (_LEFT_FINGERS, _RIGHT_FINGERS))
    groups = [np.array(low_group), np.array(high_group)]
    # fingers of each hand sorted by offset (keyboard order), thumb mask, offsets
    order = [np.argsort(off[g]) for g in groups]
    fingers_sorted = [groups[h][order[h]] for h in range(2)]
    off_sorted = [off[fingers_sorted[h]] for h in range(2)]
    thumb_mask = [np.isin(fingers_sorted[h], list(_THUMBS)) for h in range(2)]
    # v15: hard feasibility -- measured (finger, key) reach mask and excluded fingers
    fk_mask = np.ones((NUM_FINGERS, NUM_KEYS), dtype=bool)
    if finger_key_mask is not None:
        fk_mask &= np.asarray(finger_key_mask, dtype=bool).reshape(NUM_FINGERS, NUM_KEYS)
    for f in exclude_fingers:
        fk_mask[int(f), :] = False
    def allowed_for(h, ks):
        return fk_mask[fingers_sorted[h]][:, np.asarray(ks, dtype=int)]      # (F, K)
    if verbose:
        n_ex = [int((~fk_mask[f]).sum()) for f in range(NUM_FINGERS)]
        print(f"[fingering:seq] feasibility mask: forbidden keys per finger {n_ex}"
              f"{' (excluded fingers ' + str(tuple(int(f) for f in exclude_fingers)) + ')' if len(exclude_fingers) else ''}")
    # palm grids in the key frame: low hand left of the keyboard middle, high hand right
    mid = 0.5 * (float(key_y.min()) + float(key_y.max()))
    P = [np.arange(mid - base_offset - rail_limit, mid + 1e-9, grid),
         np.arange(mid, mid + base_offset + rail_limit + 1e-9, grid)]
    G0, G1 = len(P[0]), len(P[1])
    # transition costs per hand: (from, to)
    C = [move_cost * (np.abs(P[h][:, None] - P[h][None, :]) > 1e-9)
         + travel_weight * np.abs(P[h][:, None] - P[h][None, :]) for h in range(2)]
    cross = (P[1][None, :] - P[0][:, None]) < min_hand_sep                # (G0, G1) forbidden

    onset = act & ~np.concatenate([np.zeros((1, NUM_KEYS), bool), act[:-1]])
    extra_onsets = [set() for _ in range(T)]      # staggered re-issues
    dropped_once: set[tuple[int, int]] = set()

    def onset_keys(t):
        ks = set(np.nonzero(onset[t])[0].tolist()) | extra_onsets[t]
        return np.array(sorted(k for k in ks if act[t, k]), dtype=int)

    def constraint_sets():
        """Per step: the keys the fingers must be on (newest onset group, held)."""
        sets = [None] * T
        cur = np.array([], dtype=int)
        for t in range(T):
            ks = onset_keys(t)
            if ks.size:
                cur = ks
            else:
                cur = np.array([k for k in cur if act[t, k]], dtype=int)
            sets[t] = cur
        return sets

    def emission(keys):
        """(G0, G1) min over splits of low-hand cost + high-hand cost."""
        K = len(keys)
        E = np.full((G0, G1), _INF)
        for s in range(K + 1):
            lo, hi = keys[:s], keys[s:]
            parts = []
            for h, ks in ((0, lo), (1, hi)):
                if len(ks) > FINGERS_PER_HAND:            # >5: cover 5 spanning, drop the rest
                    idx = np.linspace(0, len(ks) - 1, FINGERS_PER_HAND).round().astype(int)
                    extra = len(ks) - FINGERS_PER_HAND
                    ks = ks[idx]
                else:
                    extra = 0
                e = _hand_cost_grid(ks, off_sorted[h], P[h], key_y, is_black, thumb_mask[h],
                                    reach_tol, reach_max, drop_cost, black_thumb_weight, thumb_reach_max,
                                    allowed_for(h, ks))
                parts.append(e + drop_cost * extra)
            E = np.minimum(E, parts[0][:, None] + parts[1][None, :])
        E[cross] = _INF
        return E

    def split_at(keys, pl, pr):
        """Best split of sorted keys between the hands at known palms."""
        best, best_s = _INF, 0
        for s in range(len(keys) + 1):
            tot = 0.0
            for h, ks in ((0, keys[:s]), (1, keys[s:])):
                if len(ks) > FINGERS_PER_HAND:
                    idx = np.linspace(0, len(ks) - 1, FINGERS_PER_HAND).round().astype(int)
                    tot += drop_cost * (len(ks) - FINGERS_PER_HAND); ks = ks[idx]
                p = np.array([pl if h == 0 else pr])
                tot += float(_hand_cost_grid(ks, off_sorted[h], p, key_y, is_black, thumb_mask[h],
                                             reach_tol, reach_max, drop_cost, black_thumb_weight, thumb_reach_max,
                                             allowed_for(h, ks))[0])
            if tot < best:
                best, best_s = tot, s
        return best_s

    def viterbi(sets):
        V = emission(sets[0])
        argL = np.zeros((T, G0, G1), dtype=np.int16); argR = np.zeros((T, G0, G1), dtype=np.int16)
        for t in range(1, T):
            tmp = V[:, None, :] + C[0][:, :, None]                # (pL, pL', pR)
            argL[t] = tmp.argmin(0); M = tmp.min(0)               # (pL', pR)
            tmp2 = M[:, :, None] + C[1][None, :, :]               # (pL', pR, pR')
            argR[t] = tmp2.argmin(1); V = tmp2.min(1) + emission(sets[t])
        # backtrack
        pl = np.zeros(T, dtype=int); pr = np.zeros(T, dtype=int)
        i, j = np.unravel_index(int(V.argmin()), V.shape); total = float(V[i, j])
        pl[T - 1], pr[T - 1] = i, j
        for t in range(T - 1, 0, -1):
            jprev = int(argR[t, i, j]); iprev = int(argL[t, i, jprev])
            pl[t - 1], pr[t - 1] = iprev, jprev
            i, j = iprev, jprev
        return pl, pr, total

    # --- DP passes: plan, re-issue dropped notes as staggered onsets, re-plan ---
    for _pass in range(3):
        sets = constraint_sets()
        pl_idx, pr_idx, total = viterbi(sets)
        # find notes dropped at onset steps and re-issue them stagger_steps later
        n_new = 0
        for t in range(T):
            ks = onset_keys(t)
            if not ks.size:
                continue
            s = split_at(ks, P[0][pl_idx[t]], P[1][pr_idx[t]])
            for h, sub in ((0, ks[:s]), (1, ks[s:])):
                got = {k for _, k in _hand_assign(sub, fingers_sorted[h], off_sorted[h], P[h][pl_idx[t] if h == 0 else pr_idx[t]],
                                                  key_y, is_black, reach_tol, reach_max, drop_cost, black_thumb_weight,
                                                  {f: None for f in range(NUM_FINGERS)}, {f: -10**6 for f in range(NUM_FINGERS)},
                                                  t, {}, 0.0, 0, thumb_reach_max, allowed_for(h, sub))}
                for k in sub:
                    if int(k) not in got and (t, int(k)) not in dropped_once:
                        dropped_once.add((t, int(k)))
                        td = t + int(stagger_steps)
                        if td < T and act[td, k]:
                            extra_onsets[td].add(int(k)); n_new += 1
        if verbose:
            print(f"[fingering:seq] pass {_pass + 1}: cost {total:.3f}, "
                  f"{n_new} dropped notes re-issued {stagger_steps} steps later")
        if n_new == 0:
            break

    # --- output pass: newest-onset groups per hand, fingers by palm offset ---
    finger_key = np.full((T, NUM_FINGERS), -1, dtype=np.int64)
    finger_active = np.zeros((T, NUM_FINGERS), dtype=bool)
    palm = np.stack([P[0][pl_idx], P[1][pr_idx]], 1)                       # (T, 2)
    last_key = {f: None for f in range(NUM_FINGERS)}
    last_step = {f: -10**6 for f in range(NUM_FINGERS)}
    group = [np.array([], dtype=int), np.array([], dtype=int)]
    hold: list[dict[int, int]] = [{}, {}]                                  # per hand: key -> finger
    n_drop = 0
    for t in range(T):
        ks = onset_keys(t)
        if ks.size:
            s = split_at(ks, palm[t, 0], palm[t, 1])
            new = [ks[:s], ks[s:]]
        else:
            new = [None, None]
        for h in range(2):
            if t > 0 and abs(palm[t, h] - palm[t - 1, h]) > 1e-9:
                # finger history is relative to the palm: a move resets it
                for f in fingers_sorted[h]:
                    last_key[int(f)] = None; last_step[int(f)] = -10**6
            if new[h] is not None:
                group[h] = new[h]
                hold[h] = {}
            else:
                group[h] = np.array([k for k in group[h] if act[t, k]], dtype=int)
                hold[h] = {k: f for k, f in hold[h].items() if act[t, k]}
            if not group[h].size:
                continue
            assigned = _hand_assign(group[h], fingers_sorted[h], off_sorted[h], palm[t, h],
                                    key_y, is_black, reach_tol, reach_max, drop_cost,
                                    black_thumb_weight, last_key, last_step, t, hold[h],
                                    switch_penalty, switch_window, thumb_reach_max,
                                    allowed_for(h, group[h]))
            n_drop += len(group[h]) - len(assigned)
            hold[h] = {k: f for f, k in assigned}
            for f, k in assigned:
                finger_key[t, f] = k
                finger_active[t, f] = True
                last_key[f] = k
                last_step[f] = t
    if verbose:
        travel = np.abs(np.diff(palm, axis=0)).sum(0)
        print(f"[fingering:seq] palm travel low {travel[0]:.2f} m, high {travel[1]:.2f} m; "
              f"{n_drop} note-steps unassigned")
    plan = FingeringPlan(finger_key=finger_key, finger_active=finger_active, home_key=_home_keys())
    plan.palm_y = palm            # (T, 2) planned palm positions in the key frame
    return plan
