"""MIDI-conditioned piano reward (framework-agnostic, vectorized over envs).

Given, at one control step:
  * ``pressed``  (..., 88) float in [0,1]  -- how far each key is depressed
                  (1 = fully down past the "sounds" threshold, 0 = at rest),
  * ``goal``     (..., 88) bool            -- which keys *should* be down now,

it returns a per-env scalar reward following the RoboPianist recipe: reward
hitting the keys that should sound, penalize keys that shouldn't, and (optionally)
add control/energy shaping. Pure NumPy/torch-agnostic via duck-typed ops, so the
same function serves the Isaac Lab env and offline analysis/tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PianoRewardCfg:
    press_threshold: float = 0.5   # depression at/above which a key "sounds"
    key_press_weight: float = 1.0  # reward for sounding the right keys
    false_press_weight: float = 0.5  # penalty for sounding wrong keys
    energy_weight: float = 0.0     # penalty coefficient on control energy
    # how sharply the "did it sound" reward saturates around the threshold
    sharpness: float = 0.05
    # --- PianoMime/RoboPianist shaping terms (added) ---
    fingering_weight: float = 1.0     # finger->target-key spatial shaping (CRITICAL:
    #                                   RoboPianist F1 stays 0 without this term)
    onset_weight: float = 0.5         # extra reward for sounding a key on its onset
    finger_close_enough: float = 0.01    # m; inside this -> full fingering reward
    finger_margin_mult: float = 25.0     # gaussian falloff reaches ~0.1 at 25x bound
    #   (~25 cm). Was 10x (10 cm): fingers start/pass too far from their target key
    #   so the shaping read ~0 (reward/finger~0.005) and gave no gradient. Widening
    #   the falloff makes the make-or-break term actually pull from realistic ranges.


# dm_control-style tolerance kernel (gaussian), vectorized & backend-agnostic.
def tolerance(x, lower=0.0, upper=0.0, margin=1.0, value_at_margin=0.1):
    """1 inside [lower, upper]; gaussian falloff to ~value_at_margin at `margin`
    distance outside. Mirrors dm_control.utils.rewards.tolerance(sigmoid='gaussian'),
    which RoboPianist uses for both key-press and fingering shaping."""
    xp, _ = _backend(x)
    # signed distance outside the band (0 when inside)
    dist = xp.maximum(lower - x, x - upper)
    dist = xp.clip(dist, 0.0, None) if not _ else dist.clamp(min=0.0)
    scaled = (dist / margin) * float((-2.0 * math.log(value_at_margin)) ** 0.5)
    return xp.exp(-0.5 * scaled * scaled)


def _backend(x):
    """Return (lib, is_torch) so this works for numpy arrays or torch tensors."""
    mod = type(x).__module__
    if mod.startswith("torch"):
        import torch
        return torch, True
    import numpy as np
    return np, False


def piano_reward(pressed, goal, cfg: PianoRewardCfg = PianoRewardCfg(),
                 energy=None):
    """Per-env reward. ``pressed`` (..., 88) float; ``goal`` (..., 88) bool/float.

    Reward = key_press_weight * mean over goal keys of a soft "is it down" term
           - false_press_weight * mean over non-goal keys of how-down they are
           - energy_weight * energy.
    Envs with no goal keys at this step get only the false-press / energy terms.
    """
    xp, is_torch = _backend(pressed)

    goal_f = goal.float() if is_torch else goal.astype("float32")
    not_goal = 1.0 - goal_f

    # soft indicator that a key has crossed the sounding threshold
    over = (pressed - cfg.press_threshold) / cfg.sharpness
    sounded = 1.0 / (1.0 + xp.exp(-over))  # sigmoid in [0,1]

    eps = 1e-6
    n_goal = goal_f.sum(-1)

    hit = (sounded * goal_f).sum(-1) / (n_goal + eps)          # want -> 1
    # COUNT of wrong keys sounding, per intended note -- NOT averaged over all ~87
    # non-goal keys. The old "/ n_not" diluted a misclick to ~1/87, so a wrong
    # note cost ~170x less than a right one was worth -> precision rotted. Now each
    # wrong key costs ~false_press_weight; a rest step (no goal, denom->1) charges
    # full weight per false press so it can't mash during silence.
    one = xp.ones_like(n_goal)
    denom = xp.maximum(n_goal, one)
    false = (sounded * not_goal).sum(-1) / denom                # want -> 0

    reward = cfg.key_press_weight * hit - cfg.false_press_weight * false

    if energy is not None and cfg.energy_weight > 0.0:
        reward = reward - cfg.energy_weight * energy

    return reward


def fingering_reward(fingertip_pos, target_pos, active_mask, cfg: PianoRewardCfg = PianoRewardCfg()):
    """The make-or-break spatial shaping term (RoboPianist `_compute_fingering_reward`).

    For each finger that is *assigned* to press a key this step, reward it for
    bringing its tip close to that key's target point. Without this, a high-DoF
    hand exploring from key-press reward alone never bootstraps (RoboPianist
    ablation: F1 stays at 0). Mean is taken over active fingers only.

    Args:
        fingertip_pos (E, F, 3): world positions of the F fingertips.
        target_pos    (E, F, 3): target key point for each finger (only meaningful
                                  where active_mask is True).
        active_mask   (E, F)   : which fingers are assigned a key this step.
    Returns: (E,) reward in [0, fingering_weight].
    """
    xp, is_torch = _backend(fingertip_pos)
    dist = ((fingertip_pos - target_pos) ** 2).sum(-1) ** 0.5          # (E, F)
    shaped = tolerance(
        dist, lower=0.0, upper=cfg.finger_close_enough,
        margin=cfg.finger_close_enough * cfg.finger_margin_mult,
    )                                                                  # (E, F)
    m = active_mask.float() if is_torch else active_mask.astype("float32")
    n = m.sum(-1)
    eps = 1e-6
    mean_over_active = (shaped * m).sum(-1) / (n + eps)
    return cfg.fingering_weight * mean_over_active


def onset_reward(pressed, onsets, cfg: PianoRewardCfg = PianoRewardCfg()):
    """Reward sounding a key on the exact step its note begins (not just holding).

    ``onsets`` (E, 88) bool marks note-start steps. Encourages crisp attacks /
    correct timing rather than smearing a held chord. Mean over onset keys.
    """
    xp, is_torch = _backend(pressed)
    onset_f = onsets.float() if is_torch else onsets.astype("float32")
    over = (pressed - cfg.press_threshold) / cfg.sharpness
    sounded = 1.0 / (1.0 + xp.exp(-over))
    eps = 1e-6
    n = onset_f.sum(-1)
    hit = (sounded * onset_f).sum(-1) / (n + eps)
    return cfg.onset_weight * hit


def press_accuracy(pressed, goal, threshold: float = 0.5):
    """Diagnostic (not reward): fraction of goal keys actually sounding, and
    fraction of sounding keys that were wanted. Returns (recall, precision)."""
    xp, is_torch = _backend(pressed)
    goal_b = goal.bool() if is_torch else goal.astype(bool)
    sounding = pressed >= threshold
    tp = (sounding & goal_b).sum(-1)
    want = goal_b.sum(-1)
    got = sounding.sum(-1)
    eps = 1e-6
    recall = tp / (want + eps)
    precision = tp / (got + eps)
    return recall, precision
