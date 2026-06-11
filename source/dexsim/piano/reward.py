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
    finger_margin_mult: float = 25.0     # gaussian falloff reaches ~0.1 at 25x (~25 cm)
    # --- idle-finger hover shaping (positive twin of the idle-clear penalty) ---
    # Reward idle fingers for sitting at their hover-home; the smooth gradient that
    # holds them UP so "press one finger, rest hovering" beats mash AND droop. 0 = off.
    idle_hover_weight: float = 0.0
    idle_hover_close: float = 0.005      # m dead-band -> full hover reward inside
    idle_hover_margin_mult: float = 5.0  # falloff ~0.1 at 2.5 cm below the band (z-only)
    idle_hover_z_only: bool = True       # score only height above the keys, not euclidean
    #   distance to the (laterally unreachable) home keys. Height is the mash axis.
    # --- PHASE 0 (gross arm positioning) ---
    arm_position_weight: float = 0.0     # reward each hand-base for covering its keys' centroid
    arm_position_close: float = 0.03     # m -> full positioning reward inside
    arm_position_margin_mult: float = 16.0   # gaussian falloff ~0.1 at 16x close (~0.5 m)


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
    # wrong keys are COUNTED per intended note (denom = #goal keys, min 1), NOT
    # averaged over all ~87 non-goal keys -- else a misclick dilutes to ~1/87 and
    # precision rots. A rest step (no goal) charges full weight per false press.
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


def idle_hover_reward(fingertip_pos, hover_target_pos, active_mask, cfg: PianoRewardCfg = PianoRewardCfg()):
    """Positive shaping that holds NON-assigned fingers at their hover-home.

    The mirror of :func:`fingering_reward`: that term pulls each *active* finger
    onto its key; this one pulls each *idle* finger to its hover point (home key
    top + HOVER_CLEARANCE -- the same targets the observation already exposes).
    Together they make "one finger down, the rest up" the shaped optimum, with a
    gradient on the idle fingers at all times -- unlike the idle-clear penalty,
    which is flat 0 until a finger has already dropped below the clearance plane.
    Mean is taken over idle fingers only; a step with all 10 fingers assigned
    contributes 0 (nothing is asked to hover).

    Args:
        fingertip_pos    (E, F, 3): world positions of the F fingertips.
        hover_target_pos (E, F, 3): hover point per finger (only meaningful where
                                     active_mask is False).
        active_mask      (E, F)   : which fingers are assigned a key this step.
    Returns: (E,) reward in [0, idle_hover_weight].
    """
    xp, is_torch = _backend(fingertip_pos)
    if cfg.idle_hover_z_only:
        # height above the keys is the mash axis; ignore the (often unreachable)
        # lateral offset to the spread home keys so every idle finger gets the
        # same clean anti-droop gradient. ONE-SIDED: at/above the hover plane is
        # full reward (a deliberately lifted hand -- lift_between_notes /
        # idle_hand_retract -- must not be pulled back down); only sinking BELOW
        # the plane toward the keys decays it.
        below = hover_target_pos[..., 2] - fingertip_pos[..., 2]      # (E, F) +ve = drooped
        dist = below.clamp(min=0.0) if is_torch else xp.clip(below, 0.0, None)
    else:
        dist = ((fingertip_pos - hover_target_pos) ** 2).sum(-1) ** 0.5   # (E, F)
    shaped = tolerance(
        dist, lower=0.0, upper=cfg.idle_hover_close,
        margin=cfg.idle_hover_close * cfg.idle_hover_margin_mult,
    )                                                                 # (E, F)
    m = active_mask.float() if is_torch else active_mask.astype("float32")
    idle = 1.0 - m
    n = idle.sum(-1)
    eps = 1e-6
    mean_over_idle = (shaped * idle).sum(-1) / (n + eps)
    return cfg.idle_hover_weight * mean_over_idle


def arm_position_reward(palm_pos, target_pos, active_mask, cfg: PianoRewardCfg = PianoRewardCfg()):
    """PHASE 0 gross-positioning shaping (the coarse precursor to `fingering_reward`).

    Reward each hand for bringing its BASE (palm) near the centroid of the keys it
    must play over the upcoming lookahead window, so the hand ends up *over* (covering)
    the right region of the keyboard. This is the only signal a 2-DoF arm needs to
    learn turn+lean placement before any finger pressing exists -- it's smooth and
    dense from anywhere over the hand's half (gaussian falloff ~0.5 m wide), unlike
    the key-press reward which is flat ~0 until a key actually sounds.

    Args:
        palm_pos    (E, 2, 3): world position of each hand base [left, right].
        target_pos  (E, 2, 3): target point for each hand (centroid of its upcoming
                                keys, lifted to the hover height). Only meaningful
                                where active_mask is True.
        active_mask (E, 2)   : which hands have upcoming notes (a hand with none
                                contributes nothing -- it isn't asked to cover anything).
    Returns: (E,) reward in [0, arm_position_weight].
    """
    xp, is_torch = _backend(palm_pos)
    dist = ((palm_pos - target_pos) ** 2).sum(-1) ** 0.5               # (E, 2)
    shaped = tolerance(
        dist, lower=0.0, upper=cfg.arm_position_close,
        margin=cfg.arm_position_close * cfg.arm_position_margin_mult,
    )                                                                  # (E, 2)
    m = active_mask.float() if is_torch else active_mask.astype("float32")
    n = m.sum(-1)
    eps = 1e-6
    mean_over_active = (shaped * m).sum(-1) / (n + eps)
    return cfg.arm_position_weight * mean_over_active


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
