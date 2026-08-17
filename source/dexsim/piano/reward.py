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


# ---------------------------------------------------------------------------
# RP1M reward (Zhao et al., CoRL 2024) -- a faithful port of the paper's terms
# ---------------------------------------------------------------------------
# The paper's composite (Sec. 3, eq. 2):
#
#     r_t = r_OT + r_Press + r_Sustain + a1 * r_Collision + a2 * r_Energy
#           a1 = 0.5,  a2 = 5e-3
#
# with
#     r_OT        = tolerance-shaped minimum finger->key transport distance,
#                   full credit under 1 cm. THIS is what replaces human fingering
#                   labels: the assignment is re-solved every step from the live
#                   fingertip geometry (see dexsim.piano.ot).
#     r_Press     = 0.5 * mean_over_wanted_keys g(||key_state - 1||)
#                 + 0.5 * (1 - 1_false_press)
#     r_Sustain   = g(pedal - pedal_target)          [our piano has no pedal; see below]
#     r_Collision = 1 - 1_collision
#     r_Energy    = sum_j |torque_j| * |velocity_j|   (a *cost*: subtracted)
#
# Deviations from the paper, all deliberate and all switchable:
#   * No sustain pedal exists in our 88-key piano articulation, so r_Sustain is
#     omitted rather than faked. Add the pedal joint and it drops straight in.
#   * r_Collision uses a geometric proximity proxy between the two hands instead
#     of PhysX contact reports (see `rp1m_collision_reward`).
#   * The transport itself is solved approximately (greedy + 2-opt, ~97.5% of
#     steps exact, 0.25 mm mean deviation) rather than by exact Jonker-Volgenant,
#     which does not batch onto the GPU. See dexsim.piano.ot.
#   * The constants *inside* the tolerance kernels are not stated in the paper --
#     it only says c matches Tassa et al. 2018. They are taken from RoboPianist,
#     whose shaping RP1M builds on, and are exposed as cfg fields.
#   * The false-press indicator uses this env's velocity-gated "is it ringing"
#     notion rather than a plain depression threshold, so it agrees with the F1
#     metric we log. RP1M's piano has no such hammer gate.


@dataclass
class RP1MRewardCfg:
    """Weights and shaping constants for :func:`rp1m_reward`. Defaults are the
    paper's values; the shaping bands are RoboPianist's, which RP1M inherits."""
    # --- r_OT: finger -> assigned key, solved online by optimal transport ---
    ot_weight: float = 1.0
    ot_close: float = 0.01          # m; inside this the paper gives full 1.0
    ot_margin_mult: float = 10.0    # gaussian falloff ~0.1 at 10x close (10 cm)
    # The paper's d_OT is the *cumulative* matched distance (a sum over the
    # transport plan), so "sum" is the 1:1 setting and the default. "mean"
    # divides by the number of demanded keys, which keeps a dense chord's shaping
    # on the same scale as a single note -- useful, but not what RP1M does.
    ot_reduce: str = "sum"
    ot_eps: float = 0.01            # Sinkhorn temperature (m)
    ot_iters: int = 50
    ot_side_weight: float = 0.0     # >0 penalizes cross-body reaches (not in RP1M)

    # --- r_Press: are the wanted keys down, and only those? ---
    press_weight: float = 1.0
    press_close: float = 0.05       # normalized key state within this of 1 = full credit
    press_margin_mult: float = 10.0
    # RP1M's false-press half is all-or-nothing: ONE wrong key sounding costs the
    # whole 0.5. Softening it to a per-key count is the single most useful knob
    # here for a hand that hasn't yet learned to lift its idle fingers.
    press_false_soft: bool = False

    # --- a1 * r_Collision ---
    collision_weight: float = 0.5
    collision_dist: float = 0.02    # m; hands nearer than this count as touching

    # --- a2 * r_Energy (subtracted) ---
    energy_weight: float = 5e-3


def rp1m_ot_reward(dist, n_active, cfg: RP1MRewardCfg = RP1MRewardCfg()):
    """RP1M eq. 3: shape the optimal-transport distance into a [0, 1] reward.

    ``exp(c * (d - 0.01)^2)`` for ``d >= 0.01`` and ``1.0`` below it -- which is
    exactly a dm_control gaussian `tolerance` with bounds ``(0, 0.01)``, so it
    reuses the same kernel as every other term here.

    Args:
        dist     (E,): transport distance from :func:`dexsim.piano.ot.ot_finger_cost`.
        n_active (E,): number of keys demanded this step (0 = a rest).
    Returns: (E,) reward in ``[0, ot_weight]``, zero on rest steps (nothing to finger).
    """
    xp, is_torch = _backend(dist)
    if cfg.ot_reduce == "mean":
        denom = n_active.clamp(min=1.0) if is_torch else xp.maximum(n_active, 1.0)
        d = dist / denom
    elif cfg.ot_reduce == "sum":
        d = dist
    else:
        raise ValueError(f"ot_reduce must be 'mean' or 'sum', got {cfg.ot_reduce!r}")
    shaped = tolerance(d, lower=0.0, upper=cfg.ot_close,
                       margin=cfg.ot_close * cfg.ot_margin_mult)
    has = (n_active > 0).float() if is_torch else (n_active > 0).astype("float32")
    return cfg.ot_weight * shaped * has


def rp1m_press_reward(key_state, sounding, goal, cfg: RP1MRewardCfg = RP1MRewardCfg()):
    """RP1M eq. 4 / RoboPianist ``_compute_key_press_reward``.

        0.5 * mean over wanted keys of g(||key_state - 1||)   +   0.5 * (1 - 1_fp)

    The first half is the continuous "push it all the way down" gradient; the
    second is the anti-mash clause that makes pressing 88 keys at once worthless.

    Args:
        key_state (E, 88) float: normalized key depression, 1.0 = fully pressed.
            Pass the RAW depression, not a sounding-gated value -- the whole point
            of this half is that a key on its way down already earns partial credit.
        sounding  (E, 88) bool/float: which keys are actually ringing. Used only
            for the false-press indicator, so it stays consistent with the F1 metric.
        goal      (E, 88) bool/float: which keys should sound now.
    Returns: (E,) reward in ``[0, press_weight]``.
    """
    xp, is_torch = _backend(key_state)
    goal_f = goal.float() if is_torch else goal.astype("float32")
    snd = sounding.float() if is_torch else sounding.astype("float32")
    eps = 1e-6

    gap = key_state - 1.0
    gap = gap.abs() if is_torch else xp.abs(gap)
    shaped = tolerance(gap, lower=0.0, upper=cfg.press_close,
                       margin=cfg.press_close * cfg.press_margin_mult)
    n_on = goal_f.sum(-1)
    hit = (shaped * goal_f).sum(-1) / (n_on + eps)     # 0 when nothing is wanted

    wrong = snd * (1.0 - goal_f)                        # keys ringing that shouldn't
    if cfg.press_false_soft:
        # softened: charge per wrong key against the number intended (>=1), so a
        # single stray note costs a fraction rather than the entire half.
        denom = n_on.clamp(min=1.0) if is_torch else xp.maximum(n_on, 1.0)
        fp = wrong.sum(-1) / denom
        fp = fp.clamp(max=1.0) if is_torch else xp.clip(fp, 0.0, 1.0)
    else:
        # the paper: ANY false press forfeits the whole half.
        fp = (wrong.sum(-1) > 0).float() if is_torch else (wrong.sum(-1) > 0).astype("float32")

    return cfg.press_weight * (0.5 * hit + 0.5 * (1.0 - fp))


def rp1m_collision_reward(collided, cfg: RP1MRewardCfg = RP1MRewardCfg()):
    """RP1M's ``r_Collision = 1 - 1_collision``.

    Args:
        collided (E,): bool/float, 1 where the hands are in collision this step.
            Produce it either from real PhysX contacts (``PianoEnv._hands_collided``,
            what the paper does) or from :func:`hands_in_proximity` below.
    Returns: (E,) reward in {0, collision_weight}.
    """
    xp, is_torch = _backend(collided)
    hit = collided.float() if is_torch else collided.astype("float32")
    return cfg.collision_weight * (1.0 - hit)


def hands_in_proximity(left_pts, right_pts, cfg: RP1MRewardCfg = RP1MRewardCfg()):
    """Geometric stand-in for a contact report: are the two hands within
    ``collision_dist`` of each other?

    A contact sensor is the honest answer (see ``PianoEnvCfg.rp1m_collision_contacts``),
    but this needs no sensor, no extra PhysX buffers, and for two hands on disjoint
    rails it flags exactly the failure the term exists to prevent -- the hands
    driving into each other at the keyboard midline. It is however blind to
    *within*-hand self-collision, which real contacts would catch.

    Args:
        left_pts (E, N, 3), right_pts (E, M, 3): world points on each hand.
    Returns: (E,) bool.
    """
    xp, is_torch = _backend(left_pts)
    d = ((left_pts.unsqueeze(2) - right_pts.unsqueeze(1)) ** 2).sum(-1) ** 0.5 \
        if is_torch else \
        ((left_pts[:, :, None, :] - right_pts[:, None, :, :]) ** 2).sum(-1) ** 0.5
    nearest = d.reshape(d.shape[0], -1).min(-1).values if is_torch else \
        d.reshape(d.shape[0], -1).min(-1)
    return nearest < cfg.collision_dist


def rp1m_energy_cost(applied_torque, joint_vel):
    """RP1M's ``r_Energy = |tau|^T |v|`` -- summed mechanical power per env.

    Returned as a positive COST; the composite subtracts it. Note this is real
    actuator power, not the ``action^2`` proxy the dexsim reward uses, so its
    magnitude (and therefore the useful weight) is completely different.
    """
    xp, is_torch = _backend(applied_torque)
    t = applied_torque.abs() if is_torch else xp.abs(applied_torque)
    v = joint_vel.abs() if is_torch else xp.abs(joint_vel)
    return (t * v).sum(-1)


def rp1m_reward(ot_dist, n_active, key_state, sounding, goal,
                collided=None, energy=None,
                cfg: RP1MRewardCfg = RP1MRewardCfg()):
    """The full RP1M composite. Returns ``(total, parts_dict)`` for logging.

    ``r_Sustain`` is absent -- our piano articulation has no pedal (see the note
    at the top of this section). Collision and energy are skipped when their
    inputs are None, so the term set degrades gracefully.
    """
    r_ot = rp1m_ot_reward(ot_dist, n_active, cfg)
    r_press = rp1m_press_reward(key_state, sounding, goal, cfg)
    parts = {"ot": r_ot, "press": r_press}
    total = r_ot + r_press
    if collided is not None and cfg.collision_weight > 0.0:
        r_col = rp1m_collision_reward(collided, cfg)
        parts["collision"] = r_col
        total = total + r_col
    if energy is not None and cfg.energy_weight > 0.0:
        r_en = -cfg.energy_weight * energy
        parts["energy"] = r_en
        total = total + r_en
    return total, parts


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
