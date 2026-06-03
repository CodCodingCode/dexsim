"""Decode a RoboPianist/RP1M action trajectory into Isaac robot0_* hand joints.

The authoritative action layout, recovered from the vendored RoboPianist source
(``data/robopianist_ref``) and the Shadow-hand E3M5 MuJoCo model:

  action = [ right_hand(H) | left_hand(H) | sustain(1) ]      (hands: RIGHT first)
  per hand  H = (n_hand_actuators) + (n_forearm_dofs)

with two configurations in the wild:
  * **full** action space   : 20 hand actuators + 2 forearm = 22/hand -> 45 total
    (RoboPianist demo, e.g. ``examples/twinkle_twinkle_actions.npy``)
  * **reduced** action space: 17 hand actuators + 2 forearm = 19/hand -> 39 total
    (RP1M's published 39-d action; A_THJ5, A_THJ1, A_LFJ5 are dropped)

The forearm dofs are always ``(forearm_tx, forearm_ty)`` in that order:
``tx`` slides the hand laterally along the keyboard (range set to the piano
length at init), ``ty`` is a small (0..6 cm) vertical adjust.

Actions are stored in **canonical [-1, 1]** space (RoboPianist's
``CanonicalSpecWrapper`` rescales to each actuator's ``ctrlrange`` before
stepping); we un-normalise with the per-actuator ctrlranges below. The four
distal finger actuators (``A_*FJ0``) drive a *coupled tendon* over the two distal
joints (J2+J1 in MuJoCo naming); we split that target equally across the two
Isaac joints, clamped to each joint's range.

MuJoCo Shadow joints are 1-indexed (``rh_FFJ4``); our Isaac instanceable hand is
0-indexed (``robot0_FFJ3``) — the ``J(n-1)`` map already used by
``dexsim.tasks.grasp.bodex_loader``. Both this and RoboPianist's own joint groups
agree, so the only work is index-shift + tendon split + un-normalise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- Authoritative per-actuator decode table -------------------------------
# Each row: (mujoco_actuator, ctrlrange, [isaac_target_joint(s)]).
# Order is the MuJoCo actuator order = the order of the action vector. The
# coupled distal actuators list TWO Isaac joints (target is split equally).
# ctrlranges are the Shadow E3M5 menagerie class defaults (verified from the XML).
_FULL_ACTUATORS: tuple[tuple[str, tuple[float, float], tuple[str, ...]], ...] = (
    ("A_WRJ2", (-0.523599, 0.174533), ("robot0_WRJ1",)),
    ("A_WRJ1", (-0.698132, 0.488692), ("robot0_WRJ0",)),
    ("A_THJ5", (-1.047200, 1.047200), ("robot0_THJ4",)),
    ("A_THJ4", (0.000000, 1.221730), ("robot0_THJ3",)),
    ("A_THJ3", (-0.209440, 0.209440), ("robot0_THJ2",)),
    ("A_THJ2", (-0.698132, 0.698132), ("robot0_THJ1",)),
    ("A_THJ1", (-0.261799, 1.570800), ("robot0_THJ0",)),
    ("A_FFJ4", (-0.349066, 0.349066), ("robot0_FFJ3",)),
    ("A_FFJ3", (-0.261799, 1.570800), ("robot0_FFJ2",)),
    ("A_FFJ0", (0.000000, 3.141500), ("robot0_FFJ1", "robot0_FFJ0")),   # coupled
    ("A_MFJ4", (-0.349066, 0.349066), ("robot0_MFJ3",)),
    ("A_MFJ3", (-0.261799, 1.570800), ("robot0_MFJ2",)),
    ("A_MFJ0", (0.000000, 3.141500), ("robot0_MFJ1", "robot0_MFJ0")),   # coupled
    ("A_RFJ4", (-0.349066, 0.349066), ("robot0_RFJ3",)),
    ("A_RFJ3", (-0.261799, 1.570800), ("robot0_RFJ2",)),
    ("A_RFJ0", (0.000000, 3.141500), ("robot0_RFJ1", "robot0_RFJ0")),   # coupled
    ("A_LFJ5", (0.000000, 0.785398), ("robot0_LFJ4",)),
    ("A_LFJ4", (-0.349066, 0.349066), ("robot0_LFJ3",)),
    ("A_LFJ3", (-0.261799, 1.570800), ("robot0_LFJ2",)),
    ("A_LFJ0", (0.000000, 3.141500), ("robot0_LFJ1", "robot0_LFJ0")),   # coupled
)
# Actuators dropped in the reduced (39-d) action space.
_REDUCED_EXCLUDED = ("A_THJ5", "A_THJ1", "A_LFJ5")
_REDUCED_ACTUATORS = tuple(a for a in _FULL_ACTUATORS if a[0] not in _REDUCED_EXCLUDED)

# Per-Isaac-joint limits, so un-normalised / split values stay physical.
# (min, max) for each robot0_* joint touched above.
ISAAC_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "robot0_WRJ1": (-0.523599, 0.174533),
    "robot0_WRJ0": (-0.698132, 0.488692),
    "robot0_THJ4": (-1.047200, 1.047200),
    "robot0_THJ3": (0.000000, 1.221730),
    "robot0_THJ2": (-0.209440, 0.209440),
    "robot0_THJ1": (-0.698132, 0.698132),
    "robot0_THJ0": (-0.261799, 1.570800),
    # finger abduction (J3), proximal (J2), and the two coupled distals (J1, J0)
    **{f"robot0_{F}J3": (-0.349066, 0.349066) for F in ("FF", "MF", "RF", "LF")},
    **{f"robot0_{F}J2": (-0.261799, 1.570800) for F in ("FF", "MF", "RF", "LF")},
    **{f"robot0_{F}J1": (0.000000, 1.570800) for F in ("FF", "MF", "RF", "LF")},
    **{f"robot0_{F}J0": (0.000000, 1.570800) for F in ("FF", "MF", "RF", "LF")},
    "robot0_LFJ4": (0.000000, 0.785398),   # little-finger metacarpal (from LFJ5)
}

# All 24 Isaac hand joints (the full Shadow joint set), grouped for readability.
ISAAC_HAND_JOINTS: tuple[str, ...] = (
    "robot0_WRJ1", "robot0_WRJ0",
    "robot0_THJ4", "robot0_THJ3", "robot0_THJ2", "robot0_THJ1", "robot0_THJ0",
    "robot0_FFJ3", "robot0_FFJ2", "robot0_FFJ1", "robot0_FFJ0",
    "robot0_MFJ3", "robot0_MFJ2", "robot0_MFJ1", "robot0_MFJ0",
    "robot0_RFJ3", "robot0_RFJ2", "robot0_RFJ1", "robot0_RFJ0",
    "robot0_LFJ4", "robot0_LFJ3", "robot0_LFJ2", "robot0_LFJ1", "robot0_LFJ0",
)

N_FOREARM_DOFS = 2          # (forearm_tx, forearm_ty)
FOREARM_TX_DEFAULT_RANGE = (-0.5, 0.5)   # tx range is the piano half-length at init


@dataclass
class DecodedTrajectory:
    """A RoboPianist action trajectory decoded into our joint conventions."""

    hand_q: dict[str, np.ndarray]    # side -> (T, 24) Isaac robot0_* hand angles
    hand_joint_names: tuple[str, ...]  # the 24 column names (== ISAAC_HAND_JOINTS)
    forearm: dict[str, np.ndarray]   # side -> (T, 2) = (tx, ty), canonical-unscaled
    sustain: np.ndarray              # (T,) sustain pedal in [0, 1]
    reduced: bool                    # whether the source used the reduced action space

    @property
    def num_steps(self) -> int:
        return int(self.sustain.shape[0])


def _unnormalize(a: np.ndarray, lo: float, hi: float, canonical: bool) -> np.ndarray:
    """Map canonical [-1,1] -> [lo,hi] (or pass through if already in ctrl units)."""
    if canonical:
        return lo + (np.clip(a, -1.0, 1.0) + 1.0) * 0.5 * (hi - lo)
    return np.clip(a, lo, hi)


def _decode_hand(block: np.ndarray, reduced: bool, canonical: bool) -> np.ndarray:
    """Decode one hand's actuator block -> (T, 24) Isaac hand angles.

    ``block`` is (T, n_hand) where n_hand = 17 (reduced) or 20 (full); the
    forearm dofs must already be stripped off.
    """
    table = _REDUCED_ACTUATORS if reduced else _FULL_ACTUATORS
    if block.shape[1] != len(table):
        raise ValueError(f"hand block has {block.shape[1]} actuators, expected "
                         f"{len(table)} for {'reduced' if reduced else 'full'} space")
    T = block.shape[0]
    q = {name: np.zeros(T, dtype=np.float32) for name in ISAAC_HAND_JOINTS}
    for col, (_act, (lo, hi), targets) in enumerate(table):
        val = _unnormalize(block[:, col], lo, hi, canonical)
        if len(targets) == 1:
            j = targets[0]
            jl, jh = ISAAC_JOINT_LIMITS[j]
            q[j] = np.clip(val, jl, jh).astype(np.float32)
        else:  # coupled tendon: split the (summed) target equally across J1, J0
            half = val * 0.5
            for j in targets:
                jl, jh = ISAAC_JOINT_LIMITS[j]
                q[j] = np.clip(half, jl, jh).astype(np.float32)
    # excluded joints (reduced space) are left un-actuated -> default rest (0).
    return np.stack([q[name] for name in ISAAC_HAND_JOINTS], axis=1)


def infer_layout(action_dim: int) -> tuple[bool, int]:
    """Given the action width, return (reduced, n_hand_actuators).

    45 -> full (20 hand), 39 -> reduced (17 hand). Both use 2 forearm dofs + 1
    sustain over two hands: dim = 2*(n_hand+2)+1.
    """
    if action_dim == 45:
        return False, 20
    if action_dim == 39:
        return True, 17
    # general fallback: solve 2*(n_hand+2)+1 = dim
    n_hand = (action_dim - 1) // 2 - N_FOREARM_DOFS
    reduced = (n_hand == 17)
    if n_hand not in (17, 20):
        raise ValueError(f"cannot infer RoboPianist layout from action_dim={action_dim} "
                         f"(expected 39 or 45, or 2*(n_hand+2)+1 with n_hand in 17/20)")
    return reduced, n_hand


def decode_actions(actions: np.ndarray, *, canonical: bool = True,
                   reduced: bool | None = None) -> DecodedTrajectory:
    """Decode a (T, A) RoboPianist/RP1M action trajectory.

    Args:
        actions: (T, A) array; A is 45 (full) or 39 (reduced) — auto-detected.
        canonical: if True (default) actions are in [-1,1] and are rescaled by
            each actuator's ctrlrange; if False they are already joint targets.
        reduced: force the reduced/full interpretation (else inferred from width).

    Returns a :class:`DecodedTrajectory` with per-side (T, 24) Isaac hand angles,
    per-side forearm (tx, ty), and the sustain pedal.
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"expected (T, A) actions, got shape {actions.shape}")
    A = actions.shape[1]
    inferred_reduced, n_hand = infer_layout(A)
    if reduced is None:
        reduced = inferred_reduced
    per_hand = n_hand + N_FOREARM_DOFS
    # layout: [right(per_hand) | left(per_hand) | sustain(1)]
    right = actions[:, :per_hand]
    left = actions[:, per_hand:2 * per_hand]
    sustain = actions[:, -1]

    out_hand, out_forearm = {}, {}
    for side, blk in (("right", right), ("left", left)):
        hand_block = blk[:, :n_hand]
        forearm_block = blk[:, n_hand:n_hand + N_FOREARM_DOFS]
        out_hand[side] = _decode_hand(hand_block, reduced, canonical)
        # forearm: tx in canonical [-1,1] maps to +/- piano-half; ty to (0,0.06).
        tx = _unnormalize(forearm_block[:, 0], *FOREARM_TX_DEFAULT_RANGE, canonical) \
            if canonical else forearm_block[:, 0]
        ty = _unnormalize(forearm_block[:, 1], 0.0, 0.06, canonical) \
            if canonical else forearm_block[:, 1]
        out_forearm[side] = np.stack([tx, ty], axis=1).astype(np.float32)

    # sustain: canonical [-1,1] -> [0,1]
    sus = (np.clip(sustain, -1.0, 1.0) + 1.0) * 0.5 if canonical else np.clip(sustain, 0, 1)
    return DecodedTrajectory(
        hand_q=out_hand, hand_joint_names=ISAAC_HAND_JOINTS,
        forearm=out_forearm, sustain=sus.astype(np.float32), reduced=reduced,
    )
