"""Retarget a decoded RoboPianist trajectory onto our UR10e+Shadow rig.

The honest split (see ``docs/RP1M_WARMSTART.md``):

  * the Shadow **hand** pose transfers directly — same morphology — so we copy
    the decoded ``robot0_*`` hand-joint angles into the reference's hand columns;
  * the **arm** does NOT: RoboPianist's hand rides a 2-dof sliding forearm, ours
    is a fixed-base UR10e. We therefore keep the *arm* columns from our own
    validated IK reference (which already positions each hand over its keys), and
    only inject RP1M's finger configuration. The forearm ``tx`` (lateral position
    along the keyboard) is exposed via :func:`forearm_to_wrist_target` so a future
    IK pass can refine arm placement to match RP1M's hand-over-keyboard position.

This module is pure NumPy (no Isaac dependency); the only thing that needs Isaac
is reading the live articulation ``joint_names`` (to know which q_ref column is
which joint) — done by ``scripts/build_rp1m_reference.py``.
"""

from __future__ import annotations

import numpy as np

from .. import geometry as geom
from .decode import DecodedTrajectory, ISAAC_HAND_JOINTS


def resample_traj(q: np.ndarray, t_new: int) -> np.ndarray:
    """Linearly resample a (T, ...) trajectory to ``t_new`` steps along axis 0.

    Used to time-align an RP1M trajectory (recorded at their control rate / song
    tempo) to our reference length. Per-channel linear interpolation.
    """
    t_old = q.shape[0]
    if t_old == t_new:
        return q.astype(np.float32, copy=True)
    src = np.linspace(0.0, 1.0, t_old)
    dst = np.linspace(0.0, 1.0, t_new)
    flat = q.reshape(t_old, -1)
    out = np.empty((t_new, flat.shape[1]), dtype=np.float32)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(dst, src, flat[:, c])
    return out.reshape((t_new,) + q.shape[1:])


def hand_named(decoded: DecodedTrajectory, side: str) -> dict[str, np.ndarray]:
    """{robot0_* joint name -> (T,) angle} for one hand side ("left"/"right")."""
    q = decoded.hand_q[side]
    return {name: q[:, j] for j, name in enumerate(decoded.hand_joint_names)}


def forearm_to_wrist_target(decoded: DecodedTrajectory, side: str, *,
                            left_center_key: int = 22, right_center_key: int = 66,
                            hover: float = 0.06, back_offset: float = 0.0) -> np.ndarray:
    """Map RP1M forearm (tx, ty) -> a wrist/palm target in OUR piano-local frame.

    RoboPianist's ``forearm_tx`` slides the hand laterally along the keyboard; we
    treat it as a lateral offset (metres) from the hand's nominal centre key and
    read OUR keyboard geometry for the rest. ``ty`` (their small vertical adjust)
    is added to the hover height. Returns (T, 3) target positions in the piano
    frame: Y = lateral (along keys), X = toward player, Z = above key tops.

    This is the transferable "which keys is the hand over" signal for a future
    UR10e IK pass; the current warm-start keeps arm columns from our own IK and
    does not require it.
    """
    center_key = left_center_key if side == "left" else right_center_key
    key_top = geom.key_local_top_positions()                 # (88, 3)
    center = key_top[center_key]
    tx = decoded.forearm[side][:, 0]                          # (T,) lateral offset (m)
    ty = decoded.forearm[side][:, 1]                          # (T,) small vertical
    T = tx.shape[0]
    out = np.zeros((T, 3), dtype=np.float32)
    y = np.clip(center[1] + tx, geom.KEY_Y.min(), geom.KEY_Y.max())
    out[:, 0] = center[0] + back_offset
    out[:, 1] = y
    out[:, 2] = center[2] + hover + ty
    return out


def reorder_to(named: dict[str, np.ndarray], joint_names, t: int,
               default: float = 0.0) -> np.ndarray:
    """Build a (T, ndof) array following ``joint_names`` from a name->(T,) dict.

    Joints not present in ``named`` (e.g. arm joints, or coupled J0 if the
    articulation drives it via tendon) are filled with ``default``. Mirrors
    ``dexsim.tasks.grasp.bodex_loader.BODexTrajectory.reorder_to``.
    """
    out = np.full((t, len(joint_names)), default, dtype=np.float32)
    for col, name in enumerate(joint_names):
        v = named.get(name)
        if v is not None:
            out[:, col] = v
    return out


def merge_hand_into_reference(q_ref: np.ndarray, joint_names, decoded: DecodedTrajectory,
                              *, left_side: str = "left", right_side: str = "right",
                              resample: bool = True) -> np.ndarray:
    """Overwrite the Shadow-hand columns of an existing reference with RP1M's pose.

    This is the warm-start assembler. ``q_ref`` is an existing reference of shape
    (T, 2, ndof) (arm 0 = left, arm 1 = right) whose **arm** columns already place
    each hand over its keys (e.g. from ``scripts/build_reference.py``). We keep
    those arm columns and replace every ``robot0_*`` hand column with the decoded
    RP1M finger configuration (time-resampled to T), giving a reference whose
    fingers are curled to *press*, not just hover.

    Args:
        q_ref: (T, 2, ndof) existing reference.
        joint_names: length-ndof list naming each column (the live articulation
            order — both arms share it).
        decoded: the RP1M :class:`DecodedTrajectory`.
        resample: time-align the RP1M hand pose to T (default True).

    Returns a new (T, 2, ndof) array; the input is not modified.
    """
    q_ref = np.asarray(q_ref, dtype=np.float32)
    T, n_arms, ndof = q_ref.shape
    if n_arms != 2 or ndof != len(joint_names):
        raise ValueError(f"q_ref {q_ref.shape} inconsistent with {len(joint_names)} joint_names")
    hand_cols = [i for i, n in enumerate(joint_names) if n in ISAAC_HAND_JOINTS]
    out = q_ref.copy()
    for arm_idx, side in ((0, left_side), (1, right_side)):
        named = hand_named(decoded, side)
        if resample:
            stacked = np.stack([named[n] for n in ISAAC_HAND_JOINTS], axis=1)  # (Td,24)
            stacked = resample_traj(stacked, T)
            named = {n: stacked[:, j] for j, n in enumerate(ISAAC_HAND_JOINTS)}
        for col in hand_cols:
            name = joint_names[col]
            if name in named:
                out[:, arm_idx, col] = named[name]
    return out
