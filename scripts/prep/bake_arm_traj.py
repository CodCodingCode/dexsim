"""Bake a SMOOTH, zero-phase arm trajectory from a recorded IK rollout.

The live IK solves the arm fresh every control step, which jitters (the DLS
solver oscillates step-to-step). But the arm's path is fully determined by the
song, so we can compute it once and smooth the WHOLE sequence offline. A
forward-backward (zero-phase) low-pass removes the jitter with NO lag -- unlike
the live EMA, which can only attenuate and always lags.

Input:  a rollout .npz from record_rollout.py --arm_ik_follow --zero
        (has 'left'/'right' (T,30) joint trajectories + 'joint_names_left').
Output: an .npz with the same 'left'/'right' (T,30) but the 6 UR10e ARM joints
        zero-phase low-pass filtered. The env loads this and plays the arm
        columns back instead of solving IK live (cfg.arm_traj_npz).

  python scripts/prep/bake_arm_traj.py --rollout logs/ik_rollout.npz \
         --out data/arm_traj/song.npz --cutoff 3.0
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.signal import butter, filtfilt

CONTROL_HZ = 20.0  # piano control rate


def _arm_cols(names):
    # UR10e arm joints are the ones NOT prefixed 'robot0_' (those are the hand)
    return [i for i, n in enumerate(names) if not str(n).startswith("robot0_")]


def _jitter(traj_arm):
    """mean |2nd-difference| (mrad) + direction-flip rate (%) over arm joints."""
    acc = np.abs(np.diff(traj_arm, axis=0, n=2)).mean() * 1000
    flips = (np.diff(np.sign(np.diff(traj_arm, axis=0)), axis=0) != 0).mean() * 100
    return acc, flips


def main():
    p = argparse.ArgumentParser(description="Bake a zero-phase-smoothed arm trajectory.")
    p.add_argument("--rollout", required=True, help="rollout .npz from record_rollout (arm_ik_follow --zero)")
    p.add_argument("--out", required=True, help="output baked-trajectory .npz")
    p.add_argument("--cutoff", type=float, default=3.0, help="low-pass cutoff (Hz); lower=smoother (default 3.0)")
    p.add_argument("--order", type=int, default=4, help="Butterworth order (default 4)")
    a = p.parse_args()

    d = np.load(a.rollout, allow_pickle=True)
    names = [str(n) for n in d["joint_names_left"]]
    arm = _arm_cols(names)
    left = d["left"].astype(np.float64).copy()    # (T,30)
    right = d["right"].astype(np.float64).copy()
    T = left.shape[0]

    # zero-phase Butterworth low-pass on each ARM joint column (forward+backward).
    wn = a.cutoff / (0.5 * CONTROL_HZ)
    if not (0.0 < wn < 1.0):
        raise SystemExit(f"cutoff {a.cutoff}Hz invalid for {CONTROL_HZ}Hz control (Nyquist {CONTROL_HZ/2}Hz)")
    b, aa = butter(a.order, wn, btype="low")
    for traj in (left, right):
        for c in arm:
            traj[:, c] = filtfilt(b, aa, traj[:, c])   # zero-phase: no lag

    # report jitter before/after on the arm joints
    for tag, raw, sm in (("left", d["left"][:, arm], left[:, arm]),
                         ("right", d["right"][:, arm], right[:, arm])):
        ra, rf = _jitter(raw); sa, sf = _jitter(sm)
        print(f"{tag}: jitter accel {ra:.1f} -> {sa:.2f} mrad | dir-flips {rf:.0f}% -> {sf:.0f}%")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, left=left.astype(np.float32), right=right.astype(np.float32),
             arm_cols=np.array(arm, dtype=np.int64), joint_names_left=np.array(names),
             cutoff=a.cutoff, control_hz=CONTROL_HZ, T=T)
    print(f"baked {T} steps -> {a.out}  (arm joints zero-phase low-pass @ {a.cutoff}Hz)")


if __name__ == "__main__":
    main()
