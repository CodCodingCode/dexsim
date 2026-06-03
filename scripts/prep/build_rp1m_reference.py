#!/usr/bin/env python
"""Turn an RP1M / RoboPianist action trajectory into a warm-start reference.

This is the "fine-tune-not-replay" prep — everything UP TO RL fine-tuning:

  1. decode the (T, 39|45) action vector -> Isaac robot0_* hand-joint trajectories
     + forearm (tx, ty) + sustain                            (dexsim.piano.rp1m.decode)
  2. (optional) merge the decoded hand pose into an existing IK reference's hand
     columns, keeping that reference's arm columns (which already position each
     hand over its keys)                                      (rp1m.retarget.merge_*)
  3. write a warm-start q_ref (T, 2, ndof) the piano env consumes as its residual
     base — exactly what scripts/bc_pretrain.py then clones into the PPO actor.

It is **sim-free**: it never launches Isaac. To merge it needs the articulation's
joint-name order; that is read from the reference .npz (new references store it)
or from data/reference/joint_names.json (written by build_reference.py). If
neither exists, run build_reference.py once for any song to populate the cache.

Examples
--------
    source env.sh
    # 1) decode only -> name-keyed hand trajectory (no reference needed)
    python scripts/build_rp1m_reference.py \
        --actions data/robopianist_ref/examples/twinkle_twinkle_actions.npy

    # 2) full warm-start: inject RP1M finger pose into our twinkle IK reference
    python scripts/build_rp1m_reference.py \
        --actions data/robopianist_ref/examples/twinkle_twinkle_actions.npy \
        --reference data/reference/twinkle.npz \
        --out data/reference/twinkle_rp1m.npz

Then (in Isaac) clone it into the actor:
    python scripts/bc_pretrain.py --midi data/midi/twinkle.mid \
        --reference data/reference/twinkle_rp1m.npz --dump data/bc/twinkle_rp1m.npz
    python scripts/train_piano.py --midi data/midi/twinkle.mid --bc_init <model.pt>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source"))

from dexsim.piano.rp1m import decode as dc, retarget as rt  # noqa: E402

REF_DIR = REPO / "data" / "reference"


def _load_joint_names(reference_npz: Path | None) -> list[str] | None:
    if reference_npz is not None:
        d = np.load(reference_npz, allow_pickle=True)
        if "joint_names" in d:
            return [str(x) for x in d["joint_names"]]
    cache = REF_DIR / "joint_names.json"
    if cache.exists():
        return json.loads(cache.read_text())
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actions", required=True, help="(T,39|45) RoboPianist/RP1M actions .npy")
    ap.add_argument("--reference", default=None,
                    help="existing IK reference .npz to inject the hand pose into "
                         "(its arm columns are kept). If omitted, only the decoded "
                         "name-keyed hand trajectory is written.")
    ap.add_argument("--out", default=None, help="output .npz")
    ap.add_argument("--raw", action="store_true",
                    help="actions are already joint targets (skip [-1,1] un-normalise)")
    ap.add_argument("--reduced", choices=["auto", "yes", "no"], default="auto",
                    help="force reduced(39)/full(45) action space (default: infer)")
    ap.add_argument("--no-resample", action="store_true",
                    help="do not time-resample the RP1M pose to the reference length")
    args = ap.parse_args()

    actions = np.load(args.actions)
    reduced = {"auto": None, "yes": True, "no": False}[args.reduced]
    decoded = dc.decode_actions(actions, canonical=not args.raw, reduced=reduced)
    print(f"[rp1m] decoded {actions.shape} -> {decoded.num_steps} steps, "
          f"{'reduced(39)' if decoded.reduced else 'full(45)'} action space")
    for side in ("left", "right"):
        q = decoded.hand_q[side]
        print(f"  {side:5s} hand: {q.shape}  flex range [{q.min():.2f}, {q.max():.2f}] rad  "
              f"forearm tx [{decoded.forearm[side][:,0].min():.3f}, "
              f"{decoded.forearm[side][:,0].max():.3f}] m")

    stem = Path(args.actions).stem
    if args.reference is None:
        # decode-only: write a name-keyed hand trajectory (the transferable signal)
        out = Path(args.out) if args.out else REF_DIR / f"{stem}_rp1m_hand.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            hand_q_left=decoded.hand_q["left"], hand_q_right=decoded.hand_q["right"],
            hand_joint_names=np.array(decoded.hand_joint_names),
            forearm_left=decoded.forearm["left"], forearm_right=decoded.forearm["right"],
            sustain=decoded.sustain, reduced=decoded.reduced,
        )
        print(f"[rp1m] wrote name-keyed hand trajectory -> {out}")
        print("[rp1m] (pass --reference <ik_reference.npz> to build a merged q_ref warm-start)")
        return

    # merge into an existing reference's hand columns
    ref_path = Path(args.reference)
    ref = np.load(ref_path, allow_pickle=True)
    q_ref = ref["q_ref"]
    joint_names = _load_joint_names(ref_path)
    if joint_names is None:
        sys.exit(
            f"ERROR: no joint-name order available. {ref_path.name} predates the "
            f"joint_names field and data/reference/joint_names.json is missing. "
            f"Run `python scripts/build_reference.py --midi <any.mid> --headless` "
            f"once to regenerate a reference + populate the cache, then retry."
        )
    if len(joint_names) != q_ref.shape[2]:
        sys.exit(f"ERROR: joint_names ({len(joint_names)}) != q_ref ndof ({q_ref.shape[2]})")

    merged = rt.merge_hand_into_reference(
        q_ref, joint_names, decoded, resample=not args.no_resample)
    n_hand = sum(1 for n in joint_names if n in dc.ISAAC_HAND_JOINTS)
    out = Path(args.out) if args.out else REF_DIR / f"{stem}_rp1m.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, q_ref=merged.astype(np.float32),
        midi=str(ref["midi"]) if "midi" in ref else "",
        control_dt=float(ref["control_dt"]) if "control_dt" in ref else 0.05,
        joint_names=np.array(joint_names),
        source_actions=str(args.actions), source_reference=str(ref_path),
    )
    print(f"[rp1m] merged RP1M hand pose into {n_hand} hand columns "
          f"(arm columns kept from {ref_path.name})")
    print(f"[rp1m] wrote warm-start reference -> {out}  q_ref{merged.shape}")
    print("[rp1m] next (in Isaac): bc_pretrain.py --reference this, then "
          "train_piano.py --bc_init <model.pt>  (RL fine-tuning — out of scope here)")


if __name__ == "__main__":
    main()
