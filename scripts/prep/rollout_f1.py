"""Score a saved record_rollout npz: key-press F1/precision/recall, per-hand, and
the mash indicator (avg keys sounding) -- no simulator needed.

Lets you compare planar-IK / constant-z configs instantly from their rollout npz:

  python scripts/prep/rollout_f1.py logs/rollout_frozen.npz
  python scripts/prep/rollout_f1.py results/*.npz        # compare several
"""
import argparse, glob
import numpy as np

# keyboard-middle split between the two reachable hand windows (matches the env
# hand mask + the fold): keys below -> left hand, at/above -> right hand.
LEFT_WINDOW, RIGHT_WINDOW = (19, 26), (63, 70)
SPLIT = 0.5 * (LEFT_WINDOW[1] + RIGHT_WINDOW[0])


def prf(sound, goal):
    s = sound.astype(bool); g = goal.astype(bool)
    tp = int((s & g).sum()); fp = int((s & ~g).sum()); fn = int((~s & g).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return dict(f1=f1, precision=p, recall=r, tp=tp, fp=fp, fn=fn)


def score(path):
    d = np.load(path, allow_pickle=True)
    if "goal" not in d or "sound" not in d:
        print(f"{path}: missing goal/sound (keys={list(d.keys())})"); return
    goal = d["goal"]; sound = d["sound"]
    T = goal.shape[0]
    m = prf(sound, goal)
    keys_sounding = float(sound.astype(bool).sum(1).mean())     # avg keys down/step (mash if >> goal density)
    goal_density = float(goal.astype(bool).sum(1).mean())
    keys = np.arange(88)
    lmask = keys < SPLIT; rmask = ~lmask
    L = prf(sound[:, lmask], goal[:, lmask]); R = prf(sound[:, rmask], goal[:, rmask])
    print(f"\n{path}  ({T} steps)")
    print(f"  micro   F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}"
          f"   (tp={m['tp']} fp={m['fp']} fn={m['fn']})")
    print(f"  left    F1={L['f1']:.3f}  P={L['precision']:.3f}  R={L['recall']:.3f}")
    print(f"  right   F1={R['f1']:.3f}  P={R['precision']:.3f}  R={R['recall']:.3f}")
    print(f"  keys_sounding/step={keys_sounding:.2f}  vs goal_density={goal_density:.2f}"
          f"   {'(MASHING - precision will suffer)' if keys_sounding > goal_density + 0.5 else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="record_rollout .npz file(s) / globs")
    args = ap.parse_args()
    files = [f for p in args.paths for f in (glob.glob(p) or [p])]
    for f in files:
        try:
            score(f)
        except Exception as e:
            print(f"{f}: ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
