"""Offline IK-quality diagnostic for a q_ref reference .npz — NO Isaac needed.

A reference is only a useful warm-start if its assigned fingertips actually land
on their target keys. The eval (`eval_reference.py`) needs a full sim boot; this
reads the `tip_err` already stored in the reference and reports the active-finger
error distribution in milliseconds, plus the fraction within white-key tolerances.

Rule of thumb: a white key is ~22mm wide (~11mm half-width). If p90 active tip_err
is many cm, the IK never reaches the keys -> low recall AND mashed neighbors, and no
warm-start can bootstrap from it. Fix the reference/reach before spending GPU.

  python scripts/diag_tip_err.py data/reference/twinkle.npz
"""

from __future__ import annotations

import sys
import numpy as np

WHITE_HALF_W_MM = 11.0  # ~half a white-key width; "on the key" tolerance


def report(path: str) -> None:
    d = np.load(path)
    if "tip_err" not in d:
        print(f"{path}: no tip_err field (cannot assess IK quality offline)")
        return
    a = np.asarray(d["tip_err"], dtype=float) * 1000.0  # -> mm
    q = d["q_ref"].shape if "q_ref" in d else "?"
    flat = a.reshape(-1)
    # idle finger/hand steps are stored as 0; "active" ~= strictly positive error.
    active = flat[flat > 1e-9]
    print(f"== {path}  (q_ref {q}, tip_err {a.shape}) ==")
    if active.size == 0:
        print("  all tip_err == 0 (no active targets recorded?)")
        return
    pct = lambda p: np.percentile(active, p)
    print(f"  active finger-steps: {active.size}/{flat.size} "
          f"({100*active.size/flat.size:.0f}%)")
    print(f"  active tip_err [mm]: mean={active.mean():.1f}  median={np.median(active):.1f}"
          f"  p90={pct(90):.1f}  p99={pct(99):.1f}  max={active.max():.1f}")
    for tol in (WHITE_HALF_W_MM, 22.0, 50.0):
        print(f"  within {tol:.0f}mm: {100*np.mean(active < tol):.1f}%")
    # Separate the two failure modes: a ~45mm "blur floor" (over-constrained DLS) is
    # tunable; a tail >100mm means unreachable/divergent targets (wrong hand-key
    # assignment or out-of-workspace) and needs a structural fix, not more iterations.
    blur = active[active <= 100.0]
    div = active[active > 100.0]
    print(f"  blur floor (<=100mm): {blur.size} steps, median={np.median(blur):.1f}mm"
          if blur.size else "  blur floor (<=100mm): none")
    print(f"  DIVERGENT/unreachable (>100mm): {div.size} steps "
          f"({100*div.size/active.size:.1f}% of active), median={np.median(div):.0f}mm"
          if div.size else "  DIVERGENT/unreachable (>100mm): none")
    if a.ndim == 2 and a.shape[1] == 2:  # per-hand columns
        for h, name in ((0, "L"), (1, "R")):
            col = a[:, h][a[:, h] > 1e-9]
            if col.size:
                print(f"  hand {name}: median={np.median(col):.1f}mm  p90={np.percentile(col,90):.0f}mm"
                      f"  >100mm={100*np.mean(col>100):.0f}%")
    verdict = ("USABLE-ish" if np.median(active) < WHITE_HALF_W_MM
               else "TOO FAR — IK does not reach the keys")
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    paths = sys.argv[1:] or ["data/reference/twinkle.npz"]
    for p in paths:
        report(p)
