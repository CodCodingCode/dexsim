#!/usr/bin/env python3
"""Prune old LOCAL wandb run dirs so the folder stays tidy.

This only deletes local `wandb/run-*` directories (they accumulate one per run).
It does NOT touch the wandb.ai cloud dashboard -- already-synced runs stay there;
to declutter the dashboard itself, delete runs in the web UI or via wandb.Api().
It never deletes a run dir modified in the last few minutes (an active run), and
the real model checkpoints live in logs/rsl_rl/, not here, so this is safe.

Defaults keep the newest N runs AND anything younger than D days; everything else
older is removed. Override with --keep-last/--keep-days or env WANDB_KEEP_LAST/
WANDB_KEEP_DAYS.

  python scripts/clean_wandb.py --dry-run          # preview
  python scripts/clean_wandb.py                     # keep newest 10 + <1d old
  WANDB_KEEP_LAST=5 WANDB_KEEP_DAYS=0.5 python scripts/clean_wandb.py
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import time

ACTIVE_GUARD_S = 600  # never delete a run touched in the last 10 min (likely active)


def _dirsize(d):
    """Bytes under d, tolerant of broken symlinks (wandb leaves dangling model_*.pt)."""
    total = 0
    for r, _, fs in os.walk(d):
        for f in fs:
            p = os.path.join(r, f)
            if os.path.islink(p):
                continue  # symlink to a real checkpoint in logs/rsl_rl; not our footprint
            try:
                total += os.path.getsize(p)
            except OSError:
                pass
    return total


def main():
    ap = argparse.ArgumentParser(description="Prune old local wandb run dirs.")
    ap.add_argument("--dir", default="wandb")
    ap.add_argument("--keep-last", type=int,
                    default=int(os.environ.get("WANDB_KEEP_LAST", 10)),
                    help="always keep this many newest runs")
    ap.add_argument("--keep-days", type=float,
                    default=float(os.environ.get("WANDB_KEEP_DAYS", 1)),
                    help="also keep any run younger than this many days")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs = sorted(glob.glob(os.path.join(args.dir, "run-*")),
                  key=os.path.getmtime, reverse=True)
    if not runs:
        print(f"no run dirs under {args.dir}/")
        return
    now = time.time()
    keep_recent = set(runs[: args.keep_last])
    removed = freed = 0
    for d in runs:
        age = now - os.path.getmtime(d)
        if d in keep_recent or age < args.keep_days * 86400 or age < ACTIVE_GUARD_S:
            continue
        sz = _dirsize(d)
        print(f"{'[dry] ' if args.dry_run else ''}rm {d}  ({sz // 1024} KB, {age / 3600:.1f}h old)")
        if not args.dry_run:
            shutil.rmtree(d, ignore_errors=True)
        removed += 1
        freed += sz
    verb = "would remove" if args.dry_run else "removed"
    kept = len(runs) - removed
    print(f"{verb} {removed} run dirs ({freed // 1024 // 1024} MB); kept {kept}")


if __name__ == "__main__":
    main()
