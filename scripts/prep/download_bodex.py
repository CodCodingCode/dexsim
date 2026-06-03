"""Download the BODex-Tabletop dataset (UR10e + Shadow Hand trajectories).

BODex-Tabletop is the primary imitation set for this project: it is literally
the UR10e + Shadow embodiment, ~3.08M MuJoCo-validated grasp *trajectories*
across 2397 objects. This pulls it from the Hugging Face Hub into data/bodex/.

The Hub repo id can change between releases, so it's a flag with a sensible
default; see docs/DATASETS.md for the canonical source links. Use
--include to grab a subset (the full set is large).

Usage:
  python scripts/download_bodex.py
  python scripts/download_bodex.py --repo-id JYChen18/BODex --include "tabletop/*"
  python scripts/download_bodex.py --list   # just show the file tree, download nothing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dexsim import DATA_DIR

DEFAULT_REPO = "JYChen18/BODex"          # see docs/DATASETS.md if this 404s
DEFAULT_DEST = DATA_DIR / "bodex"


def _ensure_hf():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        sys.exit(
            "huggingface_hub is not installed in this venv.\n"
            "  pip install huggingface_hub\n"
            "then re-run."
        )


def main():
    ap = argparse.ArgumentParser(description="Download BODex-Tabletop.")
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    ap.add_argument("--include", default=None,
                    help="glob to fetch a subset, e.g. 'tabletop/*' (default: all)")
    ap.add_argument("--list", action="store_true", help="list files, download nothing")
    ap.add_argument("--local", action="store_true",
                    help="symlink the already-on-disk BODex data instead of downloading")
    args = ap.parse_args()

    if args.local:
        from pathlib import Path as _P
        src = _P.home() / "DexGraspBench/downloads/ur10e_shadow_extracted/bodex_ur10e_shadow"
        if not src.exists():
            sys.exit(f"on-disk BODex data not found at {src}; drop --local to download.")
        dest = _P(args.dest); dest.mkdir(parents=True, exist_ok=True)
        link = dest / "bodex_ur10e_shadow"
        if not link.exists():
            link.symlink_to(src)
        print(f"[download_bodex] linked {src} -> {link}")
        print("Replay one with: python scripts/replay_bodex.py --headless --traj "
              f"{link}/succ_collect/<object>/scale*_pose000.npy")
        return

    _ensure_hf()
    from huggingface_hub import HfApi, snapshot_download

    if args.list:
        api = HfApi()
        files = api.list_repo_files(args.repo_id, repo_type=args.repo_type)
        print(f"# {args.repo_id} ({args.repo_type}) -- {len(files)} files")
        for f in files[:200]:
            print(f"  {f}")
        if len(files) > 200:
            print(f"  ... and {len(files) - 200} more")
        return

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[download_bodex] {args.repo_id} -> {dest}  (include={args.include})")
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        local_dir=str(dest),
        allow_patterns=[args.include] if args.include else None,
    )
    print(f"[OK] BODex-Tabletop at: {path}")
    print("Next: python scripts/replay_bodex.py --traj <a .npz/.npy under that dir> --headless")


if __name__ == "__main__":
    main()
