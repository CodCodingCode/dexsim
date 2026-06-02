"""Download DexGraspNet (Shadow Hand grasps) for object diversity.

DexGraspNet adds ~1.32M Shadow Hand grasps over 5355 objects across 133+
categories -- broader object coverage than BODex, though these are static grasp
poses rather than full trajectories, so they're best used as grasp *seeds* for
trajectory generation / as targets, not as drop-in playback.

DexGraspNet's canonical release is distributed via its project page / Google
Drive; some mirrors exist on the Hugging Face Hub. This script supports both:
a Hub repo id (default) or a direct --url (zip/tar) fallback. See
docs/DATASETS.md for the authoritative links.

Usage:
  python scripts/download_dexgraspnet.py                     # try the HF mirror
  python scripts/download_dexgraspnet.py --repo-id <id>
  python scripts/download_dexgraspnet.py --url https://.../dexgraspnet.zip
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from dexsim import DATA_DIR

DEFAULT_REPO = "mlfoundations/DexGraspNet"   # mirror; verify in docs/DATASETS.md
DEFAULT_DEST = DATA_DIR / "dexgraspnet"


def _from_hub(repo_id, dest, include):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("pip install huggingface_hub, or use --url for a direct download.")
    print(f"[dexgraspnet] hub: {repo_id} -> {dest}")
    snapshot_download(
        repo_id=repo_id, repo_type="dataset", local_dir=str(dest),
        allow_patterns=[include] if include else None,
    )


def _from_url(url, dest):
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / Path(url).name
    print(f"[dexgraspnet] downloading {url} -> {archive}")

    def _progress(block, bsize, total):
        if total > 0:
            pct = min(100, block * bsize * 100 // total)
            print(f"\r  {pct:3d}%", end="", flush=True)

    urlretrieve(url, archive, _progress)
    print()
    if zipfile.is_zipfile(archive):
        print("  extracting zip...")
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    elif tarfile.is_tarfile(archive):
        print("  extracting tar...")
        with tarfile.open(archive) as t:
            t.extractall(dest)
    print(f"[OK] extracted to {dest}")


def main():
    ap = argparse.ArgumentParser(description="Download DexGraspNet.")
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    ap.add_argument("--include", default=None, help="HF glob subset")
    ap.add_argument("--url", default=None, help="direct archive URL (zip/tar)")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    if args.url:
        _from_url(args.url, dest)
    else:
        _from_hub(args.repo_id, dest, args.include)
    print("These are grasp poses; use them as seeds/targets, see docs/DATASETS.md.")


if __name__ == "__main__":
    main()
