"""Convert a saved dexsim rollout to Rerun and optionally open the viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

from dexsim.visualization import rollout_npz_to_rrd


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("rollout", help="rollout .npz produced by record_rollout.py")
parser.add_argument("--out", help="output .rrd (default: beside the rollout)")
parser.add_argument("--open", action="store_true", help="open the native Rerun viewer after conversion")
args = parser.parse_args()

src = Path(args.rollout)
out = Path(args.out) if args.out else src.with_suffix(".rrd")
rollout_npz_to_rrd(src, out)
print(f"[rerun] wrote {out}")
if args.open:
    import subprocess

    viewer = shutil.which("rerun") or str(Path(sys.executable).with_name("rerun"))
    subprocess.run([viewer, str(out)], check=True)
