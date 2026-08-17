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
parser.add_argument("--name", help="recording name shown in the viewer (default: rollout stem)")
parser.add_argument("--prop", action="append", default=[], metavar="KEY=VALUE",
                    help="extra recording property, repeatable (e.g. --prop reward_mode=rp1m)")
args = parser.parse_args()

src = Path(args.rollout)
out = Path(args.out) if args.out else src.with_suffix(".rrd")
rollout_npz_to_rrd(src, out, name=args.name,
                   properties=dict(kv.split("=", 1) for kv in args.prop))
print(f"[rerun] wrote {out}")
if args.open:
    import subprocess

    viewer = shutil.which("rerun") or str(Path(sys.executable).with_name("rerun"))
    subprocess.run([viewer, str(out)], check=True)
