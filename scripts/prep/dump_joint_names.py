"""Dump the live articulation joint-name order to data/reference/joint_names.json.

The RP1M warm-start merge (scripts/build_rp1m_reference.py) runs sim-free, but it
needs to know which q_ref column is which joint. This one-off reads that order
from the env and caches it so the merge never needs Isaac again. Also verifies
the left and right arms share the same joint ordering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import dexsim.tasks  # noqa: F401
from dexsim.tasks.piano import PianoEnvCfg
from dexsim.tasks.piano.piano_env import PianoEnv
from dexsim import DATA_DIR

cfg = PianoEnvCfg()
cfg.scene.num_envs = 1
env = PianoEnv(cfg, render_mode=None)
env.reset()

left = list(env.left_robot.data.joint_names)
right = list(env.right_robot.data.joint_names)
print(f"[dump] per_arm_dof={env.per_arm_dof}")
print(f"[dump] left  joint order ({len(left)}): {left}")
print(f"[dump] right joint order ({len(right)}): {right}")
print(f"[dump] left == right order: {left == right}")

out = DATA_DIR / "reference" / "joint_names.json"
out.parent.mkdir(parents=True, exist_ok=True)
payload = left if left == right else {"left": left, "right": right}
out.write_text(json.dumps(payload, indent=0))
print(f"[dump] wrote {out}")

env.close()
app.close()
