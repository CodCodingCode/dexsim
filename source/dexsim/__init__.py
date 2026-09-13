"""dexsim: two rail-mounted Shadow Hands playing piano in MuJoCo.

Package layout:

* ``dexsim.piano``          -- sim-agnostic task logic: MIDI -> goals, fingering,
                               key geometry, reward terms.
* ``dexsim.mjcf``           -- MuJoCo scene builders (piano + Menagerie hands).
* ``dexsim.tasks.piano_mj`` -- the env, vectorised env, and PPO config.
* ``dexsim.visualization``  -- rollout -> Rerun recordings.
"""

from pathlib import Path

# Repo-relative locations used across the package.
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parents[1]          # repo root
ASSETS_DIR = PROJECT_DIR / "assets"           # vendored MuJoCo Menagerie + built MJCF
DATA_DIR = PROJECT_DIR / "data"               # MIDI songs / goal bundles

__all__ = ["PACKAGE_DIR", "PROJECT_DIR", "ASSETS_DIR", "DATA_DIR"]
