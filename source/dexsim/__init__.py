"""dexsim: UR10e + Shadow Hand dexterous manipulation in Isaac Lab.

Two training modes share one fixed embodiment (UR10e arm + Shadow Hand):

* RL in-hand reorientation  -> turnkey via Isaac Lab's built-in Shadow envs.
* Imitation from BODex/DexGraspNet trajectories on the *same* asset.

The observation/action specs stay fixed across both, which is the whole point
of standardizing on one embodiment.
"""

from pathlib import Path

# Repo-relative locations used across the package.
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parents[1]          # ~/dexsim
ASSETS_DIR = PROJECT_DIR / "assets"           # composed USDs land here
DATA_DIR = PROJECT_DIR / "data"               # BODex / DexGraspNet downloads

__all__ = ["PACKAGE_DIR", "PROJECT_DIR", "ASSETS_DIR", "DATA_DIR"]
