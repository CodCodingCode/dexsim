"""Reach-grasp on the combined UR10e + Shadow Hand (imitation from BODex).

Unlike the reorientation task, the imitation path is driven by scripts that
replay / learn from BODex-Tabletop trajectories on the *combined* embodiment,
so this package mainly exposes the tabletop scene config and the BODex loader.
The scene below is the same one ``scripts/spawn_scene.py`` and
``scripts/replay_bodex.py`` build.
"""

from .grasp_scene_cfg import TabletopGraspSceneCfg  # noqa: F401
from .bodex_loader import BODexTrajectory, load_bodex_trajectory  # noqa: F401

__all__ = [
    "TabletopGraspSceneCfg",
    "BODexTrajectory",
    "load_bodex_trajectory",
]
