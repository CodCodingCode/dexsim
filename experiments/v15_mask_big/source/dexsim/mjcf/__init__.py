"""MJCF scene builders for the MuJoCo port of the piano task.

The single source of truth for keyboard geometry stays ``dexsim.piano.geometry``
(shared with the Isaac stack); these modules turn it into a MuJoCo model:

  * :mod:`dexsim.mjcf.shadow_hand` -- adapt the MuJoCo Menagerie Shadow Hand
    (right + true left) to the repo's ``robot0_*`` naming convention.
  * :mod:`dexsim.mjcf.piano`       -- procedural 88-key spring-loaded piano.
  * :mod:`dexsim.mjcf.scene`       -- compose the full bimanual scene
    (piano + two rail-mounted hands) and save/compile it.

None of this imports Isaac; the MuJoCo stack runs in the plain ``.venv``.
"""

from .piano import (
    KEY_MAX_TRAVEL_ANGLE,
    KEY_SOUND_ANGLE,
    KEY_SPRING_STIFFNESS,
    KEY_SPRING_DAMPING,
    add_piano,
)
from .shadow_hand import load_hand_spec, FINGERTIP_SITES
from .scene import build_scene_spec, save_scene_xml, compile_scene

__all__ = [
    "KEY_MAX_TRAVEL_ANGLE",
    "KEY_SOUND_ANGLE",
    "KEY_SPRING_STIFFNESS",
    "KEY_SPRING_DAMPING",
    "add_piano",
    "load_hand_spec",
    "FINGERTIP_SITES",
    "build_scene_spec",
    "save_scene_xml",
    "compile_scene",
]
