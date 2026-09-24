"""MJCF scene builders for the piano task.

The single source of truth for keyboard geometry is ``dexsim.piano.geometry``;
these modules turn it into a MuJoCo model:

  * :mod:`dexsim.mjcf.shadow_hand` -- adapt the MuJoCo Menagerie Shadow Hand
    (right + true left) to the repo's ``robot0_*`` naming convention.
  * :mod:`dexsim.mjcf.piano`       -- procedural 88-key spring-loaded piano.
  * :mod:`dexsim.mjcf.scene`       -- compose the full bimanual scene
    (piano + two rail-mounted hands) and save/compile it.
  * :mod:`dexsim.mjcf.keyboard`    -- procedural MacBook Pro (M3 Max, 16")
    keyboard + laptop body: 78 spring-loaded scissor keys.
  * :mod:`dexsim.mjcf.keyboard_scene` -- the keyboard twin of ``scene``:
    laptop + two gantry-mounted hands over the home row
    (``scripts/mj/keyboard_demo.py`` types on it).
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
from .keyboard import add_macbook, add_keyboard, text_to_keys, TOUCH_TYPING_FINGER
from .keyboard_scene import (
    KeyboardSceneCfg,
    build_keyboard_scene_spec,
    compile_keyboard_scene,
    save_keyboard_scene_xml,
)

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
    "add_macbook",
    "add_keyboard",
    "text_to_keys",
    "TOUCH_TYPING_FINGER",
    "KeyboardSceneCfg",
    "build_keyboard_scene_spec",
    "compile_keyboard_scene",
    "save_keyboard_scene_xml",
]
