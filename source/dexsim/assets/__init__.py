"""Asset configurations for the UR10e + Shadow Hand embodiment."""

from .ur10e_shadow import (
    UR10E_CFG,
    SHADOW_HAND_CFG,
    UR10E_SHADOW_CFG,
    UR10E_SHADOW_LEFT_CFG,
    COMBINED_USD_PATH,
    COMBINED_LEFT_USD_PATH,
)
from .piano import (
    PIANO_CFG,
    PIANO_USD_PATH,
    KEY_SOUND_ANGLE,
)

__all__ = [
    "UR10E_CFG",
    "SHADOW_HAND_CFG",
    "UR10E_SHADOW_CFG",
    "UR10E_SHADOW_LEFT_CFG",
    "COMBINED_USD_PATH",
    "COMBINED_LEFT_USD_PATH",
    "PIANO_CFG",
    "PIANO_USD_PATH",
    "KEY_SOUND_ANGLE",
]
