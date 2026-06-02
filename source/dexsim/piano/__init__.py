"""Piano playing task: MIDI parsing, keyboard model, and bimanual reward.

Framework-agnostic pieces live here (MIDI -> note schedule). The sim-specific
environment (Isaac Lab) is assembled in ``dexsim.tasks.piano``.
"""

from .midi import (
    PianoSong,
    load_song,
    midi_to_key,
    key_is_black,
    NUM_KEYS,
    PIANO_MIN_MIDI,
    PIANO_MAX_MIDI,
)
from . import geometry
from .fingering import (
    plan_fingering,
    finger_targets_local,
    FingeringPlan,
    NUM_FINGERS,
    FINGERTIP_BODIES,
)

__all__ = [
    "PianoSong",
    "load_song",
    "midi_to_key",
    "key_is_black",
    "NUM_KEYS",
    "PIANO_MIN_MIDI",
    "PIANO_MAX_MIDI",
    "geometry",
    "plan_fingering",
    "finger_targets_local",
    "FingeringPlan",
    "NUM_FINGERS",
    "FINGERTIP_BODIES",
]
