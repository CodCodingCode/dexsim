"""MIDI -> piano note-activation schedule (framework-agnostic).

Turns a ``.mid`` file into the things an RL piano task needs:

  * ``key_activation`` : (T, 88) bool -- which of the 88 keys (A0..C8, MIDI
    21..108) should be held down at each control step.
  * ``onsets``         : (T, 88) bool -- the step where each note *begins*
    (used to reward hitting the key at the right moment, not just holding it).
  * ``sustain``        : (T,) bool -- sustain-pedal state per step.

Everything is sampled on a fixed control grid of ``control_dt`` seconds so it
lines up with the simulator's control rate. This module has no sim dependency,
so it's reusable by an Isaac Lab env, a MuJoCo env, or plain analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 88-key piano spans MIDI note numbers 21 (A0) .. 108 (C8).
PIANO_MIN_MIDI = 21
PIANO_MAX_MIDI = 108
NUM_KEYS = PIANO_MAX_MIDI - PIANO_MIN_MIDI + 1  # 88


def midi_to_key(note_number: int) -> int:
    """MIDI note number -> piano key index in [0, 88). -1 if off the keyboard."""
    if note_number < PIANO_MIN_MIDI or note_number > PIANO_MAX_MIDI:
        return -1
    return note_number - PIANO_MIN_MIDI


def key_is_black(key_index: int) -> bool:
    """True if piano key `key_index` (0..87) is a black key."""
    # MIDI note % 12 with C=0: black keys are C#,D#,F#,G#,A# = {1,3,6,8,10}.
    semitone = (key_index + PIANO_MIN_MIDI) % 12
    return semitone in (1, 3, 6, 8, 10)


def fold_into_reach(key_activation: np.ndarray, onsets: np.ndarray,
                    left_window: tuple[int, int] = (19, 26),
                    right_window: tuple[int, int] = (63, 70)):
    """Re-arrange a song so every note lands in a hand's *reachable* key window.

    The bimanual rig has two fixed arm bases (left over key ~22, right over key
    ~66) that each reach only a narrow ~8-key band; the middle of the keyboard is
    a no-man's-land neither hand can reach. A wide-range song (e.g. 5 octaves)
    therefore produces an un-trackable IK reference (fingertip errors of hundreds
    of mm -> zero-residual F1 ~0.05). This folds each note by OCTAVES toward the
    nearest hand window (preserving pitch class where the window allows, so the
    tune stays recognisable), clamping to the window edge when the <12-key window
    can't represent that pitch class. Notes below the split go to the left hand,
    notes at/above it to the right -- matching the planner's keyboard-middle split.

    Args:
        key_activation (T, 88) bool, onsets (T, 88) bool -- the original schedule.
        left_window, right_window: inclusive (lo, hi) reachable key-index bands.
    Returns: (new_key_activation, new_onsets), both (T, 88) bool, remapped.
    """
    active_keys = np.nonzero(key_activation.any(axis=0))[0]
    if active_keys.size == 0:
        return key_activation, onsets
    # Split by the keyboard middle between the two reachable windows (low register
    # -> left hand, high -> right), matching the env's hand mask in piano_env. Using
    # the median of active keys was a bug: it split EVERY song ~50/50 across both
    # hands, so a single-hand song (e.g. easy.mid keys {19,20,26}) sent half its
    # notes to the hand that can't reach them -> unreachable goal, depressed F1.
    split = 0.5 * (left_window[1] + right_window[0])

    def _fold(k: int, lo: int, hi: int) -> int:
        c = 0.5 * (lo + hi)
        while k < c - 6:                           # bring within an octave of centre
            k += 12
        while k > c + 6:
            k -= 12
        return int(min(hi, max(lo, k)))            # clamp into the reachable band

    remap = {}
    for k in active_keys:
        lo, hi = (left_window if k < split else right_window)
        remap[int(k)] = _fold(int(k), lo, hi)

    T = key_activation.shape[0]
    new_act = np.zeros_like(key_activation)
    new_ons = np.zeros_like(onsets)
    for k_old, k_new in remap.items():
        new_act[:, k_new] |= key_activation[:, k_old]
        new_ons[:, k_new] |= onsets[:, k_old]
    return new_act, new_ons


@dataclass
class PianoSong:
    """A MIDI piece sampled onto a fixed control grid."""

    name: str
    control_dt: float
    key_activation: np.ndarray  # (T, 88) bool
    onsets: np.ndarray          # (T, 88) bool
    sustain: np.ndarray         # (T,) bool
    source: Path

    @property
    def num_steps(self) -> int:
        return int(self.key_activation.shape[0])

    @property
    def duration_s(self) -> float:
        return self.num_steps * self.control_dt

    def goal_at(self, step: int, lookahead: int = 1) -> np.ndarray:
        """(lookahead, 88) future key-activation goal starting at `step`,
        zero-padded past the end of the song. This is what the policy observes
        so it can pre-position the fingers."""
        end = min(step + lookahead, self.num_steps)
        out = np.zeros((lookahead, NUM_KEYS), dtype=bool)
        if end > step:
            out[: end - step] = self.key_activation[step:end]
        return out

    def summary(self) -> str:
        active = self.key_activation.any(axis=0)
        n_notes = int(self.onsets.sum())
        rng = np.where(active)[0]
        lo = int(rng.min()) if rng.size else -1
        hi = int(rng.max()) if rng.size else -1
        return (f"{self.name}: {self.num_steps} steps, {self.duration_s:.1f}s @ "
                f"{1/self.control_dt:.0f}Hz, {n_notes} note onsets, "
                f"key span [{lo}..{hi}] ({int(active.sum())} distinct keys)")


def load_song(path: str | Path, control_dt: float = 0.05,
              trim_silence: bool = True) -> PianoSong:
    """Load a MIDI file and sample it onto a `control_dt` grid.

    Args:
        path: a .mid / .midi file.
        control_dt: seconds per control step (e.g. 0.05 -> 20 Hz).
        trim_silence: drop leading silence so the song starts at step 0.
    """
    import pretty_midi

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    pm = pretty_midi.PrettyMIDI(str(path))

    # collect all notes across instruments (skip drums)
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        notes.extend(inst.notes)
    if not notes:
        raise ValueError(f"No (non-drum) notes found in {path}")

    t0 = min(n.start for n in notes) if trim_silence else 0.0
    t_end = max(n.end for n in notes) - t0
    num_steps = int(np.ceil(t_end / control_dt)) + 1

    key_activation = np.zeros((num_steps, NUM_KEYS), dtype=bool)
    onsets = np.zeros((num_steps, NUM_KEYS), dtype=bool)
    for n in notes:
        k = midi_to_key(n.pitch)
        if k < 0:
            continue
        s = int(round((n.start - t0) / control_dt))
        e = int(round((n.end - t0) / control_dt))
        s = max(0, min(s, num_steps - 1))
        e = max(s + 1, min(e, num_steps))
        key_activation[s:e, k] = True
        onsets[s, k] = True

    # sustain pedal (CC64 >= 64 = down)
    sustain = np.zeros(num_steps, dtype=bool)
    for inst in pm.instruments:
        for cc in inst.control_changes:
            if cc.number != 64:
                continue
            step = int(round((cc.time - t0) / control_dt))
            if 0 <= step < num_steps:
                sustain[step:] = cc.value >= 64

    return PianoSong(
        name=path.stem, control_dt=control_dt,
        key_activation=key_activation, onsets=onsets, sustain=sustain,
        source=path,
    )
