"""Render a MIDI file to a .wav (so you can actually hear it). Uses the
fluidsynth binary + the soundfont bundled with pretty_midi. Reusable:

  python scripts/midi_to_wav.py results/easy_song_played.mid
  from midi_to_wav import midi_to_wav; midi_to_wav("x.mid")  # -> x.wav
"""
from __future__ import annotations
import os
import subprocess
from pathlib import Path

# soundfont shipped with pretty_midi
import pretty_midi as _pm
SOUNDFONT = str(Path(_pm.__file__).parent / "TimGM6mb.sf2")


def midi_to_wav(midi_path: str | Path, wav_path: str | Path | None = None,
                sample_rate: int = 44100, gain: float = 1.0) -> str:
    """Synthesize `midi_path` -> `wav_path` (default: same name, .wav)."""
    midi_path = str(midi_path)
    wav_path = str(wav_path) if wav_path else os.path.splitext(midi_path)[0] + ".wav"
    if not os.path.exists(SOUNDFONT):
        raise FileNotFoundError(f"soundfont not found: {SOUNDFONT}")
    cmd = ["fluidsynth", "-ni", "-g", str(gain), "-r", str(sample_rate),
           SOUNDFONT, midi_path, "-F", wav_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        out = midi_to_wav(f)
        sz = os.path.getsize(out)
        print(f"[midi_to_wav] {f} -> {out} ({sz} bytes)")
