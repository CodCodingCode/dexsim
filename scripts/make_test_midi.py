"""Write a simple test MIDI so the piano pipeline can be developed before the
real song arrives. Default: 'Twinkle Twinkle Little Star' (right-hand melody).

  python scripts/make_test_midi.py                 # -> data/midi/twinkle.mid
  python scripts/make_test_midi.py --scale         # -> data/midi/c_major_scale.mid
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pretty_midi

from dexsim import DATA_DIR


def _song(notes, bpm, name):
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    piano = pretty_midi.Instrument(program=0)  # acoustic grand
    beat = 60.0 / bpm
    t = 0.0
    for pitch, beats in notes:
        dur = beats * beat
        if pitch is not None:
            piano.notes.append(pretty_midi.Note(velocity=90, pitch=pitch,
                                                start=t, end=t + dur * 0.95))
        t += dur
    pm.instruments.append(piano)
    return pm


# Twinkle Twinkle (C major): C C G G A A G | F F E E D D C ...
C4, D4, E4, F4, G4, A4 = 60, 62, 64, 65, 67, 69
TWINKLE = [
    (C4, 1), (C4, 1), (G4, 1), (G4, 1), (A4, 1), (A4, 1), (G4, 2),
    (F4, 1), (F4, 1), (E4, 1), (E4, 1), (D4, 1), (D4, 1), (C4, 2),
    (G4, 1), (G4, 1), (F4, 1), (F4, 1), (E4, 1), (E4, 1), (D4, 2),
    (G4, 1), (G4, 1), (F4, 1), (F4, 1), (E4, 1), (E4, 1), (D4, 2),
    (C4, 1), (C4, 1), (G4, 1), (G4, 1), (A4, 1), (A4, 1), (G4, 2),
    (F4, 1), (F4, 1), (E4, 1), (E4, 1), (D4, 1), (D4, 1), (C4, 2),
]
SCALE = [(p, 1) for p in [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", action="store_true", help="C major scale instead of Twinkle")
    ap.add_argument("--bpm", type=float, default=120.0)
    args = ap.parse_args()

    out_dir = DATA_DIR / "midi"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.scale:
        pm = _song(SCALE, args.bpm, "c_major_scale")
        out = out_dir / "c_major_scale.mid"
    else:
        pm = _song(TWINKLE, args.bpm, "twinkle")
        out = out_dir / "twinkle.mid"
    pm.write(str(out))
    print(f"[make_test_midi] wrote {out}")


if __name__ == "__main__":
    main()
