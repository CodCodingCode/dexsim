"""Write a maximally-easy song: press a given set of keys (the ones resting under
the fingers) one at a time, slowly, in sequence — a '5-finger exercise'. This is
the most learnable possible piano task: no hand movement, each finger presses the
key under it. Scale up once this trains.

  python scripts/make_easy_song.py --midi 60 62 64 65 67 --bpm 60 --out data/midi/easy.mid
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pretty_midi
from dexsim import DATA_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--midi", type=int, nargs="+", required=True, help="MIDI notes to use (keys under fingers)")
ap.add_argument("--bpm", type=float, default=60.0)
ap.add_argument("--beats_per_note", type=float, default=1.0)
ap.add_argument("--reps", type=int, default=4, help="repeat the sequence this many times")
ap.add_argument("--gap_beats", type=float, default=0.5, help="silence between notes")
ap.add_argument("--out", default=str(DATA_DIR / "midi" / "easy.mid"))
args = ap.parse_args()

pm = pretty_midi.PrettyMIDI(initial_tempo=args.bpm)
inst = pretty_midi.Instrument(program=0)
beat = 60.0 / args.bpm
t = 0.5  # small lead-in
seq = list(args.midi) * args.reps
for pitch in seq:
    dur = args.beats_per_note * beat
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=t, end=t + dur * 0.9))
    t += (args.beats_per_note + args.gap_beats) * beat
pm.instruments.append(inst)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
pm.write(args.out)
print(f"[make_easy_song] {len(seq)} notes on keys {args.midi} -> {args.out} ({t:.1f}s)")
