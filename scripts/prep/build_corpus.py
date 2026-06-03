#!/usr/bin/env python
"""Ingest a MIDI corpus into a reusable manifest + goal generator.

The "free win" from the RoboPianist/RP1M analysis: the MIDI corpus is pure,
sim-agnostic goal data. Point this at a directory of ``.mid`` files (e.g. a
GiantMIDI-Piano download, the RoboPianist ``rousseau`` set, or PIG MIDIs you
preprocessed) and it builds ``data/corpus/manifest.json`` with per-song stats,
reachability and a difficulty score for curriculum training.

Examples
--------
    source env.sh
    # scan everything we already have on disk
    python scripts/build_corpus.py --roots data/midi data/robopianist_ref/robopianist/music/data

    # scan a GiantMIDI download and keep only short, reachable pieces
    python scripts/build_corpus.py --roots data/corpus/giantmidi \
        --max-difficulty 0.5 --max-duration 40 --out data/corpus/manifest.json

    # inspect an existing manifest (no re-parse)
    python scripts/build_corpus.py --show data/corpus/manifest.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source"))

from dexsim.piano.corpus import MidiCorpus  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="*", default=["data/midi"],
                    help="files/dirs to scan for .mid/.midi (+ PIG .proto)")
    ap.add_argument("--out", default="data/corpus/manifest.json",
                    help="manifest output path")
    ap.add_argument("--control-dt", type=float, default=0.05)
    ap.add_argument("--no-pig", action="store_true", help="skip PIG .proto files")
    ap.add_argument("--show", default=None, metavar="MANIFEST",
                    help="just print an existing manifest and exit")
    # optional filters applied to the written manifest
    ap.add_argument("--max-difficulty", type=float, default=None)
    ap.add_argument("--min-reachable", type=float, default=None)
    ap.add_argument("--max-duration", type=float, default=None)
    ap.add_argument("--require-foldable", action="store_true")
    args = ap.parse_args()

    if args.show:
        corpus = MidiCorpus.from_manifest(args.show)
        _print_table(corpus)
        return

    roots = [str((REPO / r) if not Path(r).is_absolute() else Path(r)) for r in args.roots]
    print(f"[corpus] scanning: {roots}")
    corpus = MidiCorpus.scan(roots, control_dt=args.control_dt,
                             include_pig=not args.no_pig, verbose=True)
    print(f"[corpus] loaded {len(corpus)} songs")

    if any(v is not None for v in (args.max_difficulty, args.min_reachable,
                                   args.max_duration)) or args.require_foldable:
        corpus = corpus.filter(max_difficulty=args.max_difficulty,
                               min_reachable=args.min_reachable,
                               max_duration_s=args.max_duration,
                               require_foldable=args.require_foldable)
        print(f"[corpus] after filter: {len(corpus)} songs")

    out = corpus.write_manifest((REPO / args.out) if not Path(args.out).is_absolute()
                                else Path(args.out))
    print(f"[corpus] wrote manifest -> {out}")
    _print_table(corpus)


def _print_table(corpus: MidiCorpus) -> None:
    print(f"\n{'song':30s} {'steps':>6} {'dur_s':>6} {'notes/s':>7} "
          f"{'poly':>4} {'span':>4} {'reach':>5} {'diff':>5}")
    print("-" * 78)
    for e in corpus.curriculum():
        print(f"{e.name[:30]:30s} {e.num_steps:6d} {e.duration_s:6.1f} "
              f"{e.notes_per_s:7.2f} {e.max_polyphony:4d} {e.distinct_keys:4d} "
              f"{e.reachable_frac:5.2f} {e.difficulty():5.2f}")


if __name__ == "__main__":
    main()
