"""MIDI corpus ingestion — the "song set / goal generator".

RoboPianist's training corpus was the human-fingered **PIG** dataset; RP1M scaled
it to ~2k pieces by adding a subset of the openly-downloadable **GiantMIDI-Piano**
dataset and dropping the human fingering (it auto-fingers — see
``dexsim.piano.fingering.plan_fingering(method="ot")``). The MIDI itself is pure
goal data: which keys sound at which time. It is *sim-agnostic*, so we can ingest
exactly the same corpus into our Isaac stack and turn each piece into a
:class:`~dexsim.piano.midi.PianoSong` goal — no human labels required.

This module is the front door for that:

  * scan a directory tree for ``.mid`` / ``.midi`` (and, best-effort, PIG
    ``.proto`` note-sequences) and load each into a ``PianoSong``;
  * score each piece for length, note density and — crucially for our fixed-base
    bimanual rig — *reachability* (how much of it falls inside the two hands'
    reachable key windows, see :func:`dexsim.piano.midi.fold_into_reach`);
  * write a JSON **manifest** so the set is reproducible and can be filtered /
    curriculum-ordered without re-parsing every file;
  * act as a **goal generator**: iterate, sample, or curriculum-order the songs
    for training.

It has no sim dependency, so it is reusable from an Isaac env, a MuJoCo env, or
plain analysis. Run it via ``scripts/build_corpus.py``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .midi import NUM_KEYS, PianoSong, load_song

# Default reachable key windows for the two fixed-base arms. These mirror the
# defaults in ``midi.fold_into_reach`` and the planner's keyboard-middle split;
# a song that lands mostly inside them is "easy" for this embodiment.
DEFAULT_LEFT_WINDOW = (19, 26)
DEFAULT_RIGHT_WINDOW = (63, 70)

MIDI_SUFFIXES = (".mid", ".midi")
PIG_SUFFIXES = (".proto",)


@dataclass
class SongEntry:
    """One song's manifest record (cheap stats, no note arrays)."""

    name: str
    path: str
    num_steps: int
    duration_s: float
    control_dt: float
    num_onsets: int
    key_lo: int
    key_hi: int
    distinct_keys: int
    notes_per_s: float
    max_polyphony: int           # most simultaneous keys at any step
    reachable_frac: float        # fraction of onset-notes already inside a hand window
    foldable: bool               # whether fold_into_reach can bring it in range
    source_format: str           # "midi" | "pig_proto"

    def difficulty(self) -> float:
        """A rough 0..1 difficulty score for curriculum ordering.

        Higher = harder: dense, polyphonic, long, and/or out of natural reach.
        Cheap and monotonic, not calibrated — just enough to order a curriculum.
        """
        density = min(self.notes_per_s / 12.0, 1.0)
        poly = min(self.max_polyphony / 10.0, 1.0)
        span = min(self.distinct_keys / 40.0, 1.0)
        unreach = 1.0 - self.reachable_frac
        length = min(self.duration_s / 60.0, 1.0)
        return float(0.30 * density + 0.20 * poly + 0.20 * span
                     + 0.20 * unreach + 0.10 * length)


def _song_stats(song: PianoSong, left_window, right_window,
                source_format: str) -> SongEntry:
    act = song.key_activation
    onsets = song.onsets
    per_step = act.sum(axis=1)
    active_keys = np.nonzero(act.any(axis=0))[0]
    lo = int(active_keys.min()) if active_keys.size else -1
    hi = int(active_keys.max()) if active_keys.size else -1

    # reachable fraction: of all note onsets, how many already fall in a hand
    # window (left for low notes, right for high), i.e. need no octave-folding.
    onset_keys = np.nonzero(onsets)[1]
    if onset_keys.size:
        in_left = ((onset_keys >= left_window[0]) & (onset_keys <= left_window[1]))
        in_right = ((onset_keys >= right_window[0]) & (onset_keys <= right_window[1]))
        reachable_frac = float((in_left | in_right).mean())
    else:
        reachable_frac = 0.0

    return SongEntry(
        name=song.name,
        path=str(song.source),
        num_steps=song.num_steps,
        duration_s=round(song.duration_s, 3),
        control_dt=song.control_dt,
        num_onsets=int(onsets.sum()),
        key_lo=lo,
        key_hi=hi,
        distinct_keys=int(active_keys.size),
        notes_per_s=round(float(onsets.sum()) / max(song.duration_s, 1e-6), 3),
        max_polyphony=int(per_step.max()) if per_step.size else 0,
        reachable_frac=round(reachable_frac, 3),
        # A song is foldable into reach iff each hand is asked for <=5 distinct
        # pitch-classes-per-window; we approximate with distinct_keys <= 10 OR it
        # already has decent reachability (octave folding handles the rest).
        foldable=bool(active_keys.size <= 10 or reachable_frac > 0.0),
        source_format=source_format,
    )


def _pig_proto_to_pretty_midi(path: Path):
    """Best-effort load of a PIG ``.proto`` (note_seq.NoteSequence) -> pretty_midi.

    PIG ships fingering-annotated pieces as protobuf NoteSequences. Converting
    them needs ``note_seq`` (``pip install note-seq``). If it isn't installed we
    raise a clear error telling the caller to install it (or supply .mid files
    directly under data/midi).
    """
    try:
        import note_seq  # type: ignore
    except Exception as e:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            f"Cannot read PIG proto {path.name}: install `note-seq` "
            f"(pip install note-seq), or supply .mid files directly instead."
        ) from e
    seq = note_seq.NoteSequence()
    seq.ParseFromString(path.read_bytes())
    return note_seq.note_sequence_to_pretty_midi(seq)


@dataclass
class MidiCorpus:
    """A scanned set of MIDI pieces with cheap per-song stats + a manifest.

    Use it as a goal generator: :meth:`iter_songs` yields fully-loaded
    :class:`PianoSong`s; :meth:`sample` / :meth:`curriculum` order them.
    """

    entries: list[SongEntry] = field(default_factory=list)
    control_dt: float = 0.05
    left_window: tuple[int, int] = DEFAULT_LEFT_WINDOW
    right_window: tuple[int, int] = DEFAULT_RIGHT_WINDOW

    # ----- construction ----------------------------------------------------
    @classmethod
    def scan(cls, roots: Sequence[str | Path], control_dt: float = 0.05,
             left_window=DEFAULT_LEFT_WINDOW, right_window=DEFAULT_RIGHT_WINDOW,
             include_pig: bool = True, verbose: bool = True) -> "MidiCorpus":
        """Recursively scan ``roots`` for MIDI (and optional PIG proto) files."""
        corpus = cls(control_dt=control_dt, left_window=tuple(left_window),
                     right_window=tuple(right_window))
        files: list[tuple[Path, str]] = []
        for root in roots:
            root = Path(root)
            if root.is_file():
                files.append((root, _fmt_of(root)))
                continue
            for p in sorted(root.rglob("*")):
                if p.suffix.lower() in MIDI_SUFFIXES:
                    files.append((p, "midi"))
                elif include_pig and p.suffix.lower() in PIG_SUFFIXES:
                    files.append((p, "pig_proto"))

        seen_names: set[str] = set()
        for path, fmt in files:
            try:
                song = corpus._load_one(path, fmt)
            except Exception as e:
                if verbose:
                    print(f"  [skip] {path.name}: {e}")
                continue
            # de-dup by stem so the same tune from two roots isn't double-counted
            if song.name in seen_names:
                continue
            seen_names.add(song.name)
            entry = _song_stats(song, corpus.left_window, corpus.right_window, fmt)
            corpus.entries.append(entry)
            if verbose:
                print(f"  [ok]   {song.summary()}  (reach {entry.reachable_frac:.2f}, "
                      f"diff {entry.difficulty():.2f})")
        corpus.entries.sort(key=lambda e: e.name)
        return corpus

    def _load_one(self, path: Path, fmt: str) -> PianoSong:
        if fmt == "pig_proto":
            pm = _pig_proto_to_pretty_midi(path)
            # write a temp .mid next to a cache so load_song can reuse its parser
            tmp = path.with_suffix(".pig.mid")
            pm.write(str(tmp))
            song = load_song(tmp, control_dt=self.control_dt)
            song.name = path.stem
            return song
        return load_song(path, control_dt=self.control_dt)

    # ----- manifest --------------------------------------------------------
    def to_manifest(self) -> dict:
        return {
            "control_dt": self.control_dt,
            "left_window": list(self.left_window),
            "right_window": list(self.right_window),
            "num_songs": len(self.entries),
            "songs": [asdict(e) for e in self.entries],
        }

    def write_manifest(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_manifest(), indent=2))
        return path

    @classmethod
    def from_manifest(cls, path: str | Path) -> "MidiCorpus":
        data = json.loads(Path(path).read_text())
        corpus = cls(
            control_dt=data["control_dt"],
            left_window=tuple(data.get("left_window", DEFAULT_LEFT_WINDOW)),
            right_window=tuple(data.get("right_window", DEFAULT_RIGHT_WINDOW)),
        )
        corpus.entries = [SongEntry(**s) for s in data["songs"]]
        return corpus

    # ----- goal generator --------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    def names(self) -> list[str]:
        return [e.name for e in self.entries]

    def filter(self, *, max_difficulty: float | None = None,
               min_reachable: float | None = None,
               max_duration_s: float | None = None,
               require_foldable: bool = False) -> "MidiCorpus":
        """Return a new corpus keeping only songs that pass the given gates."""
        out = MidiCorpus(control_dt=self.control_dt,
                         left_window=self.left_window, right_window=self.right_window)
        for e in self.entries:
            if max_difficulty is not None and e.difficulty() > max_difficulty:
                continue
            if min_reachable is not None and e.reachable_frac < min_reachable:
                continue
            if max_duration_s is not None and e.duration_s > max_duration_s:
                continue
            if require_foldable and not e.foldable:
                continue
            out.entries.append(e)
        return out

    def curriculum(self) -> list[SongEntry]:
        """Songs ordered easy -> hard (for curriculum training)."""
        return sorted(self.entries, key=lambda e: e.difficulty())

    def sample(self, rng: np.random.Generator, *, by_difficulty: bool = False) -> SongEntry:
        """Sample one song. If ``by_difficulty``, bias toward easier pieces."""
        if not self.entries:
            raise ValueError("empty corpus")
        if not by_difficulty:
            return self.entries[int(rng.integers(len(self.entries)))]
        diff = np.array([e.difficulty() for e in self.entries])
        w = np.exp(-3.0 * diff)            # easier -> higher weight
        w = w / w.sum()
        return self.entries[int(rng.choice(len(self.entries), p=w))]

    def load(self, entry: SongEntry) -> PianoSong:
        """Materialise a manifest entry back into a full PianoSong goal."""
        return load_song(entry.path, control_dt=self.control_dt)

    def iter_songs(self, *, curriculum: bool = False) -> Iterator[PianoSong]:
        order = self.curriculum() if curriculum else self.entries
        for e in order:
            try:
                yield self.load(e)
            except Exception as exc:  # pragma: no cover
                print(f"  [skip-load] {e.name}: {exc}")
