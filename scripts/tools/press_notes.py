"""Script note presses from the shell: which finger goes on which note.

    .venv-pyroki/bin/python scripts/tools/press_notes.py left:index:C4 right:thumb:F4
    .venv-pyroki/bin/python scripts/tools/press_notes.py right:index:C4 --render logs/press.png
    .venv-pyroki/bin/python scripts/tools/press_notes.py --reach left      # what CAN it touch?

Each argument is ``hand:finger:note`` -- note being a name ("C4", "F#3"), a MIDI
number (60), with fingers thumb/index/middle/ring/little (or th/ff/mf/rf/lf).
Every press prints the residual miss in millimetres, which is the honest
reachability signal for these arm-less rail-mounted hands.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source"))

from dexsim.kinematics import PianoHands             # noqa: E402
from dexsim.kinematics.piano_hands import key_to_name, note_to_key  # noqa: E402
from dexsim.piano.midi import NUM_KEYS               # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("presses", nargs="*", metavar="HAND:FINGER:NOTE")
    p.add_argument("--depth", type=float, default=None,
                   help="metres below the key top to aim (default: geometry.PRESS_DEPTH)")
    p.add_argument("--render", metavar="PNG",
                   help="path-trace the resulting pose through the warm render server")
    p.add_argument("--reach", choices=("left", "right"),
                   help="instead of pressing: list every note this hand can touch")
    p.add_argument("--json", metavar="PATH", help="also write the joint angles as json")
    a = p.parse_args()

    hands = PianoHands()

    if a.reach:
        print(f"reachable notes for the {a.reach} hand (miss <= 10 mm):")
        any_hit = False
        for finger in ("thumb", "index", "middle", "ring", "little"):
            hits = []
            for key in range(NUM_KEYS):
                got = hands.press(a.reach, finger, key + 21, depth=a.depth)
                if got.reachable:
                    hits.append(key_to_name(key))
                hands.release(a.reach, finger)
            any_hit |= bool(hits)
            print(f"  {finger:7s}: {' '.join(hits) if hits else '(none)'}")
        if not any_hit:
            print("\nNo note is reachable by any finger -- the hands are parked above\n"
                  "the keys rather than on them. Run with a single press to see the\n"
                  "residual, or check the base placement in piano_env_cfg.py.")
        return

    if not a.presses:
        p.error("give at least one HAND:FINGER:NOTE (or use --reach)")

    for spec in a.presses:
        try:
            hand, finger, note = spec.split(":")
        except ValueError:
            p.error(f"bad press {spec!r}; expected HAND:FINGER:NOTE like left:index:C4")
        note_to_key(note)                      # fail early on a typo'd note
        print(hands.press(hand, finger, note, depth=a.depth))

    print("\n" + hands.report())

    if a.json:
        import json
        Path(a.json).write_text(json.dumps(
            {side: hands.joints(side) for side in ("left", "right")}, indent=2))
        print(f"[joints] {a.json}")
    if a.render:
        print(f"[render] {hands.render(a.render)}")


if __name__ == "__main__":
    main()
