"""Script the piano hands: say which finger presses which note, get joint angles.

The point of this module is that pressing a note is a *scripted* statement, not
a mouse gesture:

    from dexsim.kinematics import PianoHands

    hands = PianoHands()
    hands.press("right", "index", "C4")          # note name, MIDI number, or key
    hands.press("right", "thumb", "G3")
    print(hands.report())                        # per-finger reachability
    hands.render("logs/chord.png")               # real Isaac image, warm server

Reachability is the thing to watch. These hands ride a Y rail with **no arm**,
so the depth into the keyboard (world X) is not actuated: a fingertip can only
reach a note by moving along the rail and curling, and it lands wherever on the
key that arc happens to cross. ``press`` therefore does not demand the key's
centre -- it searches along the key's top surface for the most reachable spot
and reports the residual miss in millimetres. A miss of a few mm means the
finger is genuinely on the key; tens of mm means it is not, and no amount of
solver effort will change that.

Runs in the pyroki venv (no Isaac, no GPU):

    .venv-pyroki/bin/python -c "..."
    .venv-pyroki/bin/python scripts/tools/press_notes.py right:index:C4
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dexsim.piano import geometry
from dexsim.piano.midi import NUM_KEYS, PIANO_MIN_MIDI

REPO = Path(__file__).resolve().parents[3]
CFG = REPO / "source" / "dexsim" / "tasks" / "piano" / "piano_env_cfg.py"
URDF_DIR = REPO / "assets" / "urdf"

# The "_tip" frames are emitted by scripts/tools/export_hand_urdf.py at the far
# end of each distal mesh: the pad that actually contacts a key, ~30 mm past the
# distal link's own origin. Aiming at the origin (which is what the sim's reward
# measures) makes reachable keys look 30 mm out of reach.
FINGERTIPS = {"thumb": "robot0_thdistal_tip", "index": "robot0_ffdistal_tip",
              "middle": "robot0_mfdistal_tip", "ring": "robot0_rfdistal_tip",
              "little": "robot0_lfdistal_tip"}
DISTAL_BODIES = {f: n[: -len("_tip")] for f, n in FINGERTIPS.items()}
FINGER_ALIASES = {"th": "thumb", "ff": "index", "mf": "middle", "rf": "ring",
                  "lf": "little", "pinky": "little", "pinkie": "little",
                  "fore": "index", "1": "thumb", "2": "index", "3": "middle",
                  "4": "ring", "5": "little"}
_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


# --------------------------------------------------------------------- notes
def note_to_key(note: str | int) -> int:
    """Note name ("C4", "F#3", "Bb2"), MIDI number (21-108), or key index -> key.

    Ints are read as MIDI note numbers, which is what "note" means everywhere
    else in this repo; use ``key=`` semantics only through this function's
    output. Middle C ("C4") is MIDI 60.
    """
    if isinstance(note, (int, np.integer)):
        midi = int(note)
    else:
        m = re.fullmatch(r"\s*([A-Ga-g])([#b♯♭]?)(-?\d+)\s*", str(note))
        if not m:
            raise ValueError(f"cannot parse note {note!r} (try 'C4', 'F#3', or 60)")
        letter, accidental, octave = m.group(1).upper(), m.group(2), int(m.group(3))
        midi = 12 * (octave + 1) + _SEMITONE[letter]
        midi += 1 if accidental in ("#", "♯") else -1 if accidental in ("b", "♭") else 0
    key = midi - PIANO_MIN_MIDI
    if not 0 <= key < NUM_KEYS:
        raise ValueError(f"{note!r} -> MIDI {midi} is off the 88-key board "
                         f"(A0=21 .. C8=108)")
    return key


def key_to_name(key: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    midi = key + PIANO_MIN_MIDI
    return f"{names[midi % 12]}{midi // 12 - 1}"


def _finger(name: str) -> str:
    f = FINGER_ALIASES.get(str(name).lower(), str(name).lower())
    if f not in FINGERTIPS:
        raise ValueError(f"unknown finger {name!r}; use {sorted(FINGERTIPS)}")
    return f


# ------------------------------------------------------------------- pose I/O
def read_pose_fields() -> dict:
    """The locked base/piano placement, read straight out of piano_env_cfg.py."""
    src = CFG.read_text()
    fields = ("piano_pos", "piano_rot", "left_base_pos", "left_base_rot",
              "right_base_pos", "right_base_rot")
    out = {}
    for f in fields:
        m = re.search(rf"{f}\s*=\s*(\([^)]*\))", src)
        if m is None:
            raise SystemExit(f"could not find {f} in {CFG}")
        out[f] = np.array(eval(m.group(1), {"__builtins__": {}}, {}), dtype=float)  # noqa: S307
    return out


def read_ready_pose(side: str) -> dict:
    src = CFG.read_text()
    m = re.search(rf"{side}_ready_pose\s*=\s*\{{(.*?)\}}", src, re.DOTALL)
    if m is None:
        return {}
    return {n: float(v) for n, v in re.findall(r'"([^"]+)"\s*:\s*([-\d.eE+]+)', m.group(1))}


# ------------------------------------------------------------ quaternion math
def quat_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


@dataclass
class Press:
    """One scripted finger-on-key assignment and how well it worked out."""
    hand: str
    finger: str
    key: int
    miss_mm: float = float("nan")
    landed: np.ndarray | None = None      # world position the fingertip reached

    @property
    def note(self) -> str:
        return key_to_name(self.key)

    @property
    def reachable(self) -> bool:
        return self.miss_mm < 10.0        # ~1 cm: on the key rather than near it

    def __repr__(self) -> str:
        mark = "ok " if self.reachable else "MISS"
        return (f"<{mark} {self.hand:5s} {self.finger:6s} -> {self.note:4s} "
                f"({self.miss_mm:.1f} mm)>")


@dataclass
class _Side:
    """One hand: the pyroki model plus whatever is currently being pressed."""
    side: str
    urdf: object
    robot: object
    base_pos: np.ndarray
    base_quat: np.ndarray
    rest: np.ndarray
    cfg: np.ndarray
    targets: dict = field(default_factory=dict)   # finger -> world xyz


class PianoHands:
    """Both rail-mounted Shadow Hands over the 88-key board, driven by script."""

    #: Fraction of the key's length (front-to-back) that counts as a valid
    #: landing spot. 0.9 accepts the whole key including its front lip -- fine
    #: for asking "can this finger touch the key at all", misleading if you need
    #: a press that will actually hold, since the lip is a millimetre-wide sliver
    #: the fingertip slides straight off. Narrow it to aim nearer the middle.
    key_span: float = 0.9

    def __init__(self, press_depth: float = geometry.PRESS_DEPTH,
                 hover: float = geometry.HOVER_CLEARANCE, samples: int = 9,
                 key_span: float | None = None):
        if key_span is not None:
            self.key_span = float(key_span)
        import pyroki as pk
        import yourdfpy

        self.press_depth = float(press_depth)
        self.hover = float(hover)
        self.samples = int(samples)
        self.poses = read_pose_fields()
        self.piano_R = quat_matrix(self.poses["piano_rot"])
        self.piano_t = np.asarray(self.poses["piano_pos"], float)
        self._key_top = geometry.key_local_top_positions()

        self.hands: dict[str, _Side] = {}
        for side in ("left", "right"):
            path = URDF_DIR / f"shadow_hand_{side}" / f"shadow_hand_{side}.urdf"
            if not path.exists():
                raise SystemExit(
                    f"missing {path}\nrun: source env.sh && "
                    "python scripts/tools/export_hand_urdf.py")
            urdf = yourdfpy.URDF.load(str(path), load_meshes=False,
                                      build_collision_scene_graph=False)
            robot = pk.Robot.from_urdf(urdf)
            ready = read_ready_pose(side)
            rest = np.array([ready.get(n, 0.0) for n in robot.joints.actuated_names])
            self.hands[side] = _Side(side, urdf, robot,
                                     np.asarray(self.poses[f"{side}_base_pos"], float),
                                     quat_matrix(self.poses[f"{side}_base_rot"]),
                                     rest, rest.copy())
        self.presses: list[Press] = []

    # -- geometry ---------------------------------------------------------
    def key_world(self, key: int, depth: float = 0.0) -> np.ndarray:
        """World position of a key's top-surface centre, ``depth`` m below it."""
        local = np.asarray(self._key_top[int(key)], float)
        world = self.piano_R @ local + self.piano_t
        return world - np.array([0.0, 0.0, depth])

    def _tip_world(self, hand: _Side, finger: str, cfg: np.ndarray) -> np.ndarray:
        hand.urdf.update_cfg(dict(zip(hand.robot.joints.actuated_names, cfg)))
        local = hand.urdf.get_transform(FINGERTIPS[finger], hand.urdf.base_link)
        return hand.base_pos + hand.base_quat @ local[:3, 3]

    # -- scripting --------------------------------------------------------
    def press(self, hand: str, finger: str, note: str | int,
              depth: float | None = None) -> Press:
        """Put one fingertip on a note. Solves immediately; returns the result."""
        side = self._side(hand)
        finger = _finger(finger)
        key = note_to_key(note)
        depth = self.press_depth if depth is None else float(depth)
        side.targets[finger] = ("key", key, depth)
        return self._solve_side(side)[finger]

    def chord(self, hand: str, mapping: dict, depth: float | None = None) -> list[Press]:
        """Press several notes at once, e.g. {"thumb": "C4", "middle": "E4"}."""
        side = self._side(hand)
        depth = self.press_depth if depth is None else float(depth)
        for finger, note in mapping.items():
            side.targets[_finger(finger)] = ("key", note_to_key(note), depth)
        got = self._solve_side(side)
        return [got[_finger(f)] for f in mapping]

    def release(self, hand: str, finger: str | None = None) -> None:
        """Lift one finger (or all of them) back to the hover pose."""
        side = self._side(hand)
        if finger is None:
            side.targets.clear()
        else:
            side.targets.pop(_finger(finger), None)
        self._solve_side(side)

    def reset(self) -> None:
        for side in self.hands.values():
            side.targets.clear()
            side.cfg = side.rest.copy()
        self.presses.clear()

    # -- solving ----------------------------------------------------------
    def _side(self, hand: str) -> _Side:
        key = str(hand).lower()
        if key not in self.hands:
            raise ValueError(f"hand must be 'left' or 'right', got {hand!r}")
        return self.hands[key]

    def _candidates(self, key: int, depth: float) -> np.ndarray:
        """Points along the key's top surface to aim at.

        World X (into the keyboard) is unactuated, so instead of demanding the
        key's centre we offer the solver the whole key and take whichever spot
        it can actually reach.
        """
        black = bool(geometry.KEY_IS_BLACK[int(key)])
        length = geometry.BLACK_L if black else geometry.WHITE_L
        centre = self.key_world(key, depth)
        half = 0.5 * self.key_span * length
        offsets = np.linspace(-half, half, self.samples)
        along = self.piano_R @ np.array([1.0, 0.0, 0.0])       # key's long axis
        return centre[None, :] + offsets[:, None] * along[None, :]

    def _solve_side(self, side: _Side) -> dict:
        from ._solver import solve_targets

        fingers = sorted(side.targets)
        if not fingers:
            side.cfg = side.rest.copy()
            return {}

        # For each finger, the set of spots on its key that would count as a hit.
        options = {}
        for finger in fingers:
            _, key, depth = side.targets[finger]
            options[finger] = self._candidates(key, depth)

        # Pick one spot per finger by trying each and keeping the best-scoring
        # solve. With a single finger this is a plain search over the key; with a
        # chord we keep it cheap by scoring each finger's choice against the
        # solution the previous choices produced.
        chosen = {f: options[f][len(options[f]) // 2] for f in fingers}
        best_cfg, best_score = None, np.inf
        for _ in range(2):                       # two refinement sweeps is plenty
            for finger in fingers:
                for candidate in options[finger]:
                    trial = dict(chosen, **{finger: candidate})
                    cfg = solve_targets(side, trial)
                    score = max(np.linalg.norm(self._tip_world(side, f, cfg) - t)
                                for f, t in trial.items())
                    if score < best_score:
                        best_cfg, best_score, chosen = cfg, score, trial
        side.cfg = best_cfg

        out = {}
        for finger in fingers:
            _, key, _ = side.targets[finger]
            landed = self._tip_world(side, finger, side.cfg)
            miss = float(np.linalg.norm(landed - chosen[finger]) * 1000.0)
            p = Press(side.side, finger, key, miss, landed)
            out[finger] = p
            self.presses = [q for q in self.presses
                            if not (q.hand == p.hand and q.finger == p.finger)]
            self.presses.append(p)
        return out

    # -- output -----------------------------------------------------------
    def joints(self, hand: str) -> dict:
        """{joint_name: radians} for one hand -- feed straight to the sim."""
        side = self._side(hand)
        return dict(zip(side.robot.joints.actuated_names, side.cfg.tolist()))

    def report(self) -> str:
        if not self.presses:
            return "no presses"
        rows = [f"{p!r}" for p in self.presses]
        worst = max(p.miss_mm for p in self.presses)
        return "\n".join(rows + [f"worst miss: {worst:.1f} mm"])

    def render(self, out: str | Path, eye=None, target=None, spp: int = 96,
               fast: bool = False) -> Path:
        """Path-trace the current pose through the warm render server."""
        cmd = ["python", str(REPO / "scripts" / "render" / "render.py"), "scene",
               "--out", str(out), "--spp", str(spp),
               "--left_joints", _kv(self.joints("left")),
               "--right_joints", _kv(self.joints("right"))]
        if fast:
            cmd.append("--fast")
        if eye:
            cmd += ["--eye", ",".join(str(v) for v in eye)]
        if target:
            cmd += ["--target", ",".join(str(v) for v in target)]
        subprocess.run(cmd, cwd=REPO, check=True)
        return Path(out)


def _kv(joints: dict) -> str:
    return ",".join(f"{k}={v:.6f}" for k, v in joints.items())
