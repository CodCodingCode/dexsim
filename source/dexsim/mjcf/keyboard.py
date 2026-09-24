"""Procedural MacBook Pro (M3 Max, 16-inch) keyboard + laptop body for MuJoCo.

A computer-keyboard twin of :mod:`dexsim.mjcf.piano`: every key is a PASSIVE
spring-loaded body the hands press, with a site at its cap centre so the env
/ demo code can read "where do I put a finger to press ``a``" the same way
it reads ``key_site_i`` on the piano.

Specs modelled (Apple Magic Keyboard, scissor switch, US ANSI, as on the
2023 16-inch MacBook Pro with M3 Max -- the 14-inch has the same keyboard):

  * 14.5u × 6 rows, horizontal pitch 19.05 mm, vertical pitch 18.6 mm
    -> keyboard well ~276 × 112 mm (Apple: ~275 × 110 mm);
  * keycap footprint 16.5 × 16.0 mm (~2.5 mm gap), 1.0 mm key travel,
    ~55 gf actuation at half travel; caps protrude ~1 mm over the well floor;
  * full-height function row (2021+ design), inverted-T arrows with
    half-height up/down, 1.5u ``esc`` + Touch ID key, 5u space bar;
  * 16-inch body 355.7 × 248.1 × 16.8 mm, 160 × 100 mm Force Touch trackpad,
    lid opened 110° (visual only), Space Black finish.

Frame convention matches the piano: keyboard-local +X points toward the
typist (front / space-bar row), +Y runs left->right along a row, +Z is up,
origin at the centre of the keyboard well at the *well floor* height.

Naming: body ``key_<name>``, joint ``joint_<name>``, cap-centre site
``key_site_<name>`` where ``<name>`` is an ASCII id (``a``, ``1``, ``space``,
``lshift``, ``f5`` ...; see :data:`KEY_NAMES`). Pressing a key drives its
slide joint NEGATIVE (down), the same sign convention as the piano hinges:
``qpos <= -KEY_ACTUATION_DEPTH`` == "key registered".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
LEGEND_DIR = _ROOT / "assets" / "mj" / "keycaps"      # generated PNGs (gitignored)

# --- Apple Magic Keyboard (laptop, scissor) geometry -----------------------
PITCH_X = 0.01905          # m, centre-to-centre along a row (1u)
PITCH_Y = 0.0186           # m, row-to-row
CAP_W = 0.0165             # m, keycap width along the row (for a 1u key)
CAP_D = 0.0160             # m, keycap depth (front-back)
CAP_H = 0.0030             # m, keycap box height (well floor -> cap top)
KEY_TRAVEL = 0.0010        # m, scissor mechanism travel
KEY_ACTUATION_DEPTH = 0.0005   # m, registers at half travel (Apple ~0.5 mm)
KEY_SPRING_STIFFNESS = 550.0   # N/m: 0.55 N (~55 gf) at full travel
KEY_SPRING_DAMPING = 0.6       # N·s/m, light (keys snap back in ~10 ms)
HALF_KEY_GAP = 0.0012      # m, gap between the two half-height arrow caps

# --- 16-inch MacBook Pro body -----------------------------------------------
BODY_W = 0.3557            # m, left-right
BODY_D = 0.2481            # m, front-back
BODY_H = 0.0168            # m, closed thickness (base only ~ 0.0098; we use
#   the whole 16.8 mm for the base slab so the lid can be a thin plate)
BASE_H = 0.0098
WELL_RECESS = 0.0020       # m, well floor below the palm-rest deck top
WELL_MARGIN = 0.0030       # m, black well plate margin around the caps
KEYBOARD_BACK_GAP = 0.0210 # m, hinge edge -> back edge of the function row
TRACKPAD_W, TRACKPAD_D = 0.160, 0.100
LID_ANGLE_DEG = 110.0      # opening angle of the display
LID_T = 0.0045             # m, lid thickness

# Space Black finish
_ALU_RGBA = (0.20, 0.20, 0.21, 1.0)
_WELL_RGBA = (0.035, 0.035, 0.04, 1.0)
_CAP_RGBA = (0.075, 0.075, 0.08, 1.0)
_CAP_MOD_RGBA = (0.10, 0.10, 0.105, 1.0)     # modifiers a shade lighter
_TRACKPAD_RGBA = (0.17, 0.17, 0.18, 1.0)
_SCREEN_RGBA = (0.02, 0.02, 0.03, 1.0)
# deck slabs collide with the hands (default contype/conaffinity 1/1) but not
# with the keycaps that rest on them (keycaps are contype 2 / conaffinity 1):
# collide iff (a.contype & b.conaffinity) | (b.contype & a.conaffinity).
_DECK_COLLISION = dict(contype=0, conaffinity=1)

# --- layout: rows back (function row) -> front (space row) ------------------
# (name, width in u); a tuple of two names in one slot = stacked half-height
# keys (the inverted-T up/down arrows).
_ROWS: list[list[tuple]] = [
    [("esc", 1.5), ("f1", 1), ("f2", 1), ("f3", 1), ("f4", 1), ("f5", 1),
     ("f6", 1), ("f7", 1), ("f8", 1), ("f9", 1), ("f10", 1), ("f11", 1),
     ("f12", 1), ("touchid", 1)],
    [("grave", 1), ("1", 1), ("2", 1), ("3", 1), ("4", 1), ("5", 1), ("6", 1),
     ("7", 1), ("8", 1), ("9", 1), ("0", 1), ("minus", 1), ("equal", 1),
     ("delete", 1.5)],
    [("tab", 1.5), ("q", 1), ("w", 1), ("e", 1), ("r", 1), ("t", 1), ("y", 1),
     ("u", 1), ("i", 1), ("o", 1), ("p", 1), ("lbracket", 1), ("rbracket", 1),
     ("backslash", 1)],
    [("caps", 1.75), ("a", 1), ("s", 1), ("d", 1), ("f", 1), ("g", 1), ("h", 1),
     ("j", 1), ("k", 1), ("l", 1), ("semicolon", 1), ("quote", 1),
     ("return", 1.75)],
    [("lshift", 2.25), ("z", 1), ("x", 1), ("c", 1), ("v", 1), ("b", 1),
     ("n", 1), ("m", 1), ("comma", 1), ("period", 1), ("slash", 1),
     ("rshift", 2.25)],
    [("fn", 1), ("lctrl", 1), ("lopt", 1), ("lcmd", 1.25), ("space", 5),
     ("rcmd", 1.25), ("ropt", 1), ("left", 1), (("up", "down"), 1),
     ("right", 1)],
]
ROW_UNITS = 14.5
NUM_ROWS = len(_ROWS)
KEYBOARD_W = ROW_UNITS * PITCH_X          # ~0.276 m
KEYBOARD_D = NUM_ROWS * PITCH_Y           # ~0.112 m

_MODIFIERS = {"esc", "touchid", "delete", "tab", "caps", "return", "lshift",
              "rshift", "fn", "lctrl", "lopt", "lcmd", "rcmd", "ropt",
              "left", "right", "up", "down"} | {f"f{i}" for i in range(1, 13)}

# character -> (key name, needs shift)
_UNSHIFTED = {
    "`": "grave", "-": "minus", "=": "equal", "[": "lbracket", "]": "rbracket",
    "\\": "backslash", ";": "semicolon", "'": "quote", ",": "comma",
    ".": "period", "/": "slash", " ": "space", "\n": "return", "\t": "tab",
}
_SHIFTED = {
    "~": "grave", "!": "1", "@": "2", "#": "3", "$": "4", "%": "5", "^": "6",
    "&": "7", "*": "8", "(": "9", ")": "0", "_": "minus", "+": "equal",
    "{": "lbracket", "}": "rbracket", "|": "backslash", ":": "semicolon",
    '"': "quote", "<": "comma", ">": "period", "?": "slash",
}
_KEY_TO_CHAR = {v: k for k, v in _UNSHIFTED.items()}
_SHIFT_OF = {v: k for k, v in _SHIFTED.items()}

# printed legend per key: (top line, bottom line); "" = none
_LEGENDS = {
    "esc": ("", "esc"), "touchid": ("", "( )"), "delete": ("", "delete"),
    "tab": ("", "tab"), "caps": ("", "caps lock"), "return": ("", "return"),
    "lshift": ("", "shift"), "rshift": ("", "shift"), "fn": ("", "fn"),
    "lctrl": ("", "control"), "lopt": ("", "option"), "lcmd": ("", "command"),
    "rcmd": ("", "command"), "ropt": ("", "option"), "space": ("", ""),
    "left": ("", "<"), "right": ("", ">"), "up": ("", "^"), "down": ("", "v"),
}
for _i in range(1, 13):
    _LEGENDS[f"f{_i}"] = ("", f"F{_i}")


def legend(name: str) -> tuple[str, str]:
    """(top, bottom) text printed on a keycap: shifted symbol over the base."""
    if name in _LEGENDS:
        return _LEGENDS[name]
    if len(name) == 1 and name.isalpha():
        return ("", name.upper())
    base = _KEY_TO_CHAR.get(name, name)
    return (_SHIFT_OF.get(name, ""), base)

# Standard touch-typing finger assignment: key -> (hand, finger) with
# hand in {"L", "R"} and finger in {"th", "ff", "mf", "rf", "lf"}.
_HOME_COLUMNS = {
    ("L", "lf"): ["grave", "1", "q", "a", "z", "tab", "caps", "lshift", "esc",
                  "fn", "lctrl"],
    ("L", "rf"): ["2", "w", "s", "x", "f1", "f2", "lopt"],
    ("L", "mf"): ["3", "e", "d", "c", "f3", "f4", "lcmd"],
    ("L", "ff"): ["4", "5", "r", "t", "f", "g", "v", "b", "f5", "f6"],
    ("R", "ff"): ["6", "7", "y", "u", "h", "j", "n", "m", "f7", "f8"],
    ("R", "mf"): ["8", "i", "k", "comma", "f9", "rcmd"],
    ("R", "rf"): ["9", "o", "l", "period", "f10", "ropt"],
    ("R", "lf"): ["0", "minus", "equal", "delete", "p", "lbracket", "rbracket",
                  "backslash", "semicolon", "quote", "return", "slash",
                  "rshift", "f11", "f12", "touchid", "left", "right", "up",
                  "down"],
    ("R", "th"): ["space"],
}
TOUCH_TYPING_FINGER: dict[str, tuple[str, str]] = {
    k: hf for hf, keys in _HOME_COLUMNS.items() for k in keys}


@dataclass(frozen=True)
class KeyCap:
    """One keycap in the keyboard-local frame (origin: well centre, floor)."""

    name: str
    row: int            # 0 = function row (back) .. 5 = space row (front)
    x: float            # front-back centre (+X toward the typist)
    y: float            # left-right centre
    width: float        # cap size along Y
    depth: float        # cap size along X
    is_modifier: bool

    @property
    def z_top(self) -> float:
        return CAP_H


def layout() -> list[KeyCap]:
    """All keycaps, back row first, left to right within a row."""
    caps: list[KeyCap] = []
    y0 = -KEYBOARD_W / 2.0
    for r, row in enumerate(_ROWS):
        assert abs(sum(w for _, w in row) - ROW_UNITS) < 1e-9, row
        x = (r - (NUM_ROWS - 1) / 2.0) * PITCH_Y
        cursor = 0.0
        for name, units in row:
            y = y0 + (cursor + units / 2.0) * PITCH_X
            width = CAP_W + (units - 1.0) * PITCH_X
            if isinstance(name, tuple):          # stacked half-height pair
                half = (CAP_D - HALF_KEY_GAP) / 2.0
                up, down = name
                caps.append(KeyCap(up, r, x - (half + HALF_KEY_GAP) / 2.0, y,
                                   width, half, True))
                caps.append(KeyCap(down, r, x + (half + HALF_KEY_GAP) / 2.0, y,
                                   width, half, True))
            else:
                caps.append(KeyCap(name, r, x, y, width, CAP_D,
                                   name in _MODIFIERS))
            cursor += units
    return caps


_LAYOUT = layout()
KEY_NAMES: list[str] = [c.name for c in _LAYOUT]
KEY_INDEX: dict[str, int] = {c.name: i for i, c in enumerate(_LAYOUT)}
NUM_KEYS = len(_LAYOUT)
assert NUM_KEYS == 78, NUM_KEYS      # 14+14+14+13+12+11(incl. up/down)


def key_local_top_positions() -> np.ndarray:
    """(NUM_KEYS, 3) cap-top centres in the keyboard-local frame."""
    return np.array([[c.x, c.y, c.z_top] for c in _LAYOUT], dtype=np.float32)


def text_to_keys(text: str) -> list[tuple[str, bool]]:
    """Map a string to ``[(key name, shift?), ...]`` keystrokes."""
    out = []
    for ch in text:
        if ch.isalpha() and ch.isascii():
            out.append((ch.lower(), ch.isupper()))
        elif ch.isdigit():
            out.append((ch, False))
        elif ch in _UNSHIFTED:
            out.append((_UNSHIFTED[ch], False))
        elif ch in _SHIFTED:
            out.append((_SHIFTED[ch], True))
        else:
            raise ValueError(f"no key for character {ch!r}")
    return out


def key_to_char(name: str, shift: bool = False) -> str:
    """Inverse of :func:`text_to_keys` for printable keys ('' otherwise)."""
    if len(name) == 1 and name.isalpha():
        return name.upper() if shift else name
    if len(name) == 1 and name.isdigit():
        return {v: k for k, v in _SHIFTED.items()}[name] if shift else name
    if name in _KEY_TO_CHAR:
        ch = _KEY_TO_CHAR[name]
        if shift:
            return {v: k for k, v in _SHIFTED.items()}.get(name, ch)
        return ch
    return ""


def _font(size: int):
    from PIL import ImageFont
    for f in ("/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_legends(out_dir: Path = LEGEND_DIR, px_per_m: int = 4000) -> dict[str, Path]:
    """Write one PNG per keycap (cap colour + white legend, Apple style:
    bottom-centred letters, shifted symbol above on the number/punctuation
    keys, small word labels on modifiers). Cached; returns {key: path}."""
    from PIL import Image, ImageDraw
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for c in _LAYOUT:
        path = out_dir / f"{c.name}.png"
        paths[c.name] = path
        if path.exists():
            continue
        # image u-axis == keyboard +Y (left->right), v-axis == keyboard +X
        # (back->front == top->bottom of the cap as the typist sees it)
        w, h = max(8, int(c.width * px_per_m)), max(8, int(c.depth * px_per_m))
        rgba = _CAP_MOD_RGBA if c.is_modifier else _CAP_RGBA
        img = Image.new("RGB", (w, h), tuple(int(255 * v) for v in rgba[:3]))
        draw = ImageDraw.Draw(img)
        top, bottom = legend(c.name)
        small = len(bottom) > 1 and not bottom.startswith("F")
        size = int(h * (0.24 if small else 0.52))
        font = _font(size)
        if bottom:
            bb = draw.textbbox((0, 0), bottom, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            y = h - th - int(0.12 * h) - bb[1] if (top or small) else (h - th) // 2 - bb[1]
            draw.text(((w - tw) // 2 - bb[0], y), bottom, fill=(255, 255, 255), font=font)
        if top:
            f2 = _font(int(h * 0.30))
            bb = draw.textbbox((0, 0), top, font=f2)
            tw = bb[2] - bb[0]
            draw.text(((w - tw) // 2 - bb[0], int(0.10 * h) - bb[1]), top,
                      fill=(255, 255, 255), font=f2)
        # MuJoCo maps a 2D texture's u axis onto the box's local X (front-back)
        # and v onto local Y, so rotate the typist-view image into that frame.
        img.transpose(Image.ROTATE_90).save(path)
    return paths


def add_keyboard(spec: mujoco.MjSpec, parent, pos, quat_wxyz=(1, 0, 0, 0),
                 body_name: str = "keyboard", legends: bool = True) -> None:
    """Add the keyboard well + all spring-loaded keys under ``parent``
    (a body or ``spec.worldbody``) at ``pos``/``quat`` in the parent frame.
    ``legends`` prints the key labels on the caps (PNG textures, generated
    on first use into ``LEGEND_DIR``)."""
    if legends:
        paths = render_legends()
        spec.texturedir = str(LEGEND_DIR)
        for name, path in paths.items():
            tex = spec.add_texture(name=f"tex_key_{name}",
                                   type=mujoco.mjtTexture.mjTEXTURE_2D,
                                   file=path.name)
            mat = spec.add_material(name=f"mat_key_{name}", texrepeat=[1.0, 1.0],
                                    texuniform=False)
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    kb = parent.add_body(name=body_name, pos=list(pos), quat=list(quat_wxyz))
    # well floor plate (cosmetic; the deck slab under it is the collider)
    kb.add_geom(name="kb_well", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[0.0, 0.0, -0.0003],
                size=[KEYBOARD_D / 2 + WELL_MARGIN, KEYBOARD_W / 2 + WELL_MARGIN,
                      0.0005],
                rgba=list(_WELL_RGBA), contype=0, conaffinity=0)
    for c in _LAYOUT:
        body = kb.add_body(name=f"key_{c.name}", pos=[c.x, c.y, 0.0],
                           gravcomp=1.0)
        body.add_joint(
            name=f"joint_{c.name}",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0.0, 0.0, 1.0],
            range=[-KEY_TRAVEL, 0.0],
            stiffness=KEY_SPRING_STIFFNESS,
            damping=KEY_SPRING_DAMPING,
            armature=1e-5,
            limited=True,
        )
        body.add_geom(
            name=f"key_geom_{c.name}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, 0.0, CAP_H / 2.0],
            size=[c.depth / 2.0, c.width / 2.0, CAP_H / 2.0],
            rgba=list(_CAP_MOD_RGBA if c.is_modifier else _CAP_RGBA),
            material=f"mat_key_{c.name}" if legends else "",
            density=1200.0,                      # ABS cap, ~1 g
            friction=[0.8, 0.005, 0.0001],
            # like the piano: keys collide with the hands, never each other
            contype=2,
            conaffinity=1,
        )
        body.add_site(name=f"key_site_{c.name}", pos=[0.0, 0.0, CAP_H],
                      size=[0.002, 0.002, 0.002], group=4)


def add_macbook(spec: mujoco.MjSpec, pos, quat_wxyz=(1, 0, 0, 0),
                body_name: str = "macbook") -> None:
    """Add the 16-inch MacBook Pro body (base, trackpad, open lid) with the
    keyboard at world ``pos``/``quat``. ``pos`` is the centre of the palm-rest
    deck TOP surface; the keyboard well floor sits ``WELL_RECESS`` below it.

    Body-local frame: +X toward the typist, +Y left->right, +Z up. The hinge
    runs along Y at x = -BODY_D/2.
    """
    mb = spec.worldbody.add_body(name=body_name, pos=list(pos),
                                 quat=list(quat_wxyz))
    kb_x = -BODY_D / 2.0 + KEYBOARD_BACK_GAP + KEYBOARD_D / 2.0
    kb_front = kb_x + KEYBOARD_D / 2.0 + WELL_MARGIN
    # base slab: the rear part carries the keyboard well (recessed), the
    # front part is the palm rest / trackpad deck at z = 0.
    rear_d = kb_front + BODY_D / 2.0
    mb.add_geom(name="base_rear", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[-BODY_D / 2.0 + rear_d / 2.0, 0.0,
                     -WELL_RECESS - (BASE_H - WELL_RECESS) / 2.0],
                size=[rear_d / 2.0, BODY_W / 2.0, (BASE_H - WELL_RECESS) / 2.0],
                rgba=list(_ALU_RGBA), **_DECK_COLLISION)
    # thin aluminium strips left/right of the well, flush with the deck
    strip_w = (BODY_W - (KEYBOARD_W + 2 * WELL_MARGIN)) / 2.0
    for sgn, tag in ((-1.0, "l"), (1.0, "r")):
        mb.add_geom(name=f"deck_strip_{tag}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[-BODY_D / 2.0 + rear_d / 2.0,
                         sgn * (BODY_W / 2.0 - strip_w / 2.0), -WELL_RECESS / 2.0],
                    size=[rear_d / 2.0, strip_w / 2.0, WELL_RECESS / 2.0],
                    rgba=list(_ALU_RGBA), **_DECK_COLLISION)
    front_d = BODY_D - rear_d
    mb.add_geom(name="base_front", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[BODY_D / 2.0 - front_d / 2.0, 0.0, -BASE_H / 2.0],
                size=[front_d / 2.0, BODY_W / 2.0, BASE_H / 2.0],
                rgba=list(_ALU_RGBA), **_DECK_COLLISION)
    # trackpad: glass inset, flush (visual)
    mb.add_geom(name="trackpad", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[BODY_D / 2.0 - 0.012 - TRACKPAD_D / 2.0, 0.0, 0.0001],
                size=[TRACKPAD_D / 2.0, TRACKPAD_W / 2.0, 0.0001],
                rgba=list(_TRACKPAD_RGBA), contype=0, conaffinity=0)
    # lid: hinged at the back edge, opened LID_ANGLE_DEG (visual only)
    tilt = np.deg2rad(LID_ANGLE_DEG - 90.0)          # lean-back from vertical
    lid = mb.add_body(name="lid", pos=[-BODY_D / 2.0, 0.0, 0.0],
                      quat=[np.cos(-tilt / 2), 0.0, np.sin(-tilt / 2), 0.0])
    lid_len = BODY_D
    lid.add_geom(name="lid_shell", type=mujoco.mjtGeom.mjGEOM_BOX,
                 pos=[-LID_T / 2.0, 0.0, lid_len / 2.0],
                 size=[LID_T / 2.0, BODY_W / 2.0, lid_len / 2.0],
                 rgba=list(_ALU_RGBA), contype=0, conaffinity=0)
    lid.add_geom(name="screen", type=mujoco.mjtGeom.mjGEOM_BOX,
                 pos=[0.0003, 0.0, lid_len / 2.0 + 0.004],
                 size=[0.0003, BODY_W / 2.0 - 0.006, lid_len / 2.0 - 0.012],
                 rgba=list(_SCREEN_RGBA), contype=0, conaffinity=0)
    add_keyboard(spec, mb, pos=[kb_x, 0.0, -WELL_RECESS])
