"""Menagerie Shadow Hand -> dexsim ``robot0_*`` naming (MuJoCo port).

The MuJoCo Menagerie ships tuned right/left Shadow Hand E3M5 models with the
*real* Shadow joint names (``rh_FFJ4`` = knuckle abduction, ``rh_FFJ1`` =
distal). The rest of this repo -- the fingering planner, the 🔒 locked ready
pose, CLAUDE.md -- uses the Isaac/OpenAI 0-based convention
(``robot0_FFJ3`` = abduction, ``robot0_FFJ0`` = distal). This module loads the
Menagerie XML, renames every joint/body/actuator/tendon so both stacks speak
the same names, and returns it as an ``MjSpec``:

    real Shadow            ->  dexsim (Isaac/OpenAI 0-based)
    rh_WRJ2 (deviation)    ->  robot0_WRJ1
    rh_WRJ1 (flexion)      ->  robot0_WRJ0
    rh_FFJ4..FFJ1          ->  robot0_FFJ3..FFJ0   (same MF/RF)
    rh_LFJ5..LFJ1          ->  robot0_LFJ4..LFJ0
    rh_THJ5..THJ1          ->  robot0_THJ4..THJ0
    bodies rh_palm etc.    ->  robot0_palm etc.

Ranges line up exactly (e.g. Menagerie ``rh_WRJ1`` range [-0.698, 0.489] ==
the locked pose's ``robot0_WRJ0`` range [-0.70, 0.49]), which is how the
mapping was verified. The rename is a plain token substitution on the XML
text: every source name starts with ``rh_``/``lh_`` and every target with
``robot0_``, so no replacement can corrupt another and all cross-references
(tendon->joint, actuator->tendon) stay consistent.

The four ``*J0`` distal pairs are tendon-coupled and driven by one position
actuator each (``robot0_A_FFJ0`` over tendon ``robot0_T_FFJ0``), matching the
Isaac setup where the J0 joints "are driven through the J1 tendon" -- so each
hand has 24 joints and 20 actuators, and the true LEFT hand (which Isaac never
had natively) comes straight from ``left_hand.xml``.

A fingertip site (``robot0_fftip`` etc.) is added at the tip of each distal
body; the env uses these for the fingering reward (the distal *body* origin
sits at the DIP joint, ~2.5 cm short of where the finger actually presses).
"""

from __future__ import annotations

from pathlib import Path

import mujoco

# repo root: source/dexsim/mjcf/shadow_hand.py -> three parents up
_ROOT = Path(__file__).resolve().parents[3]
MENAGERIE_DIR = _ROOT / "assets" / "mujoco_menagerie" / "shadow_hand"
MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"

# per-hand fingertip sites, in the FINGERTIP_BODIES order [th, ff, mf, rf, lf]
FINGERTIP_SITES = [
    "robot0_thtip", "robot0_fftip", "robot0_mftip",
    "robot0_rftip", "robot0_lftip",
]
# distal-body -> (site name, tip offset along the distal +Z axis). ~24 mm for
# fingers / ~27 mm for the thumb (E3M5 distal segment lengths).
_TIP_SITES = {
    "robot0_thdistal": ("robot0_thtip", 0.0265),
    "robot0_ffdistal": ("robot0_fftip", 0.024),
    "robot0_mfdistal": ("robot0_mftip", 0.024),
    "robot0_rfdistal": ("robot0_rftip", 0.024),
    "robot0_lfdistal": ("robot0_lftip", 0.024),
}


def _rename_map(p: str) -> dict[str, str]:
    """All ``p``-prefixed names in the Menagerie XML -> dexsim names."""
    m = {
        # wrist joints + their actuators
        f"{p}WRJ2": "robot0_WRJ1", f"{p}WRJ1": "robot0_WRJ0",
        f"{p}A_WRJ2": "robot0_A_WRJ1", f"{p}A_WRJ1": "robot0_A_WRJ0",
        # non-joint names (bodies/sites that would otherwise hit no rule get
        # the generic prefix swap in rename_xml below)
    }
    # finger joints: real J(n) -> isaac J(n-1)
    for f in ("FF", "MF", "RF"):
        for n in range(1, 5):
            m[f"{p}{f}J{n}"] = f"robot0_{f}J{n - 1}"
        m[f"{p}A_{f}J4"] = f"robot0_A_{f}J3"
        m[f"{p}A_{f}J3"] = f"robot0_A_{f}J2"
        m[f"{p}A_{f}J0"] = f"robot0_A_{f}J0"     # tendon actuator keeps J0
        m[f"{p}{f}J0"] = f"robot0_T_{f}J0"       # the coupling tendon itself
    for n in range(1, 6):
        m[f"{p}LFJ{n}"] = f"robot0_LFJ{n - 1}"
        m[f"{p}THJ{n}"] = f"robot0_THJ{n - 1}"
        m[f"{p}A_THJ{n}"] = f"robot0_A_THJ{n - 1}"
    m[f"{p}A_LFJ5"] = "robot0_A_LFJ4"
    m[f"{p}A_LFJ4"] = "robot0_A_LFJ3"
    m[f"{p}A_LFJ3"] = "robot0_A_LFJ2"
    m[f"{p}A_LFJ0"] = "robot0_A_LFJ0"
    m[f"{p}LFJ0"] = "robot0_T_LFJ0"
    return m


def rename_xml(xml: str, side: str) -> str:
    """Apply the naming convention to the raw Menagerie XML text."""
    p = "rh_" if side == "right" else "lh_"
    # longest names first so e.g. rh_A_FFJ0 is never half-eaten by rh_FFJ0…
    for old, new in sorted(_rename_map(p).items(), key=lambda kv: -len(kv[0])):
        xml = xml.replace(old, new)
    # generic fallback for everything else (bodies, mesh-free sites, classes)
    return xml.replace(p, "robot0_")


def ensure_menagerie() -> Path:
    """Return the Menagerie shadow_hand dir, sparse-cloning it if missing."""
    if (MENAGERIE_DIR / "right_hand.xml").exists():
        return MENAGERIE_DIR
    import subprocess

    dest = MENAGERIE_DIR.parent
    print(f"[mjcf] vendoring MuJoCo Menagerie shadow_hand -> {dest}")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
         MENAGERIE_URL, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set",
                    "shadow_hand"], check=True)
    return MENAGERIE_DIR


def load_hand_spec(side: str) -> mujoco.MjSpec:
    """Load the Menagerie hand for ``side`` ("left"/"right"), renamed to the
    dexsim ``robot0_*`` convention, with fingertip sites added. Returns an
    un-compiled ``MjSpec`` ready to be attached into a scene (the attach step
    adds the per-hand ``L_``/``R_`` prefix so both hands can coexist)."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    src = ensure_menagerie() / f"{side}_hand.xml"
    xml = rename_xml(src.read_text(), side)
    spec = mujoco.MjSpec.from_string(xml)
    spec.meshdir = str(MENAGERIE_DIR / "assets")

    # fingertip sites at the pressing point of each distal segment
    for b in spec.bodies:
        if b.name in _TIP_SITES:
            name, z = _TIP_SITES[b.name]
            b.add_site(name=name, pos=[0.0, 0.0, z], size=[0.004, 0.004, 0.004],
                       group=4)
    return spec
