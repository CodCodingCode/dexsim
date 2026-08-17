"""Procedural 88-key spring-loaded piano (MuJoCo port).

Same instrument as the Isaac articulation in ``dexsim/assets/piano.py`` /
``scripts/build/build_piano_usd.py``, generated straight from the shared
``dexsim.piano.geometry`` layout so every consumer (fingering planner, reward,
env, and now MuJoCo) agrees on where each key is.

Physics matches the Isaac tuning (see dexsim/assets/piano.py for the full
history of why these numbers):

  * keys are PASSIVE hinges -- never actuated, held up by a joint spring
    (stiffness 3, damping 4) with the rest angle at 0;
  * pressed = NEGATIVE angle; the hinge stops at ``-KEY_MAX_TRAVEL_ANGLE``
    (atan(0.01/0.15) ~ RoboPianist's white-key travel);
  * a key *sounds* below ``KEY_SOUND_ANGLE`` -- the env adds the velocity
    gate on top (a statically-resting hand rings nothing);
  * ``gravcomp=1`` on every key body == Isaac's ``disable_gravity=True``
    (real keys are balanced; without this the key mass sags past the sound
    angle at rest and everything reads as a false press).

Naming: body ``key_{i}``, joint ``joint_{i}``, press-point site ``key_site_{i}``
(the site sits at the key-top press target, so the env reads measured world
positions exactly like Isaac's ``body_pos_w`` + half-height did).
"""

from __future__ import annotations

import mujoco

from dexsim.piano import geometry as geom
from dexsim.piano.midi import NUM_KEYS

# --- mirror of dexsim/assets/piano.py (that module imports Isaac; keep the
# --- MuJoCo stack import-clean). Values must stay in sync.
KEY_MAX_TRAVEL_ANGLE = 0.0666        # rad; hinge stops here (physical max)
KEY_SOUND_ANGLE = -0.012             # ~18% travel -> sensitive (good recall)
KEY_SPRING_STIFFNESS = 3.0           # gentle: pressable by the weak fingers
KEY_SPRING_DAMPING = 0.1             # NOT the Isaac 4.0! That value was a
#   PhysX-specific slam absorber ("kills the low-hover contact explosion"), a
#   failure mode MuJoCo doesn't have. On a passive MuJoCo hinge, damping 4
#   over spring 3 is a tau=1.3s sponge: a finger STRIKE can only depress the
#   key ~15% before the contact ends (measured stall at -0.006 rad vs the
#   -0.012 sound angle) and the velocity gate alone would need >1.4 Nm. 0.1
#   matches the RoboPianist piano (their damping 0.05, the value the Isaac
#   file itself cites as the reference) with a little extra return damping.

# hinge line: the back edge shared by white and black keys (piano-local X)
_HINGE_X = -geom.WHITE_L / 2.0

_WHITE_RGBA = (0.96, 0.96, 0.94, 1.0)
_BLACK_RGBA = (0.08, 0.08, 0.09, 1.0)
_CASE_RGBA = (0.14, 0.10, 0.09, 1.0)


def add_piano(spec: mujoco.MjSpec, pos, quat_wxyz,
              key_damping: float | None = None) -> None:
    """Add the piano articulation to ``spec`` at world ``pos``/``quat_wxyz``.

    Frame convention matches the USD build script: piano-local +Y runs along
    the keyboard low->high, +X points toward the player, +Z is up; key 0 (A0)
    at local y=0.
    """
    damping = KEY_SPRING_DAMPING if not key_damping else float(key_damping)
    piano = spec.worldbody.add_body(name="piano", pos=list(pos),
                                    quat=list(quat_wxyz))

    span = geom.KEYBOARD_SPAN_Y
    mid_y = 0.5 * (span - geom.WHITE_PITCH)      # lateral center of the keys
    # keybed slab: below the keys, deep enough that a fully-pressed key front
    # (dips to z ~ -0.010) never touches it.
    piano.add_geom(name="keybed", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[0.0, mid_y, -0.022],
                   size=[geom.WHITE_L / 2 + 0.02, span / 2 + 0.03, 0.010],
                   rgba=list(_CASE_RGBA))
    # cosmetic case: back board behind the hinge + two side cheeks
    piano.add_geom(name="backboard", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[_HINGE_X - 0.025, mid_y, 0.02],
                   size=[0.025, span / 2 + 0.03, 0.052],
                   rgba=list(_CASE_RGBA), contype=0, conaffinity=0)
    for sgn, tag in ((-1.0, "lo"), (1.0, "hi")):
        piano.add_geom(name=f"cheek_{tag}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=[0.0, mid_y + sgn * (span / 2 + 0.0175), 0.01],
                       size=[geom.WHITE_L / 2 + 0.02, 0.0125, 0.042],
                       rgba=list(_CASE_RGBA), contype=0, conaffinity=0)

    for k in geom.layout():
        length = geom.BLACK_L if k.is_black else geom.WHITE_L
        width = geom.BLACK_W if k.is_black else geom.WHITE_W
        height = 2.0 * k.half_height
        z_center = k.z_top - k.half_height
        body = piano.add_body(
            name=f"key_{k.index}",
            pos=[_HINGE_X, k.y, z_center],
            gravcomp=1.0,
        )
        body.add_joint(
            name=f"joint_{k.index}",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            # axis -Y so pressing the front DOWN gives a NEGATIVE angle
            # (matching the Isaac/RoboPianist sign convention).
            axis=[0.0, -1.0, 0.0],
            range=[-KEY_MAX_TRAVEL_ANGLE, 0.0],
            stiffness=KEY_SPRING_STIFFNESS,
            damping=damping,
            armature=0.001,
            limited=True,
        )
        # key box extends forward (+X) from the hinge at the body origin
        body.add_geom(
            name=f"key_geom_{k.index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[k.x_center - _HINGE_X, 0.0, 0.0],
            size=[length / 2.0, width / 2.0, height / 2.0],
            rgba=list(_BLACK_RGBA if k.is_black else _WHITE_RGBA),
            density=500.0,                      # light wooden key ~20 g
            friction=[1.0, 0.005, 0.0001],
            # contype 2 / conaffinity 1: keys collide with the hands (default
            # 1/1 geoms) but never with EACH OTHER -- the black-key boxes
            # overlap their white neighbours by design (real keys interlock),
            # and without this mask the blacks rest on the whites with ~20 N,
            # pre-pressing every white key at rest. == Isaac's
            # enabled_self_collisions=False on the piano articulation.
            contype=2,
            conaffinity=1,
        )
        # measured press point: center of the key's playable top surface
        body.add_site(
            name=f"key_site_{k.index}",
            pos=[k.x_center - _HINGE_X, 0.0, k.half_height],
            size=[0.003, 0.003, 0.003],
            group=4,
        )
