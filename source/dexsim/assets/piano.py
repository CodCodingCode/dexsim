"""ArticulationCfg for the generated 88-key piano.

The piano is ONE articulation: a kinematic base with 88 keys, each hinged by a
revolute joint (``joint_0``..``joint_87``, index = MIDI-21). The keys are
*passive* and spring-loaded -- the policy never commands them. We register an
actuator group over all key joints with the return-spring stiffness/damping and
leave the target at 0, so PhysX holds each key up until a finger presses it down.

A key "sounds" when its joint angle drops below ``KEY_SOUND_ANGLE`` (radians,
negative = pressed down).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from dexsim import ASSETS_DIR

PIANO_USD_PATH = str(ASSETS_DIR / "piano88.usd")

# Matched to RoboPianist's piano physics (their setup actually registers presses):
#   white key max travel = atan(0.01/0.15) ~= 0.0666 rad; a key "sounds" at ~50%
#   of that travel; spring stiffness 2, damping 0.05 (ours was 8 / 0.5 -> 4x too
#   stiff, and the old -0.10 threshold was BEYOND the physical max travel, so a
#   key could literally never sound). This was the root cause of F1=0.
KEY_MAX_TRAVEL_ANGLE = 0.0666           # rad; key hinge stops here (physical max)
KEY_SOUND_ANGLE = -0.012                # ~18% travel -> sensitive (good recall).
#   Precision is handled by VELOCITY-gated sounding (see PianoEnv._key_pressed_
#   fraction + key_strike_vel): a resting hand depresses keys statically (~0 vel)
#   so they don't ring, even at this light depth. -0.033 was too stiff (9% sound);
#   a light threshold + the velocity gate gives both recall AND precision.
KEY_SPRING_STIFFNESS = 20.0   # 2.0 was TOO WEAK: keys SAGGED under their own gravity to
#   -0.0137 rad (frac 1.14) at REST -- already PAST KEY_SOUND_ANGLE, so every key counted
#   as permanently "sounding" with nothing on it (~4 baked-in false presses, precision
#   capped ~0.05 regardless of policy). At 20 the keys rest at ~frac 0 (verified via
#   scripts/prep/diag_contact.py), so keys_sounding reflects only real presses. The diag
#   sweep showed 15 already gives 0 false at rest; 20 leaves margin while staying pressable.
KEY_SPRING_DAMPING = 0.5      # raised with the stiffness to keep the key critically damped

PIANO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=PIANO_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=2.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,   # keys never touch each other
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            # base is already kinematic in the USD, so no root-fixing joint
            # needed (fix_root_link would require RigidBodyAPI on the root prim).
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={"joint_.*": 0.0},
    ),
    actuators={
        # passive return springs: never commanded, just hold keys up.
        "keys": ImplicitActuatorCfg(
            joint_names_expr=["joint_.*"],
            effort_limit=50.0,
            stiffness=KEY_SPRING_STIFFNESS,
            damping=KEY_SPRING_DAMPING,
        ),
    },
)
"""88-key spring-loaded piano. Read key angles from ``data.joint_pos``;
angle < KEY_SOUND_ANGLE => that key is sounding."""
