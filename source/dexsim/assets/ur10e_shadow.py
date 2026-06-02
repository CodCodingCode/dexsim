"""Articulation configs for the UR10e arm and the Shadow Hand, plus the
combined UR10e+Shadow embodiment that matches the BODex-Tabletop dataset.

Everything here is *native USD* from the Isaac Sim asset library -- no URDF
conversion. The two sub-assets are:

  * UR10e          {ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10e/ur10e.usd
  * Shadow Hand    {ISAAC_NUCLEUS_DIR}/Robots/ShadowHand/shadow_hand_instanceable.usd

The combined articulation (arm flange -> hand base, joined by a fixed joint
with a single articulation root) is produced by
``scripts/build_combined_usd.py`` and written to ``assets/ur10e_shadow.usd``.
``UR10E_SHADOW_CFG`` points at that file.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from dexsim import ASSETS_DIR

# Where build_combined_usd.py writes the composed arm+hand articulation.
COMBINED_USD_PATH = str(ASSETS_DIR / "ur10e_shadow.usd")

# ---------------------------------------------------------------------------
# Joint name groups (regex). Keep these in one place; tasks reference them.
# ---------------------------------------------------------------------------
UR10E_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Shadow Hand: 24 joints, 20 actuated (the *J0 coupled distal joints are driven
# through the *J1 tendon). The Isaac instanceable hand prefixes every joint with
# "robot0_". 5 wrist/finger groups: WR, FF, MF, RF, LF, TH.
SHADOW_HAND_WRIST_JOINTS = ["robot0_WRJ1", "robot0_WRJ0"]
SHADOW_HAND_FINGER_JOINTS = [
    "robot0_(FF|MF|RF|LF)J(3|2|1)",
    "robot0_(LF|TH)J4",
    "robot0_THJ(3|2|1|0)",
    "robot0_(FF|MF|RF|LF)J0",
]

# ---------------------------------------------------------------------------
# Stand-alone arm
# ---------------------------------------------------------------------------
UR10E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10e/ur10e.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,  # arm tucked, flange pointing down
            "elbow_joint": 1.712,
            "wrist_1_joint": -1.571,
            "wrist_2_joint": -1.571,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=UR10E_ARM_JOINTS,
            velocity_limit=120.0,
            effort_limit=330.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)
"""UR10e arm only (6-DOF). Native USD from the asset library."""


# ---------------------------------------------------------------------------
# Stand-alone Shadow Hand (instanceable -> clones cheaply across 1000s of envs)
# ---------------------------------------------------------------------------
SHADOW_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/ShadowHand/shadow_hand_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
    ),
    actuators={
        # one group over ALL hand joints (regex ".*" -> every DOF configured,
        # avoids the "not all actuators configured" warning). Scalar gains keep
        # coverage exact regardless of the hand's joint naming.
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=0.9,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Shadow Hand only (24-DOF, gravity-disabled) -- the canonical in-hand
reorientation embodiment. Values mirror Isaac Lab's built-in SHADOW_HAND_CFG."""


# ---------------------------------------------------------------------------
# Combined UR10e + Shadow Hand  (== BODex-Tabletop embodiment)
# ---------------------------------------------------------------------------
UR10E_SHADOW_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=COMBINED_USD_PATH,  # produced by scripts/build_combined_usd.py
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # OFF: the hand is bonded at the flange and its forearm overlaps the
            # arm's wrist link; with self-collisions on, the overlap generates
            # huge phantom contact forces that blow the arm apart (joints -> tens
            # of rad). The hand's own internal self-collisions also aren't needed.
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            # arm: reach out over the table, flange pointing down
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.2,
            "elbow_joint": 1.4,
            "wrist_1_joint": -1.77,
            "wrist_2_joint": -1.57,
            "wrist_3_joint": 0.0,
            # hand: relaxed open
            "robot0_.*": 0.0,
        },
    ),
    actuators={
        # stiff/high-effort so the arm HOLDS the hand at key height against
        # gravity (stiffness 800/effort 330 sagged it ~29cm below the keys). The
        # arm just holds a pose while the fingers play, so high gains are fine.
        "arm": ImplicitActuatorCfg(
            joint_names_expr=UR10E_ARM_JOINTS,
            velocity_limit=120.0,
            effort_limit=2000.0,
            stiffness=6000.0,
            damping=300.0,
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=["robot0_.*"],  # all 24 hand joints, no arm joints
            effort_limit=2.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""UR10e arm with a Shadow Hand mounted at the tool flange -- the exact
embodiment of the BODex-Tabletop trajectories. Requires the composed USD;
run ``python scripts/build_combined_usd.py`` once to generate it."""


# ---------------------------------------------------------------------------
# Combined UR10e + LEFT Shadow Hand  (mirrored hand for the left arm)
# ---------------------------------------------------------------------------
# Isaac ships only a right Shadow hand, so the left hand is the MuJoCo
# `left_shadow_hand` imported to USD and renamed robot0_* (drop-in body/joint
# names). Built by ``scripts/build_left_arm.py`` -> assets/ur10e_shadow_left.usd.
COMBINED_LEFT_USD_PATH = str(ASSETS_DIR / "ur10e_shadow_left.usd")

UR10E_SHADOW_LEFT_CFG = UR10E_SHADOW_CFG.replace(
    spawn=UR10E_SHADOW_CFG.spawn.replace(usd_path=COMBINED_LEFT_USD_PATH),
)
"""UR10e with a LEFT Shadow hand. Same robot0_* convention as the right combined
arm, so it drops into the env wherever UR10E_SHADOW_CFG is used. NOTE: the
hand->flange mount orientation still needs tuning (--mount-rpy in
build_left_arm.py) for full stability, since the MuJoCo hand's forearm frame
differs from the Isaac right hand's."""
