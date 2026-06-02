"""Config for the bimanual piano task: two UR10e+Shadow arms over an 88-key
piano, learning to play a MIDI song.

Embodiment: two combined UR10e+Shadow articulations (~30 DoF each = 60 action
DoF). The piano is a separate 88-key articulation; its keys are passive springs
the fingers press. The MIDI song (data/midi/<song>.mid) defines, per control
step, which keys should sound -- that's the goal the policy is rewarded on.

Follows Isaac Lab's DirectRLEnv convention: articulation cfgs are fields here and
are instantiated in ``PianoEnv._setup_scene``.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from dexsim.assets import UR10E_SHADOW_CFG, PIANO_CFG
from dexsim import DATA_DIR

# control: 20 Hz policy (decimation 6 @ 120 Hz sim) -> matches MIDI control_dt 0.05
SIM_DT = 1.0 / 120.0
DECIMATION = 6
CONTROL_DT = SIM_DT * DECIMATION  # 0.05 s -> 20 Hz

GOAL_LOOKAHEAD = 10               # steps of upcoming notes the policy sees (~0.5s)
NUM_KEYS = 88
PER_ARM_DOF = 30                  # UR10e(6) + Shadow(24)
NUM_FINGERS = 10                  # 5 per hand


@configclass
class PianoEnvCfg(DirectRLEnvCfg):
    # --- spaces ---
    decimation = DECIMATION
    episode_length_s = 30.0
    action_space = 2 * PER_ARM_DOF                       # 60 (residual on the IK ref)

    # observation is assembled in PianoEnv._get_observations; size is computed in
    # __post_init__ from the feature flags below so the two never drift.
    observation_space = 0
    state_space = 0

    # --- PianoMime-style observation features (all default ON) ---
    obs_fingertip_pos: bool = True    # 10x3 fingertip world pos (rel. to piano)
    obs_finger_targets: bool = True   # 10x3 reference fingertip targets (the
    #                                   "where fingers should go" conditioning)
    obs_goal_sdf: bool = True         # 88 analytic SDF of the current goal

    sim: SimulationCfg = SimulationCfg(dt=SIM_DT, render_interval=DECIMATION)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=4.0, replicate_physics=True
    )

    # --- articulations (instantiated in _setup_scene) ---
    left_robot_cfg: ArticulationCfg = UR10E_SHADOW_CFG.replace(
        prim_path="/World/envs/env_.*/LeftRobot"
    )
    right_robot_cfg: ArticulationCfg = UR10E_SHADOW_CFG.replace(
        prim_path="/World/envs/env_.*/RightRobot"
    )
    piano_cfg: ArticulationCfg = PIANO_CFG.replace(
        prim_path="/World/envs/env_.*/Piano"
    )

    # --- task / placement ---
    midi_path: str = str(DATA_DIR / "midi" / "song.mid")  # the user's song
    control_dt: float = CONTROL_DT
    goal_lookahead: int = GOAL_LOOKAHEAD

    # --- residual RL over an IK reference (PianoMime) ---
    # If reference_path exists, the action is a residual on the precomputed IK
    # reference joint trajectory (action_scale * a + q_ref). If it's missing, the
    # env degrades gracefully to a residual on the static ready pose. Build one
    # with scripts/build_reference.py. None -> derive from the MIDI stem.
    reference_path: str | None = None
    use_reference: bool = True

    # IK params (used by build_reference.py and any online IK)
    ik_damping: float = 0.05
    ik_max_step: float = 0.05

    # piano world pose (shift so the 1.22 m keyboard centers on Y=0, table height)
    piano_pos = (0.569, -0.606, 0.746)   # fit so LEFT-hand fingers rest on keys 19/20/26
    # robot base poses: elevated on pedestals just behind the keyboard so the
    # arms drape DOWN onto the keys. With the base at key height the wrist can't
    # get above the keys and the fingers end up below the keyboard; raising the
    # bases lets the hand come down onto the keys (fingers point -Z).
    left_base_pos = (-0.05, -0.30, 1.05)
    right_base_pos = (-0.05, 0.30, 1.05)

    # action scaling: targets = default + scale * action (action in [-1,1])
    action_scale: float = 0.5
    # for left-hand-only easy songs: freeze the right arm so it can't mash idle
    # high keys (those were the false presses tanking precision).
    mute_right_hand: bool = True

    # reward weights (PianoMime/RoboPianist composite)
    key_press_weight: float = 1.0
    # 0.5 gave the best F1 (0.29, recall 54%); 1.5 over-suppressed pressing
    # (recall 8%). Keep 0.5 -- the crash that stopped it is now fixed (NaN guard).
    false_press_weight: float = 0.5
    energy_weight: float = 0.0005
    fingering_weight: float = 1.0     # finger->target-key shaping (CRITICAL term)
    onset_weight: float = 0.5         # crisp attack on note onsets

    # Piano "ready" pose: from the raised (z=1.05) bases each hand drapes DOWN so
    # the fingertips rest ~2 cm above the white keys (key top z=0.722), centered
    # over that hand's half of the keyboard. Solved numerically by
    # scripts/tune_arm_pose.py against the current bases -- fingertip mean lands
    # within ~3-4 cm of (x=0.35, y=+/-0.30, z=0.74), i.e. ON the keys. wrist_2/3
    # are held so the palm stays vertical. Per-side (the two arms differ slightly).
    left_ready_pose = {
        "shoulder_pan_joint": -0.425,
        "shoulder_lift_joint": -0.62,
        "elbow_joint": 1.60,
        "wrist_1_joint": -1.50,
        "wrist_2_joint": -1.57,
        "wrist_3_joint": 0.0,
        "robot0_.*": 0.0,
    }
    # sweep result: fingertips land z=0.731 (9mm above keys, over y~0.66) -> a
    # ~1cm finger curl presses. Used for the easy-song demo (right hand only).
    right_ready_pose = {
        "shoulder_pan_joint": 0.5,
        "shoulder_lift_joint": -0.5,
        "elbow_joint": 1.2,
        "wrist_1_joint": -1.2,
        "wrist_2_joint": -1.57,
        "wrist_3_joint": 0.0,
        "robot0_.*": 0.0,
    }

    def __post_init__(self):
        # --- compute observation size from the feature flags (single source) ---
        obs = (
            2 * PER_ARM_DOF * 2                     # both arms pos+vel (120)
            + NUM_KEYS                              # current key angles (88)
            + self.goal_lookahead * NUM_KEYS        # upcoming note goals
        )
        if self.obs_fingertip_pos:
            obs += NUM_FINGERS * 3                  # 30
        if self.obs_finger_targets:
            obs += NUM_FINGERS * 3                  # 30
        if self.obs_goal_sdf:
            obs += NUM_KEYS                         # 88
        self.observation_space = obs

        # bake world poses into each articulation's initial state (per-env-origin
        # relative). The two robots are separate copies, so this is safe.
        self.left_robot_cfg.init_state.pos = self.left_base_pos
        self.right_robot_cfg.init_state.pos = self.right_base_pos
        self.piano_cfg.init_state.pos = self.piano_pos
        # piano-ready default arm pose (fingertips resting on the keys); the two
        # arms differ slightly so each has its own tuned pose.
        self.left_robot_cfg.init_state.joint_pos = dict(self.left_ready_pose)
        self.right_robot_cfg.init_state.joint_pos = dict(self.right_ready_pose)
