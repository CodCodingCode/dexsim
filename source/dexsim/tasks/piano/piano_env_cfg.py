"""Config for the bimanual piano task: two UR10e+Shadow arms over an 88-key piano.

Two combined UR10e+Shadow articulations (~30 DoF each = 60 action DoF) over a
separate 88-key piano articulation whose keys are passive springs. A MIDI song
defines, per control step, which keys should sound -- the goal the policy is
rewarded on. Articulation cfgs are fields here and instantiated in
``PianoEnv._setup_scene`` (Isaac Lab DirectRLEnv convention).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass

from dexsim.assets import UR10E_SHADOW_CFG, UR10E_SHADOW_LEFT_CFG, PIANO_CFG
from dexsim import DATA_DIR

# 20 Hz policy (decimation 6 @ 120 Hz sim) -> matches MIDI control_dt 0.05
SIM_DT = 1.0 / 120.0
DECIMATION = 6
CONTROL_DT = SIM_DT * DECIMATION

GOAL_LOOKAHEAD = 10               # steps of upcoming notes the policy sees (~0.5s)
NUM_KEYS = 88
PER_ARM_DOF = 30                  # UR10e(6) + Shadow(24)
NUM_FINGERS = 10                  # 5 per hand


@configclass
class PianoEnvCfg(DirectRLEnvCfg):
    # --- spaces ---
    decimation = DECIMATION
    episode_length_s = 30.0
    action_space = 2 * PER_ARM_DOF                       # 60 (residual on the ready pose)
    observation_space = 0                                # computed in __post_init__
    state_space = 0

    # --- observation features (assembled in PianoEnv._get_observations) ---
    obs_fingertip_pos: bool = True    # 10x3 fingertip world pos (rel. to piano)
    obs_finger_targets: bool = True   # 10x3 reference fingertip targets
    obs_goal_sdf: bool = True         # 88 analytic SDF of the current goal

    # PhysX GPU buffers bumped: defaults overflow with many finger/key contacts
    # across thousands of envs ("Patch buffer overflow").
    sim: SimulationCfg = SimulationCfg(
        dt=SIM_DT, render_interval=DECIMATION,
        physx=PhysxCfg(
            gpu_max_rigid_patch_count=2 ** 20,
            gpu_max_rigid_contact_count=2 ** 23,
            gpu_found_lost_pairs_capacity=2 ** 22,
            gpu_found_lost_aggregate_pairs_capacity=2 ** 23,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=4.0, replicate_physics=True
    )

    # --- articulations (instantiated in _setup_scene) ---
    # left arm = left-hand combined asset, right arm = right-hand asset.
    left_robot_cfg: ArticulationCfg = UR10E_SHADOW_LEFT_CFG.replace(
        prim_path="/World/envs/env_.*/LeftRobot"
    )
    right_robot_cfg: ArticulationCfg = UR10E_SHADOW_CFG.replace(
        prim_path="/World/envs/env_.*/RightRobot"
    )
    piano_cfg: ArticulationCfg = PIANO_CFG.replace(
        prim_path="/World/envs/env_.*/Piano"
    )

    # --- task / placement ---
    midi_path: str = str(DATA_DIR / "midi" / "song.mid")
    control_dt: float = CONTROL_DT
    goal_lookahead: int = GOAL_LOOKAHEAD

    # --- multi-song training ---
    # When set, load N songs' goals from this .npz bundle (goals (N,Tmax,88),
    # lens (N,), names (N,)) and train one policy across all of them (each env gets
    # a song, round-robin). None => single song from midi_path.
    songs_npz: str | None = None
    max_songs: int = 0            # 0 = all songs in the bundle; else cap to first N
    song_offset: int = 0          # skip the first N (held-out eval: train [0:K], test [K:])

    # --- fold a wide song into each hand's reachable key window ---
    # Fixed hands reach only a narrow band each; octave-fold every note into the
    # nearest hand window so the song is physically playable. Windows are inclusive.
    fold_to_reach: bool = True
    left_key_window: tuple[int, int] = (19, 26)
    right_key_window: tuple[int, int] = (63, 70)

    # --- layout: robots in front of the piano, facing it ---
    # Piano keys' player side faces +X; the robots sit at large +X and are rotated
    # 180deg about Z to reach -X back onto the keys.
    piano_pos = (0.61, 0.598, 0.746)     # keyboard centered at (0.60, 0, 0.756)
    piano_rot = (0.0, 0.0, 0.0, 1.0)     # 180deg about Z
    left_base_pos = (1.65, -0.30, 0.85)  # left hand over the low keys (-Y)
    right_base_pos = (1.65, 0.30, 0.85)  # right hand over the high keys (+Y)
    arm_base_rot = (0.0, 0.0, 0.0, 1.0)  # wxyz: 180deg about Z so the arm faces the piano

    # --- action scaling: target = default + scale * action (action in [-1,1]) ---
    # Per-joint (see PianoEnv.joint_scale): the stiff arm joints (stiffness 6000)
    # blow up under a large residual; the weak hand joints (stiffness 3) need a
    # generous range to travel between keys. So scale the arm gently, the hand more.
    arm_action_scale: float = 0.15    # 0.15 is the stable max (0.30 crashed physics)
    hand_action_scale: float = 0.35   # travels between keys without mashing
    action_scale: float = 0.15        # legacy global scale (unused; back-compat)

    # --- arm mode ---
    # FIXED HANDS: arms hold a constant pose; only the fingers train (RoboPianist-style).
    freeze_arms: bool = True
    # mute the right hand (hold its fingers at the ready pose) for left-hand-only
    # songs so it can't mash idle keys. MUST be False for two-handed songs.
    mute_right_hand: bool = True

    # ARM-IK-FOLLOW: WristPoseIK servos the 12 arm DoF to the per-hand note centroid
    # each step; the policy action is masked to the 48 finger DoF. Set freeze_arms=False.
    arm_ik_follow: bool = False
    # PLANAR-IK: weight z+orientation heavily so the arm slides flat at constant
    # height instead of tilting/sagging onto the keys.
    planar_ik: bool = False
    planar_weight: float = 25.0
    planar_iters: int = 6
    planar_pin_x: bool = False        # also hold world-X -> lateral-only (Y) slide
    arm_z_constant: bool = False      # pin both arms' hover to one fixed Z (wrists level)
    freeze_last_dof: bool = False     # IK holds wrist_3 at init (final wrist roll fixed)
    freeze_wrist: bool = False        # freeze wrist 1/2/3 -> slider-like 2-3 DoF arm
    freeze_elbow: bool = False        # also pin the elbow -> pure 2-DoF turn+lean arm
    arm_ik_pos_only: bool = False     # drop orientation rows -> position-only solve

    # --- PHASE 0: policy learns gross arm positioning (no IK, no pressing) ---
    # The policy drives a reduced arm (default shoulder_pan + shoulder_lift) so each
    # hand-base covers the centroid of the keys it must play; all other DoF frozen.
    # Reward = arm_position_weight alone (zero the key/finger/onset weights).
    # Run: freeze_arms=False, arm_ik_follow=False, mute_right_hand=False,
    #      phase0_arm_positioning=True, arm_position_weight=1.0, others 0.
    lane_clamp: bool = True           # clamp each hand's target to its own half (anti-jam)
    phase0_arm_positioning: bool = False
    # Target the palm BODY's measured offset from the covered keys (the palm rides
    # above + behind the fingertips, so the bare key centroid is unreachable).
    arm_pos_calibrate: bool = True
    arm_pos_palm_offset_left: tuple = (0.1515, 0.0666, 0.206)
    arm_pos_palm_offset_right: tuple = (0.1491, 0.0894, 0.206)
    # FOREARM CLEARANCE: penalize the forearm housing dipping below z (onto the table)
    # or drifting forward of x (lying flat); keep it up and back toward the base.
    forearm_clear_z: float = 0.90
    forearm_back_x: float = 0.93
    forearm_clear_weight: float = 3.0
    # INTER-ARM SEPARATION: penalize palm-palm distance below arm_sep_min (anti-collision).
    # Only acts when the POLICY drives the arms; in arm_ik_follow lane_clamp handles it.
    arm_sep_weight: float = 0.0
    arm_sep_min: float = 0.18
    # WRIST-TABLE CLEARANCE: penalize a wrist dipping below wrist_clear_z (table top
    # ~0.72, keys ~0.76). Policy-driven arms only (IK keeps the wrist up via arm_ik_hover).
    wrist_clear_weight: float = 0.0
    wrist_clear_z: float = 0.82
    # which arm joints the policy may move in Phase 0 (substring match on names).
    phase0_arm_joints: tuple = ("shoulder_pan", "shoulder_lift")
    phase0_arm_scale: float = 0.30    # residual scale for the live Phase-0 arm joints

    # --- fingering / press tweaks ---
    remap_thumb_to_middle: bool = False   # thumb fingering -> middle finger (better presser)
    solo_right_middle: bool = False       # mask action to ONLY the right middle finger
    solo_arm_dip: bool = False            # solo mode also drives shoulder_lift (arm-dip press)
    lift_between_notes: float = 0.0       # dip-to-strike: lift this many m between notes (0=off)
    strike_window: int = 4                # dip this many steps before an onset
    palm_down_servo: bool = False         # IK holds the hand palm-down/fingers-forward
    hand_tilt: float = 0.0                # rad to tilt the IK servo orientation
    hand_tilt_axis: int = 1               # world axis to tilt about (0=x,1=y,2=z)
    idle_hand_retract: float = 0.20       # m an inactive hand (no upcoming notes) lifts off the keys
    arm_ik_hover: float = 0.11            # m the servoed palm hovers above the key tops

    # --- reward weights (PianoMime/RoboPianist composite) ---
    key_press_weight: float = 2.0     # reward sounding the right keys
    false_press_weight: float = 1.0   # penalty per wrong key sounded (precision)
    energy_weight: float = 0.0005
    # IDLE-FINGER CLEARANCE: penalize idle fingers hanging low enough to strike keys.
    idle_clear_weight: float = 0.0
    idle_clear_margin: float = 0.02   # m above key tops an idle fingertip must stay
    # IDLE-FINGER HOVER (positive twin of idle_clear): reward idle fingers for sitting
    # at their hover-home. One-sided z-only: full reward at/above the plane, decay only
    # when sinking below it. Suggested 0.2-0.3 for hand training. 0 = off.
    idle_hover_weight: float = 0.0
    idle_hover_close: float = 0.005        # m dead-band -> full hover reward inside
    idle_hover_margin_mult: float = 5.0    # falloff ~0.1 at 2.5cm below the band
    idle_hover_z_only: bool = True
    # IDLE-FINGER CURL: curl the flexion joints of idle fingers up into the palm in the
    # base pose (structural anti-mash), so a clean single-finger press is possible.
    idle_finger_curl: float = 0.0
    # START-CURLED: curl ALL finger flex joints by this many rad in the reset/base pose.
    start_finger_curl: float = 0.0
    fingering_weight: float = 1.0     # fingertip -> assigned key spatial shaping
    onset_weight: float = 2.0         # reward sounding a key on its onset
    # PHASE-0 gross-positioning reward (hand-base -> covered-key centroid). 0 = off.
    arm_position_weight: float = 0.0
    arm_position_close: float = 0.03         # m -> full positioning reward inside
    arm_position_margin_mult: float = 8.0    # falloff ~0.1 at ~0.27 m
    # IDLE-HAND HOME: a hand with no notes rests at its home hover (over its own half).
    arm_home_idle: bool = True
    # ARM-HEALTH penalties: subtract jerk_weight*action_jerk + limit_weight*(1-limit_margin).
    jerk_weight: float = 0.1
    limit_weight: float = 0.2

    # --- recall-gated annealing (press-discovery curriculum) ---
    # Hold the false-press penalty low (and energy at 0) so pressing gets discovered,
    # then ramp both to their cfg values over anneal_steps once the recall EMA crosses
    # the gate. Monotonic; pauses if recall dips. (cfg values above are the finals.)
    anneal_false_press: bool = False
    false_press_start: float = 0.15
    anneal_recall_gate: float = 0.5
    anneal_recall_beta: float = 0.99
    anneal_steps: int = 2000

    arm_lookahead: int = 5            # steps of upcoming notes used for the centroid
    hand_base_body: str = "robot0_palm"

    key_damping: float = 0.0          # >0 overrides piano key return-spring damping

    # velocity-gated ("hammer") sounding: a key rings only when struck past the sound
    # angle (frac>=key_struck_frac) AND moving down faster than key_strike_vel; a
    # statically-resting hand rings nothing. Stays ringing until it springs back above
    # key_release_frac.
    key_struck_frac: float = 1.0
    key_release_frac: float = 0.8
    key_strike_vel: float = 0.35      # drop toward 0.25 if recall craters

    # ===================== 🔒 LOCKED STATIC POSE — DO NOT EDIT =====================
    # left_ready_pose / right_ready_pose are the constant ready pose for both arms.
    # wrist_1=-4.782 + shoulder_lift=-0.640 give a +70deg wrist-up tilt with the hand
    # lowered ~40cm in z; WRJ0/WRJ1 tilt the Shadow wrist up so the hands don't droop
    # into the table. Fingertips land ~4.5cm above the keys, pointing down.
    # User-declared final baseline -- do NOT change without an explicit request. See CLAUDE.md.
    # ===============================================================================
    left_ready_pose = {
        "shoulder_pan_joint": -0.275,
        "shoulder_lift_joint": -0.640,
        "elbow_joint": 2.20,
        "wrist_1_joint": -4.782,
        "wrist_2_joint": -1.570,
        "wrist_3_joint": 3.14159,
        "robot0_WRJ0": 0.45,   # wrist tilt up, range [-0.70, 0.49]
        "robot0_WRJ1": 0.13,   # range [-0.49, 0.14]
        "robot0_(?!WRJ).*": 0.0,
    }
    right_ready_pose = {
        "shoulder_pan_joint": -0.275,
        "shoulder_lift_joint": -0.640,
        "elbow_joint": 2.20,
        "wrist_1_joint": -4.782,
        "wrist_2_joint": -1.570,
        "wrist_3_joint": 3.14159,
        "robot0_WRJ0": 0.45,
        "robot0_WRJ1": 0.13,
        "robot0_(?!WRJ).*": 0.0,
    }

    def __post_init__(self):
        per_arm = PER_ARM_DOF
        # observation size from the feature flags (single source of truth)
        obs = (
            2 * per_arm * 2                        # both arms pos+vel
            + NUM_KEYS                              # current key angles
            + self.goal_lookahead * NUM_KEYS        # upcoming note goals
        )
        if self.obs_fingertip_pos:
            obs += NUM_FINGERS * 3
        if self.obs_finger_targets:
            obs += NUM_FINGERS * 3
        if self.obs_goal_sdf:
            obs += NUM_KEYS
        self.observation_space = obs

        # bake world poses into each articulation's initial state
        self.piano_cfg.init_state.pos = self.piano_pos
        self.piano_cfg.init_state.rot = getattr(self, "piano_rot", (1.0, 0.0, 0.0, 0.0))
        if self.key_damping > 0:
            self.piano_cfg.actuators["keys"].damping = self.key_damping
        self.left_robot_cfg.init_state.pos = self.left_base_pos
        self.right_robot_cfg.init_state.pos = self.right_base_pos
        _br = getattr(self, "arm_base_rot", (1.0, 0.0, 0.0, 0.0))
        self.left_robot_cfg.init_state.rot = _br
        self.right_robot_cfg.init_state.rot = _br
        self.left_robot_cfg.init_state.joint_pos = dict(self.left_ready_pose)
        self.right_robot_cfg.init_state.joint_pos = dict(self.right_ready_pose)
