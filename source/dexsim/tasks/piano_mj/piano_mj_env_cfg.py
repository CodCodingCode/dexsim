"""Config for the MuJoCo port of the bimanual piano task.

Mirror of ``dexsim.tasks.piano.piano_env_cfg.PianoEnvCfg`` (the Isaac version)
with the Isaac-only machinery removed: no PhysX buffers, no ArticulationCfgs,
no UR10e-arm IK modes (this embodiment has no arm joints -- each hand rides a
1-DoF Y rail exactly like the Isaac slider USDs). Everything task-level is
kept identical: layout constants, reachable key windows, reward weights, the
velocity-gated key sounding, and the 🔒 locked static ready pose.

Timing: MuJoCo runs its own physics rate (0.005 s, the RoboPianist-standard
step for finger/key contact) but the CONTROL rate is the same 20 Hz
(control_dt 0.05) the MIDI goal grid and the Isaac env use.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

from dexsim.piano.midi import NUM_KEYS as _NUM_KEYS

# 20 Hz policy (decimation 10 @ 200 Hz sim) -> matches MIDI control_dt 0.05
SIM_DT = 0.005
DECIMATION = 10
CONTROL_DT = SIM_DT * DECIMATION

GOAL_LOOKAHEAD = 10               # steps of upcoming notes the policy sees (~0.5s)
NUM_KEYS = _NUM_KEYS              # 88
PER_HAND_DOF = 25                 # one Y rail + Shadow Hand(24 joints)
PER_HAND_ACT = 21                 # one Y rail + 20 hand actuators (J0 pairs
#                                   are tendon-coupled -> one actuator each,
#                                   same coupling the Isaac USD had)
NUM_FINGERS = 10                  # 5 per hand


@dataclass
class PianoMjEnvCfg:
    # --- spaces ---
    sim_dt: float = SIM_DT
    decimation: int = DECIMATION
    episode_length_s: float = 30.0
    action_space: int = 2 * PER_HAND_ACT                 # 42 (rail + hand actuators)
    observation_space: int = 0                           # computed in __post_init__
    seed: int = 0

    # --- observation features (assembled in PianoMjEnv._get_obs) ---
    obs_fingertip_pos: bool = True    # 10x3 fingertip world pos
    obs_finger_targets: bool = True   # 10x3 reference fingertip targets
    obs_goal_sdf: bool = True         # 88 analytic SDF of the current goal

    # --- task / songs ---
    midi_path: str = "data/midi/song.mid"
    control_dt: float = CONTROL_DT
    goal_lookahead: int = GOAL_LOOKAHEAD
    songs_npz: str | None = None      # multi-song goal bundle (goals/lens/names)
    max_songs: int = 0                # 0 = all songs in the bundle
    song_offset: int = 0              # held-out eval split support

    # --- fold a wide song into each hand's reachable key window ---
    fold_to_reach: bool = True
    left_key_window: tuple[int, int] = (19, 26)
    right_key_window: tuple[int, int] = (63, 70)

    # --- layout: level, non-overlapping one-axis hand rails ---
    # BOARD FLIPPED 180° vs the Isaac cfg (user request 2026-08-17): identity
    # rotation + mirrored Y offset keep the keyboard centered at (0.61, 0,
    # 0.756) but turn the KEY FRONTS toward the hands at x=0.82, so the robots
    # play from the player's side instead of reaching over the back rail.
    # Bonus: low pitch now sits on the LEFT robot's side (real piano
    # convention) so the fingering guardrail reports swap_hands=False.
    piano_pos: tuple = (0.61, -0.598, 0.746)    # keyboard centered at (0.61, 0, 0.756)
    piano_rot: tuple = (1.0, 0.0, 0.0, 0.0)     # identity (wxyz)
    hand_fixed_z: float = 0.88
    left_base_pos: tuple = (0.82, -0.30, 0.88)
    right_base_pos: tuple = (0.82, 0.30, 0.88)

    # --- action scaling: target = ready + scale * action (action in [-1,1]) ---
    arm_action_scale: float = 0.12    # rail travel scale (m)
    hand_action_scale: float = 0.8    # NOT the Isaac 0.35! That value was tuned
    #   for Isaac's stiffness-45/effort-40 actuators. The Menagerie hand uses
    #   the real Shadow's weak position servos (kp 0.5-1, forcerange ~1 N), so
    #   the achievable press force scales with the target offset -- and at 0.35
    #   the MAXIMUM action bottoms out at -0.0103 rad, short of the -0.012
    #   sound angle: the policy was physically unable to sound a key (measured
    #   2026-08-16). 0.8 sounds reliably with headroom.

    # --- arm (rail) mode ---
    freeze_arms: bool = False         # rails held at 0; fingers-only policy
    mute_right_hand: bool = False     # hold the right hand at ready (left-only songs)
    # RAIL-FOLLOW: the rail is servoed analytically to the upcoming-note centroid
    # (the 1-DoF twin of the Isaac arm_ik_follow); the policy drives fingers only.
    rail_follow: bool = False
    arm_smooth: float = 0.8           # EMA on the rail servo target (0 = instant)
    arm_lookahead: int = 5            # steps of upcoming notes for the centroid
    lane_clamp: bool = True           # each rail target stays in its own half

    # --- fingering / press tweaks ---
    remap_thumb_to_middle: bool = False
    idle_finger_curl: float = 0.0     # rad: curl NON-assigned fingers up (rail_follow)
    start_finger_curl: float = 0.0    # rad: curl ALL flex joints in the ready pose

    # --- reward weights (PianoMime/RoboPianist composite; == Isaac cfg) ---
    key_press_weight: float = 2.0
    false_press_weight: float = 1.0
    energy_weight: float = 0.0005
    idle_clear_weight: float = 0.0
    idle_clear_margin: float = 0.02
    idle_hover_weight: float = 0.0
    idle_hover_close: float = 0.005
    idle_hover_margin_mult: float = 5.0
    idle_hover_z_only: bool = True
    fingering_weight: float = 1.0
    onset_weight: float = 2.0
    jerk_weight: float = 0.1

    # --- recall-gated annealing (press-discovery curriculum; == Isaac cfg) ---
    # Hold the false-press penalty low (and energy at 0) so pressing gets
    # discovered, then ramp both to their cfg values over anneal_steps once the
    # per-env recall EMA crosses the gate. Monotonic; pauses if recall dips.
    anneal_false_press: bool = False
    false_press_start: float = 0.15
    anneal_recall_gate: float = 0.5
    anneal_recall_beta: float = 0.99
    anneal_steps: int = 2000

    onset_tol_steps: int = 3          # +/-150ms window for the onset-timing metric

    key_damping: float = 0.0          # >0 overrides piano key return-spring damping

    # velocity-gated ("hammer") sounding. NOT the Isaac 0.35 strike gate: a
    # position-servo press decelerates as it approaches its target, so by the
    # time the key crosses the sound angle it moves slower than 0.25 rad/s --
    # measured: at 0.25 a deliberate max-action press NEVER sounds (0/30
    # steps) while random flail (fast transients) passes 183/500. At 0.10 the
    # deliberate press rings reliably (12/30, latched once developed) and
    # mash is no worse (192/500). RoboPianist uses no velocity gate at all;
    # 0.10 keeps a token anti-static-rest filter.
    key_struck_frac: float = 1.0
    key_release_frac: float = 0.8
    key_strike_vel: float = 0.10
    # DENSE GOAL-KEY PRESS REWARD (RoboPianist-style): feed the reward's hit
    # term with the RAW depression fraction of goal keys (continuous gradient
    # as the key travels down) instead of the velocity-latched sounding, which
    # is zero until the key fully rings -- no learning signal on the way down.
    # The latch still governs the false-press term and every metric
    # (recall/F1 = keys that actually SOUNDED). False = exact Isaac semantics.
    dense_goal_press: bool = True
    # evaluate the strike gate every PHYSICS substep instead of once per 50ms
    # control step. The velocity spike of a real strike lasts ~30ms, so the
    # control-rate snapshot misses most genuine presses (measured 5x
    # undercount: 104 substep strikes vs 20 control-rate on identical
    # trajectories). MuJoCo-stack improvement; False = exact Isaac semantics.
    substep_strike_detect: bool = True

    hand_base_body: str = "robot0_palm"

    # ===================== 🔒 LOCKED STATIC POSE — DO NOT EDIT =====================
    # left_ready_pose / right_ready_pose are the constant ready pose for both
    # hands: rail centered, robot0_WRJ0 = 0.45 / robot0_WRJ1 = 0.13 wrist tilt,
    # all fingers straight. Fingertips hover a few cm above the keys, pointing
    # down. User-declared final baseline -- do NOT change without an explicit
    # request. See CLAUDE.md. (Keys are regex patterns over per-hand joint
    # names, exactly like the Isaac cfg.)
    # ===============================================================================
    left_ready_pose: dict = field(default_factory=lambda: {
        "railJoint": 0.0,
        "robot0_WRJ0": 0.45,   # wrist tilt, range [-0.70, 0.49]
        "robot0_WRJ1": 0.13,   # range [-0.49, 0.14]
        "robot0_(?!WRJ).*": 0.0,
    })
    right_ready_pose: dict = field(default_factory=lambda: {
        "railJoint": 0.0,
        "robot0_WRJ0": 0.45,
        "robot0_WRJ1": 0.13,
        "robot0_(?!WRJ).*": 0.0,
    })

    def __post_init__(self):
        per_arm = PER_HAND_DOF
        # observation size from the feature flags (single source of truth)
        obs = (
            2 * per_arm * 2                        # both hands qpos+qvel
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

    def to_dict(self) -> dict:
        return asdict(self)
