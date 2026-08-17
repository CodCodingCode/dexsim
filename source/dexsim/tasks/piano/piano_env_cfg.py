"""Config for two physical Shadow Hands on independent Y rails over a piano.

Two Shadow Hand articulations (Y rail + 24 hand DoF each = 50 action DoF) over a
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

from dexsim.assets import (
    PIANO_SHADOW_HAND_LEFT_CFG,
    PIANO_SHADOW_HAND_RIGHT_CFG,
    PIANO_CFG,
)
from dexsim import DATA_DIR

# 20 Hz policy (decimation 6 @ 120 Hz sim) -> matches MIDI control_dt 0.05
SIM_DT = 1.0 / 120.0
DECIMATION = 6
CONTROL_DT = SIM_DT * DECIMATION

GOAL_LOOKAHEAD = 10               # steps of upcoming notes the policy sees (~0.5s)
NUM_KEYS = 88
PER_HAND_DOF = 25                 # one Y rail + Shadow Hand(24)
NUM_FINGERS = 10                  # 5 per hand


@configclass
class PianoEnvCfg(DirectRLEnvCfg):
    # --- spaces ---
    decimation = DECIMATION
    episode_length_s = 30.0
    action_space = 2 * PER_HAND_DOF                      # 50 (rail + hand joints)
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
    # Separate physical hands; each root is a world-anchored prismatic Y joint.
    left_robot_cfg: ArticulationCfg = PIANO_SHADOW_HAND_LEFT_CFG.replace(
        prim_path="/World/envs/env_.*/LeftRobot"
    )
    right_robot_cfg: ArticulationCfg = PIANO_SHADOW_HAND_RIGHT_CFG.replace(
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

    # --- layout: level, non-overlapping one-axis hand rails ---
    piano_pos = (0.6100, 0.5989, 0.7460)     # keyboard centered at (0.60, 0, 0.756)
    piano_rot = (0.0000000, 0.0000000, 0.0000000, 1.0000000)     # 180deg about Z
    # The mount hangs the hand fingers-down: the palm rides ~0.29 m below it, so
    # the mounts sit high. Palm z ~0.90 puts the (tilted) fingertips ~1.5 cm
    # above the key tops; palm x ~0.73 lands the tips at mid-key x~0.63.
    # Each hand has its OWN full pose (position incl. z, wxyz rotation) -- place
    # them independently (e.g. with scripts/tools/place_scene_viser.py).
    left_base_pos = (0.2013, 0.1278, 0.7857)
    left_base_rot = (0.5716125, 0.4318425, 0.4279437, 0.5510312)
    right_base_pos = (0.1940, -0.1786, 0.7838)
    right_base_rot = (0.5624963, 0.4394271, 0.4246139, 0.5569603)

    # --- action scaling: target = default + scale * action (action in [-1,1]) ---
    # Per-joint (see PianoEnv.joint_scale): the stiff rail joint blows up under a
    # large residual; the hand joints need a generous range to travel between keys.
    # So scale the rail gently, the hand more.
    arm_action_scale: float = 0.12    # legacy name: railJoint travel scale (m)
    hand_action_scale: float = 0.35   # travels between keys without mashing
    action_scale: float = 0.15        # legacy global scale (unused; back-compat)

    # --- base mode ---
    # FIXED HANDS: rails hold station; only the fingers train (RoboPianist-style).
    freeze_arms: bool = False
    # mute the right hand (hold its fingers at the ready pose) for left-hand-only
    # songs so it can't mash idle keys. MUST be False for two-handed songs.
    mute_right_hand: bool = False
    lane_clamp: bool = True           # clamp each hand's target to its own half (anti-jam)

    # --- WRIST LOCK: hold both hands upright, structurally ---
    # The rail already pins the hand base's orientation, so the ONLY way a hand
    # stops being upright is the policy (or a contact load) rotating the two
    # Shadow wrist joints robot0_WRJ0/WRJ1. Locking them makes "upright" a
    # property of the mechanism instead of something the reward has to keep
    # buying back: the wrist targets are pinned at the locked ready-pose tilt
    # (WRJ0=0.45 / WRJ1=0.13), the policy's residual on those joints is zeroed,
    # and the joints are stiffened so contact can't push them off pose either.
    # Costs 4 of the 50 action DoF (2 per hand); the fingers and rails are
    # untouched. Set False to hand the wrists back to the policy.
    wrist_lock: bool = True
    wrist_lock_slack: float = 0.005       # rad of travel left around the tilt
    wrist_lock_stiffness: float = 400.0   # vs the 45.0 finger default
    wrist_lock_damping: float = 15.0
    wrist_lock_effort: float = 400.0      # N*m; the stock 27 N*m ceiling saturates

    # --- fingering / press tweaks ---
    remap_thumb_to_middle: bool = False   # thumb fingering -> middle finger (better presser)
    solo_right_middle: bool = False       # mask action to ONLY the right middle finger
    arm_ik_hover: float = 0.11            # m the positioning-reward target sits above the key tops

    # ===================== REWARD MODE =====================
    # "dexsim" -- the composite grown in this repo (key press + fingering plan +
    #             onset + idle-hover + arm/jerk penalties, all the knobs below).
    # "rp1m"   -- a faithful port of RP1M (Zhao et al., CoRL 2024):
    #                 r = r_OT + r_Press + 0.5*r_Collision - 5e-3*r_Energy
    #             Nothing below the `--- reward weights` divider applies in this
    #             mode EXCEPT the anneal block; the rp1m_* fields drive it instead.
    #             The defining difference: fingering is NOT read from the
    #             precomputed plan, it is re-solved every step by optimal transport
    #             from the live fingertip positions (dexsim.piano.ot).
    reward_mode: str = "dexsim"

    # --- RP1M reward (used only when reward_mode == "rp1m") ---
    rp1m_ot_weight: float = 1.0
    rp1m_ot_close: float = 0.01        # m; full credit inside (the paper's 1 cm)
    rp1m_ot_margin_mult: float = 10.0
    # "sum" = the paper's cumulative d_OT (1:1). "mean" divides by the number of
    # demanded keys, keeping a chord on the same scale as a single note.
    rp1m_ot_reduce: str = "sum"
    rp1m_ot_eps: float = 0.01          # Sinkhorn temperature (m)
    # 0 = rank finger/key pairs nearest-first (default; within 0.25 mm of the
    # exact assignment after 2-opt repair, ~3x cheaper). >0 = Sinkhorn ranking,
    # ~0.03 mm. See dexsim.piano.ot for the measured table.
    rp1m_ot_iters: int = 0
    # Not in RP1M (its hands share a workspace): penalize matching a key to a
    # finger on the far side of the keyboard midline, which our disjoint rails
    # cannot reach. 0 = pure RP1M; ~2.5 mirrors the offline OT planner.
    rp1m_ot_side_weight: float = 0.0
    # Measure the OT distance only along actuated axes (drop X): the armless
    # rail hands cannot move a fingertip in X, so a 3D cost carries an
    # irreducible floor that parks r_OT's exponential in its gradient-dead
    # tail (observed: flat 0.065 for 5k iters). True = cost over (Y, Z) only.
    rp1m_ot_ignore_x: bool = True
    rp1m_press_weight: float = 1.0
    rp1m_press_close: float = 0.05     # normalized key state within this of 1 = full
    rp1m_press_margin_mult: float = 10.0
    # The paper forfeits the WHOLE 0.5 false-press half on any wrong key. With
    # fingers that haven't learned to lift yet that half is just constant 0 and
    # carries no gradient -- flip this on to charge per stray key instead.
    rp1m_press_false_soft: bool = False
    rp1m_collision_weight: float = 0.5    # alpha_1
    # How r_Collision decides the hands are colliding:
    #   True  -- real PhysX contact reports (what the paper does). One
    #            ContactSensor per left-hand body in `contact_bodies`, each
    #            filtered against the whole right hand; Isaac Lab only supports
    #            one-to-many filtering, hence one sensor per body.
    #   False -- geometric proximity: any left point within rp1m_collision_dist
    #            of any right point. No sensors, no extra PhysX contact buffers.
    rp1m_collision_contacts: bool = True
    rp1m_collision_force: float = 1.0     # N; contact force that counts as a hit
    rp1m_collision_dist: float = 0.02     # m; proximity fallback threshold
    # Bodies the contact sensors watch (left hand only -- contact is symmetric,
    # so left-vs-right catches every hand/hand collision).
    contact_bodies: tuple[str, ...] = (
        "robot0_palm", "robot0_thdistal", "robot0_ffdistal",
        "robot0_mfdistal", "robot0_rfdistal", "robot0_lfdistal",
    )
    rp1m_energy_weight: float = 5e-3      # alpha_2 (subtracted)
    # r_Sustain is omitted: our piano articulation has no pedal joint.

    # --- automatic fingering used for the PLAN (observation + dexsim reward) ---
    # "heuristic" -- the pitch-split rule; "ot" -- RP1M-style optimal transport
    # precomputed offline over the song. reward_mode="rp1m" additionally solves OT
    # online each step for its reward, independent of this setting.
    fingering_method: str = "heuristic"

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
    # F1 BONUS (applies in BOTH reward modes): per-step, per-env F1 of pressed-vs-
    # goal keys added to the reward with this weight. A metric-alignment term on
    # top of shaping, never a replacement -- it pays ~0 until the policy earns
    # nonzero F1 (self-annealing), then pulls the objective toward the number we
    # actually evaluate. 0 = off.
    f1_weight: float = 0.0
    # PHASE-0 gross-positioning reward (hand-base -> covered-key centroid). 0 = off.
    arm_position_weight: float = 0.0
    arm_position_close: float = 0.03         # m -> full positioning reward inside
    arm_position_margin_mult: float = 8.0    # falloff ~0.1 at ~0.27 m
    # IDLE-HAND HOME: a hand with no notes rests at its home hover (over its own half).
    arm_home_idle: bool = True
    # ARM-HEALTH penalties: subtract jerk_weight*action_jerk + limit_weight*(1-limit_margin).
    jerk_weight: float = 0.1
    limit_weight: float = 0.0

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
    # left_ready_pose / right_ready_pose are the constant ready pose for both hands.
    # WRJ0=0.45 / WRJ1=0.13 tilt the Shadow wrist up so the hands don't droop into
    # the table; fingers relaxed at 0. (The UR10e arms -- and their locked arm-joint
    # pose -- were removed 2026-08-13 at the user's request; the hands now ride
    # world-anchored Y rails. The WRJ tilt is carried over from that baseline.)
    # Do NOT change without an explicit request. See CLAUDE.md.
    # ===============================================================================
    left_ready_pose = {
        "railJoint": 0.0,
        "robot0_WRJ0": 0.4500,
        "robot0_WRJ1": 0.1300,
        "robot0_FFJ3": 0.0000,
        "robot0_MFJ3": 0.0000,
        "robot0_RFJ3": 0.0000,
        "robot0_LFJ3": 0.0000,
        "robot0_FFJ2": 0.3500,
        "robot0_MFJ2": 0.3500,
        "robot0_RFJ2": 0.3500,
        "robot0_LFJ2": 0.3500,
        "robot0_FFJ1": 0.3000,
        "robot0_MFJ1": 0.3000,
        "robot0_RFJ1": 0.3000,
        "robot0_LFJ1": 0.3000,
        "robot0_FFJ0": 0.2500,
        "robot0_MFJ0": 0.2500,
        "robot0_RFJ0": 0.2500,
        "robot0_LFJ0": 0.2500,
        "robot0_LFJ4": 0.0500,
        "robot0_THJ4": 0.3000,
        "robot0_THJ3": 0.3000,
        "robot0_THJ2": 0.0000,
        "robot0_THJ1": 0.0000,
        "robot0_THJ0": -0.3000,
    }
    right_ready_pose = {
        "railJoint": 0.0,
        "robot0_WRJ0": 0.4500,
        "robot0_WRJ1": 0.1300,
        "robot0_FFJ3": 0.0000,
        "robot0_MFJ3": 0.0000,
        "robot0_RFJ3": 0.0000,
        "robot0_LFJ3": 0.0000,
        "robot0_FFJ2": 0.3500,
        "robot0_MFJ2": 0.3500,
        "robot0_RFJ2": 0.3500,
        "robot0_LFJ2": 0.3500,
        "robot0_FFJ1": 0.3000,
        "robot0_MFJ1": 0.3000,
        "robot0_RFJ1": 0.3000,
        "robot0_LFJ1": 0.3000,
        "robot0_FFJ0": 0.2500,
        "robot0_MFJ0": 0.2500,
        "robot0_RFJ0": 0.2500,
        "robot0_LFJ0": 0.2500,
        "robot0_LFJ4": 0.0500,
        "robot0_THJ4": 0.3000,
        "robot0_THJ3": 0.3000,
        "robot0_THJ2": 0.0000,
        "robot0_THJ1": 0.0000,
        "robot0_THJ0": -0.3000,
    }

    def __post_init__(self):
        per_arm = PER_HAND_DOF
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
        self.left_robot_cfg.init_state.rot = self.left_base_rot
        self.right_robot_cfg.init_state.rot = self.right_base_rot
        self.left_robot_cfg.init_state.joint_pos = dict(self.left_ready_pose)
        self.right_robot_cfg.init_state.joint_pos = dict(self.right_ready_pose)
