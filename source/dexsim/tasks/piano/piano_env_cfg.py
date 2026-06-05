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
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass

from dexsim.assets import UR10E_SHADOW_CFG, SHADOW_SLIDER_CFG, PIANO_CFG
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

    # PhysX GPU buffers: the default rigid-patch/contact buffers overflow when many
    # fingers contact many keys across thousands of envs ("Patch buffer overflow,
    # increase to at least 167936"). Bump the contact/patch caps generously.
    sim: SimulationCfg = SimulationCfg(
        dt=SIM_DT, render_interval=DECIMATION,
        physx=PhysxCfg(
            gpu_max_rigid_patch_count=2 ** 20,        # was ~163840 default -> overflow
            gpu_max_rigid_contact_count=2 ** 23,
            gpu_found_lost_pairs_capacity=2 ** 22,
            gpu_found_lost_aggregate_pairs_capacity=2 ** 23,
        ),
    )
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

    # --- SLIDER embodiment (RP1M/RoboPianist): replace the heavy UR arm with a
    # 2-DoF prismatic slider that places the hand to ~0mm (solves the arm's placement
    # wall). Each hand = 26 DoF (slider_y, slider_z + 24 fingers). The policy drives the
    # fingers (residual); the slider is positioned analytically to the note (calibrated
    # at env init). See dexsim-slider-embodiment memory + scripts/smoke/slider_play.py.
    use_slider: bool = False
    slider_press_finger: int = 1     # which finger strikes (0=th,1=ff,...) for placement
    slider_hand_x: float = 0.55      # hand depth over the keys
    slider_hand_z: float = 1.19      # hand height (tip just above key tops)
    # PIANIST-POSTURE TILT: angle the hand off straight-down so pressing strikes with the
    # angled FINGERTIPS instead of mashing the whole hand down (the geometric fix for the
    # precision ceiling -- same idea as the arm's hand_tilt that doubled recall).
    slider_hand_tilt: float = 0.0    # rad, tilt about the lateral (Y/slider) axis
    slider_idle_curl: float = 0.8    # tip-curl (J1/J2) of idle fingers
    slider_idle_mcp: float = 0.0     # MCP (J3) curl of idle fingers (lift whole finger clear)
    slider_teleport_once: bool = False  # snap slider to IK target once/control-step (tight onset placement)
    slider_stiffness: float = 0.0    # >0 overrides slider PD stiffness (reach target in 1 step)
    slider_residual: float = 0.05    # policy residual scale on the 2 slider DoF (0 = pure IK placement)
    key_damping: float = 0.0         # >0 overrides piano key return-spring damping (lower=faster release)
    # FINGER-STRIKE press: angle the strike finger at the MCP (knuckle) down-FORWARD,
    # place its tip on the key via the slider, then STRIKE by flexing the PIP to drive
    # the tip onto the key while the hand stays at HOVER -> only that finger contacts
    # (no hand descent = no mash). THE candidate fix for the precision ceiling.
    slider_finger_strike: bool = False
    slider_strike_mcp: float = 0.6   # base MCP (J3) flex -> angle the strike finger forward
    slider_strike_pip: float = 0.9   # PIP (J2) flex when active -> strike the tip down
    slider_strike_hover: float = 0.02  # m the angled tip hovers above the key (strike spans this)

    # --- MULTI-SONG training ---
    # When set, load N real songs' note-goals from this precomputed .npz bundle
    # (keys: goals (N,Tmax,88), lens (N,), names (N,)) and train ONE policy across
    # all of them: each env is assigned a song (round-robin) so the rollout always
    # covers every song. This is what makes the policy generalize instead of being
    # a per-song specialist. None => single-song training from midi_path.
    songs_npz: str | None = None
    max_songs: int = 0            # 0 = use all songs in the bundle; else cap to first N
    song_offset: int = 0          # skip the first N songs (held-out eval: train [0:K], test [K:])

    # --- fold a wide-range song into the rig's reachable key windows ---
    # The two fixed arm bases each reach only a narrow ~8-key band (left over key
    # ~22, right over key ~66) with an unreachable gap between. A wide song (e.g.
    # song.mid spans keys 13..72) is otherwise un-trackable -> the IK reference
    # collapses (zero-residual F1 ~0.05). Octave-fold every note into the nearest
    # hand window so the reference is physically playable. Windows are inclusive
    # (lo, hi) key indices; the left band is the proven-reachable easy-song span.
    fold_to_reach: bool = True    # fixed hands can only reach the keys under them,
    # so fold every note into each hand's window (= where the fixed hand hovers).
    left_key_window: tuple[int, int] = (19, 26)
    right_key_window: tuple[int, int] = (63, 70)

    # --- residual RL over a ready-pose base ---
    # The action is a residual on the static ready pose: arm columns are overwritten
    # each control step by WristPoseIK (arm_ik_follow) or the analytic slider rail,
    # and the policy learns finger pressing on top. No precomputed reference needed.

    # PLAYERS-IN-FRONT layout: piano at identity rotation (keys' player side = local +X
    # faces world +X), keyboard centered at world (0.60, 0.0, 0.756). The two robots sit
    # IN FRONT of the keys (large +X) and are rotated 180deg about Z (base_rot below) to
    # FACE the piano, raised above key height -> arm_ik_follow reaches down-and-back onto
    # the keys with a big elbow bend, bodies pointing at the piano (no draping over).
    piano_pos = (0.59, -0.598, 0.746)    # keyboard centered at (0.60, 0, 0.756)
    piano_rot = (1.0, 0.0, 0.0, 0.0)     # identity (keys face +X / the robots); piano is NOT the problem
    # bases in front of the keyboard (+X), facing it; left hand over the LOW keys (now -Y),
    # right hand over the HIGH keys (+Y). base_rot makes each UR10e reach -X toward the piano.
    left_base_pos = (1.25, -0.30, 1.05)
    right_base_pos = (1.25, 0.30, 1.05)
    arm_base_rot = (0.0, 0.0, 0.0, 1.0)  # wxyz: 180deg about Z so the arm faces the piano

    # action scaling: targets = default + scale * action (action in [-1,1]).
    # Per-joint (see PianoEnv.joint_scale): the stiff arm joints (stiffness 6000)
    # explode under a large residual -> NaN, but the weak hand joints (stiffness 3)
    # need a generous range to travel between keys and actually press them. One
    # global scale can't serve both: 0.5 blew up the arm; 0.15 froze the fingers
    # (F1 stuck ~0.02). So scale the arm gently and the hand generously.
    arm_action_scale: float = 0.15    # 0.30 crashed physics (stiff arm joints blow
    #   up under a large residual -> PhysX hard crash). 0.15 is the stable max; the
    #   arm IK must instead be precise so the arm barely needs to correct.
    hand_action_scale: float = 0.35   # was 0.6: too large -> with exploration noise the
    #   fingers flailed and struck ~23 keys at once (precision 3%, reward/key pinned at
    #   -2, learning stalled). 0.35 still travels between keys in a window but stops the
    #   mash. Raise again once the fingers reliably press single keys.
    # legacy global scale; not used by the env (kept for back-compat with old configs)
    action_scale: float = 0.15

    # --- curriculum: arms first, then hands ---
    # Phase 1 (freeze_hands=True): zero the 48 hand DoF so the policy ONLY drives the
    # 12 arm DoF -> it learns to SWEEP the arms so each hand's (frozen, pressed-pose)
    # fingers land on the right keys, scored by the fingering + arm-position reward.
    # Phase 2 (freeze_hands=False, warm-started from phase 1): unfreeze the hands so
    # they learn to press while the arms hold position. Sidesteps the IK-precision
    # wall by LEARNING arm positioning instead of relying on the (imperfect) IK ref.
    freeze_hands: bool = False
    # FIXED-HANDS mode: arms hold a constant pose hovering over the keyboard; only
    # the fingers are trained to press (RoboPianist-style). Sidesteps the whole arm
    # positioning/tracking problem to first prove the fingers can hit notes.
    freeze_arms: bool = True
    # for left-hand-only easy songs: freeze the right arm so it can't mash idle
    # high keys. BUT 'song.mid' is two-handed -- it needs the right hand on 943/999
    # steps (keys span 13..72, crossing the middle). Muting it hard-capped F1 near
    # zero (half the notes unplayable). Must be False for any two-handed song.
    mute_right_hand: bool = True   # CURRICULUM (easy.mid is LEFT-hand only, keys 19/20/26):
    #   hold the right fingers at the ready pose so they can't mash idle right-window keys
    #   and generate pure false-press noise. MUST be reverted to False for the two-handed
    #   song.mid (its notes cross into the right window).

    # --- ARM-IK-FOLLOW mode (the clean decoupling: math moves the arms, RL the fingers) ---
    # Drive the 12 arm DoF ONLINE with WristPoseIK: each control step the well-posed
    # palm-servo (proven to reach within ~1cm across the whole keyboard,
    # scripts/diag_wrist_ik.py) tracks the per-hand fingering centroid, while the policy
    # action is masked to the 48 finger DoF only. The policy never touches the stiff arm
    # joints, so the arm-blowup failure mode disappears. Set with freeze_arms=False;
    # supersedes freeze_arms (static hold) when both are set.
    arm_ik_follow: bool = False
    # PLANAR-IK: make WristPoseIK behave like an XY gantry -- weight z+orientation
    # heavily (so it never trades the plane for XY travel) and iterate to convergence,
    # so the arm slides flat at constant height instead of tilting/sagging onto the keys.
    planar_ik: bool = False
    planar_weight: float = 25.0
    planar_iters: int = 6
    # PIN DEPTH: also hold world-X so the arm slides ONLY laterally (world Y), like the
    # slider's single rail. Plain planar lets the arm chase each key's depth (white/black
    # keys differ in X) -> the wrist wanders in depth; pinning X removes that.
    planar_pin_x: bool = False
    # CONSTANT ALIGNED Z: pin BOTH arms' hover target to ONE fixed height (max key top +
    # arm_ik_hover) so the two wrists stay level and never move in Z; only X/Y track notes.
    arm_z_constant: bool = False
    # FREEZE the UR10e's last DoF (wrist_3_joint, the final wrist roll): WristPoseIK
    # leaves it out of the solve so it holds its init value EXACTLY while the other 5
    # arm joints still servo. Pairs with planar_ik (constant world-Z) to visualise the
    # arm sliding flat in XY with a fixed wrist roll.
    freeze_last_dof: bool = False
    # FREEZE THE WHOLE WRIST (wrist_1/2/3): leaves only shoulder_pan (turn) + shoulder_lift
    # (lean) [+ elbow] moving -> the slider-like 2-3 DoF arm. The wrist is exactly what the
    # IK swings to fix orientation, so freezing it removes the "fling"; pair with
    # arm_ik_pos_only (you can't control orientation with no wrist, so don't try).
    freeze_wrist: bool = False
    freeze_elbow: bool = False        # also pin the elbow -> pure 2-DoF turn+lean arm
    # POSITION-ONLY arm IK: drop the orientation rows, solve a 3-DoF position target on the
    # remaining (proximal) joints. Well-posed when the wrist is frozen; smooth, no fling.
    arm_ik_pos_only: bool = False
    # ONE-FINGER-PER-NOTE mode (monophonic redesign). The arm centroid-servo positions the
    # hand only GROSSLY -> the assigned fingertip lands ~13cm from its key (precision capped).
    # Here we instead aim ONE designated fingertip directly at the current note (offset the
    # palm target by the palm->fingertip vector), curl the other 4 fingers up, and let that
    # finger strike. For a monophonic melody this plays each note cleanly. Requires arm_ik_follow.
    single_finger: bool = False
    primary_finger: int = 1       # which finger presses, per-hand order [0=th,1=ff,2=mf,3=rf,4=lf]
    single_press_z: float = -0.006  # m vs key top to drive the fingertip to (negative = into key -> strike)
    single_curl: float = 2.0      # rad to curl the 4 non-primary fingers up out of the way
    single_align_thresh: float = 0.015  # m xy-distance under which the finger dips to press (else hovers)
    single_press_flex: float = 0.0   # rad to curl the ACTIVE primary finger DOWN to strike its key (sign tested)
    single_hover: float = 0.012   # m above key top the finger hovers while moving between notes
    # HAND-TILT redesign: rotate the servoed hand from palm-straight-down toward a real
    # pianist posture so a finger CURL drives its tip DOWN onto a key (individual keystroke),
    # instead of pressing by lowering the whole hand (which mashes all fingers -> precision wall).
    hand_tilt: float = 0.0        # rad to tilt the hand-servo target orientation
    hand_tilt_axis: int = 1       # world axis to tilt about (0=x,1=y,2=z) -- empirically chosen
    idle_hand_retract: float = 0.20  # m above the keys an INACTIVE hand (no upcoming
    #   notes) lifts to, so its resting fingers stop ringing keys (the muted right hand
    #   on a left-only song was mashing ~5-7 false keys from holding station at key level).
    arm_ik_hover: float = 0.11   # m the servoed palm hovers above the key tops. Raised
    #   0.05->0.09: at 0.05 the whole hand (palm+relaxed fingers) sat ON the keyboard and
    #   mashed ~15 keys/step (precision pinned ~0.05, F1 flat). Lifting the palm makes idle
    #   fingers CLEAR the keys so pressing becomes a deliberate finger extension the policy
    #   chooses. Capped at 0.09 (not higher) so an extended Shadow finger (~7cm) can still
    #   reach the keys; raise/lower off wandb play/keys_sounding (want ~#active, not 15).

    # ARM-FINGERTIP-TRACK: drive the arm via POSITION-ONLY IK on the PRIMARY FINGERTIP
    # (not the palm) so the striking finger's tip lands ON the key, closing the ~90mm
    # palm-vs-tip (one-finger-length) gap that capped precision. diag_posik proved the
    # fingertip converges to ~18mm under PD (vs 93mm for palm-centroid). Requires
    # arm_ik_follow. ftip_max_step raises per-step arm travel so it tracks fast notes.
    arm_ftip_track: bool = False
    ftip_max_step: float = 0.12

    # reward weights (PianoMime/RoboPianist composite)
    key_press_weight: float = 2.0   # was 1.0: PRESSING the right key must dominate
    #   the (positioning) shaping, else the policy hovers near keys for finger
    #   reward and never presses (F1 flat ~0.04, reward/finger 0.44 >> press).
    # NOTE: the old tuning (0.5 best, 1.5 over-suppressed) was under the BUGGED
    # penalty that averaged misclicks over all 88 keys. reward.py now counts wrong
    # keys PER INTENDED NOTE, which is ~40-80x stronger at the same weight, so 0.75
    # here ≈ "one wrong note nearly cancels one right note". Retune off wandb
    # play/precision; drop toward 0.5 if recall collapses.
    false_press_weight: float = 1.0   # PUNISH wrong notes hard for PRECISION. Raised
    #   0.5->1.0 alongside the arm_ik_hover lift: with the hand no longer forced to mash,
    #   a stronger false-press penalty pushes the policy to sound ONLY the assigned key
    #   instead of catching the target amid ~15 rung keys (precision was pinned ~0.05).
    #   If recall collapses (policy stops pressing at all), drop back toward 0.5.
    energy_weight: float = 0.0005
    # IDLE-FINGER CLEARANCE ("the fingered thing"): penalize fingers NOT assigned a
    # note this step for hanging low enough to strike keys. fingering_reward already
    # pulls ACTIVE fingers onto their key; this is the symmetric term that lifts the
    # other 4 out of the way, so pressing one finger stops ringing the whole ~12-key
    # hand footprint (the mash that pins precision ~0.05). 0 = off.
    idle_clear_weight: float = 0.0
    idle_clear_margin: float = 0.02   # m above the key tops an idle fingertip must stay
    # IDLE-FINGER CURL (structural anti-mash): the reward-only idle-clear term was 3% of
    # the dominant false-press penalty and changed keys_sounding by 0 -- the mash is
    # PHYSICAL (the hand footprint rings ~8 keys, the policy can't beat it). So curl the
    # flexion joints (J1/J2/J3) of fingers NOT assigned a note THIS step up into the palm
    # in the base pose, making a clean single-finger press physically possible. rad added
    # per idle-finger flexion joint; sign flips the curl direction (set off=0). The policy
    # residual still rides on top, and the ACTIVE finger stays straight to press.
    idle_finger_curl: float = 0.0
    fingering_weight: float = 1.0     # PHASE 2 (hands in): positioning is auxiliary
    #   now -- the fingers do fine placement while key/onset (pressing) lead. (Phase
    #   1 used 3.0 for a strong arms-only positioning signal.)
    onset_weight: float = 2.0         # was 0.5: strongly reward sounding the right
    #   key on its onset -> the press signal the policy was ignoring (onset ~0.01).

    # --- arm gross-positioning shaping (60-DoF only; RoboPianist skips it) ---
    # The arms must place each hand over the right span of keys before the fingers
    # can reach them. This pulls each hand base toward the horizontal centroid of
    # that hand's upcoming notes, layered UNDER the (fingertip-only) fingering term
    # so the arm coarsely positions while the fingers do the fine reach. Keep it
    # well below fingering_weight or it dominates. Set 0.0 to disable. Retune off
    # wandb reward/arm vs reward/finger.
    arm_base_weight: float = 0.3
    arm_close_enough: float = 0.05    # m; within 5 cm of the note span -> full reward
    arm_margin_mult: float = 6.0      # gaussian falloff ~0.1 at 6x bound (~30 cm)
    arm_lookahead: int = 5            # steps of upcoming notes used for the centroid
    hand_base_body: str = "robot0_palm"  # Shadow-hand base body (the "hand center")

    # velocity-gated ("hammer") sounding: a key rings only if struck with downward
    # joint velocity past this (rad/s); a statically-resting hand/forearm (~0 vel)
    # rings nothing -> fixes the 52-key precision collapse. See _key_pressed_fraction.
    # a key SOUNDS only when pressed PAST the sound angle (frac>=1.0). These latch the
    # velocity-gated sounding to the REAL sound angle (was hardcoded 0.5/0.25 = half/
    # quarter depth, which counted merely-brushed keys as sounding forever).
    key_struck_frac: float = 1.0    # frac to start sounding (1.0 = at the sound angle)
    key_release_frac: float = 0.8   # frac to stop sounding as the key springs back up
    key_strike_vel: float = 0.35   # 0.15->0.35: under ARM-IK-FOLLOW the arm servo drives
    #   the WHOLE hand down onto its ~12-key footprint, ringing all of them with the servo's
    #   downward velocity (keys_sounding pinned ~12-15 on a 3-key song -> precision ~0.05).
    #   Requiring a fast deliberate strike decouples "sounding" from the slow servo descent,
    #   so only a finger the policy actively jabs rings. If recall craters (policy can't
    #   strike fast enough with the weak hand), drop toward 0.25.

    # Piano "ready" pose: from the raised (z=1.05) bases each hand drapes DOWN so
    # the fingertips rest ~2 cm above the white keys (key top z=0.722), centered
    # over that hand's half of the keyboard. Solved numerically by
    # scripts/tune_arm_pose.py against the current bases -- fingertip mean lands
    # within ~3-4 cm of (x=0.35, y=+/-0.30, z=0.74), i.e. ON the keys. wrist_2/3
    # are held so the palm stays vertical. Per-side (the two arms differ slightly).
    # FIXED-HANDS hover poses (tune_arm_pose, targets over each window at ~2cm above
    # the keys): left fingertips land 12mm over window (19,26); right 37mm over
    # (63,70). The arms HOLD these; only the fingers move to press.
    # ELBOW-DOWN / EE-STRAIGHT-DOWN seed (grid-searched): bigger elbow bend + wrist_1
    # rotated so the hand drops onto the keys from ABOVE instead of leaning over them.
    # palm lands over the keys with the flange ~0.29m above it (fingers point -Z).
    left_ready_pose = {
        "shoulder_pan_joint": -0.275,
        "shoulder_lift_joint": -1.20,
        "elbow_joint": 2.00,
        "wrist_1_joint": -2.20,
        "wrist_2_joint": -1.570,
        "wrist_3_joint": 0.0,
        "robot0_.*": 0.0,
    }
    # sweep result: fingertips land z=0.731 (9mm above keys, over y~0.66) -> a
    # ~1cm finger curl presses. Used for the easy-song demo (right hand only).
    # mirror of the (working) left reach-down pose: the old pose (lift -0.5 /
    # elbow 1.2 / wrist1 -1.2) left the right hand 30cm ABOVE the keys and 0.5m
    # off in +Y, so IK couldn't converge (126mm median err). Copy the left arm's
    # reach-down joints (it lands fingertips at keyboard height) with the
    # shoulder_pan sign flipped so it reaches toward center from the +Y base.
    right_ready_pose = {
        "shoulder_pan_joint": -0.275,
        "shoulder_lift_joint": -1.20,
        "elbow_joint": 2.00,
        "wrist_1_joint": -2.20,
        "wrist_2_joint": -1.570,
        "wrist_3_joint": 0.0,
        "robot0_.*": 0.0,
    }

    def __post_init__(self):
        # --- SLIDER embodiment: swap both robots to the 26-DoF slider hand and force
        # the arm_ik_follow control path (math positions, RL presses). per_arm=26. ---
        per_arm = PER_ARM_DOF
        if self.use_slider:
            per_arm = 26
            self.left_robot_cfg = SHADOW_SLIDER_CFG.replace(
                prim_path="/World/envs/env_.*/LeftRobot")
            self.right_robot_cfg = SHADOW_SLIDER_CFG.replace(
                prim_path="/World/envs/env_.*/RightRobot")
            self.arm_ik_follow = True       # reuse the "math positions / RL presses" path
            # NB: do NOT force fold_to_reach here -- the slider moves across the WHOLE
            # keyboard, so folding full-keyboard songs into the 8-key windows just creates
            # dense clusters that force the hand to mash. Respect cfg.fold_to_reach (default
            # True for back-compat; set False / --no_fold for sparse real-keyboard play).
            self.action_space = 2 * per_arm   # 52 (slider cols are zero-scaled in env)
            # optionally stiffen the slider so its PD reaches the IK target within ONE
            # control step (tighter sub-key placement + deeper press -> higher recall/prec).
            if self.slider_stiffness > 0:
                import math as _mm
                for _rc in (self.left_robot_cfg, self.right_robot_cfg):
                    _a = _rc.actuators["slider"]
                    _a.stiffness = self.slider_stiffness
                    _a.damping = 2.0 * _mm.sqrt(self.slider_stiffness)
                    _a.effort_limit = max(_a.effort_limit, self.slider_stiffness)
        # --- compute observation size from the feature flags (single source) ---
        obs = (
            2 * per_arm * 2                        # both arms pos+vel
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
        self.piano_cfg.init_state.pos = self.piano_pos
        self.piano_cfg.init_state.rot = getattr(self, "piano_rot", (1.0, 0.0, 0.0, 0.0))
        if self.key_damping > 0:
            self.piano_cfg.actuators["keys"].damping = self.key_damping
        if self.use_slider:
            # fingers point straight DOWN at rot (0,1,0,0)=180deg about X (native +Z).
            # Left hand over the low (left-window) keys, right over the high keys; each
            # slider_y (+/-0.6) covers its window. Joint pose: sliders & fingers at 0.
            x, z = self.slider_hand_x, self.slider_hand_z
            py = self.piano_pos[1]
            self.left_robot_cfg.init_state.pos = (x, py + 0.25, z)
            self.right_robot_cfg.init_state.pos = (x, py + 0.85, z)
            # base (0,1,0,0)=180deg about X (fingers straight down); tilt about Y (lateral)
            # so the fingers angle forward over the keys -> a strike, not a palm-mash.
            import math as _m
            _t = self.slider_hand_tilt
            _rot = (0.0, _m.cos(_t / 2), 0.0, _m.sin(_t / 2))   # (0,1,0,0) ⊗ tilt-about-Y
            self.left_robot_cfg.init_state.rot = _rot
            self.right_robot_cfg.init_state.rot = _rot
            self.left_robot_cfg.init_state.joint_pos = {"slider_.*": 0.0, "robot0_.*": 0.0}
            self.right_robot_cfg.init_state.joint_pos = {"slider_.*": 0.0, "robot0_.*": 0.0}
        else:
            self.left_robot_cfg.init_state.pos = self.left_base_pos
            self.right_robot_cfg.init_state.pos = self.right_base_pos
            # face the piano: rotate the bases (default reach is +X; arm_base_rot turns
            # them so the arm reaches toward the piano at -X).
            _br = getattr(self, "arm_base_rot", (1.0, 0.0, 0.0, 0.0))
            self.left_robot_cfg.init_state.rot = _br
            self.right_robot_cfg.init_state.rot = _br
            # piano-ready default arm pose (fingertips resting on the keys); the two
            # arms differ slightly so each has its own tuned pose.
            self.left_robot_cfg.init_state.joint_pos = dict(self.left_ready_pose)
            self.right_robot_cfg.init_state.joint_pos = dict(self.right_ready_pose)
