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

    # --- residual RL over an IK reference (PianoMime) ---
    # If reference_path exists, the action is a residual on the precomputed IK
    # reference joint trajectory (action_scale * a + q_ref). If it's missing, the
    # env degrades gracefully to a residual on the static ready pose. Build one
    # with scripts/build_reference.py. None -> derive from the MIDI stem.
    reference_path: str | None = None
    use_reference: bool = False   # fixed-hands mode: arms hold the ready pose (no
    #   per-note IK trajectory needed); only the fingers move to press.

    # IK params (used by build_reference.py and any online IK)
    ik_damping: float = 0.05    # 0.02 diverges the LEFT arm at singular configs
    #   (145mm+ even on folded keys, while the right converged to 17mm). 0.05 is the
    #   stable singularity handling -> both hands converge on the reachable windows.
    ik_max_step: float = 0.05   # 0.12 overshoots/oscillates -> WORSE convergence
    #   (right 271mm vs 57mm at 0.05). Small step + many substeps converges tighter.

    # piano world pose (shift so the 1.22 m keyboard centers on Y=0, table height)
    piano_pos = (0.569, -0.606, 0.746)   # fit so LEFT-hand fingers rest on keys 19/20/26
    # robot base poses: elevated on pedestals just behind the keyboard so the
    # arms drape DOWN onto the keys. With the base at key height the wrist can't
    # get above the keys and the fingers end up below the keyboard; raising the
    # bases lets the hand come down onto the keys (fingers point -Z).
    left_base_pos = (-0.05, -0.30, 1.05)
    right_base_pos = (-0.05, 0.30, 1.05)

    # action scaling: targets = default + scale * action (action in [-1,1]).
    # Per-joint (see PianoEnv.joint_scale): the stiff arm joints (stiffness 6000)
    # explode under a large residual -> NaN, but the weak hand joints (stiffness 3)
    # need a generous range to travel between keys and actually press them. One
    # global scale can't serve both: 0.5 blew up the arm; 0.15 froze the fingers
    # (F1 stuck ~0.02). So scale the arm gently and the hand generously.
    arm_action_scale: float = 0.15    # 0.30 crashed physics (stiff arm joints blow
    #   up under a large residual -> PhysX hard crash). 0.15 is the stable max; the
    #   reference must instead be made precise so the arm barely needs to correct.
    hand_action_scale: float = 0.35   # was 0.6: too large -> with exploration noise the
    #   fingers flailed and struck ~23 keys at once (precision 3%, reward/key pinned at
    #   -2, learning stalled). 0.35 still travels between keys in a window but stops the
    #   mash. Raise again once the fingers reliably press single keys.
    # legacy global scale (still used by scripts/bc_pretrain.py); not used by the env
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
    # Instead of learning a 60-DoF residual on a precomputed q_ref (whose FingertipIK
    # arm trajectory diverges, capping zero-residual F1 at 0.03), drive the 12 arm DoF
    # ONLINE with WristPoseIK: each control step the well-posed palm-servo (proven to
    # reach within ~1cm across the whole keyboard, scripts/diag_wrist_ik.py) tracks the
    # per-hand fingering centroid, while the policy action is masked to the 48 finger
    # DoF only. No reference trajectory needed; the arm-blowup failure mode disappears
    # (the policy never touches the stiff arm joints). Set with freeze_arms=False and
    # use_reference=False. Supersedes freeze_arms (static hold) when both are set.
    arm_ik_follow: bool = False
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
    finger_ik_base: bool = False  # PARKED: also pose the fingers analytically (hand-only
    #   FingertipIK) instead of leaving them at ready pose for RL. A relative one-step DLS
    #   target can't drive the weak hand actuators (stiffness 3) the way it drives the stiff
    #   arm, so it made no measurable difference; finger pressing is the policy's job. Kept
    #   for experimentation (e.g. if paired with a stiffer hand or iterated-to-convergence IK).

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
    left_ready_pose = {
        "shoulder_pan_joint": -0.275,
        "shoulder_lift_joint": -0.525,
        "elbow_joint": 1.150,
        "wrist_1_joint": -1.275,
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
        "shoulder_pan_joint": -0.325,
        "shoulder_lift_joint": -0.675,
        "elbow_joint": 1.300,
        "wrist_1_joint": -1.350,
        "wrist_2_joint": -1.570,
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
