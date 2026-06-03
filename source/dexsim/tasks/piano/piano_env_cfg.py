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
    arm_ik_hover: float = 0.05   # m the servoed palm hovers above the key tops (matches
    #   diag_wrist_ik's 0.05, which converged the palm to 4-14mm). Fingers reach down from there.

    # reward weights (PianoMime/RoboPianist composite)
    key_press_weight: float = 2.0   # was 1.0: PRESSING the right key must dominate
    #   the (positioning) shaping, else the policy hovers near keys for finger
    #   reward and never presses (F1 flat ~0.04, reward/finger 0.44 >> press).
    # NOTE: the old tuning (0.5 best, 1.5 over-suppressed) was under the BUGGED
    # penalty that averaged misclicks over all 88 keys. reward.py now counts wrong
    # keys PER INTENDED NOTE, which is ~40-80x stronger at the same weight, so 0.75
    # here ≈ "one wrong note nearly cancels one right note". Retune off wandb
    # play/precision; drop toward 0.5 if recall collapses.
    false_press_weight: float = 0.5   # PUNISH wrong notes hard for PRECISION. Safe now
    #   ONLY because we BC-warm-start from the IK fingering expert (--bc_init): the policy
    #   begins in a press-the-right-key basin, so a strong false penalty sharpens precision
    #   instead of cratering reward/key to -2 (which is what happened from a COLD start).
    #   If recall collapses, drop toward 0.3.
    energy_weight: float = 0.0005
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
    key_strike_vel: float = 0.15   # was 0.10: nudged up so a finger merely brushing/passing
    #   over a key doesn't ring it -- only a deliberate downward strike counts. Cuts the
    #   accidental false-ring count that inflated keys_sounding to ~23. Kept moderate (not
    #   0.20) so a BC-placed finger can still ring with a small deliberate press.

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
