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
    # 0 = one episode covers the LONGEST loaded song (2026-09-12; was a fixed
    # 30 s, so a 64 s song never trained past its first half).
    episode_length_s: float = 0.0
    # Reset each env at a uniformly random song step instead of step 0, so a
    # PPO batch from 64 envs samples 64 different places in the song rather
    # than 64 copies of the same bar (the envs otherwise run in lockstep from
    # song start to song end forever). The episode still ends at song end.
    random_song_start: bool = True
    random_start_min_steps: int = 40      # never start within 2 s of the end
    action_space: int = 2 * PER_HAND_ACT + 1             # 42 (rail + hand actuators) + sustain pedal
    # --- SUSTAIN PEDAL (2026-09-12, RoboPianist-style) ---
    # One extra action (the LAST dim): pedal is DOWN while it is > 0. While
    # down, a sounding key keeps sounding after the finger lifts (the hammer
    # latch does not release); when the pedal comes up, keys that are
    # physically released stop. No pedal goal in the MIDI: the policy decides
    # when to pedal (wrong keys kept ringing still cost the false-press
    # penalty). Without it, held bass notes that overlap far-away notes of the
    # same hand (nettspend keys 13/28/44 = 35% of goal steps) are unplayable.
    sustain_pedal: bool = False   # v14: replaced by score_sustain (per-note sustain from the score)
    # v14 (2026-09-27): a struck GOAL note keeps ringing until its goal ends, even
    # after the finger leaves; non-goal keys stop when released. Measured on v13
    # @2300: the all-or-nothing pedal was lifted 102x/64 s as a 'clear' button
    # and silenced 50 of the 108 early-cut notes; 18% of goal key-steps were
    # lost inside struck notes. The score decides what is held, not the policy.
    score_sustain: bool = False   # superseded by hold_per_finger (policy-decided sustain)
    # v14 (user request 2026-09-27): sustain is a POLICY decision, per finger. The
    # last NUM_FINGERS action dims are hold bits: while hold_f > hold_threshold,
    # the key finger f most recently STRUCK keeps ringing after the finger lifts
    # (any key, so a held wrong note still pays the false-press penalty). The
    # striker is the fingertip within hold_owner_tol laterally and lowest over
    # the key at the strike. Obs adds per finger: holding a ringing note, that
    # note is a goal now, steps until its goal ends (/ego_time_cap).
    hold_per_finger: bool = True
    hold_threshold: float = 0.5
    hold_owner_tol: float = 0.015     # m, lateral tolerance for striker identification
    hold_goal_weight: float = 0.3     # reward: hold == (owned key is a goal), per ringing owned key
    # Pedal engages only when the action exceeds this. At 0.0 the Gaussian
    # exploration noise (std ~0.5 around a mean near 0) flipped it every
    # other step: random sustain kept wrong keys ringing, the false-press
    # penalty rose, and the policy learned to press LESS (run
    # nettspend_pedal_a100: pedal down 51% of steps, 61% of onsets missed).
    pedal_threshold: float = 0.5
    # PEDAL GOAL (stand-in for a MIDI CC64 track, which this song lacks):
    # pedal_goal[t] = 1 when some goal note active at t cannot be held by a
    # finger -- its hand's active keys span more than pedal_goal_span metres
    # (a hand spans ~0.135 m thumb tip to little tip). Rewarded with
    # pedal_goal_weight * [pedal_down == pedal_goal] and shown in the obs, so
    # the policy gets a direct signal for WHEN to pedal.
    # 0.07 (v11, 2026-09-25; was 0.14): with 0.14 the pedal was only requested for
    # spans > 14 cm, but the little finger leaves key 28 for keys 6-9 cm away and
    # 12 of its 22 onsets died 1-6 steps after the strike with the pedal UP
    # (measured on v8 @1200). 0.07 requests the pedal on 87% of key-28 goal steps
    # (was 38%); pedal-goal-on rises from 35% to 75% of the song.
    pedal_goal_span: float = 0.07
    pedal_goal_weight: float = 0.3
    observation_space: int = 0                           # computed in __post_init__
    seed: int = 0

    # --- observation mode ---
    # "ego"    -- egocentric, position-invariant (2026-09-10): each hand sees
    #             only the keys under it, everything else is per-finger /
    #             per-hand relative to where the hand is. ~314 dims regardless
    #             of keyboard coverage. Layout (PianoMjEnv._get_obs_ego):
    #               hand qpos (+ qvel if ego_hand_vel)   50 (100)
    #               rail position per hand                2
    #               all 88 key angles (ego_all_keys)     88
    #               goal piano roll, goal_lookahead steps
    #                 x 88 keys (ego_piano_roll)        880
    #               per hand, ego_keys nearest keys:
    #                 dy to palm, angle, vel, sounding   2 x K x 4  (K=12 -> 96)
    #               per hand, dy palm -> next assigned note  2
    #               per finger: target xyz - tip xyz     30
    #               per finger: press-now flag           10
    #               per finger: steps to next onset,
    #                           steps to current release 20
    #               per hand, next ego_upcoming notes:
    #                 (dy to palm, steps to onset)       2 x U x 2 (U=3 -> 12)
    #               previous action (obs_prev_action)    42
    # "global" -- the per-key layout below (all observed keys, goal lookahead).
    obs_mode: str = "ego"
    ego_keys: int = 12            # keys observed per hand (nearest the palm)
    ego_upcoming: int = 3         # upcoming notes per hand for travel planning
    ego_time_cap: int = 40        # steps (2 s); timings are clipped and /cap
    obs_prev_action: bool = True
    # 2026-09-12: bring the ego input closer to RoboPianist / RP1M. Those
    # policies see joint POSITIONS only (no velocities), the full 88-key state,
    # and a binary goal piano roll over the lookahead; ours had hand velocities
    # and a 12-key window per hand instead.
    ego_hand_vel: bool = False    # hand joint velocities (was always on)
    ego_all_keys: bool = True     # all 88 key angles, absolute key order
    ego_piano_roll: bool = True   # goal[t : t+goal_lookahead] x 88, binary

    # --- observation features ("global" mode; assembled in PianoMjEnv._get_obs) ---
    # Layout (2026-09-09 slim-down; was 1216 dims, 880 of them the goal
    # lookahead over all 88 keys -- 72 of every 88 always zero once the song is
    # folded into the two hand windows):
    #   hand qpos+qvel (100) | fingertip xyz (30) | key angles (K) |
    #   key vel (K) | key sounding (K) | goal lookahead (L*K) |
    #   target fingertip xyz (30) | [goal SDF (K)]
    # K = number of OBSERVED keys (see obs_reachable_keys_only).
    obs_fingertip_pos: bool = True    # 10x3 fingertip world pos
    obs_finger_targets: bool = True   # 10x3 reference fingertip targets
    # Slice every per-key chunk to the union of left/right key windows (16 keys)
    # instead of all 88. Only meaningful with fold_to_reach; falls back to 88.
    obs_reachable_keys_only: bool = True
    # (K) key joint velocities: the hammer gate needs angle AND downward speed,
    # so without these the policy can't tell a slow silent press from a strike.
    obs_key_vel: bool = True
    # (K) hammer-gated SOUNDING latch -- the exact quantity reward/F1 score.
    # Hysteretic (rings until the key springs back above key_release_frac), so
    # not derivable from instantaneous angle+vel. A digital piano reports it as
    # MIDI note-on/off, so the ACTOR may see it (not privileged).
    obs_key_sounding: bool = True
    # (K) analytic SDF of the current goal; off = derivable from goal + targets.
    obs_goal_sdf: bool = False

    # --- critic-only (privileged) observations: asymmetric actor-critic ---
    # The vec env emits a second obs group "critic_priv" = sim ground truth the
    # actor never sees; rsl_rl concatenates it after the policy obs for the
    # critic (obs_groups critic = ["policy", "critic_priv"]). Only exists in
    # training, so it may read anything MuJoCo knows.
    critic_obs: bool = True
    # (10) net contact force magnitude on each fingertip body, order
    # [L th,ff,mf,rf,lf, R th,ff,mf,rf,lf], as |F| / critic_tip_force_clip in [0,1]
    critic_obs_tip_forces: bool = True
    critic_tip_force_clip: float = 20.0   # N; steady press 1-5 N, strike peaks ~20 N (measured)
    # (1) hand-vs-hand contact flag (any contact between an L_ and an R_ body)
    critic_obs_collision: bool = True
    state_space: int = 0                  # policy obs + critic extras; computed

    # --- task / songs ---
    midi_path: str = "data/midi/song.mid"
    # Playback tempo of midi_path: 1.0 = as written, 0.5 = half speed (each
    # note spans twice the control steps; control_dt is unchanged). For songs
    # the rig physically cannot play at tempo. "jeans - 2hollis" (2026-09-27):
    # 143 bpm 16ths = one onset every 2 steps (0.1 s), and the left hand must
    # hop 15-19 cm between the 16/17 bass notes and the 28/31 line 69 times with
    # 2 steps to do it -- the rail needs ~0.2 s (4 steps) to move and settle
    # (see stagger_steps). Train it at 0.5 first; ignored for songs_npz.
    song_speed: float = 1.0
    control_dt: float = CONTROL_DT
    goal_lookahead: int = GOAL_LOOKAHEAD
    songs_npz: str | None = None      # multi-song goal bundle (goals/lens/names)
    max_songs: int = 0                # 0 = all songs in the bundle
    song_offset: int = 0              # held-out eval split support

    # --- reach: how far each hand's rail can slide (m, +/- from its base) ---
    # 0.12 (the Isaac-era default, kept as --legacy_reach) covers ~10 keys per
    # hand and leaves the middle of the keyboard unreachable, which is why
    # fold_to_reach existed. 0.32 lets each hand cover its whole half of the
    # 1.22 m keyboard (bases sit at y = +/-0.30). arm_action_scale maps the
    # policy's [-1, 1] rail action onto the same range.
    rail_limit: float = 0.32
    # rail servo gains (position actuator on the 4.3 kg hand+carriage). The
    # Isaac-era 1200/120/500 settled a 30 cm step in ~250 ms -- as long as the
    # song's short notes. 6000 N/m critically damped (2*sqrt(k*m) ~ 320) with a
    # 2 kN force limit settles in ~100 ms; a real linear rail does this easily.
    rail_stiffness: float = 6000.0
    rail_damping: float = 320.0
    rail_force: float = 2000.0

    # --- fold a wide song into each hand's reachable key window ---
    # OFF by default (2026-09-10): folding octave-wraps every note into the two
    # 8-key windows below, which rewrites the song (nettspend: 359 notes / 17
    # pitches -> 236 notes / 11 pitches in two clusters 4 octaves apart). With
    # full rail travel the real pitches are reachable, so train on the song as
    # written. Only turn on with --legacy_reach.
    fold_to_reach: bool = False
    left_key_window: tuple[int, int] = (19, 26)      # only used when folding
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
    hand_fixed_z: float = 0.83        # pose G (2026-09-23); was 0.88 with straight fingers
    # Extra world-X push toward the piano applied AFTER the mount's white-key
    # self-calibration (see mjcf/scene.py build_scene_spec). 0.03 puts all four
    # long fingers onto the black-key surface when curled down to press; at 0
    # the little finger and thumb cannot reach a black key (2026-09-23). The X
    # component of left/right_base_pos is a dead knob -- calibration cancels it.
    tip_shift_extra: float = 0.065    # pose G: a plain press from the hover lands on the white press line
    left_base_pos: tuple = (0.82, -0.30, 0.88)
    right_base_pos: tuple = (0.82, 0.30, 0.88)

    # --- action scaling: target = ready + scale * action (action in [-1,1]) ---
    arm_action_scale: float = 0.32    # rail travel scale (m); == rail_limit so
    #                                   action +/-1 reaches the end of the rail
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
    # ON by default (2026-09-10): with the policy driving the rail from scratch
    # on the full keyboard, both hands parked where they started and learned
    # only the keys under them (61% of onsets never attempted at iter 700;
    # keys 30 cm away at recall 0.00). Reaching a far key needs a consistent
    # rail action over dozens of steps, which Gaussian exploration never finds.
    # The servo does the coarse travel; the policy keeps a +/- rail_residual
    # residual for fine placement (0 = fingers only, the RoboPianist split).
    rail_follow: bool = True
    rail_residual: float = 0.05       # m; policy residual around the servo target
    arm_smooth: float = 0.0           # EMA on the rail servo target (0 = instant;
    #                                   was 0.8 = another ~250 ms of lag)
    arm_lookahead: int = 5            # (legacy centroid servo only)
    # LEAVE-EARLY servo (2026-09-11, default): the hand targets its CURRENT
    # notes, but switches to the NEXT onset's notes as soon as the time left
    # before that onset is <= the travel time it needs (|dy| / rail_speed +
    # rail_settle_s). With no current notes it pre-positions immediately. The
    # old centroid servo only started moving when a note entered a 250 ms
    # lookahead (and then smoothed the move): 34% of hits arrived >= 3 steps
    # late and the missed notes were exactly the 250 ms ones.
    rail_leave_early: bool = True
    # v14: with the seq planner, the servo targets the PLANNED palm (bank.seq_palm)
    # instead of averaging finger offsets (which parked 1.6-2.8 cm short on
    # stretched chords: the 44+56 octave was 0/14 at v13 @2300).
    rail_follow_plan: bool = True
    # Servo centres the hand on the hand's MOST RECENTLY STARTED notes only
    # (ties -> all notes that began on that step). Earlier notes still
    # sounding are holdovers the pedal carries. Averaging over ALL active
    # notes put every finger 2-4.5 cm off whenever a new note was far from a
    # held one (nettspend key 28: 18/22 onsets never had a finger over it).
    rail_target_newest: bool = True
    # Don't leave for the next onset until the current onset group has had
    # this many steps to be struck (the pedal carries it afterwards). Without
    # it, a note followed 300 ms later by a far chord was abandoned before the
    # finger reached the key (nettspend key 28 via the middle finger: hand
    # already 5 cm away at the onset, 4/4 missed).
    rail_min_dwell_steps: int = 0
    rail_speed: float = 1.5           # m/s assumed travel speed for the lead time
    rail_settle_s: float = 0.10       # s added to the lead for the servo to settle
    lane_clamp: bool = True           # each rail target stays in its own half

    # --- fingering plan (which finger the reward/obs targets for each note) ---
    # "heuristic": lowest note -> lowest finger in hand order. With one note per
    #   hand at a time this sends EVERY right-hand note to the thumb (and every
    #   left-hand note to the little finger), so the hand must slide the rail
    #   for each key change -- measured on nettspend: right-hand recall 0.43,
    #   key 63 recall 0.00 (5 keys from where the thumb parks).
    # "ot": RP1M-style nearest-finger assignment (dexsim.piano.fingering.
    #   plan_fingering_ot, needs scipy): each note goes to the finger that has
    #   to move the least, so a hand covers 4-5 adjacent keys without moving.
    # "hand": hand-RELATIVE assignment (2026-09-10, default): per hand, centre
    #   the hand on its notes and match notes to fingers by their physical
    #   offset from the palm. The only planner consistent with a moving hand:
    #   with fold off, "ot" and "heuristic" hand one chord to fingers 30 cm
    #   apart (their homes are absolute keys), which no hand can play.
    # The thumb cannot depress a key on this embodiment: at its joint limit
    # (THJ3 = 1.22 rad) the fingertip reaches only the key TOP (+0.0 cm) with
    # no travel left, and the action scale (0.8 rad) cannot even command that
    # limit. Any note assigned to a thumb is unplayable, so withhold them.
    # Measured false: a coordinate sweep of the thumb's own actuators from the
    # ready pose reaches 0.45 cm BELOW the key top, so the thumb can press. It
    # starts 5.7 cm up against 2.0-2.3 cm for the other fingers, which is why it
    # rarely gets there. Excluding it moved servo placement 89.9% -> 87.5%.
    exclude_thumbs: bool = False
    # Stagger a chord when the best hand position still leaves a finger further
    # than this from its key (metres; 0 disables). hand_span only catches chords
    # too WIDE for the hand -- this catches chords whose spacing the fingers
    # cannot match at any position, which is what the little finger kept losing.
    hand_tol: float = 0.0
    # ARPEGGIATE: an onset group wider than the hand (e.g. the 44+56 octave the
    # right hand plays 14x in nettspend, 16.5 cm vs a 13.5 cm max span) is
    # split: the half nearer the hand plays at t, the other half this many
    # steps later with the pedal carrying the first. 2 steps (0.1 s) was at the
    # rail's physical limit (1.5 m/s + 0.1 s settle) and every staggered 56
    # onset went unstruck (2026-09-23); 4 steps = 0.2 s is reachable.
    stagger_steps: int = 4
    # Hand split boundary for the planner and the pedal goal: keys >= this go to
    # the RIGHT hand. 44 = keyboard middle (old behaviour). 40 (v11, 2026-09-25):
    # key 40 (C#4) was assigned to the left thumb, which cannot reach the keyboard,
    # yet the 0.750 model's hits on it came from the RIGHT index finger (11/12).
    hand_split_key: int = 40
    # Extra assignment cost (metres) for handing a key to a finger other than the
    # one that already plays it. OFF: first-come ownership consolidated keys onto
    # whichever finger touched them first regardless of competence -- at 0.01 it
    # gave all 22 notes of key 28 to the little finger (21% press rate, then 0/22)
    # and moved key 33 off the middle finger (92%) onto the ring finger (20%).
    sticky_weight: float = 0.0
    # Extra cost (metres) a finger pays at 100% usage share, so work spreads
    # across fingers instead of collapsing onto middle+ring. 0 = off.
    balance_weight: float = 0.0
    fingering_method: str = "hand"
    # --- "seq" planner (2026-09-27, dexsim.piano.fingering_seq): whole-song
    # Viterbi over palm positions, then fingers by offset. Costs in metres of
    # finger-to-key error. None = the planner's defaults.
    seq_reach_tol: float | None = None       # free lateral finger stretch (0.010)
    seq_reach_max: float | None = None       # beyond this a long finger cannot take the key (0.030)
    seq_thumb_reach_max: float | None = None # same for the thumb (0.045)
    seq_travel_weight: float | None = None   # cost per metre of palm travel (0.5)
    seq_move_cost: float | None = None       # flat cost per palm move (0.01)
    seq_drop_cost: float | None = None       # cost of an unassignable note -> staggered (0.15)
    seq_min_hand_sep: float | None = None    # palms may not come closer than this (0.20)
    seq_switch_penalty: float | None = None  # same finger, different key within window (0.01)
    seq_switch_window: int | None = None     # steps (10)
    # v15 (2026-09-28): HARD feasibility. reach_check.py measured, through the real
    # env.step path at pose G, which (finger, key) pairs can sound under the policy's
    # maximum press: thumbs sound NOTHING (tip 2.5-6.5 cm above every key), long
    # fingers sound almost everything in rail range. The planner never assigns a
    # forbidden pair (the note is dropped/staggered or given to another finger/hand).
    seq_reach_mask: str | None = "reach_mask_poseG.npy"   # (10, 88) bool npy; cwd or tree root
    seq_exclude_thumbs: bool = True
    # 2026-09-12: ONLINE fingering reward (RP1M). Each step, match the live
    # fingertip positions to the keys due now by min-cost assignment and
    # reward THAT matching's distances, instead of pulling each finger to its
    # pre-planned table entry. The table still drives the obs and rail servo.
    fingering_online: bool = False   # v14: shaping pulls the PLANNED finger (the thumb got 7 strikes/song under live matching)
    # --- fingering / press tweaks ---
    remap_thumb_to_middle: bool = False
    idle_finger_curl: float = 0.0     # rad: curl NON-assigned fingers up (rail_follow)
    # --- LEAP CURRICULUM (v12, 2026-09-25) ---
    # Measured on the 0.750 model: 93% of false strikes happen while the hand's rail
    # is moving (median 0.65 m/s), 60% of them 3+ cm from any goal key -- fingers
    # dragged across keys in transit. Scaffold-then-penalise:
    #   * while a hand's rail speed > leap_move_speed, add leap_lift rad of curl to
    #     ALL of that hand's flex actuators (lifts the fingers clear), scaled by
    #     (1 - leap_s);
    #   * false key-steps struck under a travelling hand cost an extra
    #     leap_penalty_weight * leap_s each;
    #   * leap_s ramps 0 -> 1 over leap_anneal_steps once the recall EMA passes
    #     anneal_recall_gate (same gate as the false-press anneal).
    # The scaffold is TRAINING-ONLY (random_song_start envs); deterministic evals
    # and playback run without it, so the score is always the policy's own lift.
    leap_lift: float = 0.35
    leap_move_speed: float = 0.10
    leap_penalty_weight: float = 0.5
    leap_anneal_steps: int = 2000
    start_finger_curl: float = 0.0    # rad: curl ALL flex joints in the ready pose

    # --- reward weights (PianoMime/RoboPianist composite; == Isaac cfg) ---
    key_press_weight: float = 2.0
    false_press_weight: float = 1.0
    energy_weight: float = 0.0005
    idle_hover_weight: float = 0.2        # was 0: reward idle fingers for staying
    #                                       above the keys (zero-action rollouts
    #                                       brush NOTHING; the false presses are
    #                                       the policy hammering and never lifting)
    idle_hover_close: float = 0.005
    idle_hover_margin_mult: float = 5.0
    idle_hover_z_only: bool = True
    fingering_weight: float = 1.0
    onset_weight: float = 2.0
    jerk_weight: float = 0.3          # was 0.1: logged jerk ROSE during training
    # ACTION LOW-PASS: commanded targets are an EMA of the policy output with
    # this time constant (s). 0 = off. The raw policy emits a new target every
    # 50 ms with jumps up to the full action scale -> visible twitching.
    action_filter_tau: float = 0.10

    # --- per-key reward weighting: point the reward at what is still missed ---
    # The press/onset rewards are paid per step, so a key that is a goal on
    # half the song earns ~8x more than a key that appears in a few short
    # notes; the policy camps on the frequent keys (nettspend: keys 20/68 at
    # recall 0.97, keys 63/66 at 0.01-0.11). Fix: weight each goal key.
    #   "none"     -- all goal keys weight 1 (original behaviour)
    #   "inv_freq" -- static: w_k ~ 1 / (steps key k is a goal), times
    #                 1 / note_len ** key_weight_note_len_pow, normalized so
    #                 the mean weight over all goal key-steps is 1
    #   "adaptive" -- inv_freq times (floor + (1-floor) * (1 - recall_ema_k)):
    #                 keys the policy already plays devalue toward the floor,
    #                 keys it still misses keep full weight (focal-loss style)
    key_weight_mode: str = "adaptive"
    key_weight_note_len_pow: float = 0.5   # 0 = pure per-step, 1 = pure per-note
    # 0.10 (v11; was 0.25): weak-key curriculum -- a mastered key keeps only 10% of
    # its static weight so the reward concentrates on the keys still being missed.
    key_weight_floor: float = 0.10         # a mastered key keeps this fraction
    key_weight_beta: float = 0.995         # per-key recall EMA (per env)
    key_weight_max: float = 8.0            # clip on the static weight
    # phase the weighting in: effective w = 1 + (w - 1) * s, with
    # s = clip(recall_ema / anneal_recall_gate, 0, 1). Early on every key pays
    # the same (the frequent keys are the easiest to discover, so cutting their
    # pay from step one just slows the start); by the time recall reaches the
    # gate the full weighting is on and pulls the policy toward the rare keys.
    key_weight_ramp: bool = True
    # The inverse-frequency prior must only ever BOOST rare keys, never starve
    # frequent ones: nettspend key 28 (the most-played key, recall 0.02 in
    # every run) was getting weight 0.36 while unlearned. Static weights are
    # clamped to >= 1 before the adaptive factor applies.
    key_weight_min_static: float = 1.0
    # per-FINGER adaptive factor on top: a goal key's weight is also scaled by
    # (floor + (1-floor) * (1 - recall_ema of the finger assigned to it)), so
    # a finger the policy never uses (v5: both little fingers, recall ~0.01-0.1)
    # is worth more until it starts landing notes.
    finger_weight_adaptive: bool = False
    finger_weight_floor: float = 0.4
    # onset bonus window (steps either side of the true onset). The exact
    # 1-step window paid ~0.02/step on average and never steered anything.
    onset_window_steps: int = 4   # was 2; widened with stagger_steps so the rolled note is still paid

    # --- recall-gated annealing (press-discovery curriculum; == Isaac cfg) ---
    # Hold the false-press penalty low (and energy at 0) so pressing gets
    # discovered, then ramp both to their cfg values over anneal_steps once the
    # per-env recall EMA crosses the gate. Monotonic; pauses if recall dips.
    anneal_false_press: bool = False
    false_press_start: float = 0.15       # tried 0.5 (2026-09-10): on the full
    #                                       keyboard the policy got charged for
    #                                       exploring before finding any press;
    #                                       3x slower start. Keep it low and let
    #                                       the anneal ramp it once recall > gate.
    anneal_recall_gate: float = 0.25      # was 0.5: with the rail servo, reach no
    #                                       longer depends on exploration, and at
    #                                       0.5 the policy sat at precision 0.34
    #                                       (2000+ false key-steps) waiting for
    #                                       the penalty to ramp. Ramp from 0.25.
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
    key_strike_vel: float = 0.10          # only used by sounding_gate="hammer"
    # 2026-09-12: "position" (default) = RoboPianist semantics, a key sounds
    # once it is depressed past key_struck_frac of the sound angle, no speed
    # requirement; still hysteretic (rings until it lifts above
    # key_release_frac). "hammer" = the velocity-gated latch above, which the
    # measurements show fires on only 12/30 deliberate presses -- the reward
    # and F1 disagreed with what the hand actually did.
    sounding_gate: str = "position"
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
    # hands. "Pose G" (user-approved 2026-09-23, replacing the straight-finger
    # pose): palm at hand_fixed_z = 0.83 (5 cm lower), wrist flattened to
    # robot0_WRJ0 = -0.20 (was 0.45), long fingers curled 30 deg at MCP (J2)
    # and PIP (J1), thumb turned down (THJ4 +60 deg, THJ3 70 deg). Measured:
    # every fingertip incl. the thumb hovers 3.8-4.1 cm above the white key
    # tops, and with tip_shift_extra = 0.065 all five fingers reach every white
    # press point with zero error and the four long fingers every black one
    # (thumb 1.3 cm off black centres). The old pose could not put the thumb on
    # any key (bottomed out 0.4 cm above white tops) nor the little finger on
    # a black key. Do NOT change without an explicit request. See CLAUDE.md
    # and docs/MUJOCO.md. (Keys are regex patterns over per-hand joint names,
    # first match wins -- keep the catch-all LAST.)
    # ===============================================================================
    left_ready_pose: dict = field(default_factory=lambda: {
        "railJoint": 0.0,
        "robot0_WRJ0": -0.20,  # wrist tilt, range [-0.70, 0.49]
        "robot0_WRJ1": 0.13,   # range [-0.49, 0.14]
        "robot0_THJ4": 1.05,   # thumb turned down (+60 deg)
        "robot0_THJ3": 1.22,   # (70 deg)
        "robot0_(FF|MF|RF|LF)J[12]": 0.5236,   # 30 deg claw at MCP + PIP
        "robot0_(?!WRJ).*": 0.0,
    })
    right_ready_pose: dict = field(default_factory=lambda: {
        "railJoint": 0.0,
        "robot0_WRJ0": -0.20,
        "robot0_WRJ1": 0.13,
        "robot0_THJ4": 1.05,
        "robot0_THJ3": 1.22,
        "robot0_(FF|MF|RF|LF)J[12]": 0.5236,
        "robot0_(?!WRJ).*": 0.0,
    })

    def obs_key_indices(self) -> list[int]:
        """Key indices (0..87) every per-key observation chunk is sliced to:
        the union of the two inclusive hand windows when obs_reachable_keys_only
        AND fold_to_reach, else all 88."""
        if self.obs_reachable_keys_only and self.fold_to_reach:
            l0, l1 = self.left_key_window
            r0, r1 = self.right_key_window
            return sorted(set(range(l0, l1 + 1)) | set(range(r0, r1 + 1)))
        return list(range(NUM_KEYS))

    def n_obs_keys(self) -> int:
        return len(self.obs_key_indices())

    def critic_extra_dim(self) -> int:
        """Number of privileged features in the "critic_priv" obs group."""
        if not self.critic_obs:
            return 0
        n = 0
        if self.critic_obs_tip_forces:
            n += NUM_FINGERS
        if self.critic_obs_collision:
            n += 1
        return n

    def ego_obs_dim(self) -> int:
        K, U = int(self.ego_keys), int(self.ego_upcoming)
        n = (2 * PER_HAND_DOF              # hand qpos
             + (2 * PER_HAND_DOF if self.ego_hand_vel else 0)   # hand qvel
             + 2                            # rail positions
             + (NUM_KEYS if self.ego_all_keys else 0)          # all key angles
             + (self.goal_lookahead * NUM_KEYS if self.ego_piano_roll else 0)
             + 2 * K * 4                    # window keys: dy, angle, vel, sounding
             + 2                            # dy palm -> next assigned note
             + NUM_FINGERS * 3              # target - tip
             + NUM_FINGERS                  # press now
             + NUM_FINGERS * 2              # steps to onset, steps to release
             + 2 * U * 2)                   # upcoming notes: dy, steps
        if self.sustain_pedal:
            n += 2                          # pedal state, pedal goal (now)
        if getattr(self, "hold_per_finger", False):
            n += NUM_FINGERS * 3            # holding, owned key is goal, steps to its goal end
        if self.obs_prev_action:
            n += self.action_space
        return n

    def __post_init__(self):
        self.action_space = (2 * PER_HAND_ACT + (1 if self.sustain_pedal else 0)
                             + (NUM_FINGERS if getattr(self, "hold_per_finger", False) else 0))
        if self.obs_mode == "ego":
            self.observation_space = self.ego_obs_dim()
            self.state_space = self.observation_space + self.critic_extra_dim()
            return
        per_arm = PER_HAND_DOF
        K = self.n_obs_keys()
        # observation size from the feature flags (single source of truth);
        # order mirrors PianoMjEnv._get_obs
        obs = 2 * per_arm * 2                       # both hands qpos+qvel
        if self.obs_fingertip_pos:
            obs += NUM_FINGERS * 3
        obs += K                                    # current key angles
        if self.obs_key_vel:
            obs += K
        if self.obs_key_sounding:
            obs += K
        obs += self.goal_lookahead * K              # upcoming note goals
        if self.obs_finger_targets:
            obs += NUM_FINGERS * 3
        if self.obs_goal_sdf:
            obs += K
        self.observation_space = obs
        self.state_space = obs + self.critic_extra_dim()

    def to_dict(self) -> dict:
        return asdict(self)
