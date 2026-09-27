# dexsim — Claude instructions

**This repo is MuJoCo only.** The stack is `source/dexsim/mjcf/` +
`source/dexsim/tasks/piano_mj/` + `scripts/mj/`, running in the plain `.venv`
(mujoco + rsl-rl ≥5.x + torch). An earlier implementation on a different
simulator is archived on the `isaac-legacy` branch; do not reintroduce any of
it (no USD assets, no arm code). See `docs/MUJOCO.md` for the physics
decisions (key damping, fingertip sites, mount calibration — each documented
with its reason; don't "fix" them).

## 🔒 LOCKED: the constant static hand pose — DO NOT EDIT

`left_ready_pose` / `right_ready_pose` in
`source/dexsim/tasks/piano_mj/piano_mj_env_cfg.py` are the constant ready
pose — **"pose G"**, user-approved 2026-09-23: `railJoint = 0`, wrist
`robot0_WRJ0 = -0.20` / `robot0_WRJ1 = 0.13`, long fingers curled 30° at
MCP+PIP (`(FF|MF|RF|LF)J[12] = 0.5236`), thumb turned down (`THJ4 = 1.05`,
`THJ3 = 1.22`), palm at `hand_fixed_z = 0.83`, `tip_shift_extra = 0.065`.
Every fingertip (thumb included) hovers ~4 cm above the keys and can reach
every white press point; the long fingers reach every black one. The
previous straight-finger pose (WRJ0 0.45, z 0.88) could not put the thumb on
any key or the little finger on a black key — see `docs/MUJOCO.md`. Do NOT
edit these unless the user explicitly asks in a new request.

## Task setup that matters

- **Versions.** The trained recipes are documented in `docs/README_v13.md`
  (whole-song `seq` planner + pose G) and `docs/README_v14.md` (per-finger
  HOLD actions, plan-following rail servo, planned-finger reward); the box
  patch/launch/verify scripts they came from are in `scripts/box/`. This
  tree IS that code (merged 2026-09-27) plus the v15 changes below.
- **v15 changes (2026-09-27, not yet trained, `docs/README_v15.md`).**
  (1) the `seq` planner honours `hand_split_key` (one-sided: keys >= 40 must
  go right; the right hand may still reach lower) -> key 40 lands on the
  right index instead of the left thumb (0.11-0.17 recall in v13/v14 vs 0.43
  on the right in v12). (2) `seq_keep_held`: an onset group with nothing for
  a hand no longer wipes that hand's held notes (unassigned key-steps 1370
  -> 985 on nettspend). (3) `--std_cap_start/end/final`: a decaying upper
  bound on the actor's action std (v12's learned std GREW 0.50 -> 0.67; the
  greedy policy scored 0.05 above the noisy one). Replaying v13/v14
  checkpoints needs `split=-1 keep_held=0` (diag) or
  `--hand_split_key -1 --seq_keep_held 0` (play).
- **song_speed.** cfg / `--song_speed` / diag `speed=` stretches a song
  that is too fast for the rig (0.5 = half tempo, control rate unchanged).
- **Rails.** `rail_limit = 0.32` m: each hand covers its half of the keyboard
  (left keys 0–53, right 33–87). `arm_action_scale` must equal `rail_limit` so
  the policy's ±1 rail action reaches the rail ends. `fold_to_reach` is OFF:
  songs train at their real pitches. `--legacy_reach` restores the old
  ±0.12 m rails + folding for checkpoints trained before 2026-09-10.
- **Fingering.** `fingering_method="hand"` is the cfg default, but every
  run since v13 trains with `--fingering seq` (`dexsim.piano.fingering_seq`):
  a whole-song Viterbi over palm positions, fingers by offset, dropped
  notes re-issued `stagger_steps` later. `hand` re-centres the palm every
  step (one finger ended up playing 74% of the right hand's onsets).
- **Rail servo.** `rail_follow=True` + `rail_leave_early`: a scripted servo
  does the travel (leaves for the next onset when time-left <= travel
  time), the policy keeps a `rail_residual` of 5 cm. Rail gains are stiff
  (6000 N/m, 2 kN). Policy-driven rails never learned to travel.
- **Sustain.** v14 (`hold_per_finger=True`, default): 10 HOLD actions, one
  per finger -- while hold_f > 0.5 the key finger f most recently struck
  keeps ringing after it lifts. `sustain_pedal=False` by default (the old
  all-or-nothing pedal was used as a clear button: 102 lifts / 64 s in
  v13). Actions: 42 actuators + 10 holds = 52 (+1 if the pedal is on).
- **PPO.** `entropy_coef` 0.001 (0.006 let the action std run 0.5 -> 1.5
  over 3000 iters and flattened F1).
- **Observation.** `obs_mode="ego"` (default, 1272 dims with the v14
  hold block; 1235 for v13 checkpoints with `hold_per_finger=False,
  sustain_pedal=True`): everything key-related is relative to the hand --
  the 12 keys nearest each palm (offset, angle, vel, sounding), per-finger
  target-minus-tip and timing, per-hand upcoming notes, hold state,
  previous action. `ego_finger_obs` (60 dims) and `ego_rail_obs` (2 dims,
  a duplicate of the rail qpos) are ON because every box checkpoint trained
  with them; `--no_ego_finger_obs --no_ego_rail_obs` gives the 1173-dim
  local layout of 2026-09-23. Sized in `PianoMjEnvCfg.ego_obs_dim`; the
  assembly order in `PianoMjEnv._get_obs_ego` must match it. Critic gets
  `critic_priv` via rsl_rl obs_groups -- asymmetric actor-critic.
- **Vec env.** Always train with `--workers N` (`PianoMjSubprocVecEnv`). The
  threaded env is GIL-bound at ~400 steps/s.
- **Metric.** Judge runs by deterministic-rollout F1
  (`scripts/mj/diag_rollout.py`), not by training-time `play/F1`, which is
  ~0.08 lower because of exploration noise. Report per-key recall when
  diagnosing: the policy tends to mash the most frequent key per hand.

## Rendering & audio

- Scene compiles in ~0.3 s; `play_piano_mj.py --video` renders in-process.
  `MUJOCO_GL=egl` on Linux with an NVIDIA driver; leave unset on macOS.
- `scripts/mj/midi_to_audio_video.py` synthesizes the exported played-notes
  MIDI and muxes it into the video (no fluidsynth needed).

## General

- `source env.sh` before anything.
- `source/dexsim/piano/` is sim-agnostic task logic (MIDI → goal schedule,
  fingering, rewards, key geometry). `dexsim.piano.geometry` is the single
  source of truth for key positions — don't hardcode them elsewhere.
- rsl-rl is ≥5.x: runner config is the dict in
  `source/dexsim/tasks/piano_mj/ppo_cfg.py`.
- `logs/`, `assets/mujoco_menagerie/` (auto-vendored) and `assets/mj/`
  (generated XML) are gitignored.
- GPU box workflow: `docs/A100_TRAINING.md`.
