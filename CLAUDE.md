# dexsim — Claude instructions

**This repo is MuJoCo only.** The stack is `source/dexsim/mjcf/` +
`source/dexsim/tasks/piano_mj/` + `scripts/mj/`, running in the plain `.venv`
(mujoco + rsl-rl ≥5.x + torch). The earlier Isaac Lab implementation was
removed on 2026-09-10; it exists only in git history and on `master`. Do not
reintroduce Isaac Sim, Isaac Lab, USD assets, or UR10e arm code here. See
`docs/MUJOCO.md` for the physics decisions (key damping, fingertip sites,
mount calibration — each documented with its reason; don't "fix" them).

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

- **Rails.** `rail_limit = 0.32` m: each hand covers its half of the keyboard
  (left keys 0–53, right 33–87). `arm_action_scale` must equal `rail_limit` so
  the policy's ±1 rail action reaches the rail ends. `fold_to_reach` is OFF:
  songs train at their real pitches. `--legacy_reach` restores the old
  ±0.12 m rails + folding for checkpoints trained before 2026-09-10.
- **Fingering.** `fingering_method="hand"` (default): hand-relative
  assignment by measured fingertip offsets, the only planner consistent
  with a moving hand. `ot` (nearest-finger) produced the folded 0.84 result.
- **Rail servo.** `rail_follow=True` + `rail_leave_early`: a scripted servo
  does the travel (leaves for the next onset when time-left <= travel
  time), the policy keeps a `rail_residual` of 5 cm. Rail gains are stiff
  (6000 N/m, 2 kN). Policy-driven rails never learned to travel.
- **Sustain pedal.** Last action dim (43 total). While > 0, sounding keys
  keep sounding after the finger lifts. Needed because held bass notes
  overlap far-away notes of the same hand (35% of nettspend's goal steps).
- **PPO.** `entropy_coef` 0.001 (0.006 let the action std run 0.5 -> 1.5
  over 3000 iters and flattened F1).
- **Observation.** `obs_mode="ego"` (default, 316 dims): everything
  key-related is relative to the hand -- the 12 keys nearest each palm
  (offset, angle, vel, sounding), per-finger target-minus-tip and timing,
  per-hand upcoming notes, rail positions, pedal state, previous action.
  Sized in `PianoMjEnvCfg.ego_obs_dim`; the assembly order in
  `PianoMjEnv._get_obs_ego` must match it. `global` is the old per-key
  layout. Critic gets `critic_priv` (fingertip forces, collision flag) via
  rsl_rl obs_groups -- asymmetric actor-critic.
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
