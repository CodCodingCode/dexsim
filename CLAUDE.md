# dexsim (MuJoCo port) — Claude instructions

**This checkout is the MuJoCo rewrite.** The default stack lives in
`source/dexsim/mjcf/` + `source/dexsim/tasks/piano_mj/` + `scripts/mj/` and
runs in this repo's plain `.venv` (mujoco + rsl-rl, **no Isaac**). The Isaac
Lab implementation is kept for reference but its venv lives in the original
`~/dexsim` checkout, not here. See `docs/MUJOCO.md` for the full port map and
the deliberate physics deviations (key damping, fingertip sites, mount
calibration — each is documented with its reason; don't "fix" them back to
the Isaac values).

## 🔒 LOCKED: the constant static hand pose — DO NOT EDIT

`left_ready_pose` / `right_ready_pose` — defined identically in
`source/dexsim/tasks/piano/piano_env_cfg.py` (Isaac) and
`source/dexsim/tasks/piano_mj/piano_mj_env_cfg.py` (MuJoCo) — are the
**constant static config the rail-mounted hands must ALWAYS have**:
`railJoint = 0`, the hand wrist tilt `robot0_WRJ0 = 0.45` /
`robot0_WRJ1 = 0.13`, all fingers at 0. Palms ride at the configured base
positions (z = 0.88), fingertips hover a few cm above the keys, pointing
down. **Do NOT edit these poses** in either cfg; treat them as frozen unless
the user explicitly says otherwise in a new request. (The historical UR10e
arm-joint values — `wrist_1_joint = -4.782`, `shoulder_lift_joint = -0.640`,
`wrist_3_joint = π` — belong to the legacy Isaac arm embodiment and are
likewise frozen where they appear.)

## Rendering & measurement

- **MuJoCo (default):** no warm server needed — the scene compiles in ~0.3 s
  and renders in-process (`MUJOCO_GL=egl`). Stills/queries: extend
  `scripts/mj/smoke_piano_mj.py`-style probes; videos:
  `scripts/mj/play_piano_mj.py --video`.
- **Isaac (legacy only):** every cold render/diag script boots the whole
  Isaac app (~30 s). If you ever work the Isaac stack, use the warm render
  server (`scripts/render/render_server.py` + `scripts/render/render.py`) as
  described in `README.md` — never write new one-shot `AppLauncher` scripts.

## General

- `source env.sh` before anything (activates the MuJoCo venv + PYTHONPATH).
- The task recipe is shared: `source/dexsim/piano/` (MIDI → goal schedule,
  fold-to-reach, fingering, rewards, key geometry) is framework-agnostic and
  used by BOTH stacks — changes there affect both.
- Embodiment: two Shadow Hands on Y rails (Menagerie E3M5, right + true
  left, renamed to the `robot0_*` convention in `dexsim/mjcf/shadow_hand.py`).
- MuJoCo scene geometry comes from `dexsim.piano.geometry` — the single
  source of truth. Don't hardcode key positions anywhere else.
- rsl-rl here is ≥5.x: its runner config is the dict in
  `source/dexsim/tasks/piano_mj/ppo_cfg.py`, NOT the Isaac-Lab-era
  `RslRlOnPolicyRunnerCfg` classes.
- `logs/` is gitignored; `assets/mujoco_menagerie/` (auto-vendored) and
  `assets/mj/` (generated XML) are gitignored too.
