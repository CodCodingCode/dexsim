# The MuJoCo stack — bimanual piano without Isaac

This repo carries a full MuJoCo port of the bimanual piano task alongside the
original Isaac Lab implementation. Same task recipe, same 🔒 locked ready
pose, same reward composition, same MIDI goal pipeline — no Isaac Sim boot,
no Vulkan driver staging, no warm render server (MuJoCo compiles the scene in
~0.3 s and renders offscreen in-process).

## Quickstart

```bash
source env.sh          # activates .venv (plain python + mujoco, no Isaac)

# sanity: compile the scene, check the locked pose, sound a key, random-step
python scripts/mj/smoke_piano_mj.py --render        # -> logs/mj_smoke.png

# train (CPU-vectorized sim; policy on GPU when one is free)
python scripts/mj/train_piano_mj.py --num_envs 64 --midi data/midi/song.mid

# roll out + record
python scripts/mj/play_piano_mj.py --zero --rail_follow --video results/mj_zero.mp4
python scripts/mj/play_piano_mj.py \
    --checkpoint logs/piano_mj/<run>/model_final.pt --video results/mj_play.mp4

# inspect the model interactively (needs a display / VNC)
python scripts/mj/build_scene.py
python -m mujoco.viewer --mjcf=assets/mj/piano_scene.xml
```

## What maps to what

| Isaac stack                             | MuJoCo stack                                   |
|-----------------------------------------|------------------------------------------------|
| `tasks/piano/piano_env_cfg.py`          | `tasks/piano_mj/piano_mj_env_cfg.py`           |
| `tasks/piano/piano_env.py` (DirectRLEnv)| `tasks/piano_mj/piano_mj_env.py` (+ `vec_env`) |
| `assets/piano.py` + piano USD           | `dexsim/mjcf/piano.py` (procedural MJCF)       |
| slider USDs (`build_shadow_hand_sliders`)| rail mounts built in `dexsim/mjcf/scene.py`   |
| NVIDIA Shadow Hand USD (right-only)     | Menagerie E3M5 right + **true left** hand      |
| `agents/rsl_rl_ppo_cfg.py`              | `tasks/piano_mj/ppo_cfg.py` (rsl_rl ≥5.x dict) |
| `train/train_piano.py`                  | `scripts/mj/train_piano_mj.py`                 |
| `train/play_piano.py`                   | `scripts/mj/play_piano_mj.py`                  |
| warm render server                      | not needed (in-process `mujoco.Renderer`)      |

Shared, unchanged: **everything in `dexsim/piano/`** (MIDI → goal schedule,
fold-to-reach, fingering planner incl. the OT variant, reward functions, key
geometry, SDF goal encoding). The reward functions were already
backend-agnostic; the MuJoCo env calls them with numpy, the Isaac env with
torch.

## The embodiment

Two Shadow Hands, each with one world-Y prismatic `railJoint` (±0.12 m, the
Isaac slider's travel), palm-down over the flipped piano, fingers toward the
keys. 24 joints/hand; 20 position actuators/hand (the four `*J0` distal pairs
are tendon-coupled, as on the real hand and in the Isaac USD) + 1 rail ⇒
**42-dim action**. Observations (1216-dim, same composition as Isaac): both
hands' qpos+qvel, 88 key angles, 10×88 goal lookahead, 10 fingertip positions,
10 fingering targets, 88-dim analytic goal SDF.

Naming: Menagerie's real Shadow names are renamed to the repo convention
(`rh_FFJ4→robot0_FFJ3`, `rh_WRJ1→robot0_WRJ0`, …, see
`dexsim/mjcf/shadow_hand.py`), then prefixed `L_`/`R_` per hand at attach
time. The 🔒 locked ready pose (`railJoint=0`, `robot0_WRJ0=0.45`,
`robot0_WRJ1=0.13`, fingers 0) applies verbatim.

## Deliberate deviations from the Isaac implementation

* **Key damping 0.1, not 4.0.** Isaac's 4.0 was a PhysX contact-explosion
  absorber; on a passive MuJoCo hinge it makes the key a τ=1.3 s sponge that a
  strike can't depress past the sound angle (measured). 0.1 matches the
  RoboPianist reference values the Isaac file itself cites. Everything else
  about key physics is identical (stiffness 3, travel 0.0666 rad, sound angle
  −0.012, gravity-compensated keys, velocity-gated hammer sounding).
* **Physics at 200 Hz (dt 0.005), not 120 Hz.** The standard MuJoCo step for
  finger/key contact (RoboPianist uses it too). The control rate is the same
  20 Hz / `control_dt=0.05` the MIDI grid uses.
* **Fingertip sites, not distal body origins.** The distal body origin sits at
  the DIP joint ~2.5 cm short of the pad; tip sites make the fingering reward
  target where the finger actually presses.
* **Self-calibrating mount.** The scene builder measures (two-pass) where the
  ready-pose fingertips land and shifts the hands along X so the tips sit over
  the white keys' press line (~50 % hinge leverage). The Menagerie hand's
  palm→tip reach differs from the NVIDIA USD, so a fixed offset would press
  at the hinge and never sound.
* **Key–key collisions masked** (`contype 2 / conaffinity 1`): black-key boxes
  interlock with white ones by design; Isaac had
  `enabled_self_collisions=False` for the same reason.
* **`hand_action_scale` 0.8, not 0.35.** The Menagerie hand uses the real
  Shadow's weak position servos (kp 0.5–1, ~1 N forcerange), so press force
  scales with target offset — at 0.35 the *maximum* action bottoms out at
  −0.0103 rad, short of the −0.012 sound angle: the policy physically could
  not sound a key (measured; runs 1–3 all failed on this).
* **Substep strike detection** (`substep_strike_detect`, default on): the
  strike's velocity spike lasts ~30 ms; sampling the hammer gate once per
  50 ms control step undercounts real strikes 5× (104 vs 20 on identical
  trajectories).
* **`key_strike_vel` 0.10, not 0.35/0.25.** A position-servo press
  decelerates near its target, so at 0.25 a deliberate max-action press
  NEVER sounds (0/30) while random flail passes the gate 183/500 — the gate
  selected for flailing. At 0.10 deliberate presses ring reliably with no
  extra mash. (RoboPianist has no velocity gate at all.)
* **Dense goal-key press reward** (`dense_goal_press`, default on,
  RoboPianist-style): the reward's hit term uses the goal keys' raw
  depression fraction (continuous gradient on the way down); the velocity
  latch still governs the false-press term and all recall/F1 metrics.
* **Recall-gated anneal ported** (`--anneal_false_press`): false-press
  penalty starts at 0.15 (energy 0) and ramps to full once the recall EMA
  crosses 0.5 — without it the policy converges to "hover, never press"
  (run1: 5000 iterations, F1 exactly 0).
* **Rail-follow replaces WristPoseIK.** With a 1-DoF rail the "arm servo" is
  analytic: slide to the world-Y centroid of the upcoming notes (EMA-smoothed,
  lane-clamped). `--rail_follow` = fingers-only policy, the RoboPianist-style
  decoupling. The UR10e-specific machinery (planar IK, phase-0 arm curriculum,
  baked arm trajectories, wrist caps) has no rail equivalent and was not
  ported.

## Environment

The `.venv` here is MuJoCo-only (no Isaac): `mujoco`, `gymnasium`,
`pretty_midi`, `imageio[-ffmpeg]`, `rsl-rl-lib` (≥5.x — note its config
format differs from the 2.x Isaac Lab bundles), numpy 1.26 (pinned: the
system torch 2.7 predates numpy 2), system-site torch with CUDA. The
Menagerie `shadow_hand` model is auto-vendored (sparse clone) into
`assets/mujoco_menagerie/` on first use.

Headless rendering uses EGL (`MUJOCO_GL=egl`, set by the scripts). Training
sim is CPU-threaded — ~2.8 ms/control-step per env, ~16 workers saturate; on
this 30-core box 64 envs collect 2048 steps in ~15 s. The policy trains on
`--device cuda` when the GPU has room and falls back to CPU automatically.
