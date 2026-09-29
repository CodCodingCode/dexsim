# dexsim — bimanual Shadow-Hand piano in MuJoCo

Two rail-mounted Shadow Hands over an 88-key spring piano learn to play a
MIDI song with PPO, from scratch. Success is measured by the
**F1 of actual key presses** against the song, not by reward.

Results so far on `results/nettspend - we not like you (1).mid`:

Version history (2026, deterministic whole-song F1 from
`scripts/mj/diag_rollout.py`; v10 onward live as `patch_vN.py` trees on the
training box, not in master's sources; the v15 tree is snapshotted in
`experiments/v15_mask_big/`):

| version | dates | key change | det. F1 |
|---|---|---|---|
| folded baseline | Sep 10 | OT fingering, ±0.12 m rails, song transposed | 0.84 |
| v1 | Sep 10-11 | unfold: ±0.32 m rails, rail servo, hand-relative fingering, ego obs | 0.34 |
| v2 | Sep 11-12 | entropy 0.006 to 0.001, stiff leave-early servo | 0.44 |
| RP1M A/B | Sep 12-13 | random song start, whole-song episodes, position gate, online fingering reward | 0.52 |
| v3 | Sep 12-13 | sustain pedal action + geometry pedal goal | 0.52 |
| v4 | Sep 14-15 | "nostay" changes, 16-hour run | 0.50 |
| v5 | Sep 15-16 | pedal-aware planner, chord stagger, onset window, action filter | 0.673 |
| v6 | Sep 15-16 | tight fingering falloff, per-finger weights, moved press points | 0.34 (failed) |
| v7 | Sep 17 | sticky finger ownership | 0.59 (killed) |
| v8 | Sep 23-24 | hands 3 cm deeper over the keys | 0.678 |
| v9 | Sep 23-24 | pose G ready pose | 0.556 (killed) |
| v10 | Sep 24 | 4-step chord roll + 4-step onset window, 7000 iters | 0.750 |
| v11 | Sep 25-26 | pedal span 0.07, key 40 to right hand, weight floor 0.10 | 0.702 |
| v12 | Sep 25-26 | v11 + leap curriculum | 0.775 (best) |
| v13 | Sep 27 | v12 + whole-song "seq" planner + pose G | 0.709 (killed) |
| v14 | Sep 27-29 | v13 + per-finger hold actions + servo follows plan | 0.725 (killed at 5530) |
| v15 | Sep 28-29, running | v14 + per-(finger,key) reach mask, thumbs excluded, 1024-512-256 MLP | **0.803** at iter 5346 |

## Quickstart

```bash
bash scripts/mj/setup_venv.sh      # one-time: uv venv + mujoco, rsl_rl, torch, ...
source env.sh                      # activates .venv, sets PYTHONPATH
python scripts/mj/smoke_piano_mj.py --songs_npz data/multisong/repertoire40.npz

# train (laptop: --device cpu --workers <cores-2>; GPU box: --device cuda)
python scripts/mj/train_piano_mj.py \
    --midi "results/nettspend - we not like you (1).mid" --episode_s 65 \
    --num_envs 1024 --workers 28 --device cuda \
    --max_iterations 3000 --anneal_false_press --tag my_run

# roll out a checkpoint: per-key recall/precision, MIDI export, video with audio
python scripts/mj/diag_rollout.py logs/piano_mj/<run>/model_final.pt "<song>.mid" hand
python scripts/mj/play_piano_mj.py --checkpoint logs/piano_mj/<run>/model_final.pt \
    --midi "<song>.mid" --episode_s 65 \
    --video results/played.mp4 --export_midi results/played.mid
python scripts/mj/midi_to_audio_video.py results/played.mid results/played.mp4 results/played_audio.mp4
```

Always pass `--workers N` when training: the single-process env serializes on
the GIL at ~400 steps/s regardless of env count; N worker processes scale
almost linearly with cores (~5,500 steps/s on 30 cores).

Checkpoints trained before 2026-09-10 used short rails, a folded song, the
per-key observation, policy-driven rails and no pedal; replay them with
`--legacy_reach --fingering ot`.

## Layout

```
dexsim/
  env.sh                          # source first (venv + PYTHONPATH + MUJOCO_GL)
  source/dexsim/
    piano/                        # sim-agnostic: MIDI -> goals, fingering (hand / OT / heuristic),
                                  #   key geometry, reward terms, goal encodings
    mjcf/                         # MuJoCo scene: procedural piano + Menagerie hands on rails
    tasks/piano_mj/               # PianoMjEnv, vec envs (threaded / process-sharded),
                                  #   rsl_rl wrapper, PPO config, song bank
    visualization/                # rollout npz -> Rerun .rrd
  scripts/
    mj/                           # smoke test, train, play/export, diagnostics, audio mux,
                                  #   scene XML build, venv setup
    prep/                         # MIDI helpers (test songs, corpus manifest, midi -> wav)
    render/view_rollout_rerun.py  # open a rollout in the Rerun viewer
    train/snapshot_run.py         # parse a training log into a one-glance summary
  data/multisong/repertoire40.npz # 40-song goal bundle
  results/                        # saved checkpoints, played MIDIs, videos
  docs/MUJOCO.md                  # stack notes and design decisions
  docs/A100_TRAINING.md           # setting up and running on the GPU box
```

## How it works

- **Embodiment.** Each hand is a MuJoCo Menagerie Shadow Hand E3M5 (right, and
  a mirrored true left) on a 1-DoF prismatic rail along the keyboard
  (`rail_limit` ±0.32 m: each hand covers its half). 21 actuators per hand
  (rail + 20 hand actuators; the J0 finger pairs are tendon-coupled), plus a
  sustain pedal action (43 total). The rail is driven by a scripted servo that
  leaves for the next note early; the policy adds a ±5 cm residual.
- **Goal.** A MIDI file becomes an (T, 88) key-activation grid at 20 Hz. A
  fingering plan assigns each note to a finger: `hand` (default) centres the
  hand on its notes and assigns by each finger's measured offset from the
  palm; `ot` is the RP1M-style nearest-finger assignment; `heuristic` is the
  pitch-order rule.
- **Observation (policy).** Egocentric, 316 dims: hand joint pos+vel, rail
  positions, the 12 keys nearest each palm (offset, angle, velocity, sounding
  latch), per-finger target-minus-tip and press timing, each hand's next 3
  notes, pedal state, previous action. The critic additionally sees fingertip
  contact forces and a hand-collision flag (asymmetric actor-critic).
- **Reward.** RoboPianist-style composite: goal-key press (per-key adaptive
  weights favouring rare / not-yet-learned keys), false-press penalty
  (annealed in once recall crosses a gate), fingertip-to-key shaping, onset
  bonus, idle-finger hover, jerk and energy penalties.
- **Metric.** Per-step key-state recall / precision / F1 against the goal
  grid, exactly RoboPianist's. `diag_rollout.py` adds per-key, per-hand, and
  onset-timing breakdowns.

## History

An earlier implementation on a different simulator (arm-mounted hands, later
rail hands) is archived on the `isaac-legacy` branch. Nothing in this tree
depends on it.
