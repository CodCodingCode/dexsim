# dexsim — UR10e + Shadow Hand in Isaac Lab

Pure-sim dexterous manipulation on the UR10e + Shadow Hand embodiment, on an
H100. Two things live here:

### 🎹 Current goal — bimanual piano (see [docs/PIANO.md](docs/PIANO.md))
Two UR10e + Shadow arms (**60 action DoF**) over an 88-key piano, trained with
PPO to play a specific MIDI song. The env builds and steps on GPU today
(`scripts/smoke/piano_env_smoke.py` passes). Drop your `.mid` in `data/midi/` and:
```bash
source env.sh
python scripts/train/train_piano.py --headless --num_envs 2048 --midi data/midi/<song>.mid
python scripts/train/play_piano.py  --num_envs 1 --video --export_midi logs/played.mid
```

### Foundation — manipulation on the same embodiment
- **RL in-hand reorientation** — turnkey via Isaac Lab's built-in Shadow env.
- **Imitation from BODex-Tabletop** — the dataset *is* this exact embodiment
  (UR10e + Shadow), so trajectories drop in; DexGraspNet adds object diversity.

> ⚙️ This box is a **compute-only container** — Isaac Sim's Vulkan/GPU foundation
> needed a one-time driver fix (staged locally, no sudo). `env.sh` wires it up;
> the full story and the `setup_nvidia_gl.sh` recipe are in [docs/SETUP.md](docs/SETUP.md).

## Layout

```
dexsim/
  env.sh                       # source this first (venv + EULA + PYTHONPATH)
  source/dexsim/
    assets/ur10e_shadow.py     # UR10e, Shadow Hand, and combined ArticulationCfgs
    assets/piano.py            # 88-key piano ArticulationCfg
    piano/                     # MIDI->goal parser + reward (framework-agnostic)
    tasks/piano/               # Dexsim-Piano-Bimanual-v0 (DirectRLEnv) + PPO cfg
    tasks/reorient/            # RL cube reorientation (Dexsim-Reorient-Cube-Shadow-v0)
    tasks/grasp/               # tabletop scene + BODex trajectory loader
  scripts/
    setup_nvidia_gl.sh         # stage NVIDIA Vulkan/GLX driver locally (the env fix)
    train/                     # PPO train + playback + eval + run monitoring
                               #   train_piano.py / play_piano.py, train_rl.py / play_rl.py,
                               #   eval_reference.py, ...
    build/                     # one-shot USD asset builders
                               #   build_combined_usd.py, build_piano_usd.py,
                               #   build_shadow_slider_usd.py, *_left_hand asset provenance
    prep/                      # MIDI/corpus prep + dataset downloads
                               #   make_test_midi.py, build_corpus.py,
                               #   download_bodex.py, replay_bodex.py, ...
    render/                    # rendering, video, and camera-rig tools
                               #   render_scene.py, record_rollout.py / render_rollout.py, ...
                               #   render_server.py + render.py  <- WARM cache (boot once, render in s)
    smoke/                     # sanity / integration tests (no training)
                               #   smoke_test.py, piano_env_smoke.py, smoke_slider.py, ...
  assets/                      # composed USDs land here (gitignored)
  data/                        # datasets (gitignored)
  IsaacLab/                    # Isaac Lab v2.1.0 checkout (installed editable)
  .venv/                       # Python 3.10 venv with Isaac Sim 4.5 + Isaac Lab
```

## Quickstart

```bash
cd ~/dexsim
source env.sh

# 0. sanity: boot the sim and spawn the hand
python scripts/smoke/smoke_test.py --headless

# 1a. RL reorientation (no dataset needed) — just hit run
python scripts/train/train_rl.py --headless --num_envs 8192
python scripts/train/play_rl.py --num_envs 16 --video

# 1b. imitation path — get the dataset, then replay it on the embodiment
python scripts/prep/download_bodex.py --list          # inspect the repo first
python scripts/prep/download_bodex.py --include "..."  # grab a subset
python scripts/build/build_combined_usd.py --inspect    # verify mount frames
python scripts/build/build_combined_usd.py              # build assets/ur10e_shadow.usd
python scripts/prep/replay_bodex.py --traj data/bodex/<file>.npz --headless
```

### Fast iteration — the warm render server (cache the Isaac boot)

Every cold render/diagnostic script (`render_scene.py`, `render_rollout.py`,
`diag_*.py`, `verify_palm.py`) boots the **whole** Isaac Sim app (~30 s, longer
under GPU contention) and rebuilds the scene from scratch on *every* run. The
warm server pays that cost **once**, then serves jobs from a file-queue so each
render/measurement takes seconds. (`render_scene.py` etc. still work standalone —
the server is the fast path on top of the shared `dexsim.render.studio` builders.)

```bash
source env.sh
# boot ONCE (~30 s), leave it running in the background:
python scripts/render/render_server.py --headless > logs/render_server.log 2>&1 &
#   ...wait for "READY" in logs/render_server.log (or logs/render_jobs/server.ready)

# then iterate — each job is seconds, no reboot, no scene rebuild:
python scripts/render/render.py scene   --eye 2.2,-1.5,1.8 --target 0.45,0,0.78 --spp 160 --out logs/x.png
python scripts/render/render.py rollout --rollout logs/rollout.npz --out results/v.mp4 --spp 96
python scripts/render/render.py query   --kind layout  --out logs/layout.json   # diag_layout
python scripts/render/render.py query   --kind orient  --out logs/orient.json   # diag_hand_orient
python scripts/render/render.py query   --kind palm --rollout logs/r.npz --out logs/palm.json  # verify_palm
python scripts/render/render.py shutdown            # stop the server
```

Measured on this box (while a training swarm shared the GPU): boot 28 s; then a
`layout` query 1.6 s, an `orient` query 1.7 s, a 64-spp still 10 s (pure
path-tracing — drop `--spp` for faster previews). The RTX *shader* cache
(`~/.cache/ov`) already persists across runs; the server adds the missing
app-process + built-scene cache, which is what dominated cold iteration.

The combined UR10e+Shadow articulation is built once by `build_combined_usd.py`
(the genuinely fiddly part — two separate articulations bonded into one tree by
a fixed joint at the tool flange). The reorientation path doesn't need it.

## Stack

| piece     | choice                                   |
|-----------|------------------------------------------|
| Arm       | UR10e (6-DOF, native USD)                |
| Hand      | Shadow Hand (24-DOF, instanceable USD)   |
| Sim       | Isaac Sim 4.5.0 (pip)                    |
| Framework | Isaac Lab v2.1.0 (rsl_rl PPO, Mimic)     |
| Dataset   | BODex-Tabletop (primary) + DexGraspNet   |

See `docs/SETUP.md` for how the environment was installed (and the gotchas that
were fixed) and `docs/DATASETS.md` for dataset sources.
