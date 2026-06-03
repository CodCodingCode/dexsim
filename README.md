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
    train/                     # PPO/SFT/BC train + playback + eval + run monitoring
                               #   train_piano.py / play_piano.py, train_rl.py / play_rl.py,
                               #   sft_rp1m.py, bc_pretrain.py, eval_reference.py, ...
    build/                     # one-shot USD asset builders
                               #   build_combined_usd.py, build_piano_usd.py,
                               #   build_shadow_slider_usd.py, *_left_hand asset provenance
    prep/                      # MIDI/corpus/reference prep + dataset downloads
                               #   make_test_midi.py, build_corpus.py, build_reference.py,
                               #   download_bodex.py, replay_bodex.py, ...
    render/                    # rendering, video, and camera-rig tools
                               #   render_scene.py, record_rollout.py / render_rollout.py, ...
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
