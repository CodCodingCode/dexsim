# dexsim — UR10e + Shadow Hand in Isaac Lab

Pure-sim dexterous manipulation on the UR10e + Shadow Hand embodiment, on an
H100. Two things live here:

### 🎹 Current goal — bimanual piano (see [docs/PIANO.md](docs/PIANO.md))
Two UR10e + Shadow arms (**60 action DoF**) over an 88-key piano, trained with
PPO to play a specific MIDI song. The env builds and steps on GPU today
(`scripts/piano_env_smoke.py` passes). Drop your `.mid` in `data/midi/` and:
```bash
source env.sh
python scripts/train_piano.py --headless --num_envs 2048 --midi data/midi/<song>.mid
python scripts/play_piano.py  --num_envs 1 --video --export_midi logs/played.mid
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
    smoke_test.py              # boot Isaac Sim, spawn Shadow Hand, step (sanity)
    build_piano_usd.py         # generate the 88-key sprung piano USD
    build_combined_usd.py      # compose UR10e + Shadow into ONE articulation USD
    make_test_midi.py          # write data/midi/twinkle.mid for development
    piano_env_smoke.py         # integration test: build+step the bimanual piano env
    train_piano.py / play_piano.py  # PPO train / playback (+ export played .mid)
    spawn_scene.py             # build the full tabletop scene and settle it
    train_rl.py / play_rl.py   # PPO train / playback for reorientation
    replay_bodex.py            # replay a BODex trajectory on the embodiment
    download_bodex.py          # fetch BODex-Tabletop
    download_dexgraspnet.py    # fetch DexGraspNet
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
python scripts/smoke_test.py --headless

# 1a. RL reorientation (no dataset needed) — just hit run
python scripts/train_rl.py --headless --num_envs 8192
python scripts/play_rl.py --num_envs 16 --video

# 1b. imitation path — get the dataset, then replay it on the embodiment
python scripts/download_bodex.py --list          # inspect the repo first
python scripts/download_bodex.py --include "..."  # grab a subset
python scripts/build_combined_usd.py --inspect    # verify mount frames
python scripts/build_combined_usd.py              # build assets/ur10e_shadow.usd
python scripts/replay_bodex.py --traj data/bodex/<file>.npz --headless
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
