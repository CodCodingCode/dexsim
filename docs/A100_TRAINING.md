# Training the MuJoCo piano task on the GPU box

The MuJoCo stack (`scripts/mj/`, `source/dexsim/tasks/piano_mj/`) needs no
Isaac Sim, no Vulkan, no EGL. On a many-core Linux box it runs the physics in
CPU worker processes and the PPO update on the GPU. Measured 2026-09-09 on a
30-core / A100-40GB machine: ~5,500 env steps/s at 1024 envs (vs ~850 on an
M3 Max laptop).

**The GPU is barely used.** Per iteration ~5.8 s is CPU physics and ~0.1 s is
the PPO update. Utilization sits near 0 % and ~1.3 GB of memory. More GPUs do
nothing; more CPU cores scale nearly linearly. Rent cores, not GPUs.

## 1. SSH

`~/.ssh/config` on the laptop:

```
Host piano
    HostName <box ip>
    User ubuntu
    IdentityFile ~/.ssh/new.pem
    IdentitiesOnly yes
```

When the provider hands out a new IP, edit `HostName`. A brand-new IP has no
`known_hosts` entry and can be accepted on first connect. A _changed_ key on
an IP you have used before is worth a second look before accepting.

## 2. One-time setup on the box

```bash
ssh piano
git clone -b mujoco-alternative https://github.com/CodCodingCode/dexsim.git ~/dexsim
cd ~/dexsim
curl -LsSf https://astral.sh/uv/install.sh | sh          # installs ~/.local/bin/uv
export PATH=$HOME/.local/bin:$PATH
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
    mujoco "rsl-rl-lib>=5" torch numpy tensorboard gymnasium pretty_midi
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"   # True
```

The MuJoCo Menagerie Shadow Hand is auto-vendored into
`assets/mujoco_menagerie/` the first time a scene compiles (needs outbound
git).

### Local (uncommitted) changes

If the laptop worktree has edits that are not on the branch yet, copy them
over instead of committing:

```bash
cd "/Users/owner/Downloads/coding projects/piano-player/dexsim-mujoco"
rsync -av --relative $(git status --short | awk '{print $2}') piano:~/dexsim/
```

## 3. Environment variables

Every command on the box needs:

```bash
cd ~/dexsim
export PYTHONPATH=source
unset MUJOCO_GL        # env.sh sets egl; this box has no EGL library and
                       # training never renders
```

Do NOT `source env.sh` on this box (it sets `MUJOCO_GL=egl` and expects the
Isaac Vulkan staging).

## 4. Smoke test

```bash
.venv/bin/python scripts/mj/smoke_piano_mj.py --songs_npz data/multisong/repertoire40.npz
```

Expect `SMOKE OK`, a scripted press that sounds key 63, and the obs summary
line (`obs 368 dims (16 observed keys), critic +11 privileged`).

## 5. Train

Always use `--workers` (process-sharded envs). The default threaded env
serializes on the GIL at ~400 steps/s no matter how many envs you give it.

```bash
mkdir -p logs/ab
nohup .venv/bin/python scripts/mj/train_piano_mj.py \
    --midi "results/nettspend - we not like you (1).mid" --episode_s 65 \
    --num_envs 1024 --workers 28 --device cuda \
    --max_iterations 3000 --save_interval 100 \
    --anneal_false_press --seed 0 --tag nettspend_slim_a100 \
    > logs/ab/nettspend_slim_a100.log 2>&1 &
```

Knobs:

| flag                                   | notes                                                                             |
| -------------------------------------- | --------------------------------------------------------------------------------- |
| `--workers N`                          | N = cores minus 2. Each worker owns `num_envs / N` envs.                          |
| `--num_envs`                           | 1024 is a good default on 30 cores. Throughput plateaus past ~32 envs per worker. |
| `--episode_s`                          | Must be at least the song length in seconds or an episode never reaches the end.  |
| `--midi` / `--songs_npz --max_songs K` | Single MIDI, or the first K songs of the 40-song bundle.                          |
| `--anneal_false_press`                 | Required. Without it the policy converges to "hover, never press".                |
| `--legacy_obs`                         | A/B baseline: the old 1216-dim obs with a symmetric critic.                       |
| `--resume_from path.pt`                | Continue from a checkpoint (obs layout must match).                               |
| `--logger wandb --wandb_project ...`   | Needs `wandb login` on the box first. Default is tensorboard.                     |

Checkpoints land in `logs/piano_mj/<timestamp>_<tag>/model_<iter>.pt` and
`model_final.pt`.

## 6. Watch it

From the laptop:

```bash
ssh piano 'tail -f ~/dexsim/logs/ab/nettspend_slim_a100.log' \
  | grep --line-buffered -E "Learning iteration|play/F1:|play/recall:|play/precision:|reward/total:|ETA"
```

Output arrives in bursts of ~5 iterations because Python block-buffers stdout
to a file. `play/F1` per iteration is noisy (swings 0.08 to 0.18 around a
0.13 mean); average over 50 iterations before drawing conclusions.

TensorBoard, tunnelled to the laptop:

```bash
ssh -L 6006:localhost:6006 piano '~/dexsim/.venv/bin/tensorboard --logdir ~/dexsim/logs/piano_mj --port 6006'
# then open http://localhost:6006
```

GPU / CPU check:

```bash
ssh piano 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv; uptime'
```

## 7. Stop / clean up

```bash
ssh piano 'pkill -f "tag nettspend_slim_a100"'
```

Killing the parent also kills the 28 daemon worker processes.

## 8. Play / export a checkpoint

```bash
.venv/bin/python scripts/mj/play_piano_mj.py --help
```

Rendering (`--video`) needs a GL backend; on this box there is none, so copy
the checkpoint to the laptop and render there, or use the MIDI export path.
