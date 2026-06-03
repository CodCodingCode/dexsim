# Using the RoboPianist / RP1M data in our (different) sim

This documents the pre-training pipeline set up from the RoboPianist analysis —
everything **up to** RL fine-tuning. The premise: the "different sim" (we are
Isaac/PhysX, they are MuJoCo) is *not* the blocker. The blockers are the
embodiment (their sliding-forearm hands vs our fixed-base UR10e) and the physics
gap (which only RL fine-tuning closes). So we transfer what transfers and leave
the press for RL.

| What | Transfer | Where |
|------|----------|-------|
| MIDI corpus (PIG / GiantMIDI) — *goal data* | ✅ direct, sim-agnostic | `dexsim.piano.corpus` |
| Fingering — *which finger plays which key* | ✅ re-derived from geometry (no human labels) | `dexsim.piano.fingering` OT |
| Shadow **hand** joint pose | ✅ same morphology, remapped | `dexsim.piano.rp1m.decode` |
| **Arm** placement | ⚠️ retarget (their forearm → our UR10e) | `dexsim.piano.rp1m.retarget` + our IK |
| The actual key *press* | ❌ left to RL fine-tuning (out of scope) | `train_piano.py` |

---

## 1. MIDI corpus — the free win

The MIDI is pure goal data (which keys sound when); it ports to any sim. Point
the ingester at a folder of `.mid` (a GiantMIDI-Piano download, the RoboPianist
`rousseau` set, PIG MIDIs you `robopianist preprocess`-ed, …):

```bash
source env.sh
python scripts/build_corpus.py \
    --roots data/midi data/robopianist_ref/robopianist/music/data \
    --out data/corpus/manifest.json
# filter to short, reachable, easy pieces for a first curriculum:
python scripts/build_corpus.py --roots data/corpus/giantmidi \
    --max-difficulty 0.5 --max-duration 40
```

The manifest stores per-song stats — length, note density, polyphony,
**reachability** (how much falls in the two hands' reachable key windows) and a
**difficulty** score. `MidiCorpus` is also a *goal generator*: `iter_songs()`,
`curriculum()`, `sample(by_difficulty=True)`, `filter(...)`. PIG `.proto` files
load best-effort via `note-seq` (else use `robopianist preprocess` to get `.mid`).

## 2. OT auto-fingering — drop the human labels

RoboPianist needed human PIG fingering (without a fingering signal F1 = 0). RP1M
removed that dependence by solving finger→key assignment as **optimal transport**
(minimum total movement, a linear-sum / Jonker-Volgenant matching). We ported the
same trick:

```python
from dexsim.piano import fingering as fg
plan = fg.plan_fingering(song.key_activation, method="ot")   # vs "heuristic"
```

Per step it assigns active keys to fingers minimising movement cost from the
current finger positions, plus hand-side and black-key-on-thumb penalties; idle
fingers drift back to a ready spread. On our songs it gives full note coverage at
**~4–5× less fingertip travel** than the old heuristic (smoother, more learnable).
It is a drop-in `FingeringPlan`, so the IK reference builder and reward shaping
use it unchanged.

## 3. RP1M motion → warm-start reference

RP1M ships ~1M expert trajectories of two **Shadow hands** on 2-dof sliding
forearms. Each action is `[right(H) | left(H) | sustain]` where `H = n_hand + 2`:
**full** = 20 hand actuators + 2 forearm = 22/hand → **45**-d (the RoboPianist
demo, `examples/twinkle_twinkle_actions.npy`); **reduced** = 17 + 2 = 19/hand →
**39**-d (RP1M's published action). Actions are canonical `[-1,1]` and are
rescaled per-actuator before stepping.

### a) decode + Shadow remap (`rp1m/decode.py`)
Splits the action, un-normalises with the authoritative Shadow E3M5 ctrlranges,
maps MuJoCo 1-indexed joints → our Isaac 0-indexed `robot0_*` (the `J(n-1)` map
also used by `bodex_loader`), and splits each coupled distal tendon (`A_*FJ0`)
across its two Isaac joints. Output: per-side `(T, 24)` hand angles + forearm
`(tx, ty)` + sustain. The reduced/full split is auto-detected from the width.

### b) arm retarget (`rp1m/retarget.py`)
Their forearm `tx` (lateral position along the keyboard) → a wrist target in
**our** piano frame via `forearm_to_wrist_target` (uses our key geometry). The
arm itself is *not* their forearm, so the warm-start **keeps the arm columns from
our own validated IK reference** (which already positions each hand over its
keys) and injects only the RP1M finger configuration. The wrist-target hook is
there for a future IK pass that nudges arm placement to match RP1M exactly.

### c) assemble the warm-start (`scripts/build_rp1m_reference.py`)
Sim-free. Merges the decoded hand pose into an existing reference's 24 hand
columns (time-resampled), keeping its 6 arm columns:

```bash
source env.sh
# decode only -> name-keyed hand trajectory (no reference needed):
python scripts/build_rp1m_reference.py \
    --actions data/robopianist_ref/examples/twinkle_twinkle_actions.npy

# full warm-start: inject RP1M finger pose into our twinkle IK reference:
python scripts/build_rp1m_reference.py \
    --actions data/robopianist_ref/examples/twinkle_twinkle_actions.npy \
    --reference data/reference/twinkle.npz \
    --out data/reference/twinkle_rp1m.npz
```

It needs the articulation's joint-name order (to know which `q_ref` column is
which joint). New references store it; otherwise it reads
`data/reference/joint_names.json` (written by `scripts/dump_joint_names.py` or
`build_reference.py`). The merge is fully sim-free once that cache exists.

---

## Then (this is the RL fine-tuning boundary — NOT done here)

```bash
# clone the warm-start reference into the PPO actor (needs Isaac):
python scripts/bc_pretrain.py --midi data/midi/twinkle.mid --headless \
    --reference data/reference/twinkle_rp1m.npz --out logs/bc/twinkle_rp1m.pt
# RL fine-tuning to actually close the physics gap (out of scope of this setup):
python scripts/train_piano.py --midi data/midi/twinkle.mid \
    --bc_init logs/bc/twinkle_rp1m.pt
```

## Where the real RP1M data goes

The 1M-trajectory dataset is on MPG Edmond
(`doi:10.17617/3.XCE8NX`, project: <https://rp1m.github.io/>). Drop per-song
action `.npy` files anywhere and point `--actions` at them; the decoder
auto-detects the 39-d (reduced) RP1M layout. **Check the Edmond license before
using it in a pipeline.**

## Self-check

`python scripts/test_pretrain_pipeline.py` runs all three parts sim-free and
asserts correctness (16 checks). No GPU needed.

## Honest status

Validated **offline** on the local twinkle fixture: decode (in-range joints),
reduced/full paths, OT fingering, and the merge (arm kept, hand replaced — mean
Δ ≈ 0.6 rad as the IK hover pose becomes RP1M's curled press pose). What is
*not* claimed: that replaying the warm-start zero-shot sounds the notes — it
won't (different physics, retargeted arm). That is exactly the RL fine-tuning
step this setup stops before.
