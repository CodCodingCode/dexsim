# Bimanual piano task

Two UR10e + Shadow Hand arms (~30 DoF each → **60 action DoF**) over an 88-key
spring-loaded piano, trained with PPO to play a specific MIDI song.

## How it fits together

```
data/midi/<song>.mid
        │  scripts/prep/make_test_midi.py writes a stand-in (Twinkle) for development
        ▼
dexsim.piano.load_song(...)        -> (T,88) key goal + (T,88) onsets
dexsim.piano.plan_fingering(...)   -> per-step finger->key assignment (10 fingers)
        ▼
Dexsim-Piano-Bimanual-v0  (DirectRLEnv)        [decoupled: math arms, RL fingers]
  arms (analytic):  WristPoseIK servos each palm onto the upcoming-note centroid
  action (60):      RESIDUAL on a ready-pose base (arm cols overwritten by IK):
                    target = base[t] + scale·a   (policy effectively drives 48 fingers)
  observation:  arms pos+vel (120) | key angles (88) | goal lookahead (880)
                | fingertip pos (30) | fingering targets (30)
                | SDF goal encoding (88)              = 1236
  reward:       key-press (right keys, none wrong) + FINGERING shaping
                (finger→assigned key, the make-or-break term) + onset + energy
        ▼
scripts/train/train_piano.py   (rsl_rl PPO)
scripts/train/eval_reference.py(F1/recall/precision of a policy or the zero-residual base)
scripts/train/play_piano.py    (roll out + EXPORT what it played back to .mid)
```

## What makes the 60-DoF problem trainable

Every piece below is structure-injection so RL doesn't explore 60 DoF from
scratch (code in `source/dexsim/piano/`):
- **Fingering** (`fingering.py`): assigns each note to a finger. RoboPianist showed
  the policy scores **F1=0 without a fingering signal** — this is the critical term.
- **Decoupled control** (`ik.py`): pure-math `WristPoseIK` servos the arms so each
  palm tracks the centroid of its upcoming notes; the policy learns only the finger
  press as a **residual** on a static ready pose. The policy never touches the stiff
  arm joints, so the arm-residual blow-up (NaN) mode is gone, and zero action already
  tracks competent positioning.
- **Composite reward** (`reward.py`): key-press + `fingering_reward` (gaussian
  `tolerance`, RoboPianist constants) + `onset_reward` + energy.
- **Rich obs** incl. fingertip positions, the fingering targets ("where fingers
  should go"), and an **SDF goal encoding** (`goal_encoding.py`).

### Typical workflow
```bash
python scripts/train/eval_reference.py --midi data/midi/twinkle.mid --arm_ik_follow --zero --headless  # sanity
python scripts/train/train_piano.py    --midi data/midi/twinkle.mid --arm_ik_follow --headless --num_envs 4096
```

Key assets (generated once, already built here):
- `assets/piano88.usd` — 88 keys, each a sprung hinge (`joint_0..joint_87`,
  index = MIDI−21). A key "sounds" when its hinge angle ≤ `KEY_SOUND_ANGLE`.
  Rebuild: `python scripts/build/build_piano_usd.py`.
- `assets/ur10e_shadow.usd` — UR10e + Shadow bonded into one articulation
  (`wrist_3_link → robot0_forearm`). Rebuild:
  `python scripts/build/build_combined_usd.py --inspect` then without `--inspect`.

## Use your own song

```bash
cp /path/to/your_song.mid data/midi/your_song.mid
python scripts/train/train_piano.py --headless --num_envs 2048 --arm_ik_follow --midi data/midi/your_song.mid
python scripts/train/play_piano.py  --num_envs 1 --video --arm_ik_follow --midi data/midi/your_song.mid \
       --export_midi logs/your_song_played.mid
```
`play_piano.py` records the keys the policy actually pressed and writes them to a
MIDI file so you can hear what it learned.

## Why the arms are driven by IK, not RL (the decoupling)

An earlier design learned a **60-DoF residual on a precomputed reference trajectory**
whose arm path was built by a **multi-fingertip** IK (5 fingertip targets on the
6-DoF arm). That over-constrains the arm and **diverges** (left hand ~305 mm off),
capping zero-residual F1 at **0.03**. The diagnostic (`scripts/diag_wrist_ik.py`,
`scripts/prep/diag_posik.py`) proved the arms *can* reach within ~1 cm everywhere
using the **well-posed** `WristPoseIK` (one palm/fingertip target on the 6-DoF arm).
So the multi-fingertip reference was removed and the two jobs decoupled:

- **Arms (analytic, no learning):** every control step `PianoEnv._ik_follow_base()`
  drives the 12 arm DoF with `WristPoseIK` so each palm tracks the **centroid of its
  upcoming notes** (`_hand_note_centroids`), hovering `arm_ik_hover` above the keys at
  the ready-pose down orientation. One DLS step/step = closed-loop tracking.
- **Fingers (learned):** the policy action is masked to the **48 finger DoF**; it
  rides as a residual on the static ready pose.

Enable with `--arm_ik_follow` (sets `arm_ik_follow`, clears `freeze_arms`). The
**slider embodiment** (`--use_slider`) is the same recipe with an analytic 1-DoF
prismatic rail in place of the arm servo (0 mm placement; no octave-folding needed).

**Validated (easy.mid, zero finger residual, 2026-06-03):** stable over 839 steps,
no NaN; **recall 0.633** purely from arm positioning. Precision is still low (~0.02)
and the assigned fingertip→key distance ~134 mm because zero-residual fingers don't
press selectively — **that is the exact job RL now learns**, on a smaller, well-
conditioned action space with the hands already on the keys. `eval_reference.py`
reports the fingertip→key distance (mm) so finger-placement progress is visible
independent of pressing.

## What's solid vs. what needs tuning

**Solid / done:** Isaac Sim + Isaac Lab on this H100 (incl. the Vulkan fix, see
`SETUP.md`), the 88-key sprung piano, the combined arm+hand articulation, the
MIDI→goal pipeline, the decoupled control (analytic arm IK + residual-RL fingers),
the composite shaped reward, and rich obs — the env builds, steps, and *trains* on
GPU (reward climbs immediately).

**Needs tuning for a *good* player (clearly-marked knobs):**
1. **Hand→flange mount + base poses / slider calibration.** Tune
   `build_combined_usd.py`'s `--mount-xyz/--mount-rpy` and
   `PianoEnvCfg.left/right_base_pos` (or the slider calibration map) so each hand's
   workspace actually covers its half of the keyboard. Better placement → higher
   ceiling for the policy.
2. **Reward weights.** `fingering_weight` (the critical shaping term) vs
   `key_press_weight`/`onset_weight`/`energy_weight`.
3. **Curriculum.** Train on a short/slow song first (Twinkle), then harder pieces.
4. **Scale + time.** Use `--num_envs 4096+` on the H100 and budget real GPU-hours.

## Honest expectations

Bimanual dexterous piano from RL is hard — RoboPianist (the reference) used
floating hands, not full arms, and still needed careful shaping. Adding two
6-DoF arms enlarges the search a lot. Expect to iterate on the mount transform /
slider calibration and reward before the policy plays cleanly; budget real
GPU-hours for training. The scaffold here is built so that iteration is the only
thing left.
