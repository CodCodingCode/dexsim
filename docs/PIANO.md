# Bimanual piano task

Two fixed-base Shadow Hands (24 DoF each → **48 action DoF**) over an 88-key
spring-loaded keyboard, trained with PPO to play a specific MIDI song.

This is **Phase 1: pure dexterous hands, without robot arms**. It deliberately
follows the simpler hands-only setup used by RoboPianist so that fingering and
reliable key pressing are solved before full arm–hand coordination. Adding UR10e
arms is a later Phase 2 milestone.

The keyboard has a realistic 88-key layout and approximately real key dimensions
(52 white/36 black keys, 23.5 mm white-key pitch, ~1.22 m span, 145 mm white-key
length). It is not a full piano digital twin: keys are simple hinged rigid bodies
with return springs; hammers, escapement, strings, pedals, and acoustics are absent.

## How it fits together

```
data/midi/<song>.mid
        │  scripts/prep/make_test_midi.py writes a stand-in (Twinkle) for development
        ▼
dexsim.piano.load_song(...)        -> (T,88) key goal + (T,88) onsets
dexsim.piano.plan_fingering(...)   -> per-step finger->key assignment (10 fingers)
        ▼
Dexsim-Piano-Bimanual-v0  (DirectRLEnv)        [fixed hand roots, RL fingers]
  action (48):      residual on a ready-pose base: target = base + scale·action
  observation:  hands pos+vel (96) | key angles (88) | goal lookahead (880)
                | fingertip pos (30) | fingering targets (30)
                | SDF goal encoding (88)              = 1212
  reward:       key-press (right keys, none wrong) + FINGERING shaping
                (finger→assigned key, the make-or-break term) + onset + energy
        ▼
scripts/train/train_piano.py   (rsl_rl PPO)
scripts/train/eval_reference.py(F1/recall/precision of a policy or the zero-residual base)
scripts/train/play_piano.py    (roll out + EXPORT what it played back to .mid)
```

## What makes the 48-DoF problem trainable

Every piece below is structure-injection so RL doesn't explore 48 DoF from
scratch (code in `source/dexsim/piano/`):
- **Fingering** (`fingering.py`): assigns each note to a finger. RoboPianist showed
  the policy scores **F1=0 without a fingering signal** — this is the critical term.
- **Hands-only control:** fixed root transforms place both hands over reachable
  key windows; the policy learns wrist/finger motion as a **residual** on a static
  ready pose. Phase 1 has no arm joints or arm controller.
- **Composite reward** (`reward.py`): key-press + `fingering_reward` (gaussian
  `tolerance`, RoboPianist constants) + `onset_reward` + energy.
- **Rich obs** incl. fingertip positions, the fingering targets ("where fingers
  should go"), and an **SDF goal encoding** (`goal_encoding.py`).

### Typical workflow
```bash
python scripts/train/eval_reference.py --midi data/midi/twinkle.mid --zero --headless  # sanity
python scripts/train/train_piano.py    --midi data/midi/twinkle.mid --headless --num_envs 4096
```

Key assets (generated once, already built here):
- `assets/piano88.usd` — 88 keys, each a sprung hinge (`joint_0..joint_87`,
  index = MIDI−21). A key "sounds" when its hinge angle ≤ `KEY_SOUND_ANGLE`.
  Rebuild: `python scripts/build/build_piano_usd.py`.
- The hands use Isaac Sim's instanceable standalone Shadow Hand USD directly;
  no combined UR10e asset is required for this task.

## Use your own song

```bash
cp /path/to/your_song.mid data/midi/your_song.mid
python scripts/train/train_piano.py --headless --num_envs 2048 --midi data/midi/your_song.mid
python scripts/train/play_piano.py  --num_envs 1 --video --midi data/midi/your_song.mid \
       --export_midi logs/your_song_played.mid
```
`play_piano.py` records the keys the policy actually pressed and writes them to a
MIDI file so you can hear what it learned.

## Current control model

- Each hand root is fixed over a narrow, reachable key window.
- PPO controls the 48 Shadow Hand wrist/finger joints as residual position targets.
- Wide songs are octave-folded into those windows by default.
- `--arm_ik_follow` and Phase-0 arm training are intentionally unsupported.

## Next validation gate

Open the scene in Isaac Sim and tune `left_base_pos`, `right_base_pos`, and
`hand_base_rot` until fingertips sit above the intended keys. Then require a
scripted one-finger strike to sound one target key reliably before retraining PPO.
