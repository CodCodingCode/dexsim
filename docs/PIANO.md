# Bimanual piano task

Two UR10e + Shadow Hand arms (~30 DoF each → **60 action DoF**) over an 88-key
spring-loaded piano, trained with PPO to play a specific MIDI song.

## How it fits together

```
data/midi/<song>.mid
        │  scripts/make_test_midi.py writes a stand-in (Twinkle) for development
        ▼
dexsim.piano.load_song(...)        -> (T,88) key goal + (T,88) onsets
dexsim.piano.plan_fingering(...)   -> per-step finger->key assignment (10 fingers)
        ▼
scripts/build_reference.py  (multi-fingertip IK)  -> data/reference/<song>.npz
        │  drives the fingertips onto their assigned keys, records q_ref(T,2,30)
        ▼
Dexsim-Piano-Bimanual-v0  (DirectRLEnv)            [PianoMime-style]
  action (60):  RESIDUAL on the IK reference:  target = q_ref[t] + scale·a
  observation:  arms pos+vel (120) | key angles (88) | goal lookahead (880)
                | fingertip pos (30) | reference fingertip targets (30)
                | SDF goal encoding (88)              = 1236
  reward:       key-press (right keys, none wrong) + FINGERING shaping
                (finger→assigned key, the make-or-break term) + onset + energy
        ▼
scripts/bc_pretrain.py   (optional: BC the IK expert -> actor warm-start)
scripts/train_piano.py   (rsl_rl PPO, --bc_init optional)
scripts/eval_reference.py(F1/recall/precision of reference or a policy)
scripts/play_piano.py    (roll out + EXPORT what it played back to .mid)
scripts/distill_generalist.py (PianoMime generalist: distill songs -> diffusion)
```

## PianoMime port (what makes the 60-DoF problem trainable)

Every piece below is structure-injection so RL doesn't explore 60 DoF from
scratch (code in `source/dexsim/piano/`):
- **Fingering** (`fingering.py`): assigns each note to a finger. RoboPianist showed
  the policy scores **F1=0 without a fingering signal** — this is the critical term.
- **IK reference** (`ik.py` + `build_reference.py`): the arm's redundancy is absorbed
  by IK; the policy only learns a **residual** on the recorded `q_ref`. Zero action
  already tracks the reference, so the policy starts competent.
- **Composite reward** (`reward.py`): key-press + `fingering_reward` (gaussian
  `tolerance`, RoboPianist constants) + `onset_reward` + energy.
- **Rich obs** incl. fingertip positions, the reference fingertip targets ("where
  fingers should go"), and an **SDF goal encoding** (`goal_encoding.py`).
- **Generalist** (`generalist.py` + `distill_generalist.py`): hierarchy + conditional
  DDPM to distill many song specialists into one policy for unseen songs.

Validated: building the Twinkle reference runs the IK pipeline end-to-end; a
15-iteration PPO smoke test moved **mean reward −0.11 → 10.94** (monotonic) — the
shaped reward gives an immediate, strong learning signal.

### Typical workflow
```bash
python scripts/build_reference.py --midi data/midi/twinkle.mid --headless   # once per song
python scripts/eval_reference.py  --midi data/midi/twinkle.mid --zero --headless  # sanity
python scripts/train_piano.py --headless --num_envs 4096 --midi data/midi/twinkle.mid
```

Key assets (generated once, already built here):
- `assets/piano88.usd` — 88 keys, each a sprung hinge (`joint_0..joint_87`,
  index = MIDI−21). A key "sounds" when its hinge angle ≤ `KEY_SOUND_ANGLE`.
  Rebuild: `python scripts/build_piano_usd.py`.
- `assets/ur10e_shadow.usd` — UR10e + Shadow bonded into one articulation
  (`wrist_3_link → robot0_forearm`). Rebuild:
  `python scripts/build_combined_usd.py --inspect` then without `--inspect`.

## Use your own song

```bash
cp /path/to/your_song.mid data/midi/your_song.mid
python scripts/train_piano.py --headless --num_envs 2048 --midi data/midi/your_song.mid
python scripts/play_piano.py  --num_envs 1 --video --midi data/midi/your_song.mid \
       --export_midi logs/your_song_played.mid
```
`play_piano.py` records the keys the policy actually pressed and writes them to a
MIDI file so you can hear what it learned.

## What's solid vs. what needs tuning

**Solid / done:** Isaac Sim + Isaac Lab on this H100 (incl. the Vulkan fix, see
`SETUP.md`), the 88-key sprung piano, the combined arm+hand articulation, the
MIDI→goal pipeline, and the full **PianoMime port** (fingering, IK reference,
residual action, composite shaped reward, rich obs, generalist scaffold) — the
env builds, steps, and *trains* on GPU (reward climbs immediately).

**Needs tuning for a *good* player (clearly-marked knobs):**
1. **Hand→flange mount + base poses.** This is now the #1 lever. The IK reference
   only got active fingertips to ~80 mm of their keys on average (54% within
   2 cm); the rest is reach/mount geometry. Tune `build_combined_usd.py`'s
   `--mount-xyz/--mount-rpy` and `PianoEnvCfg.left/right_base_pos` so each hand's
   workspace actually covers its half of the keyboard, then rebuild the reference.
   Lower reference fingertip error → higher ceiling for the policy.
2. **Reward weights.** `fingering_weight` (the critical shaping term) is now
   implemented; tune it vs `key_press_weight`/`onset_weight`/`energy_weight`.
3. **Curriculum.** Train on a short/slow song first (Twinkle), then harder pieces.
4. **Scale + time.** Use `--num_envs 4096+` on the H100 and budget real GPU-hours;
   15 iters only proves the signal, not a finished player.

## Honest expectations

Bimanual dexterous piano from RL is hard — RoboPianist (the reference) used
floating hands, not full arms, and still needed careful shaping. Adding two
6-DoF arms enlarges the search a lot. Expect to iterate on the mount transform
and reward before the policy plays cleanly; budget real GPU-hours for training.
The scaffold here is built so that iteration is the only thing left.
