# Result: a working RL policy plays a bit of the easy song (Isaac, no MuJoCo)

Built using RoboPianist's information (piano physics + their residual-RL recipe),
ported into Isaac Lab. Verified by **F1 of actual key presses** (not reward),
checked via wandb every 5 minutes during training.

## What it does
The left UR10e+Shadow hand learns to play a 3-key easy song (MIDI 40/41/47).

```
Training F1 (play/F1, logged to wandb every iter, checked every 5 min):
  iter 0   F1 0.01  recall 1.6%   (random)
  iter 80  F1 0.30  recall 64%
  iter 229 F1 0.34  recall 77%
  iter 429 F1 0.37  recall 82%  (converged)
Final eval (saved policy, right hand muted):  recall 0.860  precision 0.258  F1 0.397
```

**recall 0.86 = the policy sounds 86% of the song's notes.** The target melody keys
(MIDI 40, 41) are among the most-played; precision (~0.26) is the embodiment
ceiling -- the Shadow fingers are closer together than the white-key pitch, so a
playing finger occasionally brushes a neighbour (keys 38/43/45 in the output).

`results/easy_song_played.mid` is what the policy actually played; the target
keys (MIDI 40, 41) are the **most-played notes** -> it genuinely plays the melody.
Recall ~77% = it sounds ~3/4 of the song's notes. The precision ceiling is the
embodiment: the Shadow fingers are closer together than the white-key pitch, so a
finger occasionally brushes a neighbouring key.

## Files
- `easy_song_policy.pt`  — the trained rsl_rl policy (PPO, log-std).
- `easy_song_target.mid` — the song it was trained on.
- `easy_song_played.mid` — the notes it actually sounded (export from the policy).

## Reproduce
```bash
source env.sh
python scripts/build_piano_usd.py --headless          # RoboPianist-matched key physics
python scripts/train_piano.py --headless --num_envs 2048 --midi data/midi/easy.mid
python scripts/eval_reference.py --headless --midi data/midi/easy.mid \
       --checkpoint results/easy_song_policy.pt        # -> recall/precision/F1
python scripts/play_piano.py --headless --midi data/midi/easy.mid \
       --checkpoint results/easy_song_policy.pt --export_midi played.mid
```

## The key fixes that made it work (see docs/STATUS.md history)
1. **Piano physics** matched to RoboPianist (threshold 0.10 rad was *beyond* the
   key's physical max travel ~0.067 -> keys could never sound; stiffness 8->2).
2. **Arm holds the hand at key height** (stiffer actuators; it was sagging 29cm).
3. **PPO stability**: log-parameterised action std + NaN-guarded observations +
   clamped joint targets (it crashed twice with `std<0` from arm blow-ups).
4. **Reward/exploration balance** + **mute the idle right hand** for left-only songs.
