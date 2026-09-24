# Best result: deterministic F1 0.673 on the REAL nettspend song

`results/nettspend - we not like you (1).mid` at its real pitches, full
keyboard, no folding. Deterministic whole-song rollout of `model_final.pt`:

    F1 0.673   recall 0.548   precision 0.871      (iteration 2984)

For scale, RoboPianist reports ~0.8 on this metric with free-flying hands,
human fingerings and a pedal. This embodiment slides each hand on a 1-DoF rail.

## How this run was produced

Trained as `nettspend_v5_a100` from scratch to iteration 1423, where the
process was killed externally (no error, no OOM). Resumed from its own
`model_1400.pt` as `v5_continue` with byte-identical environment code and
run to 3000. The resume was verified exact: training F1 read 0.547 on the
first iteration after loading, matching 0.542 where the original left off.

Run from an isolated tree (`~/dexsim_piano_v6` on the box) because a second
session was editing `~/dexsim` concurrently.

    python scripts/mj/train_piano_mj.py \
        --midi "results/nettspend - we not like you (1).mid" --episode_s 65 \
        --num_envs 1024 --workers 28 --device cuda \
        --max_iterations 3000 --save_interval 100 --eval_every 100 \
        --anneal_false_press --seed 0

Recipe: egocentric 1235-dim observation, asymmetric critic, `fingering_method
= "hand"`, rail servo with leave-early + newest-note targeting, sustain pedal
action, adaptive per-key reward weights with `key_weight_min_static = 1.0`,
+/-2-step onset window, 100 ms action low-pass, `jerk_weight` 0.3,
`entropy_coef` 0.001.

## Deterministic evaluation history

| iter | F1        | recall    | precision |
| ---- | --------- | --------- | --------- |
| 1499 | 0.624     | 0.518     | 0.784     |
| 1895 | 0.652     | 0.538     | 0.827     |
| 2390 | 0.625     | 0.500     | 0.834     |
| 2588 | 0.650     | 0.521     | 0.864     |
| 2687 | 0.659     | 0.544     | 0.835     |
| 2885 | 0.670     | 0.547     | 0.865     |
| 2984 | **0.673** | **0.548** | **0.871** |

**It was still improving when the budget ran out.** The best score is the
final evaluation, and the last three are the three highest. Between 1900 and
2600 the curve looked converged around 0.64 while precision rose and recall
fell; it then broke out when both rose together. A longer run is the single
cheapest next experiment.

## Progress on this song

|                               | deterministic F1 | note                           |
| ----------------------------- | ---------------- | ------------------------------ |
| folded into two 8-key windows | 0.84             | transposed; not the real track |
| real song, v2                 | 0.44             | no pedal                       |
| real song + pedal (pedal2)    | 0.52             |                                |
| real song, v5 @1400           | 0.605            |                                |
| **real song, this run**       | **0.673**        |                                |

## What did not work: the v6 experiment

Three changes aimed at the two little-finger keys (28 and 54, stuck near zero
recall in every run): white-key press points moved 3.5 cm toward the front
edge, fingering-shaping falloff tightened 25 cm -> 4 cm, and a per-finger
adaptive reward weight. Run from scratch under the same settings, it
plateaued at training F1 0.30 from iteration 700 and its deterministic evals
settled near 0.34 (recall 0.207, precision 0.975).

The failure mode is instructive: it learned to play almost flawlessly clean
while attempting only a fifth of the notes. Recall was pinned at ~0.21 from
iteration 150 onward. The tighter falloff stops paying a finger for being
near a key it has not mastered, so the policy never explores those keys. The
per-finger weighting was meant to offset that and could not.

The moved press points remain untested in isolation and still have
independent evidence behind them: a scripted press showed they eliminate the
ring finger catching F#4 while playing F4, which was 99 false key-steps in
v5. Retest that one alone, with the falloff left at 25.
