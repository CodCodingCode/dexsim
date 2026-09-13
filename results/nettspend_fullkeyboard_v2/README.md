# nettspend_fullkeyboard_v2 — F1 0.44 on the REAL (unfolded) nettspend song

Run `nettspend_fastservo_a100`, 2026-09-11/12, 3000 iters x 32k env steps
(~7.5 h on the 30-core A100 box). Same task as `nettspend_fullkeyboard_v1`
(F1 0.34); two changes, both now cfg defaults:

- **entropy_coef 0.006 -> 0.001** (`ppo_cfg.py`). v1's action std ran away
  0.79 -> 1.47 from iter 800 and F1 went flat; here it held 0.50 -> 0.35 and
  F1 kept climbing to the end (training F1 0.23 @800 -> 0.38 @3000).
- **Fast rail servo**: stiffer rail gains (6000 N/m, critically damped,
  2 kN; 30 cm settles in 150 ms not 250) + `rail_leave_early` planner that
  leaves for the next onset when time-left <= travel time and pre-positions
  when idle. Zero-policy: finger on its key at onset 18% -> 42%.

Deterministic rollout of model_final.pt (`scripts/mj/diag_rollout.py ... hand`):

    F1 0.441  recall 0.313  precision 0.751
    LEFT  recall 0.260 precision 0.815 | RIGHT recall 0.441 precision 0.675
    onsets 327: hit 178, missed 149 (46%); hit latency median 0 steps

Precision is now high; recall is the whole gap. The three most-played keys
are barely played: 28 (470 goal steps, recall 0.01), 44 (565, 0.12),
13 (320, 0.00). Those three are 35% of all goal key-steps. They are long
bass/held notes: 38-50% of their duration the SAME hand must also play keys

> 14 cm away. A pianist holds them with the SUSTAIN PEDAL (RoboPianist has a
> pedal action; this piano has no pedal joint). With the leave-early servo the
> hand abandons them to reach the next onset, so they are lost by construction.

Next levers: (1) a sustain pedal (env action + latch: a sounding key stays
sounding while the pedal is down) -- likely the single biggest recall gain;
(2) per-note onset reward so skipping a note costs a fixed amount;
(3) the other under-played keys (54: 1.5 s notes, no far conflict, recall
0.06) need a per-note look at why the servo/policy misses them.
