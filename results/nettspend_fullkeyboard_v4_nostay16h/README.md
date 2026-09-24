# nettspend_fullkeyboard_v4_nostay16h — F1 0.50, the 16-hour run

Run `nettspend_nostay16h_a100` (2026-09-14/15), launched from the other
session's checkout `~/dexsim_ab` with its "nostay" setup (1174-dim ego obs
variant; see that checkout's git log for what it changes), 5200 iterations
x 32k env steps, 24 workers, ~16 h. Only the budget differs from the
1700-iteration `nettspend_nostay_a100` it replaced.

Training-time F1 vs the pedal run (pedal2) at the same iteration: ahead by
up to 18% until ~1000, converged by ~2000, then a flat plateau at 0.42-0.43
from iteration ~2400 to the end (2800 iterations with no gain). Action std
drifted 0.44 -> 0.54 over the run.

Deterministic rollout of model_final.pt, whole song from step 0:

    F1 0.504  recall 0.353  precision 0.878
    LEFT  recall 0.309 precision 0.863 | RIGHT recall 0.461 precision 0.903
    onsets 327: hit 209, missed 118 (36%); hit latency median 0 steps

vs the 6-hour pedal run (v3, F1 0.518 / 0.370 / 0.864): LEVEL. Precision
+0.014, recall -0.017. 16 hours bought nothing over 6.

Per key it is a reshuffle, not an addition: keys 30/32/37/47/49 are now at
0.8-1.0 (were 0.3-0.8), while 33 fell 0.57 -> 0.06 and 26 fell 1.00 -> 0.27.
The structural misses are unchanged: 28 (0.05), 54 (0.00), 40 (0.11), 13
(0.27). More training time does not touch those; that needs the weighting
fix (inverse-frequency prior starves the most-played keys), a wider onset
window, and the per-note key-28 analysis (see docs/research_journal.md).
