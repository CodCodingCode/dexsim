# nettspend_fullkeyboard_v1 — F1 0.34 on the REAL (unfolded) nettspend song

First full-keyboard result (2026-09-11). Same song as `nettspend_ot_a100`
(F1 0.84) but played at its real pitches: 17 keys spanning 13..72 instead of
11 keys folded into two 8-key windows. Rails +/-0.32 m, fold off.

Recipe (all now cfg defaults): egocentric 314-dim obs, hand-relative
fingering (`fingering_method="hand"`), rail servo + 5 cm policy residual,
adaptive per-key reward weights (ramped), false-press anneal from recall
0.25, idle-finger hover reward 0.2, asymmetric critic. Trained 600 iters
(run nettspend_servo_a100) then resumed to 3000 with the reward change
(nettspend_servo2_a100). Replay: `--fingering hand` (default), no flags.

Deterministic rollout of model_final.pt:

    F1 0.341  recall 0.233  precision 0.634
    LEFT  recall 0.162 precision 0.594 | RIGHT recall 0.405 precision 0.679
    onsets 327: hit 172, missed 155 (47%); hit latency median 1 step
    never sounded: keys 13, 40, 54, 72

Training-time F1 was flat at 0.21-0.27 from iter 800 to 3000 while action
std climbed 0.79 -> 1.47 (entropy_coef 0.006 too high for a long run) and
keys-down/step fell 2.4 -> 1.2: the policy learned to press LESS (precision
up, recall down) rather than to press more of the right keys.

Next levers, in order: entropy_coef 0.006 -> ~0.001 (or cap std); servo
lookahead by travel time (hand leaves early; 34% of hits >= 3 steps late);
per-note reward normalization. See docs/research_journal.md.
