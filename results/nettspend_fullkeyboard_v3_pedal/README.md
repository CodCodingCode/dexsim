# nettspend_fullkeyboard_v3_pedal — F1 0.52 on the REAL nettspend song

Run `nettspend_pedal2_a100`, 2026-09-12/13, 3000 iters x 32k env steps
(16 workers, shared box). Builds on v2 (F1 0.44) with:

- **Sustain pedal** (43rd action, RoboPianist-style): while > `pedal_threshold`
  (0.5, so exploration noise doesn't flap it) a sounding key keeps sounding
  after the finger lifts. A geometry-derived **pedal goal** (down when one
  hand's active notes span > 14 cm) is in the obs and rewarded at 0.3/step.
- From the other session's commits (8480f80, 48e173e, 33daf37): random song
  start per episode, episode = whole song, egocentric obs + absolute 88-key
  angles + 10-step piano roll (1235 dims), idle-clear term removed.

Deterministic rollout of model_final.pt, whole song from step 0
(`diag_rollout.py ... hand`; note the script now forces random_song_start off):

    F1 0.518  recall 0.370  precision 0.864
    LEFT  recall 0.360 precision 0.890 | RIGHT recall 0.395 precision 0.812
    onsets 327: hit 204, missed 123 (38%); hit latency median 0 steps
    pedal down 34% of steps vs goal 37%

vs v2: F1 0.44 -> 0.52, precision 0.75 -> 0.86, onsets missed 46% -> 38%.
Key 13 (held bass, 1 s notes) went 0.00 -> 0.41: the pedal works where it
is used. Still near zero: key 28 (470 goal steps, 0.02), 54 (0.03),
40 (0.07); key 56 fell 0.46 -> 0.22. These are the remaining recall gap;
the next step is a per-note look at 28 and 54 specifically (why the finger
never strikes them even though the servo arrives).

Training-time F1 0.43 (vs 0.38 no-pedal). Action std stayed ~0.5 (the
other session's setup may have changed entropy handling -- check ppo_cfg).
