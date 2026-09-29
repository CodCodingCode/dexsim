# v13: sequence fingering planner ("seq") + pose G, on the v12 recipe (2026-09-27)

Tree = copy of dexsim_piano_v12 (v11 split/pedal + v12 leap curriculum) plus:

* `source/dexsim/piano/fingering_seq.py` -- `fingering_method="seq"`: whole-song
  Viterbi over both palms (1 cm grid, travel + per-move cost, palms may not cross,
  note->hand split inside the DP), then fingers by offset from the planned palm
  with continuity (held key keeps its finger; same finger on a different key
  within 10 steps pays 1 cm; history resets when the palm moves). Notes no finger
  can reach are re-issued `stagger_steps` later. Knobs: cfg `seq_*`.
* pose G ready pose (palm z 0.83, wrist -0.20, 30 deg claw, thumb down,
  tip_shift_extra 0.065) ported from local master so the thumb can press.

Measured on nettspend (verify_v13.py): onsets per finger
L th/ff/mf/rf/lf 15/45/25/15/46, R 21/52/17/22/40 (was L 0/28/26/16/18, R 0/8/0/46/8);
planned palm travel L 7.7 m / R 5.6 m (was ~11-12 m each); 44/45 passages on
ff/mf with the palm still; 44+56 octave = thumb + little (16.4 cm, ~1.5 cm stretch each).
