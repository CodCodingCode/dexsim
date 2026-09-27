# v14 = v13 (seq planner + pose G + v12 leap curriculum) + 3 setup changes (2026-09-27)

See patch_v14.py (applied on top of the v13 tree). Measured on v13 @2300 before the change:
recall 0.587 = 24% of goal key-steps never struck + 18% lost INSIDE struck notes (pedal lifted
102x/64 s as a clear button) + right-hand 2-note chords 16/44 (octave 44+56 0/14, palm 1.6-2.8 cm short).

1. `score_sustain=True`, `sustain_pedal=False`: a struck GOAL note keeps sounding until its goal
   ends; non-goal keys stop on release. Obs 1235 -> 1232, actions 43 -> 42.
2. `rail_follow_plan=True`: the servo targets `bank.seq_palm` (leave-early on the plan's next
   change). Zero-action tracking error vs plan: median 0.4 cm (was 2.0-2.6 cm).
3. `fingering_online=False`: fingering shaping pulls the PLANNED finger (thumb got its gradient).

Score with `diag_rollout.py <ckpt> "results/nettspend - we not like you (1).mid" seq`.

## Revision (same day, user request): sustain is a POLICY decision, per finger
`patch_v14_hold.py` replaces the automatic score sustain with 10 hold actions (last dims):
while hold_f > 0.5 the key finger f most recently struck keeps ringing after it lifts (any key,
so held wrong notes still pay the false-press penalty). Striker = tip within 1.5 cm laterally,
lowest over the key, at the strike. Obs +30 per-finger dims (holding a ringing note, it is a
goal now, steps until its goal ends /40) + 10 prev-action dims -> 1272; actions 52.
Reward: hold_goal_weight 0.3 * mean over ringing owned keys of [hold == key is a goal].
Verified (verify_hold.py): R.mf strikes key 63 -> owner set; lifted 1 s with hold on -> still
ringing; hold off -> silent within 0.25 s. Run tag: v14_hold_servo.
