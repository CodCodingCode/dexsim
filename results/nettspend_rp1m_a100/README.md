# nettspend_rp1m_a100 — F1 0.52 on the real (unfolded) nettspend song

Run `nettspend_rp1m_a100`, 2026-09-13, commit 33daf37, 1700 iters x 32k env
steps (~56M steps, 8 h 23 min on the 30-core A100 box with 12 workers while
another run shared the machine). Same task and fingering ("hand") as
`nettspend_fullkeyboard_v2` (F1 0.44 at 3000 iters / ~98M steps).

    python scripts/mj/train_piano_mj.py \
        --midi "results/nettspend - we not like you (1).mid" \
        --num_envs 1024 --workers 12 --device cuda \
        --max_iterations 1700 --save_interval 100 --anneal_false_press --seed 0

Four changes vs v2, all now cfg defaults (RP1M-style setup):

- **random_song_start**: each env resets at a random song step; episode
  length = the song (was a fixed 30 s of a 64 s song, envs in lockstep).
- **sounding_gate = position**: a key sounds when depressed past the sound
  angle, no strike-speed requirement (was the velocity-gated "hammer" latch).
- **fingering_online**: the fingering reward matches the live fingertips to
  the keys due now (min-cost assignment) instead of the planned table.
  The table still drives the obs and rail servo.
- **ego obs reshaped**: no hand joint velocities, +88 absolute key angles,
  +goal piano roll (10 x 88); 1234 dims. Sustain pedal on (pending WIP).

Deterministic full-song rollout of model_final.pt (`~/eval_full.py` on the
box, steps the env directly from song step 0):

    F1 0.516  recall 0.369  precision 0.856
    onsets 327: hit 193, missed 134 (41%)
    same policy scored under the hammer gate: F1 0.518 (gate-insensitive)

Training-time F1 (with exploration noise), 50-iter means, vs the
`nettspend_pedal_a100` baseline (pedal WIP without these changes, stopped at
iter 678):

    iters   base   new
    0-49    0.077  0.103
    200-249 0.150  0.203
    400-449 0.188  0.253
    650-699 0.203  0.296
    1600-1699   -  0.431

Deterministic at iter 600: new 0.294. (The baseline's model_600.pt scored
0.09 when evaluated under this commit with `--legacy_ego`, but its training
env was a different WIP state, so that number is not trustworthy; use the
training curves for the matched comparison.)

Same shape of result as v2: precision is high, recall is the whole gap, and
41% of onsets are never attempted. finger/online_agree ended at 0.52, i.e.
the live matching endorses a different finger than the plan on half of the
assigned notes. Next levers: (1) drop the fingering obs and go RP1M-pure
(obs = proprio + roll only) now that the reward is online; (2) trim the obs
back to the papers' blocks; (3) per-note onset reward for the missed notes.
