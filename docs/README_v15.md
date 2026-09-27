# v15 = v14 + planner split fix + held-note fix + exploration cap (prepared 2026-09-27, not yet trained)

Measured on v14 @300/@500 vs v12 at the same iterations (deterministic full-song rollouts):
keys on index/middle/ring fingers 0.71 recall vs 0.32 on thumbs/little fingers; key 40 (C#4)
on the LEFT THUMB 0.11-0.17 (v12: 0.43 on the right hand); v12's learned action std grew
0.50 -> 0.67 over the run while its greedy policy scored 0.05 above the noisy one.

1. `hand_split_key` (cfg default 40) now reaches the `seq` planner, one-sided: keys >= 40 may
   only go to the right hand, the right hand may still take lower keys. nettspend: key 40 ->
   right index (103 key-steps, 0 on the left thumb); onsets covered 307/327 (was 313);
   unassigned key-steps 1000 (was 985 with the free split after fix 2). A two-sided split
   was tried first and cost key 13 190 key-steps -- that turned out to be fix 2's bug.
2. `seq_keep_held` (planner `keep_held=True`): an onset group with no notes for a hand used
   to WIPE that hand's held notes from the fingering table (key 28 lost 270 key-steps; key 13
   kept its notes only when a drop/re-issue happened to land after the other hand's onset).
   Unassigned key-steps 1370 -> 985 on nettspend with the free split.
3. `--std_cap_start 1000 --std_cap_end 6000 --std_cap_init 0.5 --std_cap_final 0.15`: after
   each learn chunk (<= 50 iters) the Gaussian actor's `std_param` and `std_range[1]` are
   clamped to a linearly decaying cap. Logged as `train/std_cap`. Smoke-tested locally
   (std 0.50 -> 0.45 when the cap hit 0.45).

Replaying v13/v14 checkpoints needs their plan: `diag_rollout.py <ckpt> <mid> seq split=-1
keep_held=0` (play: `--fingering seq --hand_split_key -1 --seq_keep_held 0`). Parity check of
the merged tree: v14 model_500 replays at F1 0.593 locally vs 0.583 on the box.

Launch: `scripts/box/launch_v15.sh` (tag `v15_split_stdcap`, same args as v14 + the cap).
