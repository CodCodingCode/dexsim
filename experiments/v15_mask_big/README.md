# v15_mask_big — the code behind F1 0.803 (2026-09-28/29)

Snapshot of the training-box tree `~/dexsim_piano_v15` that produced the best
result so far on the real nettspend song: deterministic whole-song
**F1 0.803 (recall 0.715, precision 0.914)** at iteration 5346 of the 7000-iteration
run `v15_mask_big` (1024 envs x 32 steps, PPO, from scratch, no demos).

This tree is the v10 -> v15 chain of patches applied to the 2026-09-15 box
tree, NOT the repo's `master` sources (which diverged: keyboard scene,
`ego_rail_obs`, `song_speed`). Use it as-is to reproduce or score v13/v14/v15
checkpoints; do not mix its `source/` with master's.

## What it contains

| path | what |
|---|---|
| `source/`, `scripts/` | the exact env / planner / train / play code of the run (Isaac-era leftovers dropped; package inits taken from master) |
| `patches/patch_v10..v15.py` | the ordered patch chain: v10 4-step roll, v11 pedal span + split key + weight floor, v12 leap curriculum, v14 plan-following servo + planned-finger reward, v14_hold per-finger hold actions, v15 reach mask + thumb exclusion + `--hidden_dims` |
| `README_v13.md`, `README_v14.md` | design notes for the seq planner and the hold actions |
| `reach_mask_poseG.npy` | (10, 88) bool feasibility mask per (finger, key) at pose G, from `scripts/mj/reach_check.py` |
| `verify_v13/v14/v15.py`, `verify_hold.py` | offline checks of the plan / hold mechanics |
| `launch/` | the box launch scripts (v13, v14, v15) and the v14 chain script |

## Recipe (v15)

v12 recipe (leap curriculum, adaptive key weights, recall-gated false-press
anneal, `entropy_coef` 0.001) + whole-song `seq` fingering planner (Viterbi
over palm positions) + pose G + per-finger HOLD actions instead of a pedal
(52 actions, 1272 obs) + rail servo following the planned palm + planned-finger
fingering reward + hard per-(finger,key) reach mask with thumbs excluded from
assignment + a 1024-512-256 MLP (`--hidden_dims 1024,512,256`).

```bash
# train (box):  see launch/launch_v15_mask.sh  ->  bash launch_v15_mask.sh v15_mask_big 1024,512,256
# score / render a checkpoint from THIS directory (the reach mask resolves from cwd):
export PYTHONPATH=source
python scripts/mj/diag_rollout.py <model.pt> "../../results/nettspend - we not like you (1).mid" seq
python scripts/mj/play_piano_mj.py --checkpoint <model.pt> \
    --midi "../../results/nettspend - we not like you (1).mid" --episode_s 65 \
    --fingering seq --hidden_dims 1024,512,256 --camera front --video out.mp4 --export_midi out.mid
```

Caveat for comparisons: the per-finger hold is a sim affordance more lenient
than a real pedal. v12 (0.776, real pedal) is the cleanest number against
RoboPianist / RP1M.
