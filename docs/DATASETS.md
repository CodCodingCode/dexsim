# Datasets — BODex-Tabletop & DexGraspNet (UR10e + Shadow)

The imitation path uses MuJoCo-validated grasp **trajectories** on the exact
UR10e+Shadow embodiment. **The data is already on this box** (downloaded in a
prior session) — no need to re-fetch:

```
~/DexGraspBench/downloads/
  ur10e_shadow_extracted/bodex_ur10e_shadow/succ_collect/<object>/scaleNNN_poseNNN.npy
  dexgraspnet/extracted/dexgraspnet/<object>.npy
  object_assets/ , synthesized_grasps/
```

## BODex file format (per `.npy`, a 0-d object array → dict)

| key | shape | meaning |
|---|---|---|
| `approach_qpos` | (G, T, 30) | G grasps × T waypoints × 30 DoF — **the trajectory** |
| `pregrasp/grasp/squeeze/lift_qpos` | (G, 30) | keyframe poses |
| `obj_pose` | (7,) | object pose (xyz + wxyz) |
| `obj_scale` | scalar | mesh scale |
| `obj_path` | str | mesh path |

The 30 DoF are in **MuJoCo order** (6 arm + `WRJ2,WRJ1` + 4×finger + `LFJ5..1` +
`THJ5..1`). `dexsim.tasks.grasp.bodex_loader` parses this and — crucially —
remaps to the Isaac articulation order: arm joints are identical, and
`hand:rh_<F>J<n>` → `robot0_<F>J<n-1>` (MuJoCo is 1-indexed, the Isaac
instanceable hand is 0-indexed). `BODexTrajectory.reorder_to(robot.data.joint_names)`
does the permutation. Verified on real files (mug: 31 frames, 30 DoF).

## Use it

```bash
source env.sh
# replay a real BODex grasp on the combined UR10e+Shadow embodiment
python scripts/prep/replay_bodex.py --headless \
  --traj ~/DexGraspBench/downloads/ur10e_shadow_extracted/bodex_ur10e_shadow/succ_collect/core_mug_*/scale*_pose000.npy
```

`scripts/prep/download_bodex.py --local` symlinks the on-disk data into `data/bodex/`
for convenience; pass `--list` to browse, or a `--repo-id` to fetch fresh from
the Hub if you ever need to.

## DexGraspNet

`~/DexGraspBench/downloads/dexgraspnet/extracted/dexgraspnet/*.npy` — 5355 objects,
**static** grasp poses (not trajectories), good as grasp *seeds/targets* for
trajectory generation or as object diversity. Same Shadow joint conventions.

> Note: **Dex1B is not publicly downloadable** as of 2026-06 (project page only).
> Don't plan the pipeline around it.
