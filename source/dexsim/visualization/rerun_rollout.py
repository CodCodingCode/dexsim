"""Convert dexsim's compact rollout files into scrub-able Rerun recordings."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _safe_entity_name(name: object) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


def rollout_npz_to_rrd(npz_path: str | Path, rrd_path: str | Path) -> Path:
    """Write a Rerun recording from ``record_rollout.py`` output.

    The resulting timeline contains the two palm and target positions, reach
    errors, requested/sounding piano keys, and all recorded arm/hand joints.
    This function deliberately has no Isaac imports, so conversion is instant.
    """
    import rerun as rr

    npz_path, rrd_path = Path(npz_path), Path(rrd_path)
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(npz_path, allow_pickle=True) as data:
        required = {"left", "right", "keys", "goal", "sound", "palm", "target"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{npz_path} is missing rollout fields: {', '.join(missing)}")

        left, right = np.asarray(data["left"]), np.asarray(data["right"])
        keys = np.asarray(data["keys"])
        goal, sound = np.asarray(data["goal"], bool), np.asarray(data["sound"], bool)
        palm, target = np.asarray(data["palm"]), np.asarray(data["target"])
        active = np.asarray(data["target_active"], bool) if "target_active" in data else np.ones(goal.shape[:1] + (2,), bool)
        dt = float(data["control_dt"]) if "control_dt" in data else 1.0 / 60.0
        names = list(data["joint_names_left"]) if "joint_names_left" in data else [f"joint_{i}" for i in range(left.shape[1])]
        piano_pos = np.asarray(data["piano_pos"], dtype=np.float32) if "piano_pos" in data else np.zeros(3, np.float32)

    lengths = {len(x) for x in (left, right, keys, goal, sound, palm, target, active)}
    if len(lengths) != 1:
        raise ValueError(f"rollout arrays have inconsistent frame counts: {sorted(lengths)}")

    # Imported here to keep the module cheap until conversion is actually used.
    from dexsim.piano.geometry import KEY_IS_BLACK, key_local_top_positions

    rec = rr.RecordingStream("dexsim-rollout")
    rec.save(str(rrd_path))
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    key_xyz = key_local_top_positions() + piano_pos
    rec.log("world/piano/white_keys", rr.Points3D(key_xyz[~KEY_IS_BLACK], radii=0.007, colors=[220, 220, 220]), static=True)
    rec.log("world/piano/black_keys", rr.Points3D(key_xyz[KEY_IS_BLACK], radii=0.006, colors=[30, 30, 30]), static=True)

    for t in range(len(left)):
        rec.set_time("step", sequence=t)
        rec.set_time("sim_time", duration=t * dt)
        rec.log("world/hands/palms", rr.Points3D(palm[t], radii=0.018, colors=[[70, 140, 255], [255, 110, 90]], labels=["left palm", "right palm"]))
        rec.log("world/hands/targets", rr.Points3D(target[t], radii=0.014, colors=[[70, 255, 180], [255, 210, 70]], labels=["left target", "right target"]))
        for hand, label in enumerate(("left", "right")):
            err = float(np.linalg.norm(palm[t, hand] - target[t, hand])) if active[t, hand] else np.nan
            rec.log(f"metrics/reach_error_m/{label}", rr.Scalars(err))
        rec.log("metrics/notes/goal_count", rr.Scalars(int(goal[t].sum())))
        rec.log("metrics/notes/sounding_count", rr.Scalars(int(sound[t].sum())))
        rec.log("metrics/notes/correct_count", rr.Scalars(int(np.logical_and(goal[t], sound[t]).sum())))
        rec.log("world/piano/goal_keys", rr.Points3D(key_xyz[goal[t]], radii=0.010, colors=[40, 230, 80]))
        rec.log("world/piano/sounding_keys", rr.Points3D(key_xyz[sound[t]], radii=0.013, colors=[255, 80, 50]))
        for j, name in enumerate(names):
            path = _safe_entity_name(name)
            rec.log(f"joints/left/{path}", rr.Scalars(float(left[t, j])))
            if j < right.shape[1]:
                rec.log(f"joints/right/{path}", rr.Scalars(float(right[t, j])))
        rec.log("piano/key_travel/max_abs_rad", rr.Scalars(float(np.abs(keys[t]).max())))

    rec.flush()
    return rrd_path
