"""Convert dexsim's compact rollout files into scrub-able Rerun recordings."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def _safe_entity_name(name: object) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=Path(__file__).resolve().parent,
                              capture_output=True, text=True, timeout=5,
                              check=True).stdout.strip()
    except Exception:
        return None


# --- falling-note roll ("waterfall") above the keyboard --------------------- #
NOTE_LOOKAHEAD_S = 2.0        # how far ahead of the cursor upcoming notes appear
NOTE_FALL_SPEED = 0.25        # m/s of box travel -> 2 s lookahead = 0.5 m column
NOTE_GAP = 0.015              # clearance between key top and an arriving note


def _note_segments(goal: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous True-runs of goal (T,88) as (key, start, end) with end exclusive."""
    segments = []
    padded = np.zeros((goal.shape[0] + 2,), dtype=np.int8)
    for k in range(goal.shape[1]):
        padded[1:-1] = goal[:, k]
        edges = np.flatnonzero(np.diff(padded))
        for s, e in zip(edges[::2], edges[1::2]):
            segments.append((k, int(s), int(e)))
        padded[:] = 0
    return segments


def note_roll_frames(goal: np.ndarray, key_xyz: np.ndarray,
                     key_is_black: np.ndarray, dt: float):
    """Per-frame note boxes: {upcoming,active} -> (centers, half_sizes, counts).

    Synthesia-style: a note's box slides down at NOTE_FALL_SPEED, its bottom
    face reaches the key exactly at onset, then stays anchored there and is
    consumed (shrinks) while the hold lasts. Box height = hold duration.
    numpy-only on purpose: rrd_from_dump.py (rerun venv) imports this too.
    """
    half_w = np.where(key_is_black, 0.006, 0.010)
    segments = _note_segments(goal)
    out = {kind: ([], [], []) for kind in ("upcoming", "active")}
    for t in range(goal.shape[0]):
        now = t * dt
        boxes = {kind: ([], []) for kind in out}
        for k, s, e in segments:
            onset, end = s * dt, e * dt
            if onset - now >= NOTE_LOOKAHEAD_S or end <= now:
                continue
            bottom = NOTE_GAP + max(onset - now, 0.0) * NOTE_FALL_SPEED
            top = NOTE_GAP + min(end - now, NOTE_LOOKAHEAD_S) * NOTE_FALL_SPEED
            if top <= bottom:
                continue
            centers, halves = boxes["active" if onset <= now else "upcoming"]
            centers.append((key_xyz[k, 0], key_xyz[k, 1],
                            key_xyz[k, 2] + 0.5 * (bottom + top)))
            halves.append((0.008, half_w[k], 0.5 * (top - bottom)))
        for kind, (centers, halves) in boxes.items():
            out[kind][0].extend(centers)
            out[kind][1].extend(halves)
            out[kind][2].append(len(centers))
    return out


def _blueprint():
    """Curated layout saved into the .rrd: 3D scene left, metrics right."""
    import rerun.blueprint as rrb

    # Joint plots are dense; a cursor-relative window keeps them readable.
    window = rrb.VisibleTimeRange(
        "sim_time",
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-2.0),
        end=rrb.TimeRangeBoundary.cursor_relative(seconds=2.0))
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/world", name="Scene"),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="/metrics/reach_error_m", name="Reach error (m)"),
                rrb.TimeSeriesView(origin="/metrics/notes", name="Notes: goal / sounding / correct"),
                rrb.Tabs(
                    rrb.TimeSeriesView(origin="/joints/left", name="Left joints", time_ranges=window),
                    rrb.TimeSeriesView(origin="/joints/right", name="Right joints", time_ranges=window),
                    rrb.TimeSeriesView(origin="/piano/key_travel", name="Key travel (rad)"),
                ),
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def rollout_npz_to_rrd(npz_path: str | Path, rrd_path: str | Path,
                       name: str | None = None,
                       properties: dict | None = None) -> Path:
    """Write a Rerun recording from ``record_rollout.py`` output.

    The resulting timeline contains the two palm and target positions, reach
    errors, requested/sounding piano keys, and all recorded arm/hand joints,
    plus a baked-in blueprint layout and recording properties (source file,
    frame count, git commit, and anything passed via ``properties``).
    This function deliberately has no Isaac imports, so conversion is instant.
    All time series go through ``send_columns`` (one batched call per entity),
    not a per-frame log loop.
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
    n = len(left)

    # Imported here to keep the module cheap until conversion is actually used.
    from dexsim.piano.geometry import KEY_IS_BLACK, key_local_top_positions

    rec = rr.RecordingStream("dexsim-rollout")
    rec.save(str(rrd_path))
    rec.send_blueprint(_blueprint())
    rec.send_recording_name(name or npz_path.stem)
    commit = _git_commit()
    rec.send_property("dexsim", rr.AnyValues(
        source=str(npz_path), frames=n, control_dt=dt,
        **({"git_commit": commit} if commit else {}), **(properties or {})))

    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    key_xyz = key_local_top_positions() + piano_pos
    rec.log("world/piano/white_keys", rr.Points3D(key_xyz[~KEY_IS_BLACK], radii=0.007, colors=[220, 220, 220]), static=True)
    rec.log("world/piano/black_keys", rr.Points3D(key_xyz[KEY_IS_BLACK], radii=0.006, colors=[30, 30, 30]), static=True)

    # Per-frame styling never changes -> log it once as static; the columns
    # below then only carry positions.
    rec.log("world/hands/palms",
            rr.Points3D.from_fields(radii=0.018, colors=[[70, 140, 255], [255, 110, 90]],
                                    labels=["left palm", "right palm"]), static=True)
    rec.log("world/hands/targets",
            rr.Points3D.from_fields(radii=0.014, colors=[[70, 255, 180], [255, 210, 70]],
                                    labels=["left target", "right target"]), static=True)
    rec.log("world/piano/goal_keys", rr.Points3D.from_fields(radii=0.010, colors=[40, 230, 80]), static=True)
    rec.log("world/piano/sounding_keys", rr.Points3D.from_fields(radii=0.013, colors=[255, 80, 50]), static=True)
    rec.log("world/piano/roll/upcoming",
            rr.Boxes3D.from_fields(colors=[130, 175, 255, 130], fill_mode="solid"), static=True)
    rec.log("world/piano/roll/active",
            rr.Boxes3D.from_fields(colors=[40, 230, 80, 210], fill_mode="solid"), static=True)
    rec.log("metrics/reach_error_m",
            rr.SeriesLines(names=["left", "right"], colors=[[70, 140, 255], [255, 110, 90]]), static=True)
    rec.log("metrics/notes",
            rr.SeriesLines(names=["goal", "sounding", "correct"],
                           colors=[[40, 230, 80], [255, 80, 50], [235, 235, 235]]), static=True)
    joint_names = [_safe_entity_name(j) for j in names]
    rec.log("joints/left", rr.SeriesLines(names=joint_names[:left.shape[1]]), static=True)
    rec.log("joints/right", rr.SeriesLines(names=joint_names[:right.shape[1]]), static=True)

    steps = np.arange(n)
    times = [rr.TimeColumn("step", sequence=steps),
             rr.TimeColumn("sim_time", duration=steps * dt)]

    rec.send_columns("world/hands/palms", indexes=times,
                     columns=rr.Points3D.columns(positions=palm.reshape(-1, 3)).partition([2] * n))
    rec.send_columns("world/hands/targets", indexes=times,
                     columns=rr.Points3D.columns(positions=target.reshape(-1, 3)).partition([2] * n))
    for entity, mask in (("world/piano/goal_keys", goal), ("world/piano/sounding_keys", sound)):
        rec.send_columns(entity, indexes=times,
                         columns=rr.Points3D.columns(positions=key_xyz[np.nonzero(mask)[1]])
                         .partition(mask.sum(axis=1)))

    roll = note_roll_frames(goal, key_xyz, KEY_IS_BLACK, dt)
    for kind, (centers, halves, counts) in roll.items():
        rec.send_columns(f"world/piano/roll/{kind}", indexes=times,
                         columns=rr.Boxes3D.columns(
                             centers=np.asarray(centers, np.float32).reshape(-1, 3),
                             half_sizes=np.asarray(halves, np.float32).reshape(-1, 3))
                         .partition(counts))

    reach_err = np.linalg.norm(palm - target, axis=2)
    reach_err[~active] = np.nan                            # inactive hand -> gap in the plot
    rec.send_columns("metrics/reach_error_m", indexes=times,
                     columns=rr.Scalars.columns(scalars=reach_err))
    notes = np.stack([goal.sum(axis=1), sound.sum(axis=1),
                      np.logical_and(goal, sound).sum(axis=1)], axis=1)
    rec.send_columns("metrics/notes", indexes=times,
                     columns=rr.Scalars.columns(scalars=notes.astype(np.float64)))
    rec.send_columns("joints/left", indexes=times, columns=rr.Scalars.columns(scalars=left))
    rec.send_columns("joints/right", indexes=times, columns=rr.Scalars.columns(scalars=right))
    rec.send_columns("piano/key_travel/max_abs_rad", indexes=times,
                     columns=rr.Scalars.columns(scalars=np.abs(keys).max(axis=1)))

    rec.flush()
    return rrd_path
