"""Loader for BODex-Tabletop trajectories (UR10e + Shadow Hand).

BODex (https://github.com/JYChen18/BODex) / DexGraspBench emit MuJoCo-validated
grasp *trajectories* for the UR10e+Shadow embodiment. On disk each object dir
holds ``scaleNNN_poseNNN.npy`` files; each is a 0-d object array wrapping a dict:

    obj_pose       (7,)            object pose (xyz + wxyz quat)
    obj_scale      scalar
    obj_path       str             mesh path
    approach_qpos  (G, T, 30)      G grasps x T waypoints x 30 DoF  <- the traj
    pregrasp_qpos  (G, 30)         keyframes ...
    grasp_qpos     (G, 30)
    squeeze_qpos   (G, 30)
    lift_qpos      (G, 30)

The 30 DoF are in MuJoCo joint order (below). Isaac names the same joints
differently (``hand:rh_FFJ4`` -> ``robot0_FFJ3``: MuJoCo is 1-indexed, the Isaac
instanceable hand is 0-indexed; arm joints are identical). ``reorder_to`` bridges
that automatically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

# Canonical BODex / DexGraspBench ur10e_shadow 30-DoF order (from the MuJoCo XML).
BODEX_UR10E_SHADOW_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    "hand:rh_WRJ2", "hand:rh_WRJ1",
    "hand:rh_FFJ4", "hand:rh_FFJ3", "hand:rh_FFJ2", "hand:rh_FFJ1",
    "hand:rh_MFJ4", "hand:rh_MFJ3", "hand:rh_MFJ2", "hand:rh_MFJ1",
    "hand:rh_RFJ4", "hand:rh_RFJ3", "hand:rh_RFJ2", "hand:rh_RFJ1",
    "hand:rh_LFJ5", "hand:rh_LFJ4", "hand:rh_LFJ3", "hand:rh_LFJ2", "hand:rh_LFJ1",
    "hand:rh_THJ5", "hand:rh_THJ4", "hand:rh_THJ3", "hand:rh_THJ2", "hand:rh_THJ1",
]

_HAND_RE = re.compile(r"hand:rh_([A-Z]{2})J(\d)")


def bodex_to_isaac_name(name: str) -> str:
    """Map a BODex/MuJoCo joint name to the Isaac instanceable Shadow name.

    Arm joints are unchanged. ``hand:rh_<finger>J<n>`` -> ``robot0_<finger>J<n-1>``
    (MuJoCo 1-indexed -> Isaac 0-indexed).
    """
    m = _HAND_RE.fullmatch(name)
    if not m:
        return name  # arm joint, identical in both
    finger, n = m.group(1), int(m.group(2))
    return f"robot0_{finger}J{n - 1}"


@dataclass
class BODexTrajectory:
    qpos: np.ndarray                 # (T, 30)
    joint_names: list[str] | None    # BODex names (len 30) if known
    object_name: str | None
    object_pose: np.ndarray | None   # (7,)
    object_scale: float
    object_path: str | None
    source: Path

    @property
    def num_frames(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def num_dofs(self) -> int:
        return int(self.qpos.shape[1])

    def reorder_to(self, target_joint_names: Sequence[str]) -> np.ndarray:
        """Permute ``qpos`` columns to follow ``target_joint_names`` (the Isaac
        articulation's DOF order), mapping BODex names -> Isaac names first."""
        if self.joint_names is None:
            raise ValueError("trajectory has no joint_names to reorder by")
        src_isaac = {bodex_to_isaac_name(n): i for i, n in enumerate(self.joint_names)}
        out = np.zeros((self.num_frames, len(target_joint_names)), dtype=self.qpos.dtype)
        missing = []
        for j, name in enumerate(target_joint_names):
            i = src_isaac.get(name)
            if i is None:
                missing.append(name)
                continue
            out[:, j] = self.qpos[:, i]
        if missing:
            print(f"[bodex_loader] WARNING: {len(missing)} target joints unmatched, "
                  f"zero-filled: {missing[:6]}{'...' if len(missing) > 6 else ''}")
        return out


def load_bodex_trajectory(path: str | Path, grasp_index: int = 0) -> BODexTrajectory:
    """Load one BODex ``.npy`` grasp file.

    Uses ``approach_qpos[grasp_index]`` as the trajectory when present; otherwise
    stitches the keyframes (pregrasp -> grasp -> squeeze -> lift) into a short one.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    raw = np.load(path, allow_pickle=True)
    data = raw.item() if (raw.dtype == object and raw.shape == ()) else raw
    if not isinstance(data, dict):
        # bare array fallback (assume already (T, D))
        arr = np.atleast_2d(np.asarray(data, dtype=np.float32))
        return BODexTrajectory(arr, None, None, None, 1.0, None, path)

    if "approach_qpos" in data:
        traj = np.asarray(data["approach_qpos"], dtype=np.float32)  # (G, T, 30)
        g = min(grasp_index, traj.shape[0] - 1)
        qpos = traj[g]                                              # (T, 30)
    else:
        # stitch keyframes
        keys = [k for k in ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos", "lift_qpos")
                if k in data]
        if not keys:
            raise KeyError(f"no approach_qpos or keyframes in {path}; keys={list(data)}")
        frames = [np.asarray(data[k], dtype=np.float32)[grasp_index] for k in keys]
        qpos = np.stack(frames, axis=0)                            # (4, 30)

    obj_pose = np.asarray(data["obj_pose"], dtype=np.float32) if "obj_pose" in data else None
    obj_scale = float(data.get("obj_scale", 1.0))
    obj_path = str(data["obj_path"]) if "obj_path" in data else None
    obj_name = path.parent.name  # object dir name, e.g. core_mug_...

    names = BODEX_UR10E_SHADOW_JOINTS if qpos.shape[1] == 30 else None
    return BODexTrajectory(qpos, names, obj_name, obj_pose, obj_scale, obj_path, path)
