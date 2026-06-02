"""Numerically search each arm's joint pose so the fingertips land ON the keys.

No rendering -- just teleport the arm to a candidate pose, settle a few physics
steps, read the fingertip world position, and coordinate-descend on the arm
joints to minimise distance to a target point above the keyboard. Prints the
best joint angles for left and right arms (paste into PianoEnvCfg.arm_ready_pose
/ per-side shoulder_pan).

  python scripts/tune_arm_pose.py
"""

from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--lx", type=float, default=0.35)
parser.add_argument("--ly", type=float, default=-0.28)
parser.add_argument("--rx", type=float, default=0.35)
parser.add_argument("--ry", type=float, default=0.28)
parser.add_argument("--z", type=float, default=0.745, help="target fingertip height (key top ~0.722)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from dexsim.assets import UR10E_SHADOW_CFG, PIANO_CFG  # noqa: E402
from dexsim.tasks.piano.piano_env_cfg import PianoEnvCfg  # noqa: E402

cfg = PianoEnvCfg()
dev = args.device
sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device=dev))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

piano = Articulation(PIANO_CFG.replace(prim_path="/World/Piano").replace(
    init_state=PIANO_CFG.init_state.replace(pos=cfg.piano_pos)))
left = Articulation(UR10E_SHADOW_CFG.replace(prim_path="/World/LeftRobot").replace(
    init_state=UR10E_SHADOW_CFG.init_state.replace(pos=cfg.left_base_pos)))
right = Articulation(UR10E_SHADOW_CFG.replace(prim_path="/World/RightRobot").replace(
    init_state=UR10E_SHADOW_CFG.init_state.replace(pos=cfg.right_base_pos)))
sim.reset()

# arm joint indices (UR10e 6-DoF) by name, within the combined articulation
ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


def setup(art):
    jn = art.data.joint_names
    arm_idx = [jn.index(n) for n in ARM]
    fis = [art.data.body_names.index(f"robot0_{f}distal") for f in ("ff", "mf", "rf", "lf")]
    return arm_idx, fis


def eval_pose(art, arm_idx, fis, arm_vals):
    """Teleport arm joints to arm_vals, settle, return fingertip-mean world xyz."""
    jp = art.data.default_joint_pos.clone()
    for k, idx in enumerate(arm_idx):
        jp[0, idx] = arm_vals[k]
    jv = torch.zeros_like(jp)
    art.write_joint_state_to_sim(jp, jv)
    art.set_joint_position_target(jp)
    art.write_data_to_sim()
    for _ in range(8):
        sim.step()
    return art.data.body_pos_w[0, fis].mean(0).cpu().numpy()


def search(art, target, start):
    arm_idx, fis = setup(art)
    # search shoulder_pan, shoulder_lift, elbow, wrist_1 (keep wrist_2/3 fixed so
    # the palm stays pointing down at the keys)
    free = [0, 1, 2, 3]
    vals = list(start)
    step = {0: 0.30, 1: 0.30, 2: 0.30, 3: 0.30}

    def err(v):
        f = eval_pose(art, arm_idx, fis, v)
        return np.linalg.norm(f - target), f

    best_e, best_f = err(vals)
    for _ in range(7):           # shrinking coordinate-descent sweeps
        improved = False
        for j in free:
            for sgn in (+1, -1):
                trial = list(vals)
                trial[j] += sgn * step[j]
                e, f = err(trial)
                if e < best_e - 1e-4:
                    best_e, best_f, vals = e, f, trial
                    improved = True
        if not improved:
            for j in free:
                step[j] *= 0.5
    return vals, best_f, best_e


ltgt = np.array([args.lx, args.ly, args.z])
rtgt = np.array([args.rx, args.ry, args.z])

# seed from the current ready pose (raised z=1.05 bases -> arm drapes down)
lvals, lf, le = search(left, ltgt, [-0.5, -0.9, 1.6, -1.2, -1.57, 0.0])
rvals, rf, re = search(right, rtgt, [0.5, -0.9, 1.6, -1.2, -1.57, 0.0])

print("\n================ TUNED ARM POSES ================")
for tag, vals, f, e, tgt in (("LEFT", lvals, lf, le, ltgt), ("RIGHT", rvals, rf, re, rtgt)):
    print(f"\n{tag}: fingertip=({f[0]:+.3f},{f[1]:+.3f},{f[2]:+.3f})  "
          f"target=({tgt[0]:+.2f},{tgt[1]:+.2f},{tgt[2]:+.2f})  err={e*1000:.0f}mm")
    for n, v in zip(ARM, vals):
        print(f"    {n:22s} {v:+.3f}")
app.close()
