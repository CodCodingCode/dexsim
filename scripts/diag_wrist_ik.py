"""Reachability proof: can the ARM servo the palm across the keyboard, fingers down?

This settles the "reach gap" question rigorously. Earlier diagnostics measured
5-fingertip IK residual (~45 mm, over-constrained on a 6-DoF arm). Here we instead
run the WELL-POSED solver (WristPoseIK: one palm 6-DoF target on the 6-DoF arm)
and sweep palm targets across each hand's reachable span at a fixed downward
orientation. For each target we report the converged position error (mm) and
orientation error (deg). Small errors everywhere => the arm reaches fine and the
real fix is wrist-pose IK (the "arm follows the wrist" design), not geometry.

  python scripts/diag_wrist_ik.py --headless [--iters 60] [--n 9]
"""

from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--iters", type=int, default=60, help="IK iterations per target")
parser.add_argument("--n", type=int, default=9, help="targets swept along each hand's span")
parser.add_argument("--hover", type=float, default=0.05, help="m above key tops")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np
import torch

import dexsim.tasks  # noqa: F401
from dexsim.tasks.piano import PianoEnvCfg
from dexsim.tasks.piano.piano_env import PianoEnv
from dexsim.piano.ik import WristPoseIK


def main():
    # line-buffer stdout: the per-target prints are otherwise block-buffered and
    # LOST if the IK sweep drives the arm into a singularity and segfaults sim.step
    # (a Python-less crash flushes nothing). With this, partial results + the exact
    # crash target survive -- and the crash point is itself reach/singularity evidence.
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    cfg = PianoEnvCfg()
    cfg.scene.num_envs = 1
    env = PianoEnv(cfg, render_mode=None)
    env.reset()
    dt = env.sim.get_physics_dt()
    dev = env.device

    key_top = env._key_top_world()[0].cpu().numpy()        # (88,3) world key tops
    palm = cfg.hand_base_body

    for robot, name, key_lo, key_hi in (
        (env.left_robot, "LEFT", 0, 43),
        (env.right_robot, "RIGHT", 44, 87),
    ):
        ik = WristPoseIK(robot, palm, damping=0.05, max_step=0.05, arm_only=True)
        # desired orientation = the ready-pose palm orientation (fingers pointing
        # down), held constant while we translate the palm across the keyboard.
        _, quat0 = ik.pose_w()
        q_des = quat0.clone()

        # sweep targets along this hand's key span (lateral), at hover height,
        # at the keyboard's X (toward player).
        idxs = np.linspace(key_lo, key_hi, args.n).round().astype(int)
        print(f"\n{name} arm — palm targets across keys {key_lo}..{key_hi} "
              f"(orientation held at ready/down):")
        print(f"  {'key':>4} {'targetY(m)':>10} {'pos_err(mm)':>11} {'ori_err(deg)':>12}")
        worst = 0.0
        for k in idxs:
            tgt = key_top[k].copy()
            tgt[2] += args.hover
            tpos = torch.tensor(tgt, device=dev, dtype=torch.float32).unsqueeze(0)
            for _ in range(args.iters):
                q = ik.solve(tpos, q_des)
                robot.write_joint_state_to_sim(q, torch.zeros_like(q))
                env.sim.step(render=False)
                env.scene.update(dt)
            pos, quat = ik.pose_w()
            perr = float(torch.linalg.norm(pos[0] - tpos[0]) * 1000.0)
            oerr = float(torch.linalg.norm(ik._orientation_error(q_des, quat)[0]))
            oerr_deg = np.degrees(oerr)
            worst = max(worst, perr)
            print(f"  {k:4d} {tgt[1]:10.3f} {perr:11.1f} {oerr_deg:12.1f}")
        print(f"  {name} worst position error across the span: {worst:.1f} mm")

    env.close()
    app.close()


main()
