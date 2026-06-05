"""Decisive placement diagnostic: can POSITION-ONLY arm IK walk a chosen FINGERTIP
onto a fixed target to ~1cm? Holds the finger pose constant, drives ONLY the arm via
pos_only WristPoseIK toward a static target near the keyboard, logs tip->target each
step. If it converges to ~1cm, fingertip placement is solvable (and the single_finger
integration has a bug); if it plateaus at ~90mm, the fingertip is outside arm reach."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--finger", type=int, default=1, help="0=th,1=ff,2=mf,3=rf,4=lf")
parser.add_argument("--steps", type=int, default=80)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from dexsim.tasks.piano import PianoEnvCfg
from dexsim.tasks.piano.piano_env import PianoEnv
from dexsim.piano.ik import WristPoseIK
from dexsim.piano import FINGERTIP_BODIES

cfg = PianoEnvCfg()
cfg.scene.num_envs = 1
cfg.arm_ik_follow = True          # arms IK-driven (we drive them ourselves below)
cfg.fold_to_reach = True
env = PianoEnv(cfg, render_mode=None)
env.reset()
dev = env.device
fb = FINGERTIP_BODIES[args.finger]

for hand, robot, mask in (("L", env.left_robot, env.left_key_mask),
                          ("R", env.right_robot, env.right_key_mask)):
    ik = WristPoseIK(robot, fb, max_step=0.06, pos_only=True)
    tip0, _ = ik.pose_w()                                   # (1,3) current tip
    # target: a key center this hand should reach, at the key top (a reachable point
    # near where the fingertip already hovers, so we test convergence not gross travel)
    kt = env._key_top_world()[0]                            # (88,3)
    hk = torch.nonzero(mask > 0).flatten()                  # this hand's keys
    # pick a key ~10cm away in xy (realistic note-to-note travel), so we test PD
    # convergence over real distance, not a trivial near-zero start.
    d_xy = torch.linalg.norm(kt[hk, :2] - tip0[0, :2], dim=-1)
    far = hk[(d_xy - 0.10).abs().argmin()]                  # ~10cm away
    target = kt[far].clone().unsqueeze(0)                   # (1,3) key top
    target[:, 2] -= 0.005                                   # 5mm into the key
    qd = torch.tensor([[1.0, 0, 0, 0]], device=dev)         # ignored (pos_only)
    dists = []
    am = ik.dof_mask
    for s in range(args.steps):
        qtgt = ik.solve(target, qd)                         # (1,30) arm cols move
        # PD-drive ONLY the arm joints toward the IK target (realistic: actuators, not
        # teleport). This is exactly what the live env does.
        jp = robot.data.joint_pos.clone()
        jp[:, am] = qtgt[:, am]
        robot.set_joint_position_target(jp)
        robot.write_data_to_sim()
        env.sim.step(render=False)
        robot.update(cfg.sim.dt)
        tip, _ = ik.pose_w()
        dists.append(float(torch.linalg.norm(tip - target)) * 1000.0)
    # also report distance at a few checkpoints to see the convergence curve
    cps = [dists[i] for i in (0, 4, 9, 19, 39, len(dists)-1) if i < len(dists)]
    print(f"    curve(mm): " + " ".join(f"{c:.0f}" for c in cps))
    print(f"[{hand}] finger={fb} key={int(far)} "
          f"start={dists[0]:.0f}mm  ->  final={dists[-1]:.0f}mm  min={min(dists):.0f}mm")

env.close()
app.close()
