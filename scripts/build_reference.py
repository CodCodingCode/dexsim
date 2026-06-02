"""Build the IK reference trajectory for a song (PianoMime's IK nominal).

For each control step we know, from the fingering plan, which key each finger
should press. This script drives the two arms' fingertips onto those keys with
the damped-least-squares IK controller and records the resulting joint trajectory
``q_ref`` (T, 2, 30). The piano env then uses ``q_ref`` as the base of a residual
action: the policy starts by *tracking* this reference and only learns the
corrections (contact, timing) on top -- exactly how PianoMime turns an
intractable from-scratch search into efficient residual RL.

  python scripts/build_reference.py --midi data/midi/twinkle.mid --headless
  # -> data/reference/twinkle.npz   (q_ref, plus diagnostics)

The reference need not be perfect: the arm (stiff) does the gross positioning,
the fingers (compliant) get close, and residual RL polishes the rest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Build IK reference trajectory.")
parser.add_argument("--midi", default=None, help="song .mid (default: cfg's)")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--ik_substeps", type=int, default=10,
                    help="physics+IK iterations per control step")
parser.add_argument("--out", default=None, help="output .npz (default: data/reference/<stem>.npz)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch

import dexsim.tasks  # noqa: F401 (registers env)
from dexsim.tasks.piano import PianoEnvCfg
from dexsim.tasks.piano.piano_env import PianoEnv
from dexsim.piano.ik import FingertipIK
from dexsim import DATA_DIR


def main():
    cfg = PianoEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.use_reference = False                 # we are *generating* it
    if args.midi:
        cfg.midi_path = args.midi

    env = PianoEnv(cfg, render_mode=None)
    env.reset()
    dt = env.sim.get_physics_dt()

    ik_left = FingertipIK(env.left_robot, damping=cfg.ik_damping, max_step=cfg.ik_max_step)
    ik_right = FingertipIK(env.right_robot, damping=cfg.ik_damping, max_step=cfg.ik_max_step)

    T = env.song_len
    q_ref = np.zeros((T, 2, env.per_arm_dof), dtype=np.float32)
    tip_err = np.zeros((T, 2), dtype=np.float32)   # mean fingertip error per hand

    print(f"[build_reference] song steps={T}, ik_substeps={args.ik_substeps}")
    for t in range(T):
        env.song_step[:] = t
        key_top = env._key_top_world()
        _, press, active = env._finger_targets_world(key_top)     # (E,10,3),(E,10)
        left_tgt, right_tgt = press[:, :5], press[:, 5:]

        # Kinematic IK: teleport joints to each DLS solution and refresh FK, so
        # the fingertips actually converge onto the keys (no PD lag). The
        # recorded pose is what residual RL then tracks with the PD controller.
        for _ in range(args.ik_substeps):
            q_l = ik_left.solve(left_tgt)
            q_r = ik_right.solve(right_tgt)
            zl = torch.zeros_like(q_l)
            env.left_robot.write_joint_state_to_sim(q_l, zl)
            env.right_robot.write_joint_state_to_sim(q_r, torch.zeros_like(q_r))
            env.sim.step(render=False)
            env.scene.update(dt)

        q_ref[t, 0] = env.left_robot.data.joint_pos[0].cpu().numpy()
        q_ref[t, 1] = env.right_robot.data.joint_pos[0].cpu().numpy()
        # diagnostic: how close did the active fingertips get?
        tips = env._fingertips_world()
        d = ((tips - press) ** 2).sum(-1).sqrt()
        am = active.float()
        tip_err[t, 0] = float((d[:, :5] * am[:, :5]).sum() / (am[:, :5].sum() + 1e-6))
        tip_err[t, 1] = float((d[:, 5:] * am[:, 5:]).sum() / (am[:, 5:].sum() + 1e-6))
        if t % max(1, T // 10) == 0:
            print(f"  step {t:4d}/{T}  mean fingertip err L={tip_err[t,0]*1000:5.1f}mm "
                  f"R={tip_err[t,1]*1000:5.1f}mm")

    out = Path(args.out) if args.out else DATA_DIR / "reference" / (Path(cfg.midi_path).stem + ".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, q_ref=q_ref, tip_err=tip_err,
                        midi=str(cfg.midi_path), control_dt=cfg.control_dt)
    print(f"[build_reference] wrote {out}  q_ref{q_ref.shape}  "
          f"final-half mean tip err {tip_err[T//2:].mean()*1000:.1f}mm")
    env.close()


main()
simulation_app.close()
