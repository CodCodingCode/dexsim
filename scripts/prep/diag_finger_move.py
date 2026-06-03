"""Do the Shadow hand fingers actually MOVE when commanded? + does the actuator
stiffness override take effect? Every finger intervention (IK, idle-reward, curl,
stiffness) produced byte-identical results -> either the fingers don't actuate or
the cfg overrides silently no-op. This settles it directly.

  python scripts/prep/diag_finger_move.py --headless [--hand_stiffness 250]
"""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--hand_stiffness", type=float, default=None)
parser.add_argument("--hand_effort", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
import dexsim.tasks  # noqa
from dexsim.tasks.piano import PianoEnvCfg
from dexsim.tasks.piano.piano_env import PianoEnv


def main():
    cfg = PianoEnvCfg()
    cfg.scene.num_envs = 2
    cfg.midi_path = "data/midi/easy.mid"
    cfg.arm_ik_follow = True
    cfg.freeze_arms = False
    cfg.use_reference = False
    if args.hand_stiffness is not None:
        for rc in (cfg.left_robot_cfg, cfg.right_robot_cfg):
            rc.actuators["hand"].stiffness = args.hand_stiffness
            if args.hand_effort is not None:
                rc.actuators["hand"].effort_limit = args.hand_effort
    env = PianoEnv(cfg, render_mode=None)
    env.reset()
    rob = env.left_robot
    names = rob.data.joint_names
    hand_idx = [i for i, n in enumerate(names) if "robot0_" in n]
    # the actuator object actually built on the articulation
    for key, act in rob.actuators.items():
        st = act.stiffness
        st = float(st.flatten()[0]) if hasattr(st, "flatten") else float(st)
        print(f"[diag] built actuator '{key}': stiffness={st}")

    q0 = rob.data.joint_pos.clone()
    # command a STRONG flexion on all hand joints (curl into a fist)
    target = rob.data.joint_pos.clone()
    lo = rob.data.soft_joint_pos_limits[..., 0]
    hi = rob.data.soft_joint_pos_limits[..., 1]
    for i in hand_idx:
        target[:, i] = hi[:, i]              # drive each hand joint to its upper limit
    for step in range(60):
        rob.set_joint_position_target(target)
        env.sim.step(render=False)
        env.scene.update(env.sim.get_physics_dt())
    q1 = rob.data.joint_pos
    dq = (q1 - q0).abs()
    hand_move = dq[:, hand_idx].mean().item()
    arm_idx = [i for i in range(len(names)) if i not in hand_idx]
    print(f"[diag] commanded all hand joints -> upper limit, 60 steps")
    print(f"[diag] mean |hand joint motion| = {hand_move:.4f} rad   "
          f"(near 0 => fingers DO NOT actuate)")
    print(f"[diag] per-finger sample (env0): " +
          ", ".join(f"{names[i]}:{(q1[0,i]-q0[0,i]).item():+.2f}" for i in hand_idx[:8]))
    env.close()
    app.close()


main()
