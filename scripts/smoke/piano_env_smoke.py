"""Integration smoke test for the bimanual piano env: construct it, reset, step
random actions, and print obs/reward/done shapes. No training, no policy --
this just proves the whole env (2 arms + 88-key piano + MIDI goal) builds and
steps on GPU with the right tensor shapes.

  python scripts/smoke/piano_env_smoke.py --headless --num_envs 4
  python scripts/smoke/piano_env_smoke.py --num_envs 4 --reward_mode rp1m

With ``--reward_mode rp1m`` it also exercises the online optimal-transport
fingering term, and it always checks that the wrist lock actually holds the hands
upright (see ``PianoEnvCfg.wrist_lock``).
"""

from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--reward_mode", choices=("dexsim", "rp1m"), default="dexsim")
parser.add_argument("--no_wrist_lock", action="store_true")
parser.add_argument("--force_collision", action="store_true",
                    help="POSITIVE CONTROL for r_Collision: park the right hand on top of "
                         "the left so they must collide, and assert the contact sensors "
                         "actually fire. Without this, a sensor that reports nothing is "
                         "indistinguishable from one that is wired correctly.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

import dexsim.tasks  # noqa: F401
from dexsim.tasks.piano import PianoEnvCfg

TASK = "Dexsim-Piano-Bimanual-v0"


def main():
    cfg = PianoEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.reward_mode = args.reward_mode
    if args.no_wrist_lock:
        cfg.wrist_lock = False
    if args.force_collision:
        # Drive the right hand into the left one. Note this is NOT "same base
        # position": the two hand USDs are mirrored, so each palm hangs off its
        # base the opposite way along the rail (Y) axis -- give both bases the
        # same pose and the hands still sit 21.7 cm apart. MIRROR_GAP is that
        # measured offset; the extra 2 cm forces interpenetration. If the hand
        # assets ever change, the assertion below prints the real gap.
        # Slide the right hand along the rail axis (Y) until the two hands
        # interpenetrate. Measured on the shipped layout: a base separation of
        # 0.306 m leaves a 0.217 m gap between the nearest hand points, so the two
        # palms hang ~0.089 m toward each other. Bases that close therefore touch;
        # 2 cm closer still guarantees contact. Rotations are left alone -- that
        # offset is a property of the shipped orientations.
        PALM_INSET = 0.089
        ly = cfg.left_base_pos[1]
        rx, _, rz = cfg.right_base_pos
        cfg.right_base_pos = (rx, ly - PALM_INSET + 0.02, rz)
        # __post_init__ baked the original pose into init_state when the cfg was
        # constructed, so the field above is now only documentation -- write the
        # spawn pose the env actually reads.
        cfg.right_robot_cfg.init_state.pos = cfg.right_base_pos
        print(f"[piano_smoke] FORCE COLLISION: right hand moved to {cfg.right_base_pos}")
    print(f"[piano_smoke] action_space={cfg.action_space} obs_space={cfg.observation_space} "
          f"reward_mode={cfg.reward_mode} wrist_lock={cfg.wrist_lock}")

    env = gym.make(TASK, cfg=cfg, render_mode=None)
    obs, _ = env.reset()
    print("[piano_smoke] reset OK")
    print(f"  obs['policy'] shape: {tuple(obs['policy'].shape)}")

    le = env.unwrapped
    print(f"  left_robot DOFs : {le.left_robot.num_joints}")
    print(f"  right_robot DOFs: {le.right_robot.num_joints}")
    print(f"  piano key joints: {le.piano.num_joints}")

    # WRIST LOCK: record where the wrist joints start, so we can prove below that
    # a full run of random actions never moved them off the ready pose.
    wj = le.wrist_joint_ids
    wnames = [le.left_robot.data.joint_names[i] for i in wj]
    wrist0 = torch.cat([le.left_robot.data.joint_pos[:, wj],
                        le.right_robot.data.joint_pos[:, wj]], dim=1).clone()
    masses = le.left_robot.root_physx_view.get_masses()[0]
    print(f"  left hand total mass: {float(masses.sum()):.2f} kg over "
          f"{masses.numel()} bodies (heaviest {float(masses.max()):.2f} kg)")

    rew_sum = torch.zeros(args.num_envs, device=le.device)
    logged = {}
    for i in range(args.steps):
        action = 2.0 * torch.rand(args.num_envs, cfg.action_space, device=le.device) - 1.0
        obs, rew, term, trunc, extras = env.step(action)
        rew_sum += rew
        logged = extras.get("log", logged) or logged
        if i == 0:
            print(f"  step0: rew shape {tuple(rew.shape)}, "
                  f"term {tuple(term.shape)}, trunc {tuple(trunc.shape)}")
        assert torch.isfinite(rew).all(), f"non-finite reward at step {i}"
        if i % 10 == 0:
            # left hand only; `tau` is the giveaway if a lock is losing an effort
            # fight rather than holding (see PianoEnv's wrist-lock comment).
            lr = le.left_robot
            fmt = lambda t: "[" + ", ".join(f"{v:+.4f}" for v in t.tolist()) + "]"
            print(f"   step{i:3d} wrist: cmd={fmt(le._left_target[0, wj])} "
                  f"actual={fmt(lr.data.joint_pos[0, wj])} "
                  f"tau={fmt(lr.data.applied_torque[0, wj])}")

    wrist1 = torch.cat([le.left_robot.data.joint_pos[:, wj],
                        le.right_robot.data.joint_pos[:, wj]], dim=1)
    drift = (wrist1 - wrist0).abs().max()
    print(f"[piano_smoke] wrist joints {wnames} (both hands): "
          f"max drift over {args.steps} random-action steps = {float(drift):.5f} rad")
    if cfg.wrist_lock:
        assert float(drift) < 0.02, (
            f"wrist_lock is on but the wrists moved {float(drift):.4f} rad -- "
            "the hands are not being held upright")
        print("  -> WRIST LOCK HOLDS (hands upright)")

    if cfg.reward_mode == "rp1m":
        parts = {k: v for k, v in logged.items() if k.startswith("rp1m/")}
        print(f"[piano_smoke] RP1M reward terms: {parts}")
        assert parts, "reward_mode=rp1m but no rp1m/* terms were logged"
        if cfg.rp1m_collision_contacts:
            # the contact sensors must actually be reporting -- a wrong prim path
            # would leave force_matrix_w None and the term silently stuck at "clean"
            assert le.hand_contacts, "rp1m_collision_contacts is on but no sensors were built"
            for s in le.hand_contacts:
                fm = s.data.force_matrix_w
                assert fm is not None, f"{s.cfg.prim_path}: force_matrix_w is None (filter not applied?)"
                assert fm.shape[0] == args.num_envs and fm.shape[1] == 1, \
                    f"{s.cfg.prim_path}: unexpected force matrix shape {tuple(fm.shape)}"
            n_filt = le.hand_contacts[0].data.force_matrix_w.shape[2]
            print(f"[piano_smoke] contact sensors: {len(le.hand_contacts)} bodies x "
                  f"{n_filt} filtered right-hand shapes, source={le._collided_src}")
            assert le._collided_src == "contact", "fell back to the proximity proxy"
            # a wildcard filter silently collapses to 1 channel -- catch that here
            assert n_filt == len(cfg.contact_bodies), (
                f"expected {len(cfg.contact_bodies)} filter channels, got {n_filt} -- "
                "PhysX likely rejected a filter pattern (check the log for "
                "'did not match the correct number of entries')")
            frac = logged.get("rp1m/collided_frac", 0.0)
            peak = max(float(s.data.force_matrix_w.norm(dim=-1).max()) for s in le.hand_contacts)
            # how far apart the hands actually are, so a null result is diagnosable
            tips = le._fingertips_world()
            palms = le._palms_world()
            lpts = torch.cat([tips[:, :5], palms[:, 0:1]], dim=1)
            rpts = torch.cat([tips[:, 5:], palms[:, 1:2]], dim=1)
            d = lpts.unsqueeze(2) - rpts.unsqueeze(1)              # (E,6,6,3)
            n = d.norm(dim=-1)                                     # (E,6,6)
            flat = n[0].flatten().argmin()
            vec = d[0].reshape(-1, 3)[flat]
            print(f"[piano_smoke] collided_frac={frac:.3f}, peak pair force={peak:.3f} N, "
                  f"closest hand-hand gap={float(n.min())*100:.2f} cm "
                  f"(vector L->R = {[round(float(v), 3) for v in vec]})")
            if args.force_collision:
                assert frac > 0.0 and peak > 0.0, (
                    "the hands were parked on top of each other but the contact sensors "
                    "reported nothing -- r_Collision would be dead weight")
                print("  -> CONTACT SENSORS FIRE ON A REAL COLLISION")

    print(f"[piano_smoke] stepped {args.steps} steps. mean return = {rew_sum.mean().item():.3f}")
    print("===== PIANO ENV SMOKE OK =====")
    env.close()


main()
simulation_app.close()
