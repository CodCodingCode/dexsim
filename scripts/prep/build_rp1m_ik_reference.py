"""Build a CONSISTENT RP1M warm-start reference: hand = RP1M pose, arm = IK.

The earlier ``build_rp1m_reference.py`` kept our own IK arm and only *injected*
RP1M's hand pose. That is inconsistent: the curled RP1M fingers, positioned by an
arm that was solved for a DIFFERENT (rest) hand pose, land off the keys
(measured: zero-residual F1 ~0.04, precision ~0.02, reward/key -4.4 — the fingers
mash wrong keys).

This builder fixes it. At each control step it CLAMPS the Shadow hand to RP1M's
decoded pose and runs damped-least-squares IK on the **arm only** (6 DoF) to land
those RP1M-posed fingertips on their target keys. The arm is therefore placed
*for* RP1M's hand, so zero residual == RP1M's expert fingering, correctly over the
keys. ``train_piano.py --freeze_arms`` then only has to learn the press.

  python scripts/build_rp1m_ik_reference.py --actions data/rp1m/twinkle_best.npy \
      --midi data/midi/twinkle.mid --out data/reference/twinkle_rp1m_ik.npz --headless
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Build consistent RP1M IK reference (hand=RP1M, arm=IK).")
parser.add_argument("--actions", required=True, help="(T,39|45) RP1M actions .npy")
parser.add_argument("--midi", default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--ik_substeps", type=int, default=12, help="arm-IK iterations per control step")
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch

import dexsim.tasks  # noqa: F401
from dexsim.tasks.piano import PianoEnvCfg
from dexsim.tasks.piano.piano_env import PianoEnv
from dexsim.piano.ik import FingertipIK
from dexsim.piano.rp1m import decode as rp1m_decode
from dexsim.piano.rp1m.decode import ISAAC_HAND_JOINTS
from dexsim.piano.rp1m.retarget import resample_traj
from dexsim import DATA_DIR


def _arm_only_dls(ik: FingertipIK, targets_w: torch.Tensor, arm_cols: torch.Tensor,
                  damping: float, max_step: float) -> torch.Tensor:
    """One DLS step that moves ONLY the arm columns to drive the (hand-fixed)
    fingertips toward ``targets_w`` (E,5,3). Returns dq over all DoF (hand=0)."""
    cur = ik.fingertips_w()                                   # (E,5,3)
    err = (targets_w - cur).clamp(-max_step, max_step)
    E = err.shape[0]
    e = err.reshape(E, -1)                                    # (E,15)
    Jp = ik._position_jacobians().reshape(E, -1, ik.num_dof)  # (E,15,Dfull)
    Ja = Jp[:, :, arm_cols]                                   # (E,15,6) arm only
    Jt = Ja.transpose(-1, -2)                                 # (E,6,15)
    JJt = Ja @ Jt                                             # (E,15,15)
    eye = torch.eye(JJt.shape[-1], device=JJt.device).expand_as(JJt)
    sol = torch.linalg.solve(JJt + (damping ** 2) * eye, e.unsqueeze(-1))
    dq_arm = (Jt @ sol).squeeze(-1)                           # (E,6)
    dq = torch.zeros(E, ik.num_dof, device=err.device)
    dq[:, arm_cols] = dq_arm
    return dq


def main():
    cfg = PianoEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.use_reference = False
    cfg.fold_to_reach = False      # RP1M actions are for the song's REAL keys
    cfg.mute_right_hand = False    # two-handed: generate targets for both hands
    if args.midi:
        cfg.midi_path = args.midi
    env = PianoEnv(cfg, render_mode=None)
    env.reset()
    dt = env.sim.get_physics_dt()
    device = env.device

    # joint layout: arm cols (not robot0_*) vs hand cols (robot0_*), in artic order
    names = list(env.left_robot.data.joint_names)
    arm_cols = torch.tensor([i for i, n in enumerate(names) if not n.startswith("robot0_")],
                            device=device)
    name_to_col = {n: i for i, n in enumerate(names)}
    # map RP1M's 24-col hand order -> this articulation's hand columns
    hand_col_for = [name_to_col[n] for n in ISAAC_HAND_JOINTS if n in name_to_col]
    rp1m_to_artic = torch.tensor(hand_col_for, device=device)
    present = [n for n in ISAAC_HAND_JOINTS if n in name_to_col]
    print(f"[rp1m-ik] arm cols={arm_cols.tolist()}  hand cols mapped={len(present)}/24")

    # decode RP1M actions, resample hand pose to song length
    actions = np.load(args.actions)
    dec = rp1m_decode.decode_actions(actions)
    T = env.song_len
    hand_rp1m = {}
    for side in ("left", "right"):
        stacked = dec.hand_q[side]                            # (Td,24) in ISAAC_HAND_JOINTS order
        stacked = resample_traj(stacked, T)
        # keep only columns whose joint exists in the articulation
        keep = [j for j, n in enumerate(ISAAC_HAND_JOINTS) if n in name_to_col]
        hand_rp1m[side] = torch.as_tensor(stacked[:, keep], dtype=torch.float32, device=device)
    print(f"[rp1m-ik] decoded {dec.reduced and 'reduced(39)' or 'full(45)'}, "
          f"resampled hand pose to T={T}")

    ik_l = FingertipIK(env.left_robot, damping=cfg.ik_damping, max_step=cfg.ik_max_step)
    ik_r = FingertipIK(env.right_robot, damping=cfg.ik_damping, max_step=cfg.ik_max_step)

    q_ref = np.zeros((T, 2, env.per_arm_dof), dtype=np.float32)
    tip_err = np.zeros((T, 2), dtype=np.float32)

    for t in range(T):
        env.song_step[:] = t
        key_top = env._key_top_world()
        _, press, active = env._finger_targets_world(key_top)   # (E,10,3),(E,10)
        for (robot, ik, side, sl) in ((env.left_robot, ik_l, "left", slice(0, 5)),
                                      (env.right_robot, ik_r, "right", slice(5, 10))):
            tgt = press[:, sl]
            for _ in range(args.ik_substeps):
                # clamp hand to RP1M pose, then arm-only IK toward the keys
                q = robot.data.joint_pos.clone()
                q[:, rp1m_to_artic] = hand_rp1m[side][t]
                robot.write_joint_state_to_sim(q, torch.zeros_like(q))
                env.sim.step(render=False)
                env.scene.update(dt)
                dq = _arm_only_dls(ik, tgt, arm_cols, cfg.ik_damping, cfg.ik_max_step)
                q = robot.data.joint_pos + dq
                q[:, rp1m_to_artic] = hand_rp1m[side][t]
                lo = robot.data.soft_joint_pos_limits[..., 0]
                hi = robot.data.soft_joint_pos_limits[..., 1]
                robot.write_joint_state_to_sim(torch.clamp(q, lo, hi),
                                               torch.zeros_like(q))
                env.sim.step(render=False)
                env.scene.update(dt)
        q_ref[t, 0] = env.left_robot.data.joint_pos[0].cpu().numpy()
        q_ref[t, 1] = env.right_robot.data.joint_pos[0].cpu().numpy()
        tips = env._fingertips_world()
        d = ((tips - press) ** 2).sum(-1).sqrt()
        am = active.float()
        tip_err[t, 0] = float((d[:, :5] * am[:, :5]).sum() / (am[:, :5].sum() + 1e-6))
        tip_err[t, 1] = float((d[:, 5:] * am[:, 5:]).sum() / (am[:, 5:].sum() + 1e-6))
        if t % max(1, T // 10) == 0:
            print(f"  step {t:4d}/{T}  active-fingertip err "
                  f"L={tip_err[t,0]*1000:5.1f}mm R={tip_err[t,1]*1000:5.1f}mm")

    out = Path(args.out) if args.out else DATA_DIR / "reference" / (Path(cfg.midi_path).stem + "_rp1m_ik.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, q_ref=q_ref, tip_err=tip_err, joint_names=np.array(names),
                        midi=str(cfg.midi_path), control_dt=cfg.control_dt,
                        source_actions=str(args.actions))
    fin = tip_err[T // 2:]
    print(f"[rp1m-ik] wrote {out}  q_ref{q_ref.shape}  "
          f"final-half mean active-fingertip err {fin[fin > 0].mean()*1000:.1f}mm")
    env.close()


main()
simulation_app.close()
