"""Behavior-cloning warm-start for the piano policy (PianoMime's BC-of-an-expert).

The residual env already makes "zero action == follow the open-loop IK reference."
This script goes one better: it rolls out the *closed-loop* IK controller (which
corrects the reference's residual tracking error online) and behavior-clones that
expert into the PPO actor. PPO then starts from an IK-tracking policy instead of
an open-loop one — exactly PianoMime's recipe of distilling a competent expert
into the policy before RL fine-tunes it.

  python scripts/bc_pretrain.py --midi data/midi/twinkle.mid --headless \
         --epochs 200 --out logs/bc/twinkle.pt
  # then:  python scripts/train_piano.py --bc_init logs/bc/twinkle.pt ...

The saved file is an rsl_rl-compatible checkpoint ({'model_state_dict': ...}) so
train_piano can load it directly into the actor-critic.
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="BC warm-start from the IK expert.")
parser.add_argument("--midi", default=None)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--rollout_passes", type=int, default=4, help="times through the song")
parser.add_argument("--ik_iters", type=int, default=3, help="online IK steps per control step")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--out", default="logs/bc/piano_bc.pt")
parser.add_argument("--dump", default=None, help="also save (obs,action) npz for generalist distillation")
parser.add_argument("--reference", default=None,
                    help="explicit IK/warm-start reference .npz to clone from "
                         "(e.g. an RP1M-merged reference from build_rp1m_reference.py); "
                         "default: derive from the MIDI stem")
parser.add_argument("--no_fold", action="store_true", help="disable fold_to_reach (RP1M real keys)")
parser.add_argument("--no_mute", action="store_true", help="disable mute_right_hand (two-handed)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

import dexsim.tasks  # noqa: F401
from dexsim.tasks.piano import PianoEnvCfg
from dexsim.tasks.piano.piano_env import PianoEnv
from dexsim.tasks.piano.agents.rsl_rl_ppo_cfg import PianoPPORunnerCfg
from dexsim.piano.ik import FingertipIK


def main():
    cfg = PianoEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.midi:
        cfg.midi_path = args.midi
    if args.reference:
        # clone from an explicit reference (e.g. RP1M-merged warm-start) instead
        # of the MIDI-stem default; use_reference makes the env load it as the
        # residual base so "zero action == follow this reference".
        cfg.reference_path = args.reference
        cfg.use_reference = True
    if args.no_fold:
        cfg.fold_to_reach = False
    if args.no_mute:
        cfg.mute_right_hand = False
    env = PianoEnv(cfg, render_mode=None)
    device = env.device

    ik_l = FingertipIK(env.left_robot, damping=cfg.ik_damping, max_step=cfg.ik_max_step)
    ik_r = FingertipIK(env.right_robot, damping=cfg.ik_damping, max_step=cfg.ik_max_step)

    obs_buf, act_buf = [], []
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    # Invert the env's PER-JOINT residual scale, not the legacy global action_scale.
    # The env applies target = q_ref + joint_scale * action, where joint_scale differs
    # per joint (arm 0.15 vs hand 0.35) and is ZERO on frozen DoF (freeze_arms/_hands).
    # Using the global 0.15 here mis-scaled the finger residual (0.35 != 0.15), so the
    # cloned action overshot and the warm start did NOT reproduce the IK fingering.
    inv = torch.where(env.joint_scale > 0, 1.0 / env.joint_scale,
                      torch.zeros_like(env.joint_scale))      # (1,30); frozen DoF -> 0

    steps = env.song_len * args.rollout_passes
    print(f"[bc] collecting {steps} steps x {args.num_envs} envs from the IK expert")
    for _ in range(steps):
        # expert: a few online DLS-IK iterations toward the current key targets
        key_top = env._key_top_world()
        _, press, _ = env._finger_targets_world(key_top)
        for _ in range(args.ik_iters):
            q_l = ik_l.solve(press[:, :5])
            q_r = ik_r.solve(press[:, 5:])
        ref = env.q_ref[env.song_step]                      # (E,2,30)
        expert = torch.cat([(q_l - ref[:, 0]) * inv, (q_r - ref[:, 1]) * inv], dim=-1)
        expert = expert.clamp(-1.0, 1.0)

        obs_buf.append(obs.detach().clone())
        act_buf.append(expert.detach().clone())
        obs_dict, _, _, _, _ = env.step(expert)
        obs = obs_dict["policy"]

    X = torch.cat(obs_buf, 0)
    Y = torch.cat(act_buf, 0)
    # A single NaN in the collected obs/expert (a physics blip, or expert=inf*0)
    # propagates through BC and produces an ALL-NaN actor -> every downstream run
    # crashes with "std>=0". Scrub them and drop any non-finite rows.
    finite = torch.isfinite(X).all(1) & torch.isfinite(Y).all(1)
    dropped = int((~finite).sum())
    X = torch.nan_to_num(X[finite]); Y = torch.nan_to_num(Y[finite]).clamp(-1.0, 1.0)
    print(f"[bc] dataset: obs {tuple(X.shape)}, act {tuple(Y.shape)} (dropped {dropped} non-finite rows)")

    if args.dump:
        import numpy as np
        os.makedirs(os.path.dirname(args.dump), exist_ok=True)
        np.savez_compressed(args.dump, obs=X.cpu().numpy(), action=Y.cpu().numpy(),
                            midi=str(cfg.midi_path))
        print(f"[bc] dumped distillation dataset -> {args.dump}")

    # build the SAME actor-critic rsl_rl will use, BC its actor, save its state
    from rsl_rl.modules import ActorCritic
    agent_cfg = PianoPPORunnerCfg()
    pol = agent_cfg.policy
    ac = ActorCritic(
        num_actor_obs=X.shape[1], num_critic_obs=X.shape[1], num_actions=Y.shape[1],
        actor_hidden_dims=pol.actor_hidden_dims, critic_hidden_dims=pol.critic_hidden_dims,
        activation=pol.activation, init_noise_std=pol.init_noise_std,
    ).to(device)

    # Standardize obs for BC: raw 1236-dim obs at lr 1e-3 diverged to NaN. Zero-mean
    # unit-var also matches training's empirical_normalization (so the warm-started
    # actor sees a similar input distribution at train time).
    mu = X.mean(0, keepdim=True)
    sd = X.std(0, keepdim=True).clamp_min(1e-6)
    Xn = (X - mu) / sd
    opt = torch.optim.Adam(ac.actor.parameters(), lr=5e-4)
    n = Xn.shape[0]
    bs = 4096
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            pred = ac.actor(Xn[idx])
            loss = torch.nn.functional.mse_loss(pred, Y[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ac.actor.parameters(), 1.0)
            opt.step()
            tot += loss.item() * idx.numel()
        if ep % max(1, args.epochs // 10) == 0:
            print(f"[bc] epoch {ep:4d}  mse={tot / n:.5f}")
    # refuse to save a NaN actor (the bug that crashed every downstream run)
    if any(not torch.isfinite(p).all() for p in ac.actor.parameters()):
        raise RuntimeError("[bc] actor diverged to NaN even after standardization+clip; aborting save")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model_state_dict": ac.state_dict()}, args.out)
    print(f"[bc] saved warm-start -> {args.out}")
    env.close()


main()
simulation_app.close()
