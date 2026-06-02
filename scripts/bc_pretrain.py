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
    env = PianoEnv(cfg, render_mode=None)
    device = env.device

    ik_l = FingertipIK(env.left_robot, damping=cfg.ik_damping, max_step=cfg.ik_max_step)
    ik_r = FingertipIK(env.right_robot, damping=cfg.ik_damping, max_step=cfg.ik_max_step)

    obs_buf, act_buf = [], []
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    scale = cfg.action_scale

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
        expert = torch.cat([(q_l - ref[:, 0]) / scale, (q_r - ref[:, 1]) / scale], dim=-1)
        expert = expert.clamp(-1.0, 1.0)

        obs_buf.append(obs.detach().clone())
        act_buf.append(expert.detach().clone())
        obs_dict, _, _, _, _ = env.step(expert)
        obs = obs_dict["policy"]

    X = torch.cat(obs_buf, 0)
    Y = torch.cat(act_buf, 0)
    print(f"[bc] dataset: obs {tuple(X.shape)}, act {tuple(Y.shape)}")

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

    opt = torch.optim.Adam(ac.actor.parameters(), lr=1e-3)
    n = X.shape[0]
    bs = 4096
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            pred = ac.actor(X[idx])
            loss = torch.nn.functional.mse_loss(pred, Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * idx.numel()
        if ep % max(1, args.epochs // 10) == 0:
            print(f"[bc] epoch {ep:4d}  mse={tot / n:.5f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model_state_dict": ac.state_dict()}, args.out)
    print(f"[bc] saved warm-start -> {args.out}")
    env.close()


main()
simulation_app.close()
