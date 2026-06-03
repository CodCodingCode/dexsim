"""Evaluate the IK reference (or any policy) by how well it actually plays.

Rolls out the env and reports key-press recall / precision / F1 against the MIDI
goal, plus mean reward. With ``--zero`` it applies a zero residual, i.e. it plays
the *pure IK reference* — the crucial sanity gate before spending GPU-hours: if
following the reference already sounds a decent fraction of the notes, residual
RL has a good starting point; if it sounds ~nothing, fix the reference / mount
first. With ``--checkpoint`` it instead rolls out a trained policy.

  python scripts/eval_reference.py --midi data/midi/twinkle.mid --zero --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate reference / policy play quality.")
parser.add_argument("--midi", default=None)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--zero", action="store_true", help="apply zero residual (pure reference)")
parser.add_argument("--checkpoint", default=None, help="rsl_rl policy checkpoint to roll out")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

import dexsim.tasks  # noqa: F401
from dexsim.tasks.piano import PianoEnvCfg
from dexsim.tasks.piano.piano_env import PianoEnv
from dexsim.piano.reward import press_accuracy


def main():
    cfg = PianoEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.midi:
        cfg.midi_path = args.midi

    env = PianoEnv(cfg, render_mode=None)
    policy = None
    if args.checkpoint and not args.zero:
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from dexsim.tasks.piano.agents.rsl_rl_ppo_cfg import PianoPPORunnerCfg
        wrapped = RslRlVecEnvWrapper(env)
        _ad = PianoPPORunnerCfg().to_dict(); _ad.setdefault("policy", {})["noise_std_type"] = "log"
        runner = OnPolicyRunner(wrapped, _ad, log_dir=None, device=env.device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.device)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
    T = env.song_len
    rec_sum = prec_sum = rew_sum = 0.0
    n_scored = 0
    # micro-averaged accumulators (RoboPianist/RP1M standard): sum TP/FP/FN over
    # ALL timesteps and envs, then compute P/R/F1 once. Robust to steps with few
    # active keys, and (unlike the macro avg) it charges false presses during
    # rests — exactly the precision leak the per-step mean over goal-steps hid.
    tp_tot = fp_tot = fn_tot = 0.0
    thresh = 0.5  # TODO(autoloop): pull from the piano's true sound-trigger depth

    for _ in range(T):
        if policy is not None:
            with torch.no_grad():
                action = policy(obs)
        else:
            action = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
        obs_dict, rew, _, _, _ = env.step(action)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        pressed = env._key_pressed_fraction()
        goal = env._goal_now()
        # micro counts over every env/key this step
        sounding = pressed >= thresh
        goal_b = goal.bool()
        tp_tot += float((sounding & goal_b).sum())
        fp_tot += float((sounding & ~goal_b).sum())
        fn_tot += float((~sounding & goal_b).sum())

        recall, precision = press_accuracy(pressed, goal)
        has_goal = goal.sum(-1) > 0
        if has_goal.any():
            rec_sum += float(recall[has_goal].mean())
            prec_sum += float(precision[has_goal].mean())
            n_scored += 1
        rew_sum += float(rew.mean())

    # micro (headline)
    eps = 1e-9
    mrec = tp_tot / (tp_tot + fn_tot + eps)
    mprec = tp_tot / (tp_tot + fp_tot + eps)
    mf1 = 2 * mrec * mprec / (mrec + mprec + eps)
    # macro (per-step mean over goal-steps; kept for comparison)
    rec = rec_sum / max(1, n_scored)
    prec = prec_sum / max(1, n_scored)
    f1 = 2 * rec * prec / (rec + prec + eps)
    mode = "ZERO-RESIDUAL (pure IK reference)" if policy is None else f"policy {args.checkpoint}"
    print("=" * 60)
    print(f"[eval] {mode}  song={cfg.midi_path}  envs={env.num_envs}")
    print(f"[eval] MICRO  recall={mrec:.3f}  precision={mprec:.3f}  F1={mf1:.3f}  "
          f"(headline; vs MIDI over all steps)")
    print(f"[eval] macro  recall={rec:.3f}  precision={prec:.3f}  F1={f1:.3f}  "
          f"mean_reward/step={rew_sum / T:.3f}")
    print("=" * 60)
    env.close()


main()
simulation_app.close()
