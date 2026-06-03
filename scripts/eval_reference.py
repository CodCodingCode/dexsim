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
parser.add_argument("--reference", default=None, help="explicit q_ref .npz to load "
                    "(overrides the default data/reference/<midi-stem>.npz)")
parser.add_argument("--arm_ik_follow", action="store_true", help="arms servoed online "
                    "by WristPoseIK to the fingering centroid; no q_ref needed")
parser.add_argument("--out", default=None, help="write metrics as JSON to this path "
                    "(so a backgrounded eval leaves a machine-readable result)")
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
    if args.reference:
        cfg.reference_path = args.reference
    if args.arm_ik_follow:
        # math drives the arms, policy/zero-residual drives the fingers: no reference
        # trajectory, arms are not frozen, they track the fingering centroid online.
        cfg.arm_ik_follow = True
        cfg.freeze_arms = False
        cfg.use_reference = False

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
    tipd_sum = 0.0; tipd_n = 0   # mean active fingertip->assigned-key distance (mm):
    #   the online tip_err, validates whether the hands are POSITIONED on the keys
    #   (independent of whether the fingers actually pressed). Key check for arm_ik_follow.
    # SOUND TRUTH: env._key_pressed_fraction() already applies the simulator's
    # velocity-gated sounding latch (struck past KEY_SOUND_ANGLE while moving down,
    # held until frac<0.25) and returns 0 for any key that is NOT sounding. So a key
    # sounds iff its returned fraction is > 0 -- thresholding at 0.5 double-counted
    # the gate and dropped softly-held sustained notes in [0.25,0.5), deflating recall.
    SOUND_EPS = 1e-6

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
        sounding = pressed > SOUND_EPS
        goal_b = goal.bool()
        tp_tot += float((sounding & goal_b).sum())
        fp_tot += float((sounding & ~goal_b).sum())
        fn_tot += float((~sounding & goal_b).sum())

        # online fingertip->assigned-key distance for ACTIVE fingers (mm)
        kt = env._key_top_world()
        surface, _, fa = env._finger_targets_world(kt)
        tips = env._fingertips_world()
        d = torch.linalg.norm(tips - surface, dim=-1)            # (E,10)
        if fa.any():
            tipd_sum += float((d * fa.float()).sum())
            tipd_n += int(fa.sum())

        recall, precision = press_accuracy(pressed, goal, threshold=SOUND_EPS)
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
    tip_mm = 1000.0 * tipd_sum / max(1, tipd_n)
    mode = "ZERO-RESIDUAL (pure IK reference)" if policy is None else f"policy {args.checkpoint}"
    if getattr(cfg, "arm_ik_follow", False):
        mode = "ARM-IK-FOLLOW + " + ("zero-residual fingers" if policy is None else f"policy {args.checkpoint}")
    print("=" * 60)
    print(f"[eval] {mode}  song={cfg.midi_path}  envs={env.num_envs}")
    print(f"[eval] MICRO  recall={mrec:.3f}  precision={mprec:.3f}  F1={mf1:.3f}  "
          f"(headline; vs MIDI over all steps)")
    print(f"[eval] macro  recall={rec:.3f}  precision={prec:.3f}  F1={f1:.3f}  "
          f"mean_reward/step={rew_sum / T:.3f}")
    print(f"[eval] active fingertip->key distance: {tip_mm:.1f} mm "
          f"(positioning quality; key half-width ~11mm)")
    print("=" * 60)
    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump({
                "mode": mode, "song": str(cfg.midi_path), "num_envs": env.num_envs,
                "steps": T,
                "micro": {"recall": mrec, "precision": mprec, "f1": mf1},
                "macro": {"recall": rec, "precision": prec, "f1": f1},
                "mean_reward_per_step": rew_sum / T,
                "tip_mm": tip_mm,
            }, f, indent=2)
        print(f"[eval] wrote metrics -> {args.out}")
    env.close()


main()
simulation_app.close()
