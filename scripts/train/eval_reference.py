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
parser.add_argument("--idle_finger_curl", type=float, default=None, help="curl idle fingers up in the base pose (rad; sign-test the anti-mash)")
parser.add_argument("--arm_ik_hover", type=float, default=None, help="palm hover height above keys")
parser.add_argument("--arm_lookahead", type=int, default=None, help="steps of upcoming notes for the arm centroid (1=track current note tightly)")
parser.add_argument("--single_finger", action="store_true", help="one-finger-per-note: aim the primary fingertip at the current note")
parser.add_argument("--primary_finger", type=int, default=None, help="which finger presses (0=th,1=ff,2=mf,3=rf,4=lf)")
parser.add_argument("--single_press_z", type=float, default=None, help="m vs key top to drive the fingertip (neg=into key)")
parser.add_argument("--single_curl", type=float, default=None, help="rad to curl the non-primary fingers up")
parser.add_argument("--idle_hand_retract", type=float, default=None, help="m an idle hand lifts off the keys")
parser.add_argument("--hand_tilt", type=float, default=None)
parser.add_argument("--hand_tilt_axis", type=int, default=None)
parser.add_argument("--hand_stiffness", type=float, default=None, help="override hand actuator stiffness")
parser.add_argument("--hand_effort", type=float, default=None, help="override hand actuator effort_limit")
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
        if args.idle_finger_curl is not None:
            cfg.idle_finger_curl = args.idle_finger_curl
        if args.arm_ik_hover is not None:
            cfg.arm_ik_hover = args.arm_ik_hover
        if args.arm_lookahead is not None:
            cfg.arm_lookahead = args.arm_lookahead
        if args.single_finger:
            cfg.single_finger = True
        if args.primary_finger is not None:
            cfg.primary_finger = args.primary_finger
        if args.single_press_z is not None:
            cfg.single_press_z = args.single_press_z
        if args.single_curl is not None:
            cfg.single_curl = args.single_curl
        if args.idle_hand_retract is not None:
            cfg.idle_hand_retract = args.idle_hand_retract
        if args.hand_tilt is not None:
            cfg.hand_tilt = args.hand_tilt
        if args.hand_tilt_axis is not None:
            cfg.hand_tilt_axis = args.hand_tilt_axis
    if args.hand_stiffness is not None or args.hand_effort is not None:
        for rc in (cfg.left_robot_cfg, cfg.right_robot_cfg):
            ha = rc.actuators["hand"]
            if args.hand_stiffness is not None:
                ha.stiffness = args.hand_stiffness
                ha.damping = max(0.1, 0.05 * args.hand_stiffness)
            if args.hand_effort is not None:
                ha.effort_limit = args.hand_effort
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
    # per-hand micro counts (left/right key split) + onset micro counts (rising-edge
    # strike vs note-start) + arm-health step sums. Same micro philosophy as above.
    tpL = fpL = fnL = tpR = fpR = fnR = 0.0
    # onset is matched with a +/-W-step tolerance (a finger takes a few 20Hz steps to
    # descend and trip the strike, so an exact-step match reads ~0). Collect the played
    # strikes and goal onsets per step, then dilate over time and match after the loop.
    played_steps = []; onset_steps = []
    W_on = int(getattr(env, "onset_tol_steps", 3))
    margin_sum = jerk_sum = 0.0
    lmask = env.left_key_mask.bool()        # (88,) left-hand keys
    rmask = env.right_key_mask.bool()       # (88,) right-hand keys
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

        # ONSET events: read the rising-edge strike (_just_struck) NOW, before the
        # _key_pressed_fraction() call below re-advances the latch and clears it.
        # played = key went silent->sounding this step; goal = note starts this step.
        played_steps.append(env._just_struck.clone())
        onset_steps.append(env._onset_now().bool().clone())
        # arm health (motion quality, invisible to F1): worst-joint limit margin and
        # policy action jerk, both already computed by the env this step.
        margin_sum += float(env._arm_limit_margin().mean())
        jerk_sum += float(env._action_jerk.mean())

        pressed = env._key_pressed_fraction()
        goal = env._goal_now()
        # micro counts over every env/key this step
        sounding = pressed > SOUND_EPS
        goal_b = goal.bool()
        tp_tot += float((sounding & goal_b).sum())
        fp_tot += float((sounding & ~goal_b).sum())
        fn_tot += float((~sounding & goal_b).sum())
        # per-hand micro counts: restrict the same sounding/goal masks to each hand's
        # keys (lmask/rmask partition all 88, so left+right sums match the overall).
        tpL += float((sounding & goal_b & lmask).sum())
        fpL += float((sounding & ~goal_b & lmask).sum())
        fnL += float((~sounding & goal_b & lmask).sum())
        tpR += float((sounding & goal_b & rmask).sum())
        fpR += float((sounding & ~goal_b & rmask).sum())
        fnR += float((~sounding & goal_b & rmask).sum())

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

    def _prf(tp, fp, fn):
        r = tp / (tp + fn + eps); p = tp / (tp + fp + eps)
        return p, r, 2 * p * r / (p + r + eps)
    # per-hand F1 (which arm is carrying / dragging the score) + onset F1 (did it
    # strike notes ON TIME, not just hold them) + arm-health step means.
    pL, rL, f1L = _prf(tpL, fpL, fnL)
    pR, rR, f1R = _prf(tpR, fpR, fnR)
    # windowed onset match: dilate both played-strike and goal-onset timelines by +/-W
    # steps, then a goal onset counts as recalled if a strike landed within the window
    # (and vice-versa for precision). Tolerates the finger's descent latency.
    played_t = torch.stack(played_steps).float()    # (T, E, 88)
    onset_t = torch.stack(onset_steps).float()       # (T, E, 88)

    def _dilate(x, w):                               # max over +/-w along time (dim 0)
        out = x.clone()
        for s in range(1, w + 1):
            fut = torch.zeros_like(x); fut[:-s] = x[s:]
            pst = torch.zeros_like(x); pst[s:] = x[:-s]
            out = torch.maximum(out, torch.maximum(fut, pst))
        return out
    played_d = _dilate(played_t, W_on)
    onset_d = _dilate(onset_t, W_on)
    on_r = float((onset_t * played_d).sum() / (onset_t.sum() + eps))   # onsets struck in-window
    on_p = float((played_t * onset_d).sum() / (played_t.sum() + eps))  # strikes near an onset
    on_f1 = 2 * on_p * on_r / (on_p + on_r + eps)
    print(f"[dbg] played_strikes_total={int(played_t.sum())}  onset_total={int(onset_t.sum())}")
    # strike vs onset counts: a quick "did it actually play?" sanity signal. The pure
    # reference only strikes during the initial settle (n_strikes << n_onsets) -> it
    # holds a static pose rather than re-striking each note, which is why onset F1 ~ 0.
    n_strikes = int(played_t.sum())
    n_onsets = int(onset_t.sum())
    avg_margin = margin_sum / T
    avg_jerk = jerk_sum / T
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
    print(f"[eval] per-hand  F1_left={f1L:.3f} (R={rL:.3f} P={pL:.3f})  "
          f"F1_right={f1R:.3f} (R={rR:.3f} P={pR:.3f})")
    print(f"[eval] onset     F1={on_f1:.3f}  recall={on_r:.3f}  precision={on_p:.3f}  "
          f"({n_strikes} strikes vs {n_onsets} onsets, +/-{W_on}-step tol; not just held)")
    print(f"[eval] arm       limit_margin={avg_margin:.3f} (1=mid-range, 0=at a limit)  "
          f"action_jerk={avg_jerk:.4f} (policy smoothness; 0 for pure reference)")
    print("=" * 60)
    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump({
                "mode": mode, "song": str(cfg.midi_path), "num_envs": env.num_envs,
                "steps": T,
                "micro": {"recall": mrec, "precision": mprec, "f1": mf1},
                "macro": {"recall": rec, "precision": prec, "f1": f1},
                "per_hand": {
                    "left": {"recall": rL, "precision": pL, "f1": f1L},
                    "right": {"recall": rR, "precision": pR, "f1": f1R},
                },
                "onset": {"recall": on_r, "precision": on_p, "f1": on_f1,
                          "tol_steps": W_on, "n_strikes": n_strikes, "n_onsets": n_onsets},
                "arm_health": {"limit_margin": avg_margin, "action_jerk": avg_jerk},
                "mean_reward_per_step": rew_sum / T,
                "tip_mm": tip_mm,
            }, f, indent=2)
        print(f"[eval] wrote metrics -> {args.out}")
    env.close()


main()
simulation_app.close()
