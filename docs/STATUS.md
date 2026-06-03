# Project status — honest assessment (2026-06-02)

**TL;DR:** the infrastructure is solid and validated; **the robot does not yet
play** — verified by F1 of actual key presses (≈0), not by reward. The climbing
PPO reward was the *shaping* term (fingers drifting toward keys), not notes
sounding. Below is the evidence and the real plan.

## ✅ What works (built + validated)

- Isaac Sim 4.5 + Isaac Lab 2.1 on this **compute-only H100** (incl. the
  Vulkan/OptiX driver staging — `docs/SETUP.md`). GPU physics + path-traced
  rendering confirmed.
- **Embodiment**: two UR10e + Shadow Hand arms, bonded into single articulations
  (30 joints each), stable, no explosion, hands hover over the keyboard.
- **LEFT hand**: imported from MuJoCo `left_hand.xml`, bonded to a UR10e
  (`ur10e_shadow_left.usd`, 30 joints, drop-in `robot0_*` names). *Still needs a
  ~90° mount-rotation tune to be stable.*
- **88-key sprung piano** USD; **MIDI → goal + onsets + auto-fingering** pipeline;
  PianoMime env (residual-over-IK-reference, composite reward, 1236-d obs);
  **wrist + head cameras**; **W&B logging**; **git repo**.

## ❌ What does NOT work: actually playing

Measured with `scripts/eval_reference.py` (recall/precision/**F1** of keys that
actually sound vs the MIDI), pure reference (zero residual):

| song | reference | F1 | mean reward/step |
|------|-----------|----|------------------|
| song.mid (5 octaves) | ready-pose fallback | **0.006** | 0.175 |
| song.mid | IK reference | **0.008** | 0.020 |
| twinkle (1 octave, reachable) | IK reference | **0.006** | 0.006 |

Reward can be *positive while F1≈0* → **reward is not evidence of playing.**

> **UPDATE 2026-06-03 (autoloop tick 4):** the numbers above were measured with a
> buggy metric AND the old `KEY_SOUND_ANGLE = -0.10` (beyond physical key travel —
> keys could *never* sound). After fixing the sound angle to **-0.012** and the eval
> metric (micro-averaged, scored by the sim's own velocity-gated sound latch —
> commits `732002b`, `3034758`), the pure IK reference on **twinkle** now reads:
>
> | metric | recall | precision | F1 |
> |--------|--------|-----------|----|
> | micro  | **0.114** | **0.018** | **0.031** |
>
> So the gate moved from "keys physically can't sound (recall 0)" to "keys sound,
> but the reference mashes ~50× more wrong keys than right (precision 1.8%)." The
> bottleneck is now **finger-placement precision / false presses**, not key physics.
> The `30 != 26` actuator warning is **benign** — the 4 unactuated joints per hand
> are the Shadow Hand's coupled distal `*J0` joints (driven by their parent). The
> `-0.10`/`-0.056` diagnostic below is STALE.

> **UPDATE 2026-06-03 (autoloop ticks 7–13) — root cause isolated to the IK SOLVER:**
> The reference's assigned fingertips never reach their keys (`diag_tip_err.py`:
> active median 48–58mm, only 0.4% within an 11mm key-width; the LEFT hand diverges,
> median ~305mm, 54% of steps >100mm under `build_reference.py`/`FingertipIK`). This
> is **NOT a reach gap and NOT the mount**: `diag_wrist_ik.py` proves the *well-posed*
> arm-servo solver (`WristPoseIK`, one palm target on the 6-DoF arm) places the palm
> within **4–14mm across the ENTIRE keyboard span, both hands** (left keys 0–43:
> 3.7–10.2mm). The culprit is `FingertipIK` **over-constraining the 6-DoF arm with 5
> fingertip targets** → singular-config divergence. **Fix:** build the reference with
> arm-servo IK (drive the palm/wrist with `WristPoseIK`, then set the assigned finger
> to a press pose), per `ik.py`'s own design note — not FingertipIK. The diagnosis
> phase is done; this is the implementation step that should finally move F1 off 0.03.

### Root-cause diagnostic (`scripts/key_press_diag.py`, Twinkle) — STALE, see update above
```
over 445 note-steps:  goal key depressed at all = 1%,  sounded = 0%
deepest any key pressed = -0.056 rad  (needs <= KEY_SOUND_ANGLE = -0.10 to sound)
typical goal-key angle  = -0.0006 rad (i.e. the finger isn't on the key)
```

## The real problems (priority order)

1. **Fingers don't contact/press the keys** — even for reachable Twinkle, goal
   keys sit at ~-0.0006 rad 99% of the time. The IK reference isn't landing the
   Shadow fingertips on their target keys (song reference fingertip error ~13cm).
2. **Press threshold mismatch** — sounding needs -0.10 rad; the deepest press ever
   achieved is -0.056. Even a good press wouldn't register. Fix one or both of:
   `KEY_SOUND_ANGLE` (assets/piano.py), `PRESS_DEPTH` (piano/geometry.py), key
   spring stiffness, finger effort/stiffness.
3. **Reach** — a fixed-base UR10e (~1.18m) can't span a 1.5m / 5-octave keyboard.
   This is the *second* problem (RoboPianist solved it with sliding bases/rails).

## Plan (bottom-up; don't train until step 3 passes)

1. **One finger, one key.** Directly command a fingertip onto a single key; tune
   threshold / press-depth / spring / finger strength until a press *reliably*
   crosses the sound threshold. Foundation for everything.
2. **Fix the IK reference** so fingertips actually land on (and press) their
   assigned keys — verify with `build_reference.py`'s fingertip-error readout
   (target: <1cm, currently ~13cm).
3. **Re-verify F1** climbs on a small reachable song (proves the full loop).
4. **Add base rails** (prismatic translate per arm) for full-keyboard reach.
5. **Then** PPO/residual training is meaningful — watch F1, not just reward.

## Parked (until the above is settled)
- Left-arm mount-rotation tune + wiring `left_robot_cfg` to the left USD.
- GitHub remote (local repo is ready; needs public/private + auth decision).
- Vision/MolmoAct camera-conditioned policy (cameras render; not in the policy yet).
