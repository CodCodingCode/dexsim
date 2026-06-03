# Autoresearch journal

Append-only. Newest entries at the top. One entry per loop tick.
Protocol: `docs/RESEARCH_LOOP.md`. Goal: key-press **F1 → 0.6–0.8** (not reward).

> **LOOP PAUSED after tick 13 (user decision).** Diagnosis complete — root cause is
> the `FingertipIK` solver (see STATUS ticks 7–13). Cron `3f8f18cd` cancelled.
> Implementation of the arm-servo reference builder proceeds **interactively**, not
> via autonomous ticks. To resume the loop later: re-run `/loop 5m <tick prompt>`.

> **2026-06-03 — ARM-IK-FOLLOW mode implemented (interactive, post-loop).** Rather
> than rebuild a `q_ref`, decouple: `WristPoseIK` servos the 12 arm DoF online to the
> per-hand fingering centroid (`PianoEnv._servo_arms`); policy action masked to the 48
> finger DoF. Wired into the env (`arm_ik_follow`/`arm_ik_hover` cfg), `eval_reference`
> (`--arm_ik_follow` + online fingertip→key-distance metric), and `train_piano`
> (`--arm_ik_follow`). Commits `5bd41fa`, `7a22a12`.
> **Smoke test (easy.mid, --arm_ik_follow --zero):** stable 839 steps, no NaN; **recall
> 0.633** (FingertipIK reference was 0.114 → 5.5×) from arm positioning ALONE. precision
> ~0.02 / fingertip→key ~134 mm at zero residual (fingers don't press selectively yet —
> that's the RL job now). Architecture thesis confirmed: the arm positioning was the
> bottleneck, and the well-posed arm-servo removes it. **Next: short PPO run with
> `--arm_ik_follow` to confirm the policy lifts precision/F1 off the 0.03 ceiling.**

---

## 2026-06-03 — tick 13 (GATE RESOLVED: it's the SOLVER, not reach/mount)
- **Decisive result** (`diag_wrist2.log`, full 18-row sweep): WristPoseIK arm-servo
  places the palm within **4–14mm across the ENTIRE keyboard, both hands** (LEFT keys
  0–43: pos_err 3.7–10.2mm, ori_err ~1–2°; RIGHT 7–14mm). The arms physically reach
  everywhere to ~1cm.
- **=> Conclusion locked:** the unplayable reference is a **SOLVER artifact** —
  `FingertipIK` over-constrains the 6-DoF arm with 5 fingertip targets and diverges at
  singular configs (LEFT 305mm). It is NOT a reach gap, NOT the mount, NOT staleness,
  NOT the metric. Every other hypothesis is now eliminated by direct measurement.
- **Change:** recorded the conclusion in `STATUS.md` (source-of-truth update, ticks
  7–13 block) and marked mission step 3 done / step 4 = implement arm-servo builder.
- **THE DIAGNOSIS PHASE IS COMPLETE.** Remaining work = implement an arm-servo
  reference builder (WristPoseIK palm servo + finger press pose), rebuild, re-grade,
  re-eval F1. That is a multi-tick *implementation*, structurally different from the
  measure-one-thing-per-tick diagnosis the loop has been doing.
- **NEXT TICK (step 4):** begin the arm-servo builder. Smallest first slice: stand up
  `scripts/build_reference_wrist.py` skeleton that, for one hand, servos the palm to
  the active-key hover target with WristPoseIK and dumps q_ref+tip_err to a FRESH npz
  (no overwrite); grade it. Iterate finger-press posing in following ticks.
- **Tick-11 experiment FAILED:** `diag_wrist_ik.py` log was 42 lines, ended at Isaac
  boot, no results, no traceback. Diagnosis: stdout was block-buffered and a **C-level
  crash in `env.sim.step`** (no Python traceback ⇒ segfault) flushed nothing. The
  crash is before the first per-target print could flush — most likely the LEFT arm
  IK driving into a **singularity / physics blow-up** (which would itself answer the
  reach gate: a singular left arm = the divergence source).
- **Change:** `diag_wrist_ik.py` now line-buffers stdout (commit `e0a861b`) so partial
  per-target errors AND the exact crash target survive. Compiles.
- **Experiment relaunched (background):** `python -u scripts/diag_wrist_ik.py
  --headless --n 9 --iters 60` (PID 329525) → `logs/autoloop/diag_wrist2.log`.
- **NEXT TICK MUST read `logs/autoloop/diag_wrist2.log` — three outcomes:**
  1. full sweep completes, LEFT worst-err small (<~20mm) → SOLVER problem: arm reaches
     fine, adopt arm-servo (rp1m_ik) reference builder for plain MIDI.
  2. full sweep, LEFT worst-err large at part of span → REACH gap: move left_base_pos
     / shrink left_key_window.
  3. crashes again mid-LEFT-sweep at a specific key → that key/config is a **singular
     pose** the left arm can't servo through → reach/singularity fix (base pose or
     window), and the crash key tells us where.
- **Process note:** this is the 2nd failed/again-launched background run in a row
  (curik rebuild was fine; wrist sweep crashed). Background Isaac jobs are fragile;
  budget ~2 ticks each (launch → read). Acceptable but slowing convergence.
- **Result (current-IK rebuild, twinkle_curik.npz, built 20:23):** STILL diverges
  the left. hand L median **305mm, 54% steps >100mm**; hand R 53mm/17%. So the left
  divergence is **LIVE under build_reference.py/FingertipIK**, not a stale artifact —
  tick 9 over-corrected (it trusted the rous_ik proxy). Lesson: a proxy from a
  different pipeline is not a substitute for grading the thing you actually run.
- **What the 72mm rous_ik file actually was:** `twinkle_rous_ik.npz` came from
  `build_rp1m_ik_reference.py`, which uses `_arm_only_dls` — **arm-columns-only IK
  with the hand CLAMPED to an RP1M-retargeted pose**. That's the arm-servo design.
  It's 4× better (left 305→72mm) — confirming FingertipIK's over-constraint is the
  problem and arm-servo is the fix (already partly implemented).
- **But even arm-servo left ~72mm / 24% divergent** → suggests a genuine **left-arm
  REACH gap**, not only solver choice. Must settle reach-vs-solver before rewriting.
- **Change:** updated ACTIVE MISSION in `RESEARCH_LOOP.md` — un-obsolete the left-arm
  issue (it's live), set the decisive gate "is it reach or solver?" ahead of the IK
  rewrite. Commit below.
- **Experiment launched (background):** `scripts/diag_wrist_ik.py --headless`
  (PID 328994) — runs the WELL-POSED WristPoseIK (palm 6-DoF target on the 6-DoF arm,
  60 iters) and sweeps targets across each hand's span. Log `logs/autoloop/diag_wrist.log`.
- **NEXT TICK MUST read `logs/autoloop/diag_wrist.log`. Decision gate:**
  * small pos err across the LEFT span (e.g. <20mm everywhere) → arm reaches fine;
    the fix is purely "switch reference builder to WristPoseIK arm-servo" (adopt the
    rp1m_ik pipeline for plain MIDI). Solver problem, not geometry.
  * large left-span err / fails at part of the span → genuine **reach gap**: move
    `left_base_pos` / shrink `left_key_window`, THEN re-grade. Geometry problem.
- **Change:** `diag_tip_err.py` now prints each reference's build mtime — the
  anti-stale guard from tick 9's lesson (verified: twinkle.npz shows "built 17:09",
  obviously pre the 17:56 IK-fix commit). Commit `d0649a8`.
- **Experiment launched (background):** `build_reference.py --midi data/midi/twinkle.mid
  --headless --out data/reference/twinkle_curik.npz` (PID 328567; default fold+mute,
  ik_substeps 10, current cfg/damping 0.05). Writes a FRESH file (no overwrite),
  stores tip_err. Log: `logs/autoloop/build_curik.log`.
- **NEXT TICK MUST:** `python scripts/diag_tip_err.py data/reference/twinkle_curik.npz`
  (tail the build log if the npz isn't there yet — full song build takes a few min).
  This is the honest current-cfg baseline. Expected from the rous_ik proxy: left
  median ~70mm, right ~30mm, both above the ~11mm key-width target → confirms the
  ~45mm blur floor is the live blocker and motivates step 3 (arm-servo IK rewrite).
  If left ≫ right again here (with CURRENT cfg), then there IS a residual left-arm
  geometry issue after all and step 4 (base pose/window) moves up.
- **Backlog:** mission step 2 in progress; step 3 (FingertipIK→WristPoseIK arm-servo)
  queued as the highest-leverage F1 change.

---

## 2026-06-03 — tick 9 (mission step 1: locate left mount → CORRECTION)
- **Question:** where is the left mount orientation set; what's the ~90° error?
- **Finding — there is no live mount bug; tick 8 was misled by a STALE file:**
  * `piano_env_cfg` sets NO rotation for either arm — both derive from the same
    `UR10E_SHADOW_CFG`, only `*_base_pos` (y ∓0.30) differ. The STATUS:17,84 "~90°
    mount-rotation" note is stale.
  * The IK fixes (ik_damping 0.02→0.05 + key-windows) landed in commit `6e15516` at
    **17:56**. The default reference I graded, `twinkle.npz`, has mtime **17:09** —
    built ~47 min EARLIER, in the damping-0.02 era that the cfg comment itself says
    "diverges the LEFT arm (145mm+)." So the 286mm left divergence = stale artifact.
  * Post-fix builds already converge the left far better: `twinkle_rous_ik.npz`
    (18:08) left-hand median **72mm** (vs stale 286mm), divergence 24% (vs 52%).
- **=> Re-scope:** drop the mount-rotation hunt. The real current ceiling is the
  shared **~45mm DLS blur floor** (over-constrained FingertipIK), with left still ~2×
  right (72 vs 31mm). `ik.py`'s own docstring already recommends the fix: arm-servo
  (WristPoseIK) on the wrist + clamp the hand, instead of over-constraining a 6-DoF
  arm with 5 fingertip targets.
- **Change:** re-scoped the ACTIVE MISSION in `RESEARCH_LOOP.md` (mount step → done/
  obsolete; new priority = rebuild-current + switch reference IK to arm-servo) and
  logged this correction. (Lesson: always grade the file the env actually LOADS, and
  check its build time vs the last IK commit — `diag_tip_err.py` should arguably print
  the file mtime.)
- **NEXT TICK (step 2):** background-rebuild a current-IK twinkle reference to a FRESH
  filename (not overwriting any existing npz; check `build_reference.py` for an output
  flag — add one if absent, that's the tick's change), then `diag_tip_err.py` it to
  get the honest current-cfg left/right split.

---

## 2026-06-03 — tick 8
- **Question (backlog #1):** WHY does the IK miss by ~5cm (p90 446mm)?
- **Finding 1 — over-constrained solver floor:** `build_reference.py` drives
  `FingertipIK` (10 substeps). But `ik.py` itself documents FingertipIK
  over-constrains the 6-DoF arm with 5 fingertip targets → **~45mm residual + occasional
  divergence**. Our 48mm median IS that inherent floor; more iterations won't fix it
  (max_step 5cm × 10 already allows 50cm of travel).
- **Finding 2 (the big one) — it's the LEFT hand, and it DIVERGES:** enhanced
  `diag_tip_err.py` to split blur(<=100mm)/divergence(>100mm) per hand. On
  `twinkle.npz`:
  * **hand R: median 48mm, p90 72mm, 1% >100mm** → just the blur floor, basically fine.
  * **hand L: median 286mm, p90 947mm, 52% of active steps >100mm** → DIVERGENT.
  The left arm is failing on half its notes (median 29cm off). This is NOT generic IK
  blur — it's the **known-unfinished left-arm mount-rotation** (`STATUS.md:17,84`:
  "LEFT hand … still needs a ~90° mount-rotation tune"). The mis-oriented mount makes
  left-arm IK targets unreachable/divergent.
- **=> Root cause chain is now complete:** F1≈0 ← reference unplayable ← left-arm IK
  diverges ← left-hand mount rotation wrong. (Right hand is at the tunable 45mm floor.)
- **Change:** `diag_tip_err.py` now reports blur-vs-divergence + per-hand split
  (commit `b7f9011`) — this is what made the asymmetry legible offline.
- **NEXT STEP IS LARGER THAN ONE TICK — surfaced to user:** fixing the left mount
  (in the combined-USD build or `left_robot_cfg` orientation) + rebuilding references
  is a multi-step structural change, not a one-small-edit. Plan:
  1. fix left-arm mount rotation so left fingertips can reach keys (target: left-hand
     median tip_err → ~right-hand's 45mm, divergence→~0);
  2. rebuild twinkle reference, re-grade with `diag_tip_err.py` (offline, seconds);
  3. if left ≈ right (~45mm floor), attack the shared 45mm blur floor (e.g.
     WristPoseIK arm-servo + RP1M-clamped hand, per ik.py's own recommendation) to
     get fingertips within a key width;
  4. only THEN re-eval F1 and resume the warm-start track.
- **Backlog — NEW #1:** "fix left-arm mount rotation (left IK diverges, 52% steps
  >100mm)". #2: "shared 45mm DLS blur floor → switch reference IK to arm-servo".

---

## 2026-06-03 — tick 7
- **A/B result (tick 6) — hypothesis REFUTED:** fresh `twinkle_rp1m.npz` ≈ stale
  `twinkle.npz` (precision 0.020 vs 0.018, recall 0.135 vs 0.114, F1 0.035 vs 0.031).
  Staleness was NOT the cause. The reference mis-places fingers even freshly built.
- **ROOT CAUSE (now quantified offline):** `tip_err` in the reference shows the
  assigned fingertips never reach their target keys. `scripts/diag_tip_err.py
  data/reference/twinkle.npz` → active finger-steps median **48mm**, p90 **446mm**,
  only **0.4% within 11mm** (a white-key half-width), 1.8% within 22mm. So recall
  ~13% (fingers occasionally close) and precision ~2% (misplaced hand mashes
  neighbors) are both explained by **IK that doesn't reach the keyboard**, NOT the
  metric / hover / staleness. `twinkle_rous_ik.npz` is marginally better (6.8%
  within 11mm) but still far.
- **Change:** added `scripts/diag_tip_err.py` — offline (no-Isaac) IK-quality
  check from a reference's stored `tip_err`. Commit `52cb51a`. Now any tick can
  grade a reference in seconds instead of a multi-min sim eval.
- **Why this matters:** the warm-start track (KL-to-BC etc.) is fully blocked until
  the reference reaches keys. No PPO warm-start can fix a reference whose fingertips
  sit 5cm off. F1 ceiling is set by IK reach.
- **NEXT TICK — diagnose WHY IK misses (p90=446mm, max ~1m suggests targets are
  unreachable / wrong, not mild under-convergence):** candidates —
  (a) run `scripts/diag_reach.py` (Isaac, background) to test whether target keys are
      even reachable from each hand's base pose / keyboard span;
  (b) inspect `source/dexsim/piano/ik.py` solver iterations + target coords, and the
      hand→key span assignment (is one hand assigned keys outside its workspace?);
  (c) check arm base placement vs keyboard width in `piano_env_cfg.py`.
- **Backlog — NEW #1:** "IK reference does not reach keys (median 48mm)". Demote
  precision/false-press (it's a symptom of this). Warm-start track: ON HOLD.

---

## 2026-06-03 — tick 6
- **Question (backlog #1, narrowed):** confirm stale-vs-fresh reference — is precision
  0.018 just a stale artifact, or a real IK problem?
- **Setup found:** stale `twinkle.npz` and fresh `twinkle_rp1m.npz` are both
  (480,2,30) → clean same-length A/B. (`twinkle_rous_ik.npz` is only 159 frames — a
  different length, left out of this comparison to avoid confounds.)
- **Change:** added `--reference <npz>` to `eval_reference.py` (sets
  `cfg.reference_path`), so we can point eval at any q_ref. Compiles; commit `e0febe7`.
- **Experiment launched (background, sequential):** PID 327023 runs two zero-residual
  evals on twinkle —
  * stale default `twinkle.npz` → `logs/autoloop/ab_stale.json`
  * fresh `twinkle_rp1m.npz` → `logs/autoloop/ab_fresh.json`
- **NEXT TICK MUST:** read both JSONs (tail `logs/autoloop/ab_*.log` if absent — two
  Isaac boots, takes a few min). Decision rule:
  * fresh precision ≫ 0.018 (e.g. >0.1) → **CONFIRMED stale artifact.** Promote
    "rebuild ALL references with current geometry + make the default twinkle.npz the
    fresh build" to #1, then resume warm-start track. Reference pipeline is fine.
  * fresh precision still ~0.018 → staleness was NOT it; the IK/finger-placement
    itself mis-places idle fingers → dig into the reference builder / HOVER targets.
- **Backlog:** unchanged pending result.

---

## 2026-06-03 — tick 5
- **Question (backlog #1):** why is reference precision 1.8% — where do false presses
  come from?
- **Finding (likely root cause = STALE reference file):** zero-residual play commands
  exactly `q_ref[step]`, loaded from `data/reference/<midi-stem>.npz`. The eval used
  `data/reference/twinkle.npz`, **mtime 17:09**. But `HOVER_CLEARANCE` (idle-finger
  lift) was raised 0.010→0.030 and committed at **17:56** (commit `6e15516`) with the
  note that 1cm "sounded ~5 wrong keys/step, precision capped at 0.077." So the loaded
  reference predates the fix → idle fingers sit ~1cm too low → false presses →
  precision 0.018. Newer post-fix builds already exist on disk and were NOT used:
  `twinkle_rp1m.npz` (17:44), `twinkle_rous_ik.npz` (18:08).
- **Change:** `PianoEnv._load_reference` now stashes the resolved path and the status
  line prints the reference *filename + frame count* (was just "loaded"/"FALLBACK").
  Staleness is now visible in every eval/train log — the gap that hid this. Compiles;
  commit `0709590`.
- **Hypothesis (to confirm next tick):** evaluating a *post-hover-fix* reference
  (`twinkle_rous_ik.npz` or `twinkle_rp1m.npz`) will show markedly higher precision
  than the stale 0.018. If so, the reference pipeline is fine — we were just eval'ing
  a stale artifact — and the path to a usable warm-start is "rebuild references with
  current geometry," not a deeper IK fix.
- **NEXT TICK:** add a `--reference <npz>` flag to `eval_reference.py` (its one
  change), then background-eval `twinkle_rous_ik.npz` and `twinkle_rp1m.npz` vs the
  stale default; compare precision. (eval_reference.py currently has no way to point
  at a non-default reference — that's why I didn't run it this tick.)
- **Backlog:** #1 still "precision/false presses" but now narrowed to "confirm stale
  vs fresh reference"; if fresh ref is good → promote "rebuild all references" + then
  resume warm-start work.

---

## 2026-06-03 — tick 4
- **Question (backlog):** read tick-3's background reference eval; is the IK
  reference good enough to warm-start from?
- **RESULT (the gate):** pure IK reference, twinkle, corrected metric →
  **micro F1 = 0.031, recall 0.114, precision 0.018**, reward/step −1.9
  (`logs/autoloop/ref_twinkle.json`). Verdict: **reference is NOT yet a usable
  warm-start.** But it's no longer 0 — recall went 0→11% (the earlier
  KEY_SOUND_ANGLE −0.10→−0.012 fix let keys physically sound).
- **Red herring ruled out:** the log's `30 != 26` actuator warning is **benign** —
  the 4 unactuated joints/hand are Shadow's coupled distal `*J0` joints (driven by
  parent), per `ur10e_shadow.py:40`. NOT undriven fingers. Good that I checked
  before "fixing" the actuator cfg.
- **Real diagnosis:** precision 1.8% ≫ dominates — the reference *mashes ~50× more
  wrong keys than right ones*. Bottleneck = **finger-placement precision / false
  presses**, NOT key physics or actuation. recall 11% = fingers reach the right key
  only sometimes.
- **Change:** corrected `docs/STATUS.md` (was citing the buggy-metric 0.006 and the
  since-fixed −0.10 sound angle) — new measured table + benign-warning note + marked
  the −0.10 diagnostic STALE. Commit `056d530`. (Source-of-truth fix so this isn't
  re-chased.)
- **Backlog update — NEW #1:** *why precision 1.8%?* Where do the false presses come
  from — idle/passing fingers resting on keys, IK targeting wrong keys, or the
  hover/clearance pose too low? Candidate next ticks: (a) eval on `easy.mid` to see
  if it's song-specific; (b) inspect IK target assignment vs fingering; (c) raise
  idle-finger hover clearance. Warm-start work (KL-to-BC etc.) is **on hold** until
  reference precision is fixed — warm-starting from a 0.03-F1 reference is pointless.

---

## 2026-06-03 — tick 3
- **Question (backlog):** launch the background pure-reference Isaac eval — the gate
  we've been deferring — now that the metric (ticks 1+2) is trustworthy.
- **Change:** added `--out <json>` to `eval_reference.py` so a backgrounded eval
  leaves a machine-readable result instead of stdout-only. Compiles; commit `ba3155e`.
- **Experiment launched (background):** `python scripts/eval_reference.py
  --midi data/midi/twinkle.mid --zero --headless --num_envs 16
  --out logs/autoloop/ref_twinkle.json` (PID 326153 at launch; stdout ->
  `logs/autoloop/ref_twinkle.log`). Isaac boots in minutes; result outlives this tick.
- **NEXT TICK MUST:** read `logs/autoloop/ref_twinkle.json` (and tail the .log if the
  json isn't there yet — Isaac may still be booting or may have errored). If F1>0:
  reference is a usable warm-start; record per-metric numbers, move to reward/warm-
  start work. If F1≈0 or the run errored: the reference/IK/mount is the blocker —
  that becomes the top backlog item (warm-start is pointless without it).
- **Backlog update:** "Is the RP1M/IK reference good?" → IN PROGRESS (awaiting run).

---

## 2026-06-03 — tick 2
- **Question (backlog):** press_threshold vs the piano's true sound-trigger depth
  (the TODO left in tick 1).
- **Finding (important):** `PianoEnv._key_pressed_fraction()` is NOT a raw depth —
  it already applies the simulator's **velocity-gated sounding latch** (key starts
  sounding only when struck past `KEY_SOUND_ANGLE=-0.012` rad while moving down
  faster than `key_strike_vel`, stays until `frac<0.25`) and returns **0 for any
  key not sounding**. So the sim's own ground-truth "this key sounds" == returned
  fraction > 0. Thresholding the eval at 0.5 re-applied the gate a second time and
  dropped softly-held sustained notes riding in [0.25, 0.5) → recall deflated.
- **Change:** `eval_reference.py` scores `sounding = pressed > SOUND_EPS(1e-6)` for
  both micro and macro (passed threshold into `press_accuracy` too). Compiles;
  commit `3034758`.
- **Hypothesis:** zero-residual reference recall will read **higher** than before on
  any song with held notes (we stop discarding sustains); precision ~unchanged since
  the gate already suppresses static resting contact. Net F1 should be a more honest,
  likely higher reference number.
- **Evaluate:** needs a sim run — `scripts/eval_reference.py --zero --headless`.
  Still queued as a background Isaac tick; the two metric fixes (tick 1+2) mean that
  run will finally produce a trustworthy reference F1 to gate warm-start on.
- **Backlog update:** closed press_threshold item. Next tick → either launch the
  background pure-reference Isaac eval, or the reward-balance ablation (does sounding
  the note out-reward mere hovering?).

---

## 2026-06-03 — tick 1
- **Question (backlog):** micro- vs macro-averaged F1 in `eval_reference.py`.
- **Finding:** the old script averaged per-step recall/precision only over steps
  that *had* a goal key, then combined — so false presses during rests never hit
  precision, and few-active-key steps were over-weighted. RoboPianist/RP1M report
  **micro** F1: sum TP/FP/FN over all steps+envs, compute P/R/F1 once.
- **Change:** `eval_reference.py` now accumulates global TP/FP/FN and prints MICRO
  P/R/F1 as the headline (macro kept as secondary). Compiles (`py_compile`); not
  yet run under Isaac. Commit `732002b`.
- **Hypothesis:** micro F1 will read *lower* than the old number for any policy
  that mashes during rests — a more honest baseline. Confirm next time the sim runs.
- **Evaluate:** run `scripts/eval_reference.py --zero --headless` (needs Isaac;
  queue as a background tick). Note: `thresh=0.5` is still arbitrary — flagged TODO,
  next-but-one backlog item (true sound-trigger depth).
- **Backlog update:** closed "micro vs macro". Next tick → "Is the RP1M/IK
  reference itself good?" (background Isaac eval) OR the press_threshold item.

---

## 2026-06-03 — tick 0 (loop initialized)
- **State at start:** F1 ≈ 0 per `docs/STATUS.md`; latest run `1jqpgsue` ran ~12min
  then stopped; no live GPU process. Reward climbing is shaping (finger/onset/arm),
  not notes sounding.
- **Action:** scaffolding only — wrote `docs/RESEARCH_LOOP.md` (per-tick protocol +
  safety rails + backlog) and this journal. No code change.
- **Next tick should:** start with backlog item "Is the RP1M/IK reference itself
  good?" — without a reference that scores F1>0 at zero residual, no warm-start can
  bootstrap PPO. Run `eval_reference.py --zero` per song and record per-song F1.
