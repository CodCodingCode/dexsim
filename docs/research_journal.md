# Autoresearch journal

Append-only. Newest entries at the top. One entry per loop tick.
Protocol: `docs/RESEARCH_LOOP.md`. Goal: key-press **F1 → 0.6–0.8** (not reward).

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
