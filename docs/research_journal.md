# Autoresearch journal

Append-only. Newest entries at the top. One entry per loop tick.
Protocol: `docs/RESEARCH_LOOP.md`. Goal: key-press **F1 → 0.6–0.8** (not reward).

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
