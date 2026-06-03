# Autoresearch journal

Append-only. Newest entries at the top. One entry per loop tick.
Protocol: `docs/RESEARCH_LOOP.md`. Goal: key-press **F1 → 0.6–0.8** (not reward).

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
