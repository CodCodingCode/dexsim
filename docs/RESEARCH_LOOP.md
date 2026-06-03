# Autoresearch loop — protocol

**Study goal:** make the two-arm UR10e+Shadow (60-DoF) agent actually *play* a MIDI
song — i.e. push **key-press F1** (`scripts/eval_reference.py`, micro-averaged,
keys that truly sound vs the MIDI) from ≈0 toward RoboPianist-specialist range
(F1 ≈ 0.6–0.8). Reward going up is NOT the goal; F1 is. Current blockers:
BC/RP1M warm-start → PPO collapse, reference/IK quality, reward shaping balance.

This file is the contract each loop tick executes. Keep each tick **bounded and
fully reversible** (git-committed, no pushes, no destructive ops).

## Per-tick procedure
1. **Sense.** Read tail of `docs/research_journal.md` (last 2 entries) and
   `docs/STATUS.md`. Check live state: `nvidia-smi` compute-apps; tail the latest
   `wandb/run-*/files/output.log`; if a checkpoint exists, note latest F1.
2. **Pick ONE open question** from the backlog at the bottom of this file (or the
   journal). Don't restart the same thread every tick — rotate.
3. **Research** it: web search for the specific technique/paper (BC→PPO collapse
   fixes — DAPG / KL-to-BC / AWAC / RLPD; RP1M optimal-transport fingering;
   residual-RL warm-start; reward shaping for sparse key presses), OR analyze the
   code/logs. Capture the concrete, actionable takeaway — not a summary.
4. **Apply ONE small, reversible change** that tests a hypothesis: a config/reward/
   script edit, or a quick eval. Commit it (`git commit`, NO push) with a message
   tagged `[autoloop]`. Prefer changes evaluable by `eval_reference.py` quickly.
5. **Experiments:** a full train run (ETA ~2h) outlasts the 5-min tick — launch it
   in the **background**, record the run id + hypothesis in the journal, and let
   later ticks monitor/eval it. Never block the tick on a long run.
6. **Log.** Append a dated entry to `docs/research_journal.md`: question,
   finding, change made (+commit hash), hypothesis, how/when it'll be evaluated,
   and any result now available. Update the backlog (close/add questions).

## Hard safety rails
- **Never push.** Commits only.
- **Never delete** anything under `data/`, `wandb/`, or any checkpoint.
- **Never kill a healthy training process** without first confirming a checkpoint
  was saved; prefer launching new runs over killing existing ones.
- One change per tick. If unsure whether a change is safe/reversible, make it
  propose-only: write the diff into the journal instead of applying it.
- Stay inside `~/dexsim`. No system/global changes.

## ACTIVE MISSION (user-authorized 2026-06-03, tick 8)
Root cause of F1≈0 is traced: **left-arm mount rotation is wrong → left-arm IK
diverges (52% of active steps >100mm, median 286mm) → reference unplayable.** Right
hand is fine (~45mm DLS blur floor). The user has authorized the loop to **fix this
autonomously across ticks** — so for this mission the "one small edit per tick" rule
is relaxed to "one coherent step per tick" (an edit may be a real mount/cfg change),
but ALL hard safety rails still apply (no push, no deletes, no killing healthy runs,
git-commit every step, stay in ~/dexsim). Grade progress OFFLINE with
`scripts/diag_tip_err.py` (seconds) before spending a full sim eval.

Plan (each step = one tick; verify before advancing):
1. [ ] Locate where the left arm/hand mount orientation is set (combined-USD build
       `scripts/build_*usd*.py`, or `left_robot_cfg` rot in `piano_env_cfg.py`).
       Identify the ~90° error. (research-only tick is fine)
2. [ ] Apply the mount-rotation fix (small cfg/build edit). Commit.
3. [ ] Rebuild the twinkle reference (background) with the fixed mount; re-grade with
       `diag_tip_err.py`. Target: left-hand median → ~right-hand's ~45mm, divergence→~0.
4. [ ] If left≈right at the 45mm floor: attack the shared blur floor — switch the
       reference IK to WristPoseIK arm-servo + RP1M-clamped hand (ik.py's own
       recommendation) to get fingertips within a white-key width (~11mm).
5. [ ] Re-eval F1 (`eval_reference.py --zero`); confirm it moved off ~0.03.
6. [ ] THEN resume warm-start track (KL-to-frozen-BC etc.).

## Parked backlog (after the mission, or if blocked)
- [ ] KL-to-frozen-BC: add `β·KL(π‖π_BC)` to PPO loss, anneal β→0 (warm-start collapse).
- [ ] Reward balance: does the weighting reward *sounding* enough vs hovering? Ablate.
- [ ] Onset-F1 as a sharper diagnostic than sustained-key F1.
- [ ] Action scale / residual magnitude — can the residual physically depress a key?
- [x] Micro vs macro F1 (tick 1); press_threshold/sound-gate (tick 2); reference is
      the blocker not the metric (tick 4); stale-vs-fresh ruled out (tick 6); IK
      reach is the ceiling (tick 7); left-mount divergence isolated (tick 8).
