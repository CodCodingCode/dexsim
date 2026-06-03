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

## ACTIVE MISSION (user-authorized 2026-06-03; re-scoped tick 9)
User authorized the loop to **fix the reference IK autonomously across ticks** — for
this mission the "one small edit per tick" rule is relaxed to "one coherent step per
tick" (an edit may be a real cfg/IK change), but ALL hard safety rails still apply
(no push, no deletes, no killing healthy runs, git-commit every step, stay in
~/dexsim). Grade progress OFFLINE with `scripts/diag_tip_err.py` before any sim eval.

**Tick-9 correction:** the tick-8 "left mount rotation is wrong" conclusion was a
GRADING ARTIFACT — `twinkle.npz` (the default reference) was built 17:09, *before* the
IK fixes landed (commit `6e15516`, 17:56: ik_damping 0.02→0.05, key-windows). Post-fix
builds (`twinkle_rous_ik.npz`) already cut left-hand median 286mm→72mm. There is **no
live ~90° mount bug** (no rotation is set in `piano_env_cfg`; the STATUS note is stale).
Real current state on the POST-fix reference: both hands limited by a shared **~45mm
DLS blur floor** (over-constrained FingertipIK), left still ~2× right (72 vs 31mm).

Re-scoped plan (each step = one tick; verify before advancing):
1. [x] (tick 9) Mount handling: none set in cfg (no ~90° bug).
2. [x] (tick 11) Current-IK rebuild (twinkle_curik.npz) STILL diverges left (305mm,
       54%) under build_reference.py/FingertipIK → left issue is LIVE, not stale
       (tick 9 over-corrected). The 72mm rous_ik came from build_rp1m_ik_reference.py's
       `_arm_only_dls` (arm-only, hand clamped) = the arm-servo design, 4× better.
3. [x] (tick 13) GATE RESOLVED = **SOLVER**, not reach. `diag_wrist_ik.py` shows
       WristPoseIK arm-servo places the palm within **4–14mm across the whole span,
       BOTH hands** (left keys 0–43: 3.7–10.2mm). FingertipIK's 5-tips-on-6-DoF-arm
       over-constraint is the divergence source. No mount/reach/base change needed.
4. [ ] **IMPLEMENT arm-servo reference builder** (the F1-moving step): a builder that
       per control step (a) uses `WristPoseIK` to servo each hand's palm over its
       active key(s) at the down orientation, (b) sets the assigned finger to a press
       pose (others to hover). Likely a new `scripts/build_reference_wrist.py` or a
       `--solver wrist` path in `build_reference.py`. Build to a FRESH npz; do not
       overwrite. NOTE: this is a multi-tick implementation — scope carefully.
5. [ ] Grade with `diag_tip_err.py` (target active median → ~11mm, divergence→~0),
       then `eval_reference.py --zero` — confirm F1 moves off ~0.03.
6. [ ] THEN resume warm-start track (KL-to-frozen-BC etc.).

## Parked backlog (after the mission, or if blocked)
- [ ] KL-to-frozen-BC: add `β·KL(π‖π_BC)` to PPO loss, anneal β→0 (warm-start collapse).
- [ ] Reward balance: does the weighting reward *sounding* enough vs hovering? Ablate.
- [ ] Onset-F1 as a sharper diagnostic than sustained-key F1.
- [ ] Action scale / residual magnitude — can the residual physically depress a key?
- [x] Micro vs macro F1 (tick 1); press_threshold/sound-gate (tick 2); reference is
      the blocker not the metric (tick 4); stale-vs-fresh ruled out (tick 6); IK
      reach is the ceiling (tick 7); left-mount divergence isolated (tick 8).
