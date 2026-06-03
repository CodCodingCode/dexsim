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

## Open-questions backlog (rotate; edit freely)
- [ ] KL-to-frozen-BC: add `β·KL(π‖π_BC)` to PPO loss, anneal β→0. Smallest fix
      for warm-start collapse. Where does it go in the rsl_rl PPO update?
- [ ] Is the RP1M/IK reference itself good? Eval pure reference (zero residual)
      F1 per song; if reference F1≈0 the warm-start can't help — fix retarget/IK first.
- [ ] press_threshold vs the piano's true sound-trigger depth — is F1 measured at
      the right threshold? (`eval_reference.py`, `press_accuracy`).
- [ ] Reward balance: does false_press/onset/fingering weighting actually reward
      *sounding the note* enough vs just hovering? Ablate.
- [ ] Micro- vs macro-averaged F1 in `eval_reference.py` — switch to micro.
- [ ] Onset-F1 as a sharper diagnostic than sustained-key F1.
- [ ] Action scale / residual magnitude — can the residual physically depress a key?
