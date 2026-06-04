# Research PLAN

## GOAL
Identify proven, concrete techniques to make RL-based **bimanual dexterous piano-playing**
(arm + Shadow Hand, RoboPianist / PianoMime lineage, Isaac Sim / MuJoCo) actually **learn to
press keys** and reach high key-press **F1** — covering reward shaping, action-space/embodiment
design, PhysX/Isaac contact-stability for finger↔sprung-key contact, and imitation/warm-start/
curriculum to escape the "hover-near-the-key-but-don't-press" local optimum. Want **sources +
specific recommended settings**, not generalities.

### Concrete context driving the goal (our live failure modes)
- Zero-residual reference (arm IK-positioned + ready fingers) reaches F1≈0.19 / recall≈0.67, but
  **PPO residual training DEGRADES it** (recall 0.67→0.28): the shaped reward pays ~0.47 for
  hovering near keys and ~0 for actually pressing, so the policy learns NOT to press.
- Low hover (fingers reach keys) **explodes PhysX** unless key joint damping is high; high key
  stiffness stops the explosion but makes keys unpressable by the weak (stiffness-3) Shadow fingers.
- Precision capped by idle fingers pressing neighbor keys (hand footprint ~ several keys).

## THREADS (independent, non-overlapping)

### Thread 1 — Reward shaping & RL algorithm for key-press F1
**Objective:** what reward terms / weights / RL algorithm (DroQ, SAC, PPO, MPO…) the published
RoboPianist/PianoMime/RP1M and related dexterous-key works use; which terms are *essential* vs
shaping; how they avoid the "hover not press" optimum; reported F1 numbers and the settings behind them.
**MUST NOT cover:** contact/physics solver tuning (Thread 3); imitation/warm-start (Thread 4);
arm vs slider embodiment (Thread 2).
**Done when:** concrete reward recipe (terms + relative weights) + algorithm choice + target F1 documented with sources.

### Thread 2 — Action space & embodiment design
**Objective:** how proven systems structure the action space for dexterous piano: floating/slider
hands vs full arms; residual-over-reference vs direct joint targets; decoupling gross arm
positioning (IK/analytic) from finger pressing (RL); control rates; per-joint action scaling.
What makes the 60-DoF problem tractable.
**MUST NOT cover:** reward weights (Thread 1); contact physics (Thread 3); BC/curriculum (Thread 4).
**Done when:** clear recommendation on arm handling + action parameterization + scaling, with sources.

### Thread 3 — Isaac Sim / PhysX contact stability for finger↔sprung-key contact
**Objective:** preventing PhysX/GPU solver explosions when many dexterous fingers contact
spring-loaded key joints: key joint stiffness/damping, contact offsets, solver iteration counts,
depenetration velocity, contact stiffness/compliance, substep/dt, articulation settings. Best
practices for stable high-DoF hand–object contact in Isaac Lab.
**MUST NOT cover:** reward (Thread 1); RL algorithm/warm-start (Thread 4); action space (Thread 2).
**Done when:** concrete Isaac/PhysX settings that give stable finger-key contact, with sources.

### Thread 4 — Imitation / warm-start / curriculum to escape local optima
**Objective:** fixes for "good initialization degraded by RL" and sparse-press local optima:
BC→PPO collapse remedies (DAPG, KL-to-frozen-BC, AWAC, Cal-QL, RLPD), residual-RL warm-start,
curricula for dexterous pressing, and how RP1M / RoboPianist multi-song datasets are used to seed policies.
**MUST NOT cover:** base reward design (Thread 1); contact physics (Thread 3); action space (Thread 2).
**Done when:** concrete method(s) to keep/improve a competent init under RL, with sources + settings.
