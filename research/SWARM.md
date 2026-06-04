# TRAINING SWARM — shared memory

GOAL: train an arm_ik_follow piano policy that PLAYS easy.mid — high key-press **F1**
(recall AND precision both high), measured by `play/F1` etc. in the training log.

## CURRENT BEST STATE (start here)
- The piano physics were BROKEN and are now FIXED (in code defaults): key gravity disabled,
  KEY_SPRING_STIFFNESS=8, damping=4 → keys rest at frac~0 (no phantom presses) and are
  pressable; low-hover PhysX blowup is tamed by the key damping.
- **Zero-residual base** (arm IK-positioned + ready fingers) at hover 0.05 now scores
  **F1≈0.19, recall≈0.67, precision≈0.11, keys_sounding≈4.7**. THIS IS THE BAR TO BEAT.
- **Core problem:** PPO residual training DEGRADES the base — recall drifts 0.67→0.28 over
  ~30 iters (policy learns to hover, not press) and precision stays ~0.08-0.11.
- So two jobs: (1) keep recall high (stop the degradation), (2) raise precision (idle fingers
  press neighbor keys → mash footprint ~2-5 keys; only ~1-3 should sound).

## HOW TO RUN AN EXPERIMENT (each agent)
```
cd /home/ubuntu/dexsim && source env.sh >/dev/null 2>&1
python -u scripts/train/train_piano.py --arm_ik_follow \
  --midi data/midi/easy.mid --headless --num_envs 512 --max_iterations 90 \
  --tag <YOURTAG> --seed <N> \
  [your flags] > logs/swarm/<YOURTAG>.log 2>&1
```
Run it in the FOREGROUND (it blocks ~10-20 min on the shared GPU, then exits — no polling needed).
Use a UNIQUE --tag and --seed per run so logs/wandb don't collide. num_envs 512 keeps GPU light
(several agents run at once). DEFAULT (no overrides) reproduces the fixed-piano baseline.

## READ THE RESULT (after the run exits)
```
TAG=<YOURTAG>
# survived? (blowup = PhysX/CUDA error)
grep -ciE "PhysX error|CUDA error" logs/swarm/$TAG.log
# final metrics (LAST values):
for m in F1 recall precision keys_sounding; do echo -n "$m="; grep -E "play/$m:" logs/swarm/$TAG.log | grep -oE "[0-9.]+$" | tail -1; done
# trajectory (did recall hold or decline?):
paste <(grep -E "Learning iteration" logs/swarm/$TAG.log|grep -oE "[0-9]+/90") <(grep -E "play/F1:" logs/swarm/$TAG.log|grep -oE "[0-9.]+$") <(grep -E "play/recall:" logs/swarm/$TAG.log|grep -oE "[0-9.]+$") <(grep -E "play/precision:" logs/swarm/$TAG.log|grep -oE "[0-9.]+$")|awk 'NR%10==1||1{print}'|tail -12
```
A run "wins" if it BEATS the base bar (F1>0.19) OR holds recall>0.5 with rising precision, AND survived.

## CLI FLAGS available on train_piano.py (override per-run; no code edits needed)
--arm_ik_hover FLOAT (palm height above keys; 0.05 reaches/presses but near blowup; 0.07+ safe but fingers barely reach)
--hand_action_scale FLOAT (finger residual size; LOW 0.05-0.15 = less jitter/blowup + less degradation)
--init_noise FLOAT (PPO initial action-noise std; lower = stays near base)
--strike_vel FLOAT (rad/s gate for a key to sound; 0.05 lenient)
--key_press_weight / --onset_weight FLOAT (reward for sounding the right key / on its onset)
--fingering_weight / --arm_base_weight FLOAT (shaping = "be near the key"; LOWER stops hover-not-press)
--false_press_weight FLOAT (penalty for wrong keys)
--idle_clear_weight FLOAT (penalty for idle fingers hanging low)
--key_stiffness FLOAT (piano key return spring; 8 default; 12-15 firmer)
--hand_stiffness / --hand_effort FLOAT (finger actuator authority)
--freeze_arms (arms held static instead of IK-followed)
(env cfg also has idle_finger_curl, but it's NOT a CLI flag — leave it unless your thread is told to add one)

## THREAD ASSIGNMENTS (stay strictly in your lever set; do not touch other threads' levers)
- **Thread 1 — ANTI-DEGRADATION / stay near the good base.** Levers ONLY: --hand_action_scale,
  --init_noise (and you may note if lower LR/entropy would help, but don't change reward/physics/hover).
  Win = recall stays ≥0.5 over 90 iters (base 0.67 doesn't collapse). Try e.g. hand_action_scale
  {0.04,0.08,0.15} × init_noise {0.05,0.12}. Keep arm_ik_hover=0.05 fixed.
- **Thread 2 — PRECISION via idle fingers.** Levers ONLY: --idle_clear_weight, and a SLIGHTLY higher
  --arm_ik_hover (0.05-0.08) to lift idle fingers. Goal: raise precision >0.2 without killing recall.
  (Idle-finger contact is now REAL, so idle_clear may finally bite — test idle_clear_weight {20,60,150}.)
- **Thread 3 — REWARD BALANCE.** Levers ONLY: --key_press_weight, --onset_weight, --fingering_weight,
  --arm_base_weight, --false_press_weight. Find weights where BOTH recall and precision rise.
  Keep arm_ik_hover=0.05, hand_action_scale=0.10 fixed.
- **Thread 4 — GEOMETRY/PHYSICS reach.** Levers ONLY: --arm_ik_hover, --key_stiffness, --strike_vel,
  --hand_stiffness/--hand_effort. Find the lowest stable hover (max recall) and whether firmer keys/
  stronger fingers raise pressing without blowup. Keep reward + exploration at defaults.

## RESULTS (each agent: APPEND your best 1-2 rows here; do NOT edit others' rows)
| thread | tag | key flags | survived | F1 | recall | precision | keys_snd | note |
|--------|-----|-----------|----------|----|--------|-----------|---------|------|
| base   | -   | zero-residual reference | yes | 0.19 | 0.67 | 0.11 | 4.7 | the bar to beat |
