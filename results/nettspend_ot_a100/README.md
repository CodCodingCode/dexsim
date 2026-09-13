# nettspend_ot_a100 — F1 0.84 on "nettspend - we not like you (1).mid"

Trained 2026-09-10 on the A100 box (30 cores), MuJoCo stack, 1024 envs x 28
worker processes, 3000 iterations (~98M env steps, ~6 h).

    python scripts/mj/train_piano_mj.py \
        --midi "results/nettspend - we not like you (1).mid" --episode_s 65 \
        --fingering ot --num_envs 1024 --workers 28 --device cuda \
        --max_iterations 3000 --save_interval 100 --anneal_false_press --seed 0

Obs: 368-dim slim layout (16 reachable keys, key vel + sounding latch);
asymmetric critic (+10 fingertip contact forces, +1 hand-collision flag).

Deterministic rollout of model_final.pt (scripts/mj/diag_rollout.py ... ot):

    F1 0.836  recall 0.767  precision 0.917
    LEFT  recall 0.835 precision 0.942 | RIGHT recall 0.609 precision 0.845
    note onsets 236: hit 218, missed 18 (8%); hit latency median 0 steps

Training-time F1 (with exploration noise) ended at 0.75.

Remaining misses are concentrated on three keys: 66 (R middle, recall 0.01),
63 (R thumb, 0.11), 25 (L index, 0.41). All are keys that appear on a small
fraction of steps; the frequent keys are at 0.9-0.99. Next lever is reward
balance (per-note rather than per-step press reward, higher false-press
start) so rare/short notes are worth chasing.

Comparison: previous best documented result was F1 0.40 (Isaac, June 2026,
3-key one-hand song). The heuristic-fingering run on this same song was
stopped at iter 1073 with training F1 0.63 (deterministic 0.69 at iter 900).
