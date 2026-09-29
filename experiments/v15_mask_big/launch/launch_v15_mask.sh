#!/bin/bash
# v15 (2026-09-28): v14 recipe + seq planner feasibility mask (reach_check.py at pose G) + thumbs excluded. Control for the model-size A/B.
TAG=${1:-v15_mask}; HD=${2:-}
cd ~/dexsim_piano_v15 && export PYTHONPATH=source && unset MUJOCO_GL && mkdir -p logs/ab
EXTRA=""; [ -n "$HD" ] && EXTRA="--hidden_dims $HD"
setsid nohup /home/ubuntu/dexsim/.venv/bin/python scripts/mj/train_piano_mj.py \
  --midi "results/nettspend - we not like you (1).mid" --episode_s 65 \
  --num_envs 1024 --workers 14 --device cuda \
  --max_iterations 7000 --save_interval 100 --eval_every 100 \
  --anneal_false_press --fingering seq --seed 0 --tag $TAG $EXTRA > logs/ab/$TAG.log 2>&1 < /dev/null &
echo "launched $TAG pgid $!"
