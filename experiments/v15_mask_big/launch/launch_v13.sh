#!/bin/bash
# v13: v12 recipe (leap curriculum, 7000 iters, from scratch) + fingering "seq" + pose G.
TAG=v13_seq_poseG
cd ~/dexsim_piano_v13 && export PYTHONPATH=source && unset MUJOCO_GL && mkdir -p logs/ab
setsid nohup /home/ubuntu/dexsim/.venv/bin/python scripts/mj/train_piano_mj.py \
  --midi "results/nettspend - we not like you (1).mid" --episode_s 65 \
  --num_envs 1024 --workers 28 --device cuda \
  --max_iterations 7000 --save_interval 100 --eval_every 100 \
  --anneal_false_press --fingering seq --seed 0 --tag $TAG > logs/ab/$TAG.log 2>&1 < /dev/null &
echo "launched pgid $!"
