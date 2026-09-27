#!/bin/bash
# v15 = v14 recipe (seq planner, pose G, per-finger HOLD, plan servo) + three changes (2026-09-27):
#   1. seq planner honours hand_split_key=40 (one-sided): key 40 -> right index, never the left thumb
#   2. seq_keep_held: an onset group with nothing for a hand keeps that hand's held notes in the table
#   3. action-std cap: 0.50 until iter 1000, linearly down to 0.15 at iter 6000 (v12's std grew to 0.67)
# Tree: rsync the repo's source/ + scripts/ into ~/dexsim_piano_v15 (symlink .venv/assets/data/results like v14).
TAG=v15_split_stdcap
cd ~/dexsim_piano_v15 && export PYTHONPATH=source && unset MUJOCO_GL && mkdir -p logs/ab
setsid nohup /home/ubuntu/dexsim/.venv/bin/python scripts/mj/train_piano_mj.py \
  --midi "results/nettspend - we not like you (1).mid" --episode_s 65 \
  --num_envs 1024 --workers 28 --device cuda \
  --max_iterations 7000 --save_interval 100 --eval_every 100 \
  --anneal_false_press --fingering seq --seed 0 \
  --std_cap_start 1000 --std_cap_end 6000 --std_cap_init 0.5 --std_cap_final 0.15 \
  --tag $TAG > logs/ab/$TAG.log 2>&1 < /dev/null &
echo "launched pgid $!"
