#!/bin/bash
# wait for the v13 run to finish, then launch v14 (so v13 keeps the whole box)
while pgrep -f "^/home/ubuntu/dexsim/.venv/bin/python scripts/mj/train_piano_mj.*v13_seq" >/dev/null; do sleep 120; done
echo "$(date -u) v13 finished, launching v14" >> ~/chain_v14.log
bash ~/launch_v14.sh >> ~/chain_v14.log 2>&1
