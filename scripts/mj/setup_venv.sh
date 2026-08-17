#!/usr/bin/env bash
# One-time setup of the MuJoCo venv for this checkout (no Isaac).
#
#   bash scripts/mj/setup_venv.sh
#
# Inherits the system-site torch (CUDA build) and pins numpy to 1.26: the
# system torch 2.7 predates numpy 2, and with numpy>=2 torch's numpy interop
# silently breaks ("Failed to initialize NumPy"). scipy is reinstalled in the
# venv because the system scipy 1.8 was built against numpy<1.25.
set -euo pipefail
cd "$(dirname "$0")/../.."

python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install \
    "numpy==1.26.4" \
    "scipy==1.11.4" \
    mujoco \
    gymnasium \
    pretty_midi \
    imageio imageio-ffmpeg \
    rsl-rl-lib

.venv/bin/python - <<'EOF'
import numpy, torch, mujoco, scipy, rsl_rl, gymnasium, pretty_midi
torch.from_numpy(numpy.ones(3))          # torch<->numpy interop must work
print(f"OK  mujoco {mujoco.__version__} | numpy {numpy.__version__} | "
      f"torch {torch.__version__} | scipy {scipy.__version__}")
EOF
echo "MuJoCo venv ready -- 'source env.sh' and run scripts/mj/smoke_piano_mj.py"
