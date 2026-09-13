#!/usr/bin/env bash
# One-time setup of the MuJoCo venv for this checkout.
#
#   bash scripts/mj/setup_venv.sh
#
# Uses uv (https://astral.sh/uv). torch resolves to the CUDA build on Linux
# with an NVIDIA driver and to the CPU/MPS build on macOS.
set -euo pipefail
cd "$(dirname "$0")/../.."

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
    mujoco "rsl-rl-lib>=5" torch numpy scipy tensorboard gymnasium \
    pretty_midi "imageio[ffmpeg]"

.venv/bin/python - <<'PY'
import numpy, torch, mujoco, scipy, rsl_rl, gymnasium, pretty_midi
print(f"OK  mujoco {mujoco.__version__} | numpy {numpy.__version__} | "
      f"torch {torch.__version__} (cuda={torch.cuda.is_available()}) | scipy {scipy.__version__}")
PY
echo "MuJoCo venv ready -- 'source env.sh' and run scripts/mj/smoke_piano_mj.py"
