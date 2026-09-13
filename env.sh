# Source this before running anything: `source env.sh`
# Activates the MuJoCo venv and makes the dexsim package importable.
DEXSIM_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]:-$0}" )" && pwd )"

source "${DEXSIM_ROOT}/.venv/bin/activate"
export PYTHONPATH="${DEXSIM_ROOT}/source:${PYTHONPATH}"

# Headless rendering backend for `play_piano_mj.py --video`:
#   Linux with an NVIDIA driver -> egl; macOS -> leave unset (CGL).
if [ "$(uname)" != "Darwin" ] && [ -z "${MUJOCO_GL:-}" ]; then
  export MUJOCO_GL=egl
fi

echo "[dexsim] env ready  (python: $(python --version 2>&1), root: ${DEXSIM_ROOT})"
