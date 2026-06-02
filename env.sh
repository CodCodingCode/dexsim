# Source this before running anything: `source env.sh`
# Activates the venv, accepts the Omniverse EULA non-interactively (needed for
# headless runs), and makes the dexsim package importable.
DEXSIM_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]:-$0}" )" && pwd )"

source "${DEXSIM_ROOT}/.venv/bin/activate"

# Accept the NVIDIA Omniverse EULA for headless / non-interactive launches.
export OMNI_KIT_ACCEPT_EULA=YES

# --- NVIDIA Vulkan/GLX userspace driver (staged locally) ---------------------
# This container ships CUDA but NOT the graphics driver libs, so Isaac Sim's RTX
# renderer + GPU-physics foundation fail at Vulkan init. We stage the matching
# 580.105.08 userspace libs (libGLX_nvidia + Vulkan ICD) under .nvidia-gl and
# point the Vulkan loader at them -- no sudo, no kernel touch, fully local.
# Re-create with scripts/setup_nvidia_gl.sh if .nvidia-gl is ever removed.
if [ -d "${DEXSIM_ROOT}/.nvidia-gl" ]; then
  export VK_ICD_FILENAMES="${DEXSIM_ROOT}/.nvidia-gl/nvidia_icd.json"
  export LD_LIBRARY_PATH="${DEXSIM_ROOT}/.nvidia-gl:${LD_LIBRARY_PATH}"
fi

# dexsim is pip-installed editable, but export PYTHONPATH too as a belt-and-braces.
export PYTHONPATH="${DEXSIM_ROOT}/source:${PYTHONPATH}"

echo "[dexsim] env ready  (python: $(python --version 2>&1), root: ${DEXSIM_ROOT})"
