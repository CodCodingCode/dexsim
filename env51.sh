# dexsim environment -- Isaac Sim 5.1 / Isaac Lab 2.3.2 (side-by-side migration).
# The original 4.5 stack stays in env.sh + .venv + IsaacLab/; this activates the
# new stack in .venv-isaac51 + IsaacLab51/. Source ONE of them per shell, never both.
DEXSIM_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]:-$0}" )" && pwd )"
source "${DEXSIM_ROOT}/.venv-isaac51/bin/activate"
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
# staged Vulkan driver (same driver files work for kit 107)
if [ -d "${DEXSIM_ROOT}/.nvidia-gl" ]; then
  export VK_ICD_FILENAMES="${DEXSIM_ROOT}/.nvidia-gl/nvidia_icd.json"
  export LD_LIBRARY_PATH="${DEXSIM_ROOT}/.nvidia-gl:${LD_LIBRARY_PATH}"
fi
export PYTHONPATH="${DEXSIM_ROOT}/source:${PYTHONPATH}"
echo "[dexsim-51] env ready  (python: $(python --version 2>&1), isaac 5.1 stack)"
