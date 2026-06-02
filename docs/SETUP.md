# Setup notes — how this environment was built (and the gotchas)

Target box: single **H100 80GB**, Ubuntu 22.04, Python 3.10, **compute-only
container** (CUDA present, graphics driver NOT — this matters, see below).

## 1. Isaac Sim + Isaac Lab

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install "isaacsim[all,extscache]==4.5.0.0" --extra-index-url https://pypi.nvidia.com
git clone --depth 1 --branch v2.1.0 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && ./isaaclab.sh -i
```

### Install gotchas that bit us (all fixed)
- **`isaaclab` core failed to build** — `flatdict` needs `pkg_resources`, removed
  in setuptools ≥ 81. Fix: `pip install "setuptools==79.0.1"`, then install the
  core with build isolation OFF so it uses that setuptools:
  `pip install -e IsaacLab/source/isaaclab --no-build-isolation`
  (also needs `pip install toml`). The other `isaaclab_*` sub-packages installed
  fine via `isaaclab.sh -i`.
- **Missing system libs** for Isaac Sim:
  `sudo apt-get install -y libsm6 libice6 libglu1-mesa libegl1 libxt6 libxrender1 libxext6`
  (without `libSM`/`libICE`, `libhdx.so` fails to load and omni.physx import dies).

## 2. The big one — Vulkan on a compute-only container

Symptom: Isaac Sim boots but logs
`carb.graphics-vulkan.plugin VkResult: ERROR_INCOMPATIBLE_DRIVER` and
`PhysXFoundation: Unable to get IGpuFoundation` — then hangs. GPU physics never
starts.

Root cause (verified): **not** a driver version mismatch — the kernel module and
`nvidia-smi` both report `580.105.08`. The container simply ships the *compute*
driver (`libnvidia-ml`, CUDA) but **none of the graphics/Vulkan userspace libs**
(`libGLX_nvidia`, `libEGL_nvidia`, `libnvidia-glcore`, the Vulkan ICD). That's a
container started with `NVIDIA_DRIVER_CAPABILITIES=compute,utility` instead of
`...,graphics`.

### Fix (no sudo, no kernel changes, fully reversible)
Stage the **userspace-only** graphics libs from the *exact matching* driver
version next to the venv, and point the Vulkan loader at them:

```bash
sudo apt-get install -y libvulkan1 vulkan-tools   # the loader (+ vulkaninfo)
bash scripts/setup_nvidia_gl.sh                   # downloads matching .run,
                                                  # extracts userspace .so's into
                                                  # .nvidia-gl/, writes the ICD
```

`env.sh` then exports:
```bash
export VK_ICD_FILENAMES=$PWD/.nvidia-gl/nvidia_icd.json
export LD_LIBRARY_PATH=$PWD/.nvidia-gl:$LD_LIBRARY_PATH
```

Verify:
```bash
vulkaninfo --summary | grep -E "deviceName|driverName"
#   deviceName = NVIDIA H100 80GB HBM3
#   driverName = NVIDIA
```

After this, Isaac Sim reports `Graphics API: Vulkan / Driver 580.105.08`, the GPU
foundation initializes, and `scripts/smoke_test.py` passes (24-DOF Shadow hand
simulates on GPU). **First run is slow** (cold RTX shader compile, a few minutes);
subsequent runs are fast once the shader cache is warm.

> The proper alternative, if you control container launch, is to relaunch with
> `NVIDIA_DRIVER_CAPABILITIES=all` (or include `graphics,display`) and the
> `nvidia-container-toolkit` injects these libs for you — then `.nvidia-gl` is
> unnecessary.

## 3. Everyday use

```bash
source env.sh                       # venv + EULA + Vulkan env + PYTHONPATH
python scripts/smoke_test.py --headless --device cuda:0
```
