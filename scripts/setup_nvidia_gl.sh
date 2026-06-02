#!/usr/bin/env bash
# Stage the NVIDIA userspace graphics/Vulkan driver locally (no sudo, no kernel
# changes). Needed because this is a compute-only container: CUDA is present but
# the graphics driver libs (libGLX_nvidia, Vulkan ICD) are not, so Isaac Sim's
# RTX renderer + GPU-physics foundation can't init Vulkan.
#
# This downloads the runfile MATCHING the loaded kernel module, extracts it
# (no install), copies the userspace .so's into dexsim/.nvidia-gl, and writes a
# Vulkan ICD json. env.sh then exports VK_ICD_FILENAMES + LD_LIBRARY_PATH.
#
# Re-run if .nvidia-gl is deleted or the host driver version changes.
set -euo pipefail

DEXSIM_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
GLDIR="${DEXSIM_ROOT}/.nvidia-gl"

# detect the loaded kernel-module version so userspace EXACTLY matches it
VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
[ -n "$VER" ] || { echo "could not read driver version from nvidia-smi"; exit 1; }
echo "[setup_nvidia_gl] matching kernel driver $VER"

RUN="/tmp/NVIDIA-Linux-x86_64-${VER}.run"
URL="https://us.download.nvidia.com/XFree86/Linux-x86_64/${VER}/NVIDIA-Linux-x86_64-${VER}.run"
[ -f "$RUN" ] || { echo "[setup_nvidia_gl] downloading $URL"; curl -sL -o "$RUN" "$URL"; }

rm -rf "/tmp/nv_${VER}"
sh "$RUN" --extract-only --target "/tmp/nv_${VER}" >/dev/null
echo "[setup_nvidia_gl] extracted"

mkdir -p "$GLDIR"
cp -f "/tmp/nv_${VER}"/*.so."${VER}" "$GLDIR"/
cp -f "/tmp/nv_${VER}"/nvidia_icd.json "$GLDIR"/
( cd "$GLDIR"
  for f in *.so."${VER}"; do
    base="${f%.so.${VER}}"
    ln -sf "$f" "${base}.so"
    ln -sf "$f" "${base}.so.0"
    ln -sf "$f" "${base}.so.1"
  done )
python3 - "$GLDIR/nvidia_icd.json" <<'PY'
import json,sys
p=sys.argv[1]; d=json.load(open(p))
d["ICD"]["library_path"]="libGLX_nvidia.so.0"
json.dump(d,open(p,"w"),indent=2)
PY

echo "[setup_nvidia_gl] staged into $GLDIR"
VK_ICD_FILENAMES="$GLDIR/nvidia_icd.json" LD_LIBRARY_PATH="$GLDIR:${LD_LIBRARY_PATH:-}" \
  vulkaninfo --summary 2>/dev/null | grep -iE "deviceName|driverName" | head -4 || true
echo "[setup_nvidia_gl] done. 'source env.sh' to use."
