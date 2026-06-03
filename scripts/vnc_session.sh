#!/usr/bin/env bash
# Start (or reuse) a localhost-only TigerVNC desktop on :1 so you can drive the
# interactive Isaac Sim GUI from a remote machine (e.g. a Mac) over an SSH tunnel.
# The H100 has no NVENC, so WebRTC livestream is impossible; VNC encodes frames on
# the CPU instead. Vulkan still renders on the GPU and presents to the X display.
#
# Usage:
#   scripts/vnc_session.sh                       # start/ensure the VNC desktop
#   scripts/vnc_session.sh scripts/train/play_piano.py --num_envs 1 --checkpoint <pt>
#                                                # ...then launch an Isaac GUI app on it
#
# Connect from your Mac:
#   ssh -L 5901:localhost:5901 <user>@<host>     # tunnel (keep this open)
#   open vnc://localhost:5901                     # built-in Screen Sharing
#   # password: see the one you set with vncpasswd (~/.vnc/passwd)
#
# NOTE: launch Isaac WITHOUT --headless so Kit opens a window on the display.
# If the 3D viewport is black (H100 real-time-RTX quirk), switch the renderer to
# "RTX - Interactive (Path Tracing)" in the viewport's render-mode dropdown.
# NB: no `-u` (nounset) — env.sh references ${PYTHONPATH}/${LD_LIBRARY_PATH}
# which may be unset in a fresh shell, and that's fine.
set -eo pipefail
DISP=":1"; GEOM="${VNC_GEOM:-1920x1080}"
PORT=$((5900 + ${DISP#:}))
cd "$(dirname "$0")/.."

# Reuse an existing desktop if the RFB port is already listening; only start one
# otherwise. (Checking the port is reliable; `vncserver -list` parsing is not.)
if ss -ltn 2>/dev/null | grep -q "127.0.0.1:${PORT}\b"; then
  echo "Reusing existing VNC desktop on ${DISP} (port ${PORT})."
else
  vncserver "${DISP}" -localhost yes -geometry "${GEOM}" -depth 24
fi
echo "VNC desktop ready on ${DISP} (port ${PORT}, localhost-only)."
echo "Mac:  ssh -L 5901:localhost:5901 $(whoami)@<this-host>   then   open vnc://localhost:5901"

if [ "$#" -gt 0 ]; then
  # shellcheck disable=SC1091
  source env.sh
  echo "Launching on ${DISP}: $*"
  exec env DISPLAY="${DISP}" python "$@"
fi
