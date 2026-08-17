"""Live-streamed Isaac Sim viewport (WebRTC) -- the piano scene in a browser.

Boots the SAME scene as the render server (dexsim.render.studio) but with
Isaac's WebRTC livestream on, so you can watch/orbit the real Isaac viewport
from a browser instead of VNC.

  source env.sh
  python scripts/render/live_scene.py            # WebRTC (default)
  python scripts/render/live_scene.py --hz 30    # slower sim, lighter stream

CLIENT (Isaac Sim 4.5): there is NO in-browser client page any more -- the old
http://host:8211/streaming/webrtc-client server was `omni.services.streamclient.webrtc`,
which this build does not ship (confirmed: no `streamclient` extension in
isaacsim/extscache). Instead install NVIDIA's standalone **Isaac Sim WebRTC
Streaming Client** on your laptop, launch it, and point it at this box's PUBLIC
IP (signaling port 49100).

🔑 THE ONE THAT BIT US: Isaac advertises its media endpoint to the client via
`--/app/livestream/publicEndpointAddress`, and Isaac Lab's AppLauncher fills that
from the **PUBLIC_IP environment variable, defaulting to 127.0.0.1**
(app_launcher.py `_resolve_experience_file`). Launch without PUBLIC_IP and the
server tells your laptop "send media to 127.0.0.1" -- so signaling connects over
TCP and the video is black forever, which looks exactly like a firewall problem
and is not one. This script now sets PUBLIC_IP itself (see --public_ip).

NETWORK: signaling is TCP 49100; the video itself rides **UDP**. An `ssh -L`
tunnel forwards TCP only, so tunnelling alone connects but shows black/frozen
video. You need direct reachability -- open TCP 49100 + the UDP media range to
your IP in the cloud security group, or put both machines on a VPN (Tailscale/
WireGuard). Local iptables here is already wide open (policy ACCEPT).

If that's more hassle than it's worth, the low-friction paths are the viser
placer (posing) and `render.py scene` stills / rerun rrds (visual checks).
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess

from isaaclab.app import AppLauncher


def _detect_public_ip() -> str:
    """Best-effort public address of this box, for publicEndpointAddress.

    Order: an already-set PUBLIC_IP, then whatever the internet says we are
    (this box is NAT'd -- its own interface only knows the 10.x LAN address, so
    asking locally would hand the client an unroutable endpoint), then the LAN
    address as a last resort for same-network clients.
    """
    if os.environ.get("PUBLIC_IP"):
        return os.environ["PUBLIC_IP"]
    try:
        ip = subprocess.run(["curl", "-s", "--max-time", "4", "https://ifconfig.me"],
                            capture_output=True, text=True).stdout.strip()
        if ip.count(".") == 3:
            return ip
    except Exception:
        pass
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--hz", type=float, default=60.0, help="target sim steps/second")
p.add_argument("--style", default="studio", choices=["studio", "simple"])
p.add_argument("--public_ip", default=None,
               help="address the client is told to send media to. Default: auto-detect. "
                    "MUST be reachable from the client -- 127.0.0.1 (the Isaac Lab "
                    "default) gives you a connection with black video.")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
# Livestream mode semantics CHANGED between stacks:
#   Isaac Sim 4.5 / Isaac Lab 2.1:  2 = WebRTC        (1 = native Omniverse client)
#   Isaac Sim 5.x / Isaac Lab 2.3:  1 = WebRTC public (uses $PUBLIC_IP as the
#                                       advertised endpoint), 2 = WebRTC private/LAN
# We want the PUBLIC_IP-advertising mode on both stacks.
import isaaclab
_LAB_MAJOR_MINOR = tuple(int(x) for x in getattr(isaaclab, "__version__", "0.36").split(".")[:2])
args.livestream = 1 if _LAB_MAJOR_MINOR >= (0, 40) else 2

# MUST be set before AppLauncher is constructed: it reads PUBLIC_IP during
# __init__ to build --/app/livestream/publicEndpointAddress.
public_ip = args.public_ip or _detect_public_ip()
os.environ["PUBLIC_IP"] = public_ip

app = AppLauncher(args).app

import time

from isaaclab.sim import SimulationCfg, SimulationContext

from dexsim.tasks.piano.piano_env_cfg import PianoEnvCfg
from dexsim.render import studio

cfg = PianoEnvCfg()
sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device=args.device))
piano, left, right = studio.build_scene(cfg, style=args.style)
sim.reset()

print(f"""
[live_scene] ================= WebRTC LIVESTREAM UP =================
[live_scene]   connect to     : {public_ip}   (signaling TCP 49100)
[live_scene]   media endpoint : {public_ip}   (publicEndpointAddress)
[live_scene]   client         : NVIDIA 'Isaac Sim WebRTC Streaming Client'
[live_scene]                    desktop app -- this build ships NO browser page.
[live_scene]
[live_scene]   If it connects but the video is black, the endpoint above is not
[live_scene]   reachable from your laptop: media rides UDP, so an `ssh -L` tunnel
[live_scene]   is not enough. Open TCP 49100 + the UDP media range to your IP, or
[live_scene]   put both machines on a VPN, then pass --public_ip <that address>.
[live_scene] ========================================================
""", flush=True)
print("[live_scene] holding the ready pose; Ctrl-C to stop", flush=True)

dt = 1.0 / max(1e-3, args.hz)
next_t = time.perf_counter()
while app.is_running():
    for a in (piano, left, right):
        a.set_joint_position_target(a.data.default_joint_pos)
        a.write_data_to_sim()
    sim.step(render=True)
    # PUMP THE KIT EVENT LOOP. This is what actually drives the livestream:
    # ICE negotiation, encoder, and frame submission all run as Kit update
    # callbacks. Stepping physics alone leaves the WebRTC session stuck at
    # "signalling connected, no media" -- a grey client window forever. Also
    # never block the loop with time.sleep(): pace it by skipping updates,
    # because a sleeping main thread is a stalled streamer.
    app.update()
    next_t += dt
    while time.perf_counter() < next_t and app.is_running():
        app.update()                    # idle-pump instead of sleeping

app.close()
