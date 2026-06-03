"""No-policy stability test: spawn the two arms at their piano-ready pose, set the
actuators to HOLD that pose, run real physics (no policy, no kinematic override),
and film it. If the arm is steady here, any twitchiness is the POLICY's residual
jitter, not the embodiment/physics.

  python scripts/hold_test_video.py --headless --out results/hold_test.mp4
"""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--out", default="results/hold_test.mp4")
p.add_argument("--frames_dir", default="logs/hold_frames")
p.add_argument("--seconds", type=float, default=8.0)
p.add_argument("--spp", type=int, default=16)
p.add_argument("--fps", type=int, default=20)
p.add_argument("--eye", default="1.15,-1.05,1.15")
p.add_argument("--target", default="0.45,-0.55,0.76")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import carb
_s = carb.settings.get_settings()
_s.set("/rtx/rendermode", "PathTracing")
_s.set("/rtx/pathtracing/spp", 4)
_s.set("/rtx/pathtracing/totalSpp", args.spp)
_s.set("/rtx/pathtracing/optixDenoiser/enabled", False)
_s.set("/rtx/pathtracing/maxBounces", 4)
_s.set("/app/asyncRendering", False)

import os, subprocess
import numpy as np, torch
from PIL import Image
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sensors import Camera, CameraCfg
from dexsim.tasks.piano.piano_env_cfg import PianoEnvCfg

cfg = PianoEnvCfg()
sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device=args.device))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=3000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=3000.0))
piano = Articulation(cfg.piano_cfg.replace(prim_path="/World/Piano"))
left = Articulation(cfg.left_robot_cfg.replace(prim_path="/World/LeftRobot"))
right = Articulation(cfg.right_robot_cfg.replace(prim_path="/World/RightRobot"))
cam = Camera(CameraCfg(prim_path="/World/cam", update_period=0, height=720, width=1280,
                       data_types=["rgb"],
                       spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 1e6))))
sim.reset()
cam.set_world_poses_from_view(
    torch.tensor([[float(v) for v in args.eye.split(",")]], device=args.device),
    torch.tensor([[float(v) for v in args.target.split(",")]], device=args.device))

# hold targets = the spawned ready pose; NEVER overridden -> pure physics
hold = {a: a.data.default_joint_pos.clone() for a in (piano, left, right)}
q0 = left.data.default_joint_pos.clone()

decim = 6                                   # 120 Hz sim -> 20 Hz frames
n_frames = int(args.seconds * args.fps)
os.makedirs(args.frames_dir, exist_ok=True)
drift_max = 0.0
for f in range(n_frames):
    for _ in range(decim):
        for a in (piano, left, right):
            a.set_joint_position_target(hold[a])
            a.write_data_to_sim()
        sim.step()
    left.update(1 / 120.0)
    drift = float((left.data.joint_pos[0] - q0[0]).abs().max())
    drift_max = max(drift_max, drift)
    for _ in range(args.spp):
        sim.render()
    cam.update(sim.get_physics_dt(), force_recompute=True)
    rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype("uint8")
    Image.fromarray(rgb).save(os.path.join(args.frames_dir, f"frame_{f:05d}.png"))
    if f % 20 == 0:
        print(f"  frame {f}/{n_frames}  left-arm max drift from ready pose = "
              f"{drift:.4f} rad", flush=True)

print(f"[hold_test] MAX joint drift over {args.seconds}s with NO policy = "
      f"{drift_max:.4f} rad ({np.degrees(drift_max):.2f} deg)", flush=True)
cmd = ["ffmpeg", "-y", "-framerate", str(args.fps), "-i",
       os.path.join(args.frames_dir, "frame_%05d.png"),
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", args.out]
subprocess.run(cmd, check=True)
print(f"[hold_test] saved -> {args.out}")
app.close()
