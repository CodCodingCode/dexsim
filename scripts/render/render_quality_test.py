"""Single-frame render-quality test for the 60-DoF bimanual rig.

Goal: a clean studio look (no grey grid, soft lighting, denoised-clean via high
spp) instead of the grainy default. Renders ONE frame of the rig at its ready
pose so the look can be tuned fast, then ported into render_rollout.py.

  python scripts/render/render_quality_test.py --headless --spp 160 --out /tmp/q.png
"""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--spp", type=int, default=160)
p.add_argument("--bounces", type=int, default=6)
p.add_argument("--width", type=int, default=1280)
p.add_argument("--height", type=int, default=720)
p.add_argument("--eye", default="1.5,-1.35,1.35")
p.add_argument("--target", default="0.5,-0.5,0.80")
p.add_argument("--rollout", default="/tmp/easy_fixed.npz", help="pose arms from this rollout's frame")
p.add_argument("--frame", type=int, default=120)
p.add_argument("--out", default="/tmp/q.png")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import carb
_s = carb.settings.get_settings()
_s.set("/rtx/rendermode", "PathTracing")
_s.set("/rtx/pathtracing/spp", 8)
_s.set("/rtx/pathtracing/totalSpp", args.spp)
_s.set("/rtx/pathtracing/optixDenoiser/enabled", False)     # nvoptix.bin absent on this box
_s.set("/rtx/pathtracing/maxBounces", args.bounces)
_s.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", args.bounces)
# filmic-ish tonemap + a touch of exposure for a polished look
_s.set("/rtx/post/tonemap/op", 1)                            # 1 = filmic/Reinhard-ish
_s.set("/rtx/post/histogram/enabled", True)
_s.set("/rtx/post/tonemap/cm2Factor", 1.0)
_s.set("/app/asyncRendering", False)

import numpy as np, torch
from PIL import Image
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sensors import Camera, CameraCfg
from dexsim.tasks.piano.piano_env_cfg import PianoEnvCfg


def quat_from_two_pts(eye, tgt):
    """quaternion (w,x,y,z) that points a light's -Z axis from eye toward tgt."""
    d = np.array(tgt) - np.array(eye); d = d / (np.linalg.norm(d) + 1e-9)
    z = -d                                  # light emits along -Z; aim -Z at the target
    up = np.array([0, 0, 1.0])
    x = np.cross(up, z); x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    w = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w < 1e-6:
        return (1.0, 0.0, 0.0, 0.0)
    x_ = (R[2, 1] - R[1, 2]) / (4 * w); y_ = (R[0, 2] - R[2, 0]) / (4 * w); z_ = (R[1, 0] - R[0, 1]) / (4 * w)
    return (float(w), float(x_), float(y_), float(z_))


def main():
    cfg = PianoEnvCfg()
    sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device=args.device))

    # --- clean studio floor (replaces the grey grid GroundPlane) ---
    floor_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.10, 0.12), roughness=0.55, metallic=0.0)
    sim_utils.CuboidCfg(size=(40.0, 40.0, 0.04), visual_material=floor_mat).func(
        "/World/Floor", sim_utils.CuboidCfg(size=(40.0, 40.0, 0.04), visual_material=floor_mat),
        translation=(0.0, 0.0, -0.02))

    # --- studio dome (soft even fill; brighter so the grey arms read) ---
    sim_utils.DomeLightCfg(intensity=700.0, color=(0.5, 0.55, 0.65)).func(
        "/World/Dome", sim_utils.DomeLightCfg(intensity=700.0, color=(0.5, 0.55, 0.65)))
    scene_c = (0.5, -0.5, 0.85)
    # Lights placed HIGH and BEHIND the camera (eye y=-1.35) so they illuminate the
    # rig but stay out of frame. Key: front-right-high; fill: front-left; rim: behind subject.
    key_pos = (1.8, -3.4, 4.2)
    sim_utils.DiskLightCfg(intensity=120000.0, radius=0.9, color=(1.0, 0.97, 0.92)).func(
        "/World/Key", sim_utils.DiskLightCfg(intensity=120000.0, radius=0.9, color=(1.0, 0.97, 0.92)),
        translation=key_pos, orientation=quat_from_two_pts(key_pos, scene_c))
    fill_pos = (-1.6, -3.2, 3.4)
    sim_utils.DiskLightCfg(intensity=40000.0, radius=1.3, color=(0.85, 0.9, 1.0)).func(
        "/World/Fill", sim_utils.DiskLightCfg(intensity=40000.0, radius=1.3, color=(0.85, 0.9, 1.0)),
        translation=fill_pos, orientation=quat_from_two_pts(fill_pos, scene_c))
    rim_pos = (0.3, 2.6, 3.6)
    sim_utils.DiskLightCfg(intensity=60000.0, radius=0.7, color=(0.8, 0.85, 1.0)).func(
        "/World/Rim", sim_utils.DiskLightCfg(intensity=60000.0, radius=0.7, color=(0.8, 0.85, 1.0)),
        translation=rim_pos, orientation=quat_from_two_pts(rim_pos, scene_c))

    piano = Articulation(cfg.piano_cfg.replace(prim_path="/World/Piano"))
    left = Articulation(cfg.left_robot_cfg.replace(prim_path="/World/LeftRobot"))
    right = Articulation(cfg.right_robot_cfg.replace(prim_path="/World/RightRobot"))
    cam = Camera(CameraCfg(prim_path="/World/cam", update_period=0, height=args.height, width=args.width,
                           data_types=["rgb"],
                           spawn=sim_utils.PinholeCameraCfg(focal_length=20.0, horizontal_aperture=20.955,
                                                            clipping_range=(0.05, 1e6))))
    sim.reset()
    eye = torch.tensor([[float(v) for v in args.eye.split(",")]], device=args.device)
    tgt = torch.tensor([[float(v) for v in args.target.split(",")]], device=args.device)
    cam.set_world_poses_from_view(eye, tgt)

    # pose the rig from a real rollout frame (arms reaching over the keyboard)
    import os
    if args.rollout and os.path.exists(args.rollout):
        d = np.load(args.rollout, allow_pickle=True)
        fr = min(args.frame, d["left"].shape[0] - 1)
        lj = torch.tensor(d["left"][fr:fr + 1], dtype=torch.float32, device=args.device)
        rj = torch.tensor(d["right"][fr:fr + 1], dtype=torch.float32, device=args.device)
        kj = torch.tensor(d["keys"][fr:fr + 1], dtype=torch.float32, device=args.device)
        left.write_joint_state_to_sim(lj, torch.zeros_like(lj))
        right.write_joint_state_to_sim(rj, torch.zeros_like(rj))
        piano.write_joint_state_to_sim(kj, torch.zeros_like(kj))
        print(f"[quality_test] posed arms from {args.rollout} frame {fr}", flush=True)
    else:
        for a in (left, right):
            a.write_joint_state_to_sim(a.data.default_joint_pos, torch.zeros_like(a.data.default_joint_pos))
    for a in (left, right, piano):
        a.write_data_to_sim()
    for _ in range(90):
        sim.step()
    for _ in range(args.spp):
        sim.render()
    cam.update(sim.get_physics_dt(), force_recompute=True)
    rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype("uint8")
    Image.fromarray(rgb).save(args.out)
    nb = int((rgb.sum(-1) > 12).mean() * 100)
    print(f"[quality_test] saved {args.out}  non-black {nb}%  spp={args.spp}", flush=True)


main()
app.close()
