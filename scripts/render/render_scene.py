"""Photoreal render of the bimanual piano scene via an explicit Camera SENSOR.

env.render() relies on a viewport that doesn't exist headless -> black frames.
Instead we spawn the scene in a plain SimulationContext, attach an
isaaclab Camera sensor (its own render product), accumulate path-traced frames
(OptiX denoiser on tensor cores; works on the RT-core-less H100), and read the
sensor's RGB output directly.

  python scripts/render_scene.py --headless --out logs/scene.png
"""

from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="logs/scene.png")
parser.add_argument("--settle", type=int, default=90)
parser.add_argument("--spp", type=int, default=160, help="path-traced frames to accumulate")
parser.add_argument("--eye", default="2.2,-1.5,1.8")
parser.add_argument("--target", default="0.45,0.0,0.78")
parser.add_argument("--pathtrace", action="store_true", default=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import carb
_s = carb.settings.get_settings()
_s.set("/rtx/rendermode", "PathTracing")
_s.set("/rtx/pathtracing/spp", 8)
_s.set("/rtx/pathtracing/totalSpp", args.spp)
# OptiX denoiser weights (/usr/share/nvidia/nvoptix.bin) are absent in this
# compute container -> denoiser errors -> black. Disable it and rely on raw
# sample accumulation (high totalSpp) for a clean-enough image.
_s.set("/rtx/pathtracing/optixDenoiser/enabled", False)
_s.set("/rtx/pathtracing/maxBounces", 4)
_s.set("/app/asyncRendering", False)

import os
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sensors import Camera, CameraCfg

from dexsim.assets import UR10E_SHADOW_CFG, PIANO_CFG
from dexsim.tasks.piano.piano_env_cfg import PianoEnvCfg


def main():
    cfg = PianoEnvCfg()
    sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device=args.device))

    # ground + bright light
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=3000.0))

    # piano + two arms using the ENV's configured cfgs (base pose + piano-ready
    # joint pose are already baked in by PianoEnvCfg.__post_init__), just
    # re-prim'd to single-env paths so the render matches what training sees.
    piano = Articulation(cfg.piano_cfg.replace(prim_path="/World/Piano"))
    left = Articulation(cfg.left_robot_cfg.replace(prim_path="/World/LeftRobot"))
    right = Articulation(cfg.right_robot_cfg.replace(prim_path="/World/RightRobot"))

    # camera sensor
    cam = Camera(CameraCfg(
        prim_path="/World/cam",
        update_period=0, height=720, width=1280, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955,
                                         clipping_range=(0.05, 1e6)),
    ))

    sim.reset()
    eye = [[float(v) for v in args.eye.split(",")]]
    tgt = [[float(v) for v in args.target.split(",")]]
    cam.set_world_poses_from_view(torch.tensor(eye, device=args.device),
                                  torch.tensor(tgt, device=args.device))

    for _ in range(args.settle):
        piano.set_joint_position_target(piano.data.default_joint_pos)
        left.set_joint_position_target(left.data.default_joint_pos)
        right.set_joint_position_target(right.data.default_joint_pos)
        for a in (piano, left, right):
            a.write_data_to_sim()
        sim.step()

    print(f"[render_scene] accumulating {args.spp} path-traced frames...", flush=True)
    for i in range(args.spp):
        sim.render()
        if i % 40 == 0:
            print(f"  ... {i}/{args.spp}", flush=True)
    cam.update(sim.get_physics_dt(), force_recompute=True)

    rgb = cam.data.output["rgb"][0].cpu().numpy()
    nonblack = int((rgb[..., :3].sum(-1) > 10).sum())
    print(f"[render_scene] rgb {rgb.shape}, non-black px = {nonblack} "
          f"({100*nonblack/(rgb.shape[0]*rgb.shape[1]):.1f}%)")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    from PIL import Image
    Image.fromarray(rgb[..., :3].astype("uint8")).save(args.out)
    print(f"[render_scene] saved -> {args.out}")


main()
simulation_app.close()
