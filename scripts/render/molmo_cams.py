"""MolmoAct-style camera rig: two egocentric WRIST cams + one CENTER 'head' cam
mounted between the two arms, all looking at the hands/keys. These three image
streams are the vision input a VLA (MolmoAct-style) policy would consume.

Renders logs/molmo/{left_wrist,right_wrist,head}.png. Uses ONE camera sensor
repositioned per shot (path tracing on this RT-core-less H100 only accumulates a
single render product reliably).

  python scripts/molmo_cams.py --headless
"""
from __future__ import annotations
import argparse, os
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--spp", type=int, default=110)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True; a.enable_cameras = True
app = AppLauncher(a).app
import carb
_s = carb.settings.get_settings()
_s.set("/rtx/rendermode", "PathTracing"); _s.set("/rtx/pathtracing/totalSpp", a.spp)
_s.set("/rtx/pathtracing/optixDenoiser/enabled", False); _s.set("/app/asyncRendering", False)

import numpy as np, torch
from PIL import Image
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.sensors import Camera, CameraCfg
from dexsim.assets import UR10E_SHADOW_CFG, PIANO_CFG
from dexsim.tasks.piano.piano_env_cfg import PianoEnvCfg

cfg = PianoEnvCfg(); dev = a.device
sim = SimulationContext(SimulationCfg(dt=1/120.0, device=dev))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=1800.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=1800.0))
piano = Articulation(PIANO_CFG.replace(prim_path="/World/Piano").replace(
    init_state=PIANO_CFG.init_state.replace(pos=cfg.piano_pos)))
left = Articulation(UR10E_SHADOW_CFG.replace(prim_path="/World/LeftRobot").replace(
    init_state=UR10E_SHADOW_CFG.init_state.replace(pos=cfg.left_base_pos, joint_pos=dict(cfg.left_ready_pose))))
right = Articulation(UR10E_SHADOW_CFG.replace(prim_path="/World/RightRobot").replace(
    init_state=UR10E_SHADOW_CFG.init_state.replace(pos=cfg.right_base_pos, joint_pos=dict(cfg.right_ready_pose))))
cam = Camera(CameraCfg(prim_path="/World/cam", height=600, width=800, data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=16.0, clipping_range=(0.01, 1e5))))
sim.reset()

def bidx(art, n): return art.data.body_names.index(n)
wl, wr = bidx(left, "wrist_3_link"), bidx(right, "wrist_3_link")
fl = [bidx(left, f"robot0_{f}distal") for f in ("ff","mf","rf")]
fr = [bidx(right, f"robot0_{f}distal") for f in ("ff","mf","rf")]

for _ in range(120):
    for art in (piano, left, right):
        art.set_joint_position_target(art.data.default_joint_pos); art.write_data_to_sim()
    sim.step()
for art in (left, right): art.update(1/120.0)

def wrist_view(art, wi, fis):
    w = art.data.body_pos_w[0, wi].cpu().numpy()
    f = art.data.body_pos_w[0, fis].mean(0).cpu().numpy()
    eye = w + np.array([0.0, 0.0, 0.12]) + 0.06*(w-f)/(np.linalg.norm(w-f)+1e-6)  # just over the knuckles
    return eye, f                                                                 # look AT the fingertips

le, lt = wrist_view(left, wl, fl)
re, rt = wrist_view(right, wr, fr)
# CENTER 'head' cam: midway between the two bases, up high, looking down the arms
mid = (np.array(cfg.left_base_pos) + np.array(cfg.right_base_pos)) / 2
head_eye = mid + np.array([0.10, 0.0, 0.35])
head_tgt = np.array([cfg.piano_pos[0], 0.0, 0.74])

os.makedirs("logs/molmo", exist_ok=True)
shots = [("left_wrist", le, lt), ("right_wrist", re, rt), ("head", head_eye, head_tgt)]
t = lambda v: torch.tensor([v], dtype=torch.float32, device=dev)
for tag, eye, tgt in shots:
    cam.set_world_poses_from_view(t(eye.astype(np.float32)), t(np.asarray(tgt, dtype=np.float32)))
    for _ in range(a.spp): sim.render()
    cam.update(1/120.0, force_recompute=True)
    rgb = cam.data.output["rgb"][0].cpu().numpy()[..., :3].astype("uint8")
    Image.fromarray(rgb).save(f"logs/molmo/{tag}.png")
    print(f"[molmo] {tag:11s} eye={eye.round(2)} -> {np.asarray(tgt).round(2)}  "
          f"({100*(rgb.sum(-1)>10).mean():.0f}% non-black) -> logs/molmo/{tag}.png", flush=True)
app.close()
