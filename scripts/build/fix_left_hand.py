"""Clean up the MJCF-imported left hand: keep ONE articulation root, drop the
empty MuJoCo worldBody, then spawn + render to confirm it's a proper LEFT hand."""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True; a.enable_cameras = True
app = AppLauncher(a).app

import carb
_s = carb.settings.get_settings()
_s.set("/rtx/rendermode", "PathTracing"); _s.set("/rtx/pathtracing/totalSpp", 96)
_s.set("/rtx/pathtracing/optixDenoiser/enabled", False); _s.set("/app/asyncRendering", False)

from pxr import Usd, UsdPhysics, PhysxSchema
from dexsim import ASSETS_DIR
USD = str(ASSETS_DIR / "shadow_hand_left.usd")

# --- inspect + consolidate articulation root ---
stage = Usd.Stage.Open(USD)
print("=== prim tree (roots/bodies) ===")
roots, bodies = [], []
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI): roots.append(prim.GetPath().pathString)
    if prim.HasAPI(UsdPhysics.RigidBodyAPI): bodies.append(prim.GetPath().pathString)
print("  articulation roots:", roots)
print("  first bodies:", bodies[:4], "...", len(bodies), "total")

default = stage.GetDefaultPrim()
keep = default.GetPath().pathString
# strip every articulation root except the default prim; deactivate worldBody
stripped = []
for prim in stage.Traverse():
    pth = prim.GetPath().pathString
    if prim.GetName() == "worldBody":
        prim.SetActive(False); stripped.append(pth + " (worldBody off)")
        continue
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI) and pth != keep:
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        stripped.append(pth)
# ensure the default prim is the articulation root
if not default.HasAPI(UsdPhysics.ArticulationRootAPI):
    UsdPhysics.ArticulationRootAPI.Apply(default)
print("  stripped/deactivated:", stripped)
stage.GetRootLayer().Save()
print(f"[fix] saved {USD}")

# --- spawn + render ---
import numpy as np, torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.sensors import Camera, CameraCfg

sim = SimulationContext(SimulationCfg(dt=1/120.0, device=a.device))
sim_utils.GroundPlaneCfg().func("/g", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2500.0).func("/l", sim_utils.DomeLightCfg(intensity=2500.0))
cfg = ArticulationCfg(prim_path="/World/LeftHand",
    spawn=sim_utils.UsdFileCfg(usd_path=USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0.5)),
    actuators={"f": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=3.0, damping=0.1)})
hand = Articulation(cfg)
cam = Camera(CameraCfg(prim_path="/World/cam", height=600, width=800, data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=22.0, clipping_range=(0.01, 1e5))))
sim.reset()
print(f"[fix] spawned OK: {hand.num_joints} joints, {hand.num_bodies} bodies")
print("  fingertips:", [b for b in hand.data.body_names if "distal" in b])
for _ in range(30):
    hand.set_joint_position_target(hand.data.default_joint_pos); hand.write_data_to_sim(); sim.step()
cam.set_world_poses_from_view(torch.tensor([[0.35, 0.0, 0.62]], device=a.device),
                              torch.tensor([[0.0, 0.0, 0.5]], device=a.device))
for _ in range(96): sim.render()
cam.update(1/120.0, force_recompute=True)
from PIL import Image
rgb = cam.data.output["rgb"][0].cpu().numpy()[..., :3].astype("uint8")
Image.fromarray(rgb).save("logs/left_hand.png")
print(f"[fix] rendered left hand -> logs/left_hand.png ({100*(rgb.sum(-1)>10).mean():.1f}% non-black)")
app.close()
