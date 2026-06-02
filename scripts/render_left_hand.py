"""Render the imported left hand from a framed angle (camera aimed at the palm
centroid from the palm-normal side), so the fingers/thumb are clearly visible."""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True; a.enable_cameras = True
app = AppLauncher(a).app
import carb
_s = carb.settings.get_settings()
_s.set("/rtx/rendermode", "PathTracing"); _s.set("/rtx/pathtracing/totalSpp", 110)
_s.set("/rtx/pathtracing/optixDenoiser/enabled", False); _s.set("/app/asyncRendering", False)

import numpy as np, torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.sensors import Camera, CameraCfg
from dexsim import ASSETS_DIR
USD = str(ASSETS_DIR / "shadow_hand_left.usd")

sim = SimulationContext(SimulationCfg(dt=1/120.0, device=a.device))
sim_utils.GroundPlaneCfg().func("/g", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=3000.0).func("/l", sim_utils.DomeLightCfg(intensity=3000.0))
hand = Articulation(ArticulationCfg(prim_path="/World/LeftHand",
    spawn=sim_utils.UsdFileCfg(usd_path=USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0.5)),
    actuators={"f": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=3.0, damping=0.1)}))
cam = Camera(CameraCfg(prim_path="/World/cam", height=720, width=720, data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=28.0, clipping_range=(0.01, 1e5))))
sim.reset()
for _ in range(30):
    hand.set_joint_position_target(hand.data.default_joint_pos); hand.write_data_to_sim(); sim.step()
hand.update(1/120.0)

names = hand.data.body_names
pos = hand.data.body_pos_w[0].cpu().numpy()
tips = np.array([pos[names.index(n)] for n in
                 ("lh_ffdistal","lh_mfdistal","lh_rfdistal","lh_lfdistal","lh_thdistal")])
palm = pos[names.index("lh_palm")]
thumb = pos[names.index("lh_thdistal")]
center = tips.mean(0)
print(f"[left] palm={palm.round(3)} center_tips={center.round(3)} thumb={thumb.round(3)}")
# camera: pull back along the +Z and toward the thumb side so we see the spread
eye = center + np.array([0.0, 0.0, 0.28]) + 0.18*(palm-center)/(np.linalg.norm(palm-center)+1e-6)
eye = center + np.array([0.22, -0.10, 0.18])
cam.set_world_poses_from_view(torch.tensor([eye], dtype=torch.float32, device=a.device),
                              torch.tensor([center], dtype=torch.float32, device=a.device))
for _ in range(110): sim.render()
cam.update(1/120.0, force_recompute=True)
from PIL import Image
rgb = hand and cam.data.output["rgb"][0].cpu().numpy()[..., :3].astype("uint8")
Image.fromarray(rgb).save("logs/left_hand2.png")
print(f"[left] rendered -> logs/left_hand2.png ({100*(rgb.sum(-1)>10).mean():.1f}% non-black)")
app.close()
