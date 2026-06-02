"""Render the combined LEFT arm (gravity off, holding the ready pose) so we can
see how the MuJoCo hand sits on the flange and pick the mount rotation."""
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
from dexsim.assets import COMBINED_LEFT_USD_PATH

sim = SimulationContext(SimulationCfg(dt=1/120.0, device=a.device))
sim_utils.GroundPlaneCfg().func("/g", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2800.0).func("/l", sim_utils.DomeLightCfg(intensity=2800.0))
robot = Articulation(ArticulationCfg(prim_path="/World/R",
    spawn=sim_utils.UsdFileCfg(usd_path=COMBINED_LEFT_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),  # freeze for the photo
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,0.75),
        joint_pos={"shoulder_lift_joint":-0.9,"elbow_joint":1.6,"wrist_1_joint":-1.2,"wrist_2_joint":-1.57,"robot0_.*":0.0}),
    actuators={"arm": ImplicitActuatorCfg(joint_names_expr=["shoulder.*","elbow.*","wrist.*"], stiffness=800.0, damping=40.0),
               "hand": ImplicitActuatorCfg(joint_names_expr=["robot0_.*"], stiffness=3.0, damping=0.1)}))
cam = Camera(CameraCfg(prim_path="/World/cam", height=720, width=900, data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01,1e5))))
sim.reset()
for _ in range(30):
    robot.set_joint_position_target(robot.data.default_joint_pos); robot.write_data_to_sim(); sim.step()
robot.update(1/120.0)
names = robot.data.body_names; pos = robot.data.body_pos_w[0].cpu().numpy()
wrist = pos[names.index("wrist_3_link")]
tips = np.array([pos[names.index(n)] for n in
                 ("robot0_ffdistal","robot0_mfdistal","robot0_rfdistal","robot0_lfdistal","robot0_thdistal")])
print(f"[left_arm] wrist={wrist.round(3)} fingertip_mean={tips.mean(0).round(3)}")
center = (wrist + tips.mean(0)) / 2
eye = center + np.array([0.5, -0.5, 0.25])
cam.set_world_poses_from_view(torch.tensor([eye],dtype=torch.float32,device=a.device),
                              torch.tensor([center],dtype=torch.float32,device=a.device))
for _ in range(110): sim.render()
cam.update(1/120.0, force_recompute=True)
from PIL import Image
rgb = cam.data.output["rgb"][0].cpu().numpy()[...,:3].astype("uint8")
Image.fromarray(rgb).save("logs/left_arm.png")
print(f"[left_arm] rendered -> logs/left_arm.png ({100*(rgb.sum(-1)>10).mean():.1f}% non-black)")
app.close()
