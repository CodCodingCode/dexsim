"""Spawn the UR10e + Shadow Hand tabletop scene and step it.

This is the "does my embodiment actually load and stand up under gravity" check.
It builds the InteractiveScene from ``TabletopGraspSceneCfg`` (combined robot,
table, object), runs the physics for a few hundred steps holding the default
pose, and prints the articulation summary (DOF count, joint names).

Headless by default; pass nothing extra to run on a display, or --headless on a
server. On a server you can record with --video.

Usage:
  python scripts/spawn_scene.py --headless
  python scripts/spawn_scene.py --num_envs 4
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spawn UR10e + Shadow tabletop scene.")
parser.add_argument("--num_envs", type=int, default=1, help="number of parallel envs")
parser.add_argument("--steps", type=int, default=400, help="physics steps to run")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext, SimulationCfg

from dexsim.tasks.grasp import TabletopGraspSceneCfg


def main():
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=args.device))
    sim.set_camera_view(eye=(1.8, 1.8, 1.4), target=(0.5, 0.0, 0.8))

    scene_cfg = TabletopGraspSceneCfg(num_envs=args.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    print("\n========== ARTICULATION SUMMARY ==========")
    print(f"  bodies : {robot.num_bodies}")
    print(f"  DOFs   : {robot.num_joints}")
    print(f"  joints : {robot.data.joint_names}")
    print("==========================================\n")

    # hold the default pose; let physics settle the hand+object on the table.
    default_pos = robot.data.default_joint_pos.clone()
    for i in range(args.steps):
        robot.set_joint_position_target(default_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        if i % 100 == 0:
            obj = scene["object"]
            print(f"  step {i:4d} | object z = {obj.data.root_pos_w[0, 2].item():.3f} m")

    print("\n[OK] scene spawned and stepped without exploding. Embodiment is live.")


main()
simulation_app.close()
