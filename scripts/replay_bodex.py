"""Replay a BODex-Tabletop trajectory on the UR10e + Shadow Hand.

Loads one BODex trajectory file, permutes its joint columns to the Isaac
articulation's DOF order, and plays the configuration sequence back on the
combined embodiment in the tabletop scene. This is the bridge between the
dataset and the sim -- and the seed for Mimic-style imitation data generation.

Usage:
  python scripts/replay_bodex.py --traj data/bodex/<something>.npz --headless
  python scripts/replay_bodex.py --traj data/bodex/<...>.npy --loop 3
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a BODex trajectory in sim.")
parser.add_argument("--traj", required=True, help="path to a BODex .npz/.npy trajectory")
parser.add_argument("--loop", type=int, default=1, help="times to replay")
parser.add_argument("--rate", type=int, default=4, help="sim substeps per traj frame")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext, SimulationCfg

from dexsim.tasks.grasp import TabletopGraspSceneCfg, load_bodex_trajectory


def main():
    traj = load_bodex_trajectory(args.traj)
    print(f"[replay_bodex] {traj.source.name}: {traj.num_frames} frames, "
          f"{traj.num_dofs} dofs, object={traj.object_name}")

    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=args.device))
    sim.set_camera_view(eye=(1.8, 1.8, 1.4), target=(0.5, 0.0, 0.8))

    scene = InteractiveScene(TabletopGraspSceneCfg(num_envs=1, env_spacing=2.5))
    sim.reset()
    robot = scene["robot"]
    joint_names = robot.data.joint_names

    # Permute BODex columns onto the articulation's DOF order when names exist;
    # otherwise trust that the file already uses the articulation order.
    if traj.joint_names is not None:
        qpos = traj.reorder_to(joint_names)
    else:
        if traj.num_dofs != robot.num_joints:
            raise ValueError(
                f"Trajectory has {traj.num_dofs} dofs but the robot has "
                f"{robot.num_joints} and the file carries no joint_names to "
                f"align them. Inspect the file or pass an aligned trajectory."
            )
        print("[replay_bodex] no joint_names in file -> assuming native DOF order")
        qpos = traj.qpos
    qpos_t = torch.as_tensor(qpos, device=args.device, dtype=torch.float32)

    for lap in range(args.loop):
        for f in range(traj.num_frames):
            target = qpos_t[f].unsqueeze(0)
            for _ in range(args.rate):
                robot.set_joint_position_target(target)
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim.get_physics_dt())
        print(f"[replay_bodex] lap {lap + 1}/{args.loop} done")

    print("[OK] replay complete.")


main()
simulation_app.close()
