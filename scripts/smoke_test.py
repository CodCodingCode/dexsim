"""Minimal end-to-end smoke test: boot Isaac Sim headless, spawn the Shadow Hand
from dexsim's config, step physics, print the articulation summary.

No combined USD or dataset required -- this just proves Isaac Sim + Isaac Lab +
dexsim's asset configs all load and simulate.

  python scripts/smoke_test.py --headless
"""

from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

from dexsim.assets import SHADOW_HAND_CFG


@configclass
class _SmokeScene(InteractiveSceneCfg):
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    robot = SHADOW_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _log(msg):
    print(f"[smoke] {msg}", flush=True)


def main():
    _log("creating SimulationContext...")
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=args.device))
    _log("building scene...")
    scene = InteractiveScene(_SmokeScene(num_envs=2, env_spacing=1.0))
    _log("sim.reset() (first GPU-physics warmup / shader compile may be slow)...")
    sim.reset()
    _log("sim.reset() done")
    robot = scene["robot"]
    print("\n===== SMOKE OK =====")
    print(f"  device   : {args.device}")
    print(f"  envs     : {scene.num_envs}")
    print(f"  hand DOFs: {robot.num_joints}")
    print(f"  joints   : {robot.data.joint_names}")
    _log("stepping physics...")
    for i in range(args.steps):
        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        if i == 0:
            _log("first sim.step() returned (GPU pipeline live)")
    print(f"  stepped  : {args.steps} steps, no explosion")
    print("====================\n")


main()
simulation_app.close()
