"""Mount a camera on each robot WRIST and render what it sees -- the fingers and
the keys beneath them -- plus a third overview camera of the whole scene.

This is the "wrist cam so we can see which keys the hands press" deliverable. The
cameras are real Camera sensors parented to each wrist_3_link, so they rigidly
track the hand exactly like a real wrist-mounted camera would.

  python scripts/wrist_cam_render.py --out logs/wristcam
"""

from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="logs/wristcam")
parser.add_argument("--settle", type=int, default=120)
parser.add_argument("--spp", type=int, default=96, help="path-traced frames to accumulate")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import carb  # noqa: E402
_s = carb.settings.get_settings()
_s.set("/rtx/rendermode", "PathTracing")
_s.set("/rtx/pathtracing/spp", 8)
_s.set("/rtx/pathtracing/totalSpp", args.spp)
_s.set("/rtx/pathtracing/optixDenoiser/enabled", False)
_s.set("/rtx/pathtracing/maxBounces", 4)
_s.set("/app/asyncRendering", False)

import os  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

from dexsim.assets import UR10E_SHADOW_CFG, PIANO_CFG  # noqa: E402
from dexsim.tasks.piano.piano_env_cfg import PianoEnvCfg  # noqa: E402


def body_idx(art, name):
    return art.data.body_names.index(name)


def save(rgb, path, tag):
    rgb = rgb[..., :3].astype("uint8")
    nonblack = int((rgb.sum(-1) > 10).sum())
    pct = 100 * nonblack / (rgb.shape[0] * rgb.shape[1])
    Image.fromarray(rgb).save(path)
    print(f"[wristcam] {tag:10s} -> {path}  ({pct:.1f}% non-black)", flush=True)


def main():
    cfg = PianoEnvCfg()
    dev = args.device
    sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device=dev))

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=1400.0, color=(1.0, 1.0, 1.0)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=1400.0))

    # use the exact articulations the env configures (bases + ready pose already
    # baked into their init_state by PianoEnvCfg.__post_init__)
    piano = Articulation(cfg.piano_cfg.replace(prim_path="/World/Piano"))
    left = Articulation(cfg.left_robot_cfg.replace(prim_path="/World/LeftRobot"))
    right = Articulation(cfg.right_robot_cfg.replace(prim_path="/World/RightRobot"))

    # wrist cameras: a real sensor parented under each wrist_3_link so it tracks
    # the hand. Pose is set explicitly after settle (below) via look-from-view.
    cam_cfg = lambda p: CameraCfg(  # noqa: E731
        prim_path=p, update_period=0, height=600, width=800, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955,
                                         clipping_range=(0.01, 1e5)))
    # ONE camera, repositioned to each viewpoint. Path tracing on the RT-core-less
    # H100 only reliably accumulates a single render product, so multiple Camera
    # sensors leave all-but-one black. A single camera moved between shots (the
    # scene is static after settle) sidesteps that entirely.
    cam = Camera(cam_cfg("/World/shot_cam"))

    sim.reset()

    wl, wr = body_idx(left, "wrist_3_link"), body_idx(right, "wrist_3_link")
    fl = [body_idx(left, f"robot0_{f}distal") for f in ("ff", "mf", "rf")]
    fr = [body_idx(right, f"robot0_{f}distal") for f in ("ff", "mf", "rf")]
    key_z = 0.722

    for i in range(args.settle):
        for a in (piano, left, right):
            a.set_joint_position_target(a.data.default_joint_pos)
            a.write_data_to_sim()
        sim.step()

    # aim each wrist cam: eye just above+behind the wrist, looking down at the
    # fingertips projected onto the key plane (so the keys under the fingers fill
    # the frame). Overview: 3/4 view of the whole rig.
    def view(art, wi, fis):
        # Steep over-the-knuckles view: hover ~0.4 m above the key plane, set back
        # toward the arm base, looking down at the keys directly under the
        # fingertips. The fingers sit between camera and keys -> we see exactly
        # which keys each finger is over / pressing.
        w = art.data.body_pos_w[0, wi].cpu().numpy()
        f = art.data.body_pos_w[0, fis].mean(0).cpu().numpy()
        back = w - f
        back[2] = 0
        n = np.linalg.norm(back) + 1e-6
        center = np.array([f[0], f[1], key_z])
        eye = center + 0.13 * back / n + np.array([0, 0, 0.40])
        return eye, center

    le, lt = view(left, wl, fl)
    re, rt = view(right, wr, fr)
    print(f"[wristcam] LEFT  eye={le.round(3)} -> tgt={lt.round(3)}", flush=True)
    print(f"[wristcam] RIGHT eye={re.round(3)} -> tgt={rt.round(3)}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    dt = sim.get_physics_dt()
    t = lambda v: torch.tensor([v], dtype=torch.float32, device=dev)  # noqa: E731

    # overhead mount camera: above the keyboard center, looking straight down
    # (slight -Y offset for a little perspective on the hands).
    over_eye = np.array([cfg.piano_pos[0], cfg.piano_pos[1] - 0.25, cfg.piano_pos[2] + 1.25])
    over_tgt = np.array([cfg.piano_pos[0], cfg.piano_pos[1], cfg.piano_pos[2] + 0.02])
    shots = [
        ("left wrist",  le, lt, f"{args.out}/left_wrist.png"),
        ("right wrist", re, rt, f"{args.out}/right_wrist.png"),
        ("overhead", over_eye, over_tgt, f"{args.out}/overhead.png"),
    ]
    for tag, eye, tgt, path in shots:
        cam.set_world_poses_from_view(t(eye), t(tgt))
        print(f"[wristcam] accumulating {args.spp} frames for {tag}...", flush=True)
        for _ in range(args.spp):
            sim.render()
        cam.update(dt, force_recompute=True)
        save(cam.data.output["rgb"][0].cpu().numpy(), path, tag)

    # report fingertip-to-key gap so we know if the hands actually reach
    for tag, art, fis in (("LEFT", left, fl), ("RIGHT", right, fr)):
        f = art.data.body_pos_w[0, fis].mean(0).cpu().numpy()
        print(f"[wristcam] {tag} fingertip mean=({f[0]:+.3f},{f[1]:+.3f},{f[2]:+.3f}) "
              f"key_z={key_z}  dz={f[2]-key_z:+.3f}")


main()
simulation_app.close()
