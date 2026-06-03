"""Phase 2 of filming: replay the recorded rollout (logs/rollout.npz from
record_rollout.py) in the path-traced Camera scene and stitch frames to an MP4.

Real-time RTX gives black frames on this H100, so we use PathTracing with the
OptiX denoiser OFF (same recipe as render_scene.py), accumulating a handful of
samples per frame. Joints are driven kinematically to the recorded trajectory.

  python scripts/render_rollout.py --headless --rollout logs/rollout.npz \
         --out results/easy_song_video.mp4 --spp 24
"""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--rollout", default="logs/rollout.npz")
p.add_argument("--out", default="results/easy_song_video.mp4")
p.add_argument("--frames_dir", default="logs/video_frames")
p.add_argument("--spp", type=int, default=24, help="path-traced samples per video frame")
p.add_argument("--settle", type=int, default=60, help="warmup sim steps before filming")
p.add_argument("--max_frames", type=int, default=0, help="0 = whole song")
p.add_argument("--stride", type=int, default=1, help="capture every Nth control step")
p.add_argument("--fps", type=int, default=20)
p.add_argument("--eye", default="1.5,-1.35,1.35")
p.add_argument("--target", default="0.5,-0.5,0.80")
p.add_argument("--width", type=int, default=1280)
p.add_argument("--height", type=int, default=720)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
args.enable_cameras = True

app = AppLauncher(args).app

import carb
_s = carb.settings.get_settings()
_s.set("/rtx/rendermode", "PathTracing")
_s.set("/rtx/pathtracing/spp", 4)
_s.set("/rtx/pathtracing/totalSpp", args.spp)
_s.set("/rtx/pathtracing/optixDenoiser/enabled", False)   # weights absent on this box
_s.set("/rtx/pathtracing/maxBounces", 4)
_s.set("/app/asyncRendering", False)

import os
import subprocess
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sensors import Camera, CameraCfg

from dexsim.tasks.piano.piano_env_cfg import PianoEnvCfg

NUM_KEYS = 88
_BLACK_IN_OCTAVE = {1, 3, 6, 8, 10}        # which semitones are black keys


def _is_black(k: int) -> bool:
    return (k % 12) in _BLACK_IN_OCTAVE


def _font(sz):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def draw_note_ui(rgb: np.ndarray, goal_row, sound_row) -> Image.Image:
    """Composite the rendered frame with a piano-keyboard UI strip that compares
    the GROUND-TRUTH notes (what should play) to what the model is PLAYING.
      blue   = should play (ground truth), not played -> MISSED
      green  = played AND should play           -> CORRECT
      red    = played but should NOT             -> WRONG
    """
    fh, fw = rgb.shape[:2]
    strip_h = 150
    canvas = Image.new("RGB", (fw, fh + strip_h), (18, 18, 22))
    canvas.paste(Image.fromarray(rgb), (0, 0))
    d = ImageDraw.Draw(canvas)

    pad, top = 24, fh + 40
    kb_w, kb_h = fw - 2 * pad, 90
    cell = kb_w / NUM_KEYS
    goal = np.asarray(goal_row).astype(bool)
    sound = np.asarray(sound_row).astype(bool)
    n_goal = int(goal.sum()); n_correct = int((goal & sound).sum()); n_wrong = int((sound & ~goal).sum())

    for k in range(NUM_KEYS):
        x0 = pad + k * cell
        x1 = x0 + cell - 1
        base = (40, 40, 48) if _is_black(k) else (225, 225, 230)
        if goal[k] and sound[k]:
            col = (40, 200, 70)        # correct = green
        elif sound[k] and not goal[k]:
            col = (220, 60, 60)        # wrong = red
        elif goal[k] and not sound[k]:
            col = (70, 130, 240)       # missed = blue
        else:
            col = base
        d.rectangle([x0, top, x1, top + kb_h], fill=col, outline=(60, 60, 66))

    f = _font(22); fs = _font(16)
    d.text((pad, fh + 8), "GROUND TRUTH vs PLAYED", fill=(235, 235, 240), font=f)
    legend = [("correct", (40, 200, 70)), ("wrong", (220, 60, 60)), ("missed", (70, 130, 240))]
    lx = pad + 360
    for name, c in legend:
        d.rectangle([lx, fh + 12, lx + 18, fh + 30], fill=c)
        d.text((lx + 24, fh + 12), name, fill=(220, 220, 225), font=fs); lx += 130
    d.text((fw - pad - 230, fh + 12),
           f"correct {n_correct}/{n_goal}  wrong {n_wrong}", fill=(235, 235, 240), font=fs)
    return canvas


def set_state(art, q):
    """Kinematically drive an articulation to joint positions q (1, ndof)."""
    art.write_joint_state_to_sim(q, torch.zeros_like(q))
    art.set_joint_position_target(q)
    art.write_data_to_sim()


def main():
    data = np.load(args.rollout, allow_pickle=True)
    left_traj = torch.tensor(data["left"], dtype=torch.float32, device=args.device)
    right_traj = torch.tensor(data["right"], dtype=torch.float32, device=args.device)
    key_traj = torch.tensor(data["keys"], dtype=torch.float32, device=args.device)
    goal_arr = data["goal"] if "goal" in data else np.zeros((left_traj.shape[0], NUM_KEYS), np.uint8)
    sound_arr = data["sound"] if "sound" in data else np.zeros((left_traj.shape[0], NUM_KEYS), np.uint8)
    n = left_traj.shape[0]
    idxs = list(range(0, n, args.stride))
    if args.max_frames:
        idxs = idxs[: args.max_frames]
    print(f"[render_rollout] {n} recorded steps -> {len(idxs)} video frames "
          f"@ {args.spp} spp", flush=True)

    cfg = PianoEnvCfg()
    sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device=args.device))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=3000.0))

    piano = Articulation(cfg.piano_cfg.replace(prim_path="/World/Piano"))
    left = Articulation(cfg.left_robot_cfg.replace(prim_path="/World/LeftRobot"))
    right = Articulation(cfg.right_robot_cfg.replace(prim_path="/World/RightRobot"))
    cam = Camera(CameraCfg(
        prim_path="/World/cam", update_period=0,
        height=args.height, width=args.width, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955,
                                         clipping_range=(0.05, 1e6)),
    ))

    sim.reset()
    eye = torch.tensor([[float(v) for v in args.eye.split(",")]], device=args.device)
    tgt = torch.tensor([[float(v) for v in args.target.split(",")]], device=args.device)
    cam.set_world_poses_from_view(eye, tgt)

    # warm up at the first recorded pose so lighting/physics settle
    for _ in range(args.settle):
        set_state(left, left_traj[0:1])
        set_state(right, right_traj[0:1])
        piano.write_joint_state_to_sim(key_traj[0:1], torch.zeros_like(key_traj[0:1]))
        piano.write_data_to_sim()
        sim.step()

    os.makedirs(args.frames_dir, exist_ok=True)
    for f, t in enumerate(idxs):
        set_state(left, left_traj[t:t + 1])
        set_state(right, right_traj[t:t + 1])
        piano.write_joint_state_to_sim(key_traj[t:t + 1], torch.zeros_like(key_traj[t:t + 1]))
        piano.write_data_to_sim()
        sim.step()                       # update transforms; restarts accumulation
        for _ in range(args.spp):
            sim.render()
        cam.update(sim.get_physics_dt(), force_recompute=True)
        rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype("uint8")
        frame = draw_note_ui(rgb, goal_arr[t], sound_arr[t])    # 3D view + note-compare UI
        frame.save(os.path.join(args.frames_dir, f"frame_{f:05d}.png"))
        if f % 25 == 0:
            nb = int((rgb.sum(-1) > 10).mean() * 100)
            print(f"  frame {f}/{len(idxs)} (step {t}) non-black {nb}%", flush=True)

    out_fps = max(1, args.fps // args.stride)
    cmd = ["ffmpeg", "-y", "-framerate", str(out_fps),
           "-i", os.path.join(args.frames_dir, "frame_%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", args.out]
    print("[render_rollout] encoding:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"[render_rollout] saved video -> {args.out}")


main()
app.close()
