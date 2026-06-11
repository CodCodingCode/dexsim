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
p.add_argument("--spp", type=int, default=96, help="path-traced samples per video frame")
p.add_argument("--settle", type=int, default=60, help="warmup sim steps before filming")
p.add_argument("--max_frames", type=int, default=0, help="0 = whole song")
p.add_argument("--stride", type=int, default=1, help="capture every Nth control step")
p.add_argument("--fps", type=float, default=0,
               help="output fps. 0 (default) = REALTIME: derived as (1/control_dt)/stride "
                    "so the clip's wall-clock matches the performance. Pass a value only to "
                    "force a non-realtime fps.")
p.add_argument("--eye", default="1.5,-1.35,1.35")
p.add_argument("--target", default="0.5,-0.5,0.80")
p.add_argument("--cams", default=None,
               help="MULTI-STILL in ONE boot: ';'-list of 'eye|target' (e.g. '1.2,0,1.3|0.5,0,0.8;...'); "
                    "saves one PNG per cam to --out given as a matching ';'-list. Renders rollout frame 0.")
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
_s.set("/rtx/pathtracing/spp", 8)
_s.set("/rtx/pathtracing/totalSpp", args.spp)
_s.set("/rtx/pathtracing/optixDenoiser/enabled", False)   # weights absent on this box -> rely on high spp
_s.set("/rtx/pathtracing/maxBounces", 6)                  # was 4: softer global illumination
_s.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 6)
_s.set("/rtx/post/tonemap/op", 1)                          # filmic-ish tonemap for a polished look
_s.set("/rtx/post/histogram/enabled", True)
_s.set("/app/asyncRendering", False)


def _quat_aim(eye, tgt):
    """quaternion (w,x,y,z) aiming a light's -Z from eye toward tgt."""
    import numpy as _np
    d = _np.array(tgt, float) - _np.array(eye, float); d /= (_np.linalg.norm(d) + 1e-9)
    z = -d; up = _np.array([0, 0, 1.0]); x = _np.cross(up, z); x /= (_np.linalg.norm(x) + 1e-9)
    y = _np.cross(z, x); R = _np.stack([x, y, z], 1)
    w = _np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w < 1e-6: return (1.0, 0.0, 0.0, 0.0)
    return (float(w), float((R[2, 1]-R[1, 2])/(4*w)), float((R[0, 2]-R[2, 0])/(4*w)), float((R[1, 0]-R[0, 1])/(4*w)))

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
    # clean studio floor (replaces the grey grid) + soft 3-point-ish lighting
    _fm = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.10, 0.12), roughness=0.55)
    sim_utils.CuboidCfg(size=(40.0, 40.0, 0.04), visual_material=_fm).func(
        "/World/Floor", sim_utils.CuboidCfg(size=(40.0, 40.0, 0.04), visual_material=_fm),
        translation=(0.0, 0.0, -0.02))
    # SUPPORTS so nothing floats: a table under the piano (top just below the keys at
    # ~0.74 m) and two pedestals under the robot bases (z=1.05). Visual props only.
    _wood = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.13, 0.08), roughness=0.7)
    _ped = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.12, 0.14), roughness=0.5)
    sim_utils.CuboidCfg(size=(0.45, 1.55, 0.72), visual_material=_wood).func(
        "/World/PianoTable", sim_utils.CuboidCfg(size=(0.45, 1.55, 0.72), visual_material=_wood),
        translation=(0.60, 0.0, 0.36))                         # under the piano (center x=0.60)
    for _nm, _y in (("PedL", -0.30), ("PedR", 0.30)):
        sim_utils.CuboidCfg(size=(0.34, 0.34, 1.05), visual_material=_ped).func(
            f"/World/{_nm}", sim_utils.CuboidCfg(size=(0.34, 0.34, 1.05), visual_material=_ped),
            translation=(1.60, _y, 0.525))                     # under each base (in front, x=1.25)
    sim_utils.DomeLightCfg(intensity=700.0, color=(0.5, 0.55, 0.65)).func(
        "/World/Dome", sim_utils.DomeLightCfg(intensity=700.0, color=(0.5, 0.55, 0.65)))
    _c = (0.5, -0.5, 0.85)
    # high overhead so the fixtures stay out of frame; intensities scaled for distance
    for nm, pos, inten, rad, col in [
        ("Key", (1.6, -2.2, 7.5), 360000.0, 1.0, (1.0, 0.97, 0.92)),
        ("Fill", (-1.6, -2.2, 6.5), 120000.0, 1.4, (0.85, 0.9, 1.0)),
        ("Rim", (0.3, 2.4, 7.0), 180000.0, 0.8, (0.8, 0.85, 1.0))]:
        sim_utils.DiskLightCfg(intensity=inten, radius=rad, color=col).func(
            f"/World/{nm}", sim_utils.DiskLightCfg(intensity=inten, radius=rad, color=col),
            translation=pos, orientation=_quat_aim(pos, _c))

    piano = Articulation(cfg.piano_cfg.replace(prim_path="/World/Piano"))
    left = Articulation(cfg.left_robot_cfg.replace(prim_path="/World/LeftRobot"))
    right = Articulation(cfg.right_robot_cfg.replace(prim_path="/World/RightRobot"))

    # Pedestals under each base so the arms don't float (cosmetic; bases are
    # world-fixed). Box runs ground -> base z, read live from the resolved cfg.
    for _nm, _rc in (("LeftPedestal", cfg.left_robot_cfg),
                     ("RightPedestal", cfg.right_robot_cfg)):
        _bx, _by, _bz = _rc.init_state.pos
        if _bz and _bz > 0.02:
            _pc = sim_utils.CuboidCfg(
                size=(0.26, 0.26, float(_bz)),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.22, 0.22, 0.25), metallic=0.1, roughness=0.5))
            _pc.func(f"/World/{_nm}", _pc,
                     translation=(_bx, _by, float(_bz) / 2.0))

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

    # MULTI-STILL: one Isaac boot, N cameras of the static frame-0 pose -> N PNGs (fast).
    if args.cams:
        from PIL import Image as _Img
        set_state(left, left_traj[0:1]); set_state(right, right_traj[0:1])
        piano.write_joint_state_to_sim(key_traj[0:1], torch.zeros_like(key_traj[0:1])); piano.write_data_to_sim()
        outs = [o.strip() for o in args.out.split(";")]
        specs = [s.strip() for s in args.cams.split(";")]
        for _ci, _spec in enumerate(specs):
            _es, _ts = _spec.split("|")
            _e = torch.tensor([[float(v) for v in _es.split(",")]], device=args.device)
            _t = torch.tensor([[float(v) for v in _ts.split(",")]], device=args.device)
            cam.set_world_poses_from_view(_e, _t)
            sim.step()                                   # restart path-trace accumulation for the new view
            for _ in range(args.spp):
                sim.render()
            cam.update(sim.get_physics_dt(), force_recompute=True)
            _rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype("uint8")
            _Img.fromarray(_rgb).save(outs[_ci])
            print(f"[render_rollout] STILL {_ci + 1}/{len(specs)} -> {outs[_ci]}", flush=True)
        return

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

    # REALTIME by construction: consecutive rendered frames are `stride` control
    # steps apart (control_dt seconds each), so the only fps that makes the clip's
    # wall-clock equal the performance is (1/control_dt)/stride. Encoding at a flat
    # fps that ignores stride is what made strided clips play sped-up by exactly the
    # stride factor (e.g. 999 steps shown in 80 frames @ 20 fps = 12.5x too fast).
    control_dt = float(data["control_dt"]) if "control_dt" in data else float(cfg.control_dt)
    control_hz = 1.0 / control_dt
    stride = max(1, args.stride)
    if args.fps and args.fps > 0:
        out_fps = float(args.fps)                       # explicit non-realtime override
        fps_arg = repr(out_fps)
    else:
        out_fps = control_hz / stride                   # realtime
        fps_arg = f"{control_hz:g}/{stride}"            # exact rational for ffmpeg, e.g. 20/12
    real_dur = len(idxs) / out_fps
    print(f"[render_rollout] control_dt={control_dt:.4f}s ({control_hz:g}Hz) stride={stride} "
          f"-> {out_fps:.3f} fps | {len(idxs)} frames = {real_dur:.1f}s clip, covering "
          f"{idxs[-1] * control_dt:.1f}s of the {n * control_dt:.1f}s performance", flush=True)
    cmd = ["ffmpeg", "-y", "-framerate", fps_arg,
           "-i", os.path.join(args.frames_dir, "frame_%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-vf", "fps=30",   # re-time to a smooth 30fps container (duration preserved)
           args.out]
    print("[render_rollout] encoding:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"[render_rollout] saved video -> {args.out}")


main()
app.close()
