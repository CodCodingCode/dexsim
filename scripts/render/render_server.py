"""Warm/persistent Isaac Sim renderer -- the iteration-speed cache.

Cold scripts (render_scene.py, render_rollout.py, diag_*.py) each boot the WHOLE
Isaac Sim app (~tens of seconds) and rebuild the scene from scratch on every run.
This server boots the app + builds the piano scene ONCE, then serves render and
diagnostic jobs from a file-queue, so each subsequent render/measure costs seconds.

  source env.sh
  # boot once, in the background:
  python scripts/render/render_server.py --headless > logs/render_server.log 2>&1 &
  # then fire as many jobs as you like, each in ~seconds:
  python scripts/render/render.py scene  --eye 2.2,-1.5,1.8 --out logs/x.png
  python scripts/render/render.py rollout --rollout logs/rollout.npz --out results/v.mp4
  python scripts/render/render.py query  --kind layout --out logs/layout.json
  python scripts/render/render.py shutdown      # stop the server

Protocol (file-queue under --jobs_dir, default logs/render_jobs):
  client writes  <id>.job.json   (atomically: *.tmp then rename)
  server writes  <id>.done.json  (result)  or  <id>.err.json (error+traceback)
  a `server.ready` marker (with pid) signals a live, booted server.
"""
from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--jobs_dir", default="logs/render_jobs")
p.add_argument("--width", type=int, default=1280)
p.add_argument("--height", type=int, default=720)
p.add_argument("--style", default="studio", choices=["studio", "simple"])
p.add_argument("--poll", type=float, default=0.2, help="idle poll interval (s)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
args.enable_cameras = True

app = AppLauncher(args).app

import glob
import json
import os
import subprocess
import time
import traceback

import numpy as np
import torch

from isaaclab.sim import SimulationCfg, SimulationContext

from dexsim.tasks.piano.piano_env_cfg import PianoEnvCfg
from dexsim.render import studio

NUM_KEYS = studio.NUM_KEYS


# --------------------------------------------------------------------------- #
# boot: app is up, build the scene exactly ONCE
# --------------------------------------------------------------------------- #
def _parse_vec(v, default):
    if v is None:
        return list(default)
    if isinstance(v, str):
        return [float(x) for x in v.split(",")]
    return [float(x) for x in v]


studio.apply_pathtrace_settings(160, max_bounces=6, tonemap=(args.style == "studio"))

cfg = PianoEnvCfg()
sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device=args.device))
piano, left, right = studio.build_scene(cfg, style=args.style)
cam = studio.make_camera(args.width, args.height)
sim.reset()
studio.aim_camera(cam, studio.HERO_EYE, studio.HERO_TARGET, args.device)
for _ in range(30):                                  # settle once at the ready pose
    for a in (piano, left, right):
        a.set_joint_position_target(a.data.default_joint_pos)
        a.write_data_to_sim()
    sim.step()

control_dt = float(cfg.control_dt)
LEFT_NAMES = list(left.data.joint_names)
RIGHT_NAMES = list(right.data.joint_names)

os.makedirs(args.jobs_dir, exist_ok=True)
READY = os.path.join(args.jobs_dir, "server.ready")
with open(READY, "w") as f:
    json.dump({"pid": os.getpid(), "width": args.width, "height": args.height,
               "style": args.style, "control_dt": control_dt}, f)
print(f"[render_server] READY pid={os.getpid()} jobs_dir={args.jobs_dir} "
      f"style={args.style} {args.width}x{args.height}", flush=True)


# --------------------------------------------------------------------------- #
# helpers shared by the job handlers
# --------------------------------------------------------------------------- #
def _drive_default():
    for a in (piano, left, right):
        studio.set_state(a, a.data.default_joint_pos)


def _apply_joint_overrides(art, names, overrides):
    """overrides: {joint_name: value}; returns the driven (1,ndof) tensor."""
    q = art.data.default_joint_pos.clone()
    for jn, v in (overrides or {}).items():
        if jn in names:
            q[0, names.index(jn)] = float(v)
    studio.set_state(art, q)
    return q


def _accumulate(spp, settle=0):
    for _ in range(max(0, settle)):
        sim.step()
    sim.step()                                # restarts path-trace accumulation
    for _ in range(int(spp)):
        sim.render()
    cam.update(sim.get_physics_dt(), force_recompute=True)
    return cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype("uint8")


def _bodies_world(art, substrs):
    """{substr: {pos:[x,y,z], quat:[w,x,y,z]}} for the first body matching each substr."""
    out = {}
    bnames = art.body_names
    for s in substrs:
        idx = next((i for i, n in enumerate(bnames) if s in n), None)
        if idx is None:
            out[s] = None
            continue
        pos = art.data.body_pos_w[0, idx].cpu().tolist()
        quat = art.data.body_quat_w[0, idx].cpu().tolist()
        out[s] = {"pos": [round(float(v), 4) for v in pos],
                  "quat": [round(float(v), 4) for v in quat]}
    return out


def _keyboard_bbox():
    kb = piano.data.body_pos_w[0].cpu()
    return {"min": [round(float(kb[:, i].min()), 3) for i in range(3)],
            "max": [round(float(kb[:, i].max()), 3) for i in range(3)],
            "center": [round(float(kb[:, i].mean()), 3) for i in range(3)],
            "top_z": round(float(kb[:, 2].max()), 3)}


# --------------------------------------------------------------------------- #
# job handlers
# --------------------------------------------------------------------------- #
def handle_scene(job):
    """Single path-traced still at the ready pose (or a rollout frame / overrides)."""
    spp = int(job.get("spp", 160))
    settle = int(job.get("settle", 60))
    studio.set_total_spp(spp)

    if job.get("rollout") and "frame" in job:
        data = np.load(job["rollout"], allow_pickle=True)
        t = int(job["frame"])
        studio.set_state(left, torch.tensor(data["left"][t:t + 1], dtype=torch.float32, device=args.device))
        studio.set_state(right, torch.tensor(data["right"][t:t + 1], dtype=torch.float32, device=args.device))
        kq = torch.tensor(data["keys"][t:t + 1], dtype=torch.float32, device=args.device)
        piano.write_joint_state_to_sim(kq, torch.zeros_like(kq)); piano.write_data_to_sim()
    else:
        _drive_default()
        if job.get("left_joints"):
            _apply_joint_overrides(left, LEFT_NAMES, job["left_joints"])
        if job.get("right_joints"):
            _apply_joint_overrides(right, RIGHT_NAMES, job["right_joints"])

    studio.aim_camera(cam, _parse_vec(job.get("eye"), studio.HERO_EYE),
                      _parse_vec(job.get("target"), studio.HERO_TARGET), args.device)
    rgb = _accumulate(spp, settle=settle)
    out = job["out"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    studio.Image.fromarray(rgb).save(out)
    nonblack = float((rgb.sum(-1) > 10).mean() * 100)
    return {"out": out, "nonblack_pct": round(nonblack, 1), "spp": spp}


def handle_rollout(job):
    """Replay a recorded npz to an MP4 with the note-compare UI (render_rollout)."""
    data = np.load(job["rollout"], allow_pickle=True)
    left_traj = torch.tensor(data["left"], dtype=torch.float32, device=args.device)
    right_traj = torch.tensor(data["right"], dtype=torch.float32, device=args.device)
    key_traj = torch.tensor(data["keys"], dtype=torch.float32, device=args.device)
    n = left_traj.shape[0]
    goal_arr = data["goal"] if "goal" in data else np.zeros((n, NUM_KEYS), np.uint8)
    sound_arr = data["sound"] if "sound" in data else np.zeros((n, NUM_KEYS), np.uint8)
    rdt = float(data["control_dt"]) if "control_dt" in data else control_dt

    spp = int(job.get("spp", 96)); settle = int(job.get("settle", 60))
    stride = max(1, int(job.get("stride", 1)))
    idxs = list(range(0, n, stride))
    if job.get("max_frames"):
        idxs = idxs[: int(job["max_frames"])]
    frames_dir = job.get("frames_dir", "logs/video_frames")
    out = job["out"]
    studio.set_total_spp(spp)
    studio.aim_camera(cam, _parse_vec(job.get("eye"), studio.HERO_EYE),
                      _parse_vec(job.get("target"), studio.HERO_TARGET), args.device)

    for _ in range(settle):                              # warm up at the first pose
        studio.set_state(left, left_traj[0:1])
        studio.set_state(right, right_traj[0:1])
        piano.write_joint_state_to_sim(key_traj[0:1], torch.zeros_like(key_traj[0:1]))
        piano.write_data_to_sim()
        sim.step()

    os.makedirs(frames_dir, exist_ok=True)
    # CLEAR stale frames first: ffmpeg globs frame_%05d.png, so leftover frames from a
    # previous (longer) render would get appended -> a second "clip" stitched onto the end.
    for _stale in glob.glob(os.path.join(frames_dir, "frame_*.png")):
        os.remove(_stale)
    for f, t in enumerate(idxs):
        studio.set_state(left, left_traj[t:t + 1])
        studio.set_state(right, right_traj[t:t + 1])
        piano.write_joint_state_to_sim(key_traj[t:t + 1], torch.zeros_like(key_traj[t:t + 1]))
        piano.write_data_to_sim()
        sim.step()
        for _ in range(spp):
            sim.render()
        cam.update(sim.get_physics_dt(), force_recompute=True)
        rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype("uint8")
        studio.draw_note_ui(rgb, goal_arr[t], sound_arr[t]).save(
            os.path.join(frames_dir, f"frame_{f:05d}.png"))

    out_fps = float(job["fps"]) if job.get("fps") else (1.0 / rdt) / stride
    fps_arg = repr(out_fps) if job.get("fps") else f"{1.0 / rdt:g}/{stride}"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cmd = ["ffmpeg", "-y", "-framerate", fps_arg,
           "-i", os.path.join(frames_dir, "frame_%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-vf", "fps=30", out]
    subprocess.run(cmd, check=True)
    return {"out": out, "n_frames": len(idxs), "fps": round(out_fps, 3),
            "covers_s": round(idxs[-1] * rdt, 1) if idxs else 0.0}


def handle_query(job):
    """Geometry/measurement queries (subsumes diag_layout / diag_hand_orient /
    verify_palm / diag_arm_links). Drives a pose, then reads world body data."""
    kind = job.get("kind", "bodies")
    res = {"kind": kind, "keyboard": _keyboard_bbox()}

    # ---- pose source -----------------------------------------------------
    if job.get("rollout"):
        data = np.load(job["rollout"], allow_pickle=True)
        Lt, Rt = data["left"], data["right"]
        fracs = job.get("frame_fracs", [0.3, 0.5, 0.7])
        frames = [int(Lt.shape[0] * fr) for fr in fracs]
    else:
        _drive_default()
        if job.get("left_joints"):
            _apply_joint_overrides(left, LEFT_NAMES, job["left_joints"])
        if job.get("right_joints"):
            _apply_joint_overrides(right, RIGHT_NAMES, job["right_joints"])
        for _ in range(5):
            sim.step()
        frames = None

    # ---- preset measurements --------------------------------------------
    if kind == "layout":
        res.update({
            "left_base": [round(v, 3) for v in cfg.left_robot_cfg.init_state.pos],
            "right_base": [round(v, 3) for v in cfg.right_robot_cfg.init_state.pos],
            "left_palm": _bodies_world(left, ["robot0_palm"]).get("robot0_palm"),
            "right_palm": _bodies_world(right, ["robot0_palm"]).get("robot0_palm"),
        })
    elif kind == "orient":
        from dexsim.piano import FINGERTIP_BODIES
        palm = _bodies_world(left, ["robot0_palm"]).get("robot0_palm")
        ppos = np.array(palm["pos"]) if palm else np.zeros(3)
        tips, zs = {}, []
        for fb in FINGERTIP_BODIES:
            bw = _bodies_world(left, [fb]).get(fb)
            if bw:
                d = (np.array(bw["pos"]) - ppos).tolist()
                tips[fb] = [round(float(v), 4) for v in d]; zs.append(d[2])
        avgz = float(np.mean(zs)) if zs else 0.0
        res.update({"palm": palm, "fingertip_minus_palm": tips,
                    "mean_fingertip_dz": round(avgz, 4),
                    "palm_above_keys": round(ppos[2] - res["keyboard"]["top_z"], 4),
                    "verdict": "FINGERS DOWN (ok)" if avgz < -0.02 else "INVERTED (BAD)"})
    elif kind == "palm" and frames is not None:
        kbx = (res["keyboard"]["min"][0], res["keyboard"]["max"][0])
        rows = []
        for t in frames:
            studio.set_state(left, torch.tensor(Lt[t:t + 1], dtype=torch.float32, device=args.device))
            studio.set_state(right, torch.tensor(Rt[t:t + 1], dtype=torch.float32, device=args.device))
            for _ in range(3):
                sim.step()
            rows.append({"frame": t,
                         "left_palm": _bodies_world(left, ["robot0_palm"]).get("robot0_palm")["pos"],
                         "right_palm": _bodies_world(right, ["robot0_palm"]).get("robot0_palm")["pos"]})
        over = all(kbx[0] - 0.1 <= r["left_palm"][0] <= kbx[1] + 0.15 for r in rows)
        res.update({"keyboard_x_range": kbx, "frames": rows,
                    "verdict": "over keys" if over else "NOT over keys (still reaching past)"})
    elif kind == "joints":
        # disambiguate L/R asset differences: applied default pose + limits per joint.
        def _lim(art, i):
            try:
                lo, hi = art.data.joint_pos_limits[0, i].tolist()
            except Exception:
                lo, hi = art.data.soft_joint_pos_limits[0, i].tolist()
            return [round(lo, 3), round(hi, 3)]
        rows = []
        for n in LEFT_NAMES:
            li = LEFT_NAMES.index(n)
            ri = RIGHT_NAMES.index(n) if n in RIGHT_NAMES else None
            rows.append({
                "joint": n,
                "L_default": round(float(left.data.default_joint_pos[0, li]), 4),
                "R_default": round(float(right.data.default_joint_pos[0, ri]), 4) if ri is not None else None,
                "L_limits": _lim(left, li),
                "R_limits": _lim(right, ri) if ri is not None else None,
            })
        res.update({"name_symdiff": sorted(set(LEFT_NAMES) ^ set(RIGHT_NAMES)),
                    "joints": rows})
    else:  # generic: dump the requested body world poses on each arm
        which = job.get("bodies", ["robot0_palm", "forearm", "wrist_3", "upper_arm"])
        res.update({"left_bodies": _bodies_world(left, which),
                    "right_bodies": _bodies_world(right, which),
                    "left_joint_names": LEFT_NAMES})

    if job.get("out"):
        os.makedirs(os.path.dirname(job["out"]) or ".", exist_ok=True)
        with open(job["out"], "w") as f:
            json.dump(res, f, indent=2)
        res = {"out": job["out"], **res}
    return res


HANDLERS = {"scene": handle_scene, "rollout": handle_rollout, "query": handle_query}


# --------------------------------------------------------------------------- #
# serve loop
# --------------------------------------------------------------------------- #
def _write_result(stem, payload, ok):
    suffix = ".done.json" if ok else ".err.json"
    tmp = stem + suffix + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, stem + suffix)


def serve():
    while app.is_running():
        jobs = sorted(glob.glob(os.path.join(args.jobs_dir, "*.job.json")))
        if not jobs:
            time.sleep(args.poll)
            continue
        path = jobs[0]
        stem = path[: -len(".job.json")]
        try:
            with open(path) as f:
                job = json.load(f)
        except Exception:
            os.remove(path)                              # unreadable -> drop
            continue
        os.remove(path)                                  # claim it
        jtype = job.get("type")
        if jtype == "shutdown":
            _write_result(stem, {"ok": True, "msg": "shutting down"}, True)
            print("[render_server] shutdown requested", flush=True)
            break
        t0 = time.time()
        try:
            handler = HANDLERS[jtype]
            result = handler(job)
            result["elapsed_s"] = round(time.time() - t0, 2)
            _write_result(stem, result, True)
            print(f"[render_server] {jtype} done in {result['elapsed_s']}s "
                  f"-> {result.get('out', '(inline)')}", flush=True)
        except Exception as e:
            _write_result(stem, {"error": str(e), "type": jtype,
                                 "traceback": traceback.format_exc()}, False)
            print(f"[render_server] {jtype} FAILED: {e}", flush=True)


try:
    serve()
finally:
    if os.path.exists(READY):
        os.remove(READY)
    app.close()
