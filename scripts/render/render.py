"""Thin client for the warm render_server.py -- submit a job, block for the result.

Imports NOTHING heavy (no isaaclab), so it returns in seconds: the cost of the
Isaac boot is paid once by the long-lived server, not per render.

  python scripts/render/render.py scene   --eye 2.2,-1.5,1.8 --out logs/x.png --spp 160
  python scripts/render/render.py rollout --rollout logs/rollout.npz --out results/v.mp4 --spp 96
  python scripts/render/render.py query   --kind layout --out logs/layout.json
  python scripts/render/render.py query   --kind palm --rollout logs/rollout.npz --out logs/palm.json
  python scripts/render/render.py shutdown

Start the server first (once):
  python scripts/render/render_server.py --headless > logs/render_server.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid


def _kv_json(s):
    """Parse a 'name=val,name=val' string into {name: float}."""
    out = {}
    for part in s.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = float(v)
    return out


def build_job(a):
    job = {"type": a.cmd}
    if a.cmd == "shutdown":
        return job
    for k in ("out", "rollout", "eye", "target", "frames_dir", "kind"):
        v = getattr(a, k, None)
        if v is not None:
            job[k] = v
    for k in ("spp", "settle", "stride", "max_frames", "frame"):
        v = getattr(a, k, None)
        if v is not None and v != -1:
            job[k] = v
    if getattr(a, "fps", 0):
        job["fps"] = a.fps
    if getattr(a, "left_joints", None):
        job["left_joints"] = _kv_json(a.left_joints)
    if getattr(a, "right_joints", None):
        job["right_joints"] = _kv_json(a.right_joints)
    if getattr(a, "bodies", None):
        job["bodies"] = [s.strip() for s in a.bodies.split(",") if s.strip()]
    if getattr(a, "frame_fracs", None):
        job["frame_fracs"] = [float(x) for x in a.frame_fracs.split(",")]
    return job


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["scene", "rollout", "query", "shutdown"])
    p.add_argument("--jobs_dir", default="logs/render_jobs")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--out")
    p.add_argument("--rollout")
    p.add_argument("--eye")
    p.add_argument("--target")
    p.add_argument("--spp", type=int, default=-1)
    p.add_argument("--settle", type=int, default=-1)
    p.add_argument("--stride", type=int, default=-1)
    p.add_argument("--max_frames", type=int, default=-1)
    p.add_argument("--frame", type=int, default=-1, help="scene: render this rollout frame")
    p.add_argument("--fps", type=float, default=0)
    p.add_argument("--frames_dir")
    p.add_argument("--kind", default="bodies",
                   help="query preset: layout | orient | palm | bodies")
    p.add_argument("--bodies", help="query bodies: comma-separated name substrings")
    p.add_argument("--frame_fracs", help="query palm: comma-separated fractions, e.g. 0.3,0.5,0.7")
    p.add_argument("--left_joints", help="scene/query: 'shoulder_lift_joint=-1.95,elbow_joint=-1.55'")
    p.add_argument("--right_joints")
    a = p.parse_args()

    ready = os.path.join(a.jobs_dir, "server.ready")
    if not os.path.exists(ready):
        sys.exit(f"[render] no server.ready in {a.jobs_dir} -- start render_server.py first:\n"
                 f"  python scripts/render/render_server.py --headless > logs/render_server.log 2>&1 &")

    os.makedirs(a.jobs_dir, exist_ok=True)
    jid = uuid.uuid4().hex[:12]
    stem = os.path.join(a.jobs_dir, jid)
    job = build_job(a)
    tmp = stem + ".job.json.tmp"
    with open(tmp, "w") as f:
        json.dump(job, f)
    os.replace(tmp, stem + ".job.json")              # atomic publish
    print(f"[render] submitted {a.cmd} job {jid}", flush=True)

    done, err = stem + ".done.json", stem + ".err.json"
    t0 = time.time()
    while time.time() - t0 < a.timeout:
        if os.path.exists(done):
            with open(done) as f:
                res = json.load(f)
            os.remove(done)
            print(json.dumps(res, indent=2))
            return
        if os.path.exists(err):
            with open(err) as f:
                res = json.load(f)
            os.remove(err)
            print("[render] FAILED:\n" + res.get("traceback", res.get("error", "?")), file=sys.stderr)
            sys.exit(1)
        time.sleep(0.1)
    sys.exit(f"[render] timed out after {a.timeout}s waiting for job {jid}")


if __name__ == "__main__":
    main()
