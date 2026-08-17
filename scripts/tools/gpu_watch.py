"""What is actually on the GPU right now, and how much of it is waste.

`nvidia-smi` tells you 15 GB is used; it does not tell you that 10 of those GB
are two forgotten Isaac processes rendering frames nobody is watching. This
merges three sources into one attributed view:

  * ``nvidia-smi --query-compute-apps``  -> per-process GPU memory
  * ``nvidia-smi pmon``                  -> per-process SM (compute) utilisation
  * ``ps``                               -> what the process actually is, and its age

and then classifies each process against how this repo is *meant* to use the GPU,
so "waste" is a claim with a reason attached rather than a vibe.

    python scripts/tools/gpu_watch.py                 # one snapshot
    python scripts/tools/gpu_watch.py --watch 30      # sample every 30 s to CSV
    python scripts/tools/gpu_watch.py --reap          # kill ORPHANED render servers

Needs no extra packages and no Isaac boot -- it shells out to nvidia-smi.

ON REAPING: `--reap` only ever kills render servers that are NOT the one named in
``logs/render_jobs/server.ready``. That marker is the single source of truth for
"the live warm server"; every other render_server process is by definition
untracked, unreachable by the job client, and pure memory. Anything else on the
GPU (training, live_scene, another user's work) is reported and never touched.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time

READY_PATH = "logs/render_jobs/server.ready"

# How this repo is supposed to use the GPU. Order matters: first match wins.
KINDS = [
    ("render_server", "warm Isaac render/measure server (CLAUDE.md says run ONE)"),
    ("live_scene",    "WebRTC livestream -- renders continuously, viewer or not"),
    ("train_",        "training run"),
    ("record_rollout", "rollout recording"),
    ("piano_env_smoke", "smoke test"),
    ("place_scene_viser", "viser scene placer"),
]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return ""


def sample() -> dict:
    """One merged snapshot of every compute process on the GPU."""
    procs: dict[int, dict] = {}

    for line in _run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                      "--format=csv,noheader,nounits"]).splitlines():
        if "," not in line:
            continue
        pid_s, mem_s = (x.strip() for x in line.split(",", 1))
        procs[int(pid_s)] = {"pid": int(pid_s), "mib": int(mem_s), "sm": 0}

    # pmon reports instantaneous SM% per process; a process can hold gigabytes at
    # 0% (parked cache) or burn the card at 90% (actively rendering).
    for line in _run(["nvidia-smi", "pmon", "-c", "1"]).splitlines():
        if line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 4 or not f[1].isdigit():
            continue
        pid = int(f[1])
        if pid in procs:
            procs[pid]["sm"] = int(f[3]) if f[3].isdigit() else 0

    for pid, p in procs.items():
        p["cmd"] = _run(["ps", "-o", "args=", "-p", str(pid)]).strip()
        p["age"] = _run(["ps", "-o", "etime=", "-p", str(pid)]).strip()
        p["kind"] = next((k for k, _ in KINDS if k in p["cmd"]), "other")

    tot = _run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits"]).strip().split(",")
    return {
        "procs": sorted(procs.values(), key=lambda p: -p["mib"]),
        "used_mib": int(tot[0]) if tot and tot[0].strip().isdigit() else 0,
        "total_mib": int(tot[1]) if len(tot) > 1 and tot[1].strip().isdigit() else 0,
        "util": int(tot[2]) if len(tot) > 2 and tot[2].strip().isdigit() else 0,
    }


def live_server_pid() -> int | None:
    """PID of the ONE render server the job client will actually talk to."""
    try:
        with open(READY_PATH) as f:
            pid = int(json.load(f)["pid"])
        os.kill(pid, 0)          # signal 0 = liveness probe, no signal delivered
        return pid
    except Exception:
        return None


def analyse(snap: dict) -> tuple[list[dict], int]:
    """Tag each process with a verdict. Returns (procs, reclaimable MiB)."""
    tracked = live_server_pid()
    waste = 0
    for p in snap["procs"]:
        kind, sm = p["kind"], p["sm"]
        if kind == "render_server":
            if tracked is None:
                p["verdict"] = "ORPHAN (no live server.ready -- untracked)"
                p["reap"] = True
            elif p["pid"] != tracked:
                p["verdict"] = f"ORPHAN (server.ready points at {tracked}, not this)"
                p["reap"] = True
            else:
                p["verdict"] = "the live warm server (keep)"
                p["reap"] = False
        elif kind == "live_scene":
            p["verdict"] = (f"streaming at {sm}% SM -- burns the GPU whether or not a "
                            "client is attached")
            p["reap"] = False
        elif kind == "other":
            p["verdict"] = "unrecognised -- not touching it"
            p["reap"] = False
        else:
            p["verdict"] = "expected workload"
            p["reap"] = False
        if p.get("reap"):
            waste += p["mib"]
    return snap["procs"], waste


def report(snap: dict) -> int:
    procs, waste = analyse(snap)
    used, total, util = snap["used_mib"], snap["total_mib"], snap["util"]
    bar = int(30 * used / max(1, total))
    print(f"\nGPU  [{'#' * bar}{'.' * (30 - bar)}]  {used/1024:.1f}/{total/1024:.1f} GiB"
          f"   {util}% util   {len(procs)} process(es)\n")
    print(f"  {'PID':>8} {'MEM':>9} {'SM':>5} {'AGE':>12}  WHAT")
    for p in procs:
        name = os.path.basename(p["cmd"].split()[-1] if p["cmd"] else "?")
        for k, _ in KINDS:
            if k in p["cmd"]:
                name = k
                break
        flag = "  <-- REAPABLE" if p.get("reap") else ""
        print(f"  {p['pid']:>8} {p['mib']/1024:>7.1f}Gi {p['sm']:>4}% {p['age']:>12}  "
              f"{name}{flag}")
        print(f"  {'':>8} {'':>9} {'':>5} {'':>12}    {p['verdict']}")
    if waste:
        print(f"\n  RECLAIMABLE NOW: {waste/1024:.1f} GiB in orphaned render servers "
              f"({waste * 100 // max(1, total)}% of the card). "
              f"Free it with:  python scripts/tools/gpu_watch.py --reap")
    else:
        print("\n  No orphaned render servers.")
    idle_hold = sum(p["mib"] for p in procs if p["sm"] == 0 and not p.get("reap"))
    if idle_hold:
        print(f"  (another {idle_hold/1024:.1f} GiB is held at 0% SM -- parked, not "
              f"necessarily wasted: a warm server is idle *by design*.)")
    return waste


def reap(snap: dict, dry: bool) -> None:
    procs, _ = analyse(snap)
    targets = [p for p in procs if p.get("reap")]
    if not targets:
        print("nothing to reap.")
        return
    for p in targets:
        if dry:
            print(f"  would kill {p['pid']} ({p['mib']/1024:.1f}Gi) -- {p['verdict']}")
            continue
        try:
            os.kill(p["pid"], signal.SIGTERM)
            print(f"  SIGTERM -> {p['pid']}  (freeing {p['mib']/1024:.1f}Gi)")
        except ProcessLookupError:
            print(f"  {p['pid']} already gone")
        except PermissionError:
            print(f"  {p['pid']} not ours to kill (different user)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--watch", type=float, default=0,
                   help="sample every N seconds instead of once, appending to --csv")
    p.add_argument("--csv", default="logs/gpu_watch.csv",
                   help="where --watch accumulates samples (for plotting later)")
    p.add_argument("--reap", action="store_true",
                   help="SIGTERM every orphaned render server (never anything else)")
    p.add_argument("--dry_run", action="store_true", help="with --reap, only say what it would do")
    a = p.parse_args()

    if a.reap:
        reap(sample(), a.dry_run)
        return
    if not a.watch:
        report(sample())
        return

    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
    new = not os.path.exists(a.csv)
    print(f"sampling every {a.watch}s -> {a.csv}   (Ctrl-C to stop)")
    with open(a.csv, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["ts", "used_mib", "total_mib", "util", "n_procs",
                        "orphan_mib", "pid", "kind", "mib", "sm"])
        while True:
            snap = sample()
            procs, waste = analyse(snap)
            ts = time.time()
            for pr in procs:
                w.writerow([f"{ts:.0f}", snap["used_mib"], snap["total_mib"], snap["util"],
                            len(procs), waste, pr["pid"], pr["kind"], pr["mib"], pr["sm"]])
            fh.flush()
            print(f"  {time.strftime('%H:%M:%S')}  {snap['used_mib']/1024:5.1f}Gi  "
                  f"{snap['util']:3d}%  {len(procs)} procs  orphaned={waste/1024:.1f}Gi")
            time.sleep(a.watch)


if __name__ == "__main__":
    sys.exit(main())
