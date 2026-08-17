"""Interactive scene-placement GUI (Viser, browser-based -- no Isaac, no VNC).

Gizmos place each hand + the piano; per-joint sliders sculpt the finger pose.
The skeleton preview is posed by the REAL simulator (warm render server), so
it's exact kinematics with NO physics -- nothing droops while you edit.

Finding your way around the joints:
  * sliders have plain-English names ("index knuckle curl" not "FFJ2");
  * floating labels on the 3D hand name each fingertip;
  * touching a slider flashes the finger it controls YELLOW in the 3D view.

Buttons write back into ``piano_env_cfg.py``:
  * base poses  -> left/right_base_pos / _rot
  * finger pose -> left/right_ready_pose (the sim starts here and the position
    actuators HOLD it once physics runs)

Run on the sim box:
    source env.sh && python scripts/tools/place_scene_viser.py
Then from your laptop:
    ssh -L 8012:localhost:8012 piano   ->  open http://localhost:8012
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import viser

from dexsim.piano import geometry

REPO = Path(__file__).resolve().parents[2]
CFG = REPO / "source" / "dexsim" / "tasks" / "piano" / "piano_env_cfg.py"
FK_JSON = REPO / "logs" / "viser_fk.json"
PORT = 8012

POSE_FIELDS = ("piano_pos", "piano_rot", "left_base_pos", "left_base_rot",
               "right_base_pos", "right_base_rot")

# (joint, lo, hi, human label, highlight-prefix). Order = display order.
# FF=index MF=middle RF=ring LF=pinky TH=thumb. railJoint excluded (trained).
JOINTS = [
    ("robot0_WRJ0", -0.698, 0.489, "WRIST pitch  (raise = hand tilts UP)", "wr"),
    ("robot0_WRJ1", -0.489, 0.140, "WRIST side tilt", "wr"),
    ("robot0_FFJ3", -0.349, 0.349, "index spread (side-to-side)", "ff"),
    ("robot0_FFJ2", 0.0, 1.571, "index knuckle curl", "ff"),
    ("robot0_FFJ1", 0.0, 1.571, "index mid curl", "ff"),
    ("robot0_FFJ0", 0.0, 1.571, "index tip curl", "ff"),
    ("robot0_MFJ3", -0.349, 0.349, "middle spread (side-to-side)", "mf"),
    ("robot0_MFJ2", 0.0, 1.571, "middle knuckle curl", "mf"),
    ("robot0_MFJ1", 0.0, 1.571, "middle mid curl", "mf"),
    ("robot0_MFJ0", 0.0, 1.571, "middle tip curl", "mf"),
    ("robot0_RFJ3", -0.349, 0.349, "ring spread (side-to-side)", "rf"),
    ("robot0_RFJ2", 0.0, 1.571, "ring knuckle curl", "rf"),
    ("robot0_RFJ1", 0.0, 1.571, "ring mid curl", "rf"),
    ("robot0_RFJ0", 0.0, 1.571, "ring tip curl", "rf"),
    ("robot0_LFJ4", 0.0, 0.785, "pinky palm-cup", "lf"),
    ("robot0_LFJ3", -0.349, 0.349, "pinky spread (side-to-side)", "lf"),
    ("robot0_LFJ2", 0.0, 1.571, "pinky knuckle curl", "lf"),
    ("robot0_LFJ1", 0.0, 1.571, "pinky mid curl", "lf"),
    ("robot0_LFJ0", 0.0, 1.571, "pinky tip curl", "lf"),
    ("robot0_THJ4", -1.047, 1.047, "thumb rotate (toward/away palm)", "th"),
    ("robot0_THJ3", 0.0, 1.222, "thumb lift", "th"),
    ("robot0_THJ2", -0.209, 0.209, "thumb mid", "th"),
    ("robot0_THJ1", -0.524, 0.524, "thumb bend", "th"),
    ("robot0_THJ0", -1.571, 0.0, "thumb tip curl (negative = curl)", "th"),
]
TIP_LABELS = {"robot0_thdistal": "thumb", "robot0_ffdistal": "index",
              "robot0_mfdistal": "middle", "robot0_rfdistal": "ring",
              "robot0_lfdistal": "pinky"}
HIGHLIGHT = (255, 225, 40)


# --------------------------------------------------------------- cfg I/O
def read_pose_fields() -> dict:
    src = CFG.read_text()
    vals: dict = {}
    for f in POSE_FIELDS:
        m = re.search(rf"{f}\s*=\s*(\([^)]*\))", src)
        vals[f] = tuple(eval(m.group(1), {"__builtins__": {}}, {}))  # noqa: S307
    return vals


def read_ready_joints(side: str) -> dict:
    src = CFG.read_text()
    m = re.search(rf"{side}_ready_pose\s*=\s*\{{(.*?)\}}", src, re.DOTALL)
    out = {name: 0.0 for name, *_ in JOINTS}
    for name, val in re.findall(r'"([^"]+)"\s*:\s*([-\d.eE]+)', m.group(1)):
        if name in out:
            out[name] = float(val)
    return out


def fmt(f: str, v: tuple) -> str:
    prec = 7 if f.endswith("_rot") else 4
    return "(" + ", ".join(f"{x:.{prec}f}" for x in v) + ")"


def write_pose_fields(vals: dict) -> str:
    src = CFG.read_text()
    for f in POSE_FIELDS:
        src = re.sub(rf"({f}\s*=\s*)\([^)]*\)", rf"\g<1>{fmt(f, vals[f])}", src)
    CFG.write_text(src)
    return "\n".join(f"{f} = {fmt(f, vals[f])}" for f in POSE_FIELDS)


def write_ready_poses(left_j: dict, right_j: dict) -> None:
    src = CFG.read_text()
    for side, jv in (("left", left_j), ("right", right_j)):
        body = '{\n        "railJoint": 0.0,\n'
        for name, *_ in JOINTS:
            body += f'        "{name}": {jv[name]:.4f},\n'
        body += "    }"
        src = re.sub(rf"({side}_ready_pose\s*=\s*)\{{.*?\}}",
                     lambda m: m.group(1) + body, src, flags=re.DOTALL)
    CFG.write_text(src)


# --------------------------------------------------- render-server FK query
def quat_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def query_skeleton(left_j: dict, right_j: dict) -> dict | None:
    kv = lambda d: ",".join(f"{k}={v:.4f}" for k, v in d.items())
    cmd = ["python", "scripts/render/render.py", "query", "--kind", "skeleton",
           "--left_joints", kv(left_j), "--right_joints", kv(right_j),
           "--out", str(FK_JSON)]
    try:
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, timeout=60)
        if r.returncode != 0:
            return None
        d = json.loads(FK_JSON.read_text())
    except Exception:
        return None
    vals = read_pose_fields()
    out = {}
    for side, key in (("left", "left_bodies"), ("right", "right_bodies")):
        base = np.array(vals[f"{side}_base_pos"])
        R = quat_mat(vals[f"{side}_base_rot"])
        out[side] = {name: (R.T @ (np.array(b["pos"]) - base)).tolist()
                     for name, b in d.get(key, {}).items()}
    return out


# ------------------------------------------------------------------- main
def main() -> None:
    vals = read_pose_fields()
    server = viser.ViserServer(host="0.0.0.0", port=PORT, label="dexsim placer")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=4.0, height=4.0, cell_size=0.1,
                          plane="xy", position=(0.6, 0.0, 0.0))
    server.scene.add_box("/table", color=(150, 130, 110),
                         dimensions=(0.6, 1.5, 0.02), position=(0.6, 0.0, 0.71))

    piano_tc = server.scene.add_transform_controls(
        "/piano", scale=0.35, position=vals["piano_pos"], wxyz=vals["piano_rot"])
    key_pos = geometry.key_local_top_positions()
    for i in range(len(key_pos)):
        black = bool(geometry.KEY_IS_BLACK[i])
        half_h = float(geometry.KEY_HALF_H[i])
        cx, cy, cz_top = (float(v) for v in key_pos[i])
        server.scene.add_box(
            f"/piano/key_{i:02d}",
            color=(40, 40, 40) if black else (235, 235, 230),
            dimensions=(geometry.BLACK_L if black else geometry.WHITE_L,
                        geometry.BLACK_W if black else geometry.WHITE_W,
                        2 * half_h),
            position=(cx, cy, cz_top - half_h))

    hands = {}
    for name, color, pk, rk in (("left", (80, 140, 255), "left_base_pos", "left_base_rot"),
                                ("right", (255, 110, 90), "right_base_pos", "right_base_rot")):
        tc = server.scene.add_transform_controls(
            f"/{name}_hand", scale=0.22, position=vals[pk], wxyz=vals[rk])
        server.scene.add_frame(f"/{name}_hand/axes", axes_length=0.09, axes_radius=0.003)
        hands[name] = {"tc": tc, "color": color, "bones": {}, "labels": {}}

    # ------------------------------------------------------------------ GUI
    server.gui.add_markdown(
        "**dexsim scene placer** — gizmos move/rotate each hand + piano. "
        "Sliders pose real Shadow Hand kinematics (no physics, nothing droops; "
        "~1 s refresh). **Touch a slider and the finger it controls flashes "
        "YELLOW** in 3D; fingertips carry name labels. Only the rail slide has "
        "no slider (the policy trains it).")
    pose_md = server.gui.add_markdown("")
    status_md = server.gui.add_markdown("*skeleton: querying server...*")

    sliders = {"left": {}, "right": {}}
    dirty = threading.Event()
    hl = {"prefix": None, "until": 0.0}          # active highlight + expiry time

    def set_highlight(prefix: str) -> None:
        hl["prefix"] = prefix
        hl["until"] = time.monotonic() + 2.5
        apply_colors()
        dirty.set()

    def apply_colors() -> None:
        for side in ("left", "right"):
            h = hands[side]
            for body, handle in h["bones"].items():
                short = body.replace("robot0_", "")
                lit = (hl["prefix"] is not None and time.monotonic() < hl["until"]
                       and (short.startswith(hl["prefix"])
                            or (hl["prefix"] == "wr" and short in ("palm", "wrist"))))
                if lit:
                    handle.color = HIGHLIGHT
                else:
                    tip = body.endswith("distal")
                    c = h["color"]
                    handle.color = tuple(min(255, int(v * 1.35)) for v in c) if tip else c

    def make_sliders(side: str) -> None:
        init = read_ready_joints(side)
        with server.gui.add_folder(f"{side.capitalize()} fingers",
                                   expand_by_default=(side == "left")):
            curl = server.gui.add_slider("CURL ALL fingers", min=0.0, max=1.571,
                                         step=0.01, initial_value=0.0)
            for name, lo, hi, label, prefix in JOINTS:
                s = server.gui.add_slider(label, min=lo, max=hi, step=0.01,
                                          initial_value=float(np.clip(init[name], lo, hi)))
                s.on_update(lambda _e, p=prefix: set_highlight(p))
                sliders[side][name] = s

            @curl.on_update
            def _(_e, side=side) -> None:
                v = curl.value
                for f in ("FF", "MF", "RF", "LF"):
                    for j in ("J2", "J1", "J0"):
                        sl = sliders[side][f"robot0_{f}{j}"]
                        sl.value = float(np.clip(v, sl.min, sl.max))
                dirty.set()

    make_sliders("left")
    make_sliders("right")
    lift_btn = server.gui.add_button("☝ LIFT all fingers (straighten + wrist up)")
    copy_btn = server.gui.add_button("⇆ Copy left fingers -> right")
    write_pose_btn = server.gui.add_button("💾 Write BASE poses (positions/rotations)")
    write_fing_btn = server.gui.add_button("💾 Write FINGER pose (sim starts + holds it)")
    reset_btn = server.gui.add_button("↩ Reset everything to cfg file")

    def slider_vals(side: str) -> dict:
        return {n: s.value for n, s in sliders[side].items()}

    def pull_pose() -> dict:
        vals["piano_pos"] = tuple(float(v) for v in piano_tc.position)
        vals["piano_rot"] = tuple(float(v) for v in piano_tc.wxyz)
        for side in ("left", "right"):
            tc = hands[side]["tc"]
            vals[f"{side}_base_pos"] = tuple(float(v) for v in tc.position)
            vals[f"{side}_base_rot"] = tuple(float(v) for v in tc.wxyz)
        return vals

    def refresh_pose_md(_=None) -> None:
        pull_pose()
        pose_md.content = ("```python\n"
                           + "\n".join(f"{f} = {fmt(f, vals[f])}" for f in POSE_FIELDS)
                           + "\n```")

    piano_tc.on_update(refresh_pose_md)
    hands["left"]["tc"].on_update(refresh_pose_md)
    hands["right"]["tc"].on_update(refresh_pose_md)
    refresh_pose_md()

    def draw_skeleton(skel: dict) -> None:
        for side in ("left", "right"):
            h = hands[side]
            for body, local in skel[side].items():
                tip = body.endswith("distal")
                if body not in h["bones"]:
                    c = h["color"]
                    col = tuple(min(255, int(v * 1.35)) for v in c) if tip else c
                    h["bones"][body] = server.scene.add_box(
                        f"/{side}_hand/skel/{body}", color=col,
                        dimensions=(0.018, 0.018, 0.018) if tip else (0.012, 0.012, 0.012),
                        position=tuple(local))
                else:
                    h["bones"][body].position = tuple(local)
                if body in TIP_LABELS:
                    lp = (local[0], local[1], local[2] - 0.03)
                    if body not in h["labels"]:
                        h["labels"][body] = server.scene.add_label(
                            f"/{side}_hand/lbl/{body}", TIP_LABELS[body], position=lp)
                    else:
                        h["labels"][body].position = lp
        apply_colors()

    def fk_worker() -> None:
        while True:
            dirty.wait()
            time.sleep(0.25)                     # debounce: coalesce slider drags
            dirty.clear()
            skel = query_skeleton(slider_vals("left"), slider_vals("right"))
            if skel is None:
                status_md.content = ("⚠ *render server not reachable — skeleton "
                                     "preview frozen (gizmos still work). Boot "
                                     "`scripts/render/render_server.py` and move a slider.*")
                continue
            draw_skeleton(skel)
            status_md.content = "*skeleton: live from simulator (no physics)*"
            if hl["prefix"] is not None and time.monotonic() >= hl["until"]:
                hl["prefix"] = None
                apply_colors()

    threading.Thread(target=fk_worker, daemon=True).start()
    dirty.set()                                   # initial skeleton fetch

    @lift_btn.on_click
    def _(_) -> None:
        for side in ("left", "right"):
            for name, s in sliders[side].items():
                if name == "robot0_WRJ0":
                    # just INSIDE the limit: a default sitting exactly on a joint
                    # limit makes Isaac Lab's _validate_cfg raise ("default
                    # positions out of the limits"), which is swallowed in some
                    # app configs and fatal in others (it killed live_scene.py).
                    s.value = s.max - 0.004            # wrist pitched fully up
                elif name == "robot0_WRJ1":
                    s.value = 0.13
                else:
                    s.value = float(np.clip(0.0, s.min, s.max))
        status_md.content = "*fingers straightened, wrists pitched up*"
        dirty.set()

    @copy_btn.on_click
    def _(_) -> None:
        for n, s in sliders["left"].items():
            sliders["right"][n].value = s.value
        dirty.set()

    @write_pose_btn.on_click
    def _(_) -> None:
        summary = write_pose_fields(pull_pose())
        status_md.content = f"✅ **base poses written**\n```\n{summary}\n```"
        print(f"[placer] wrote base poses:\n{summary}", flush=True)

    @write_fing_btn.on_click
    def _(_) -> None:
        write_ready_poses(slider_vals("left"), slider_vals("right"))
        status_md.content = ("✅ **finger pose written** — the sim starts here and "
                             "the actuators hold it once physics runs.")
        print("[placer] wrote finger ready poses", flush=True)

    @reset_btn.on_click
    def _(_) -> None:
        fresh = read_pose_fields()
        vals.update(fresh)
        piano_tc.position, piano_tc.wxyz = fresh["piano_pos"], fresh["piano_rot"]
        for side in ("left", "right"):
            hands[side]["tc"].position = fresh[f"{side}_base_pos"]
            hands[side]["tc"].wxyz = fresh[f"{side}_base_rot"]
            for n, v in read_ready_joints(side).items():
                s = sliders[side][n]
                s.value = float(np.clip(v, s.min, s.max))
        refresh_pose_md()
        dirty.set()

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (1.9, -1.3, 1.6)
        client.camera.look_at = (0.6, 0.0, 0.85)

    print(f"[placer] viser running on http://0.0.0.0:{PORT} "
          f"(tunnel: ssh -L {PORT}:localhost:{PORT} piano)", flush=True)
    server.sleep_forever()


if __name__ == "__main__":
    main()
