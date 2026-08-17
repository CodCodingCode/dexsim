"""Bimanual Shadow Hand piano IK in the browser (PyRoki + Viser).

Replaces PyRoki's stock single-arm demo with OUR embodiment: the two armless
rail-mounted Shadow Hands from the piano task, at the approved base placement,
above the real 88-key keyboard. Drag a fingertip gizmo (or snap a finger onto a
key) and PyRoki solves the 25-DoF hand -- rail included -- in milliseconds.

The kinematics come from ``assets/urdf/shadow_hand_*/`` which is generated from
the very USD the sim loads, so a solution here is valid in Isaac
(``scripts/smoke/check_urdf_fk.py`` proves the two agree to 0.000 mm).

Runs in the pyroki venv -- no Isaac, no GPU:

    .venv-pyroki/bin/python scripts/tools/piano_ik_viser.py
    # from your laptop:  ssh -L 8013:localhost:8013 piano  ->  http://localhost:8013

If the URDFs are missing, generate them first (Isaac venv, ~1 s, no app boot):

    source env.sh && python scripts/tools/export_hand_urdf.py
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source"))          # dexsim.piano.geometry (numpy only)

import jax.numpy as jnp                            # noqa: E402
import jax_dataclasses as jdc                      # noqa: E402
import jaxlie                                      # noqa: E402
import jaxls                                       # noqa: E402
import pyroki as pk                                # noqa: E402
import viser                                       # noqa: E402
import yourdfpy                                    # noqa: E402
from viser.extras import ViserUrdf                 # noqa: E402

from dexsim.piano import geometry                  # noqa: E402

CFG = REPO / "source" / "dexsim" / "tasks" / "piano" / "piano_env_cfg.py"
URDF_DIR = REPO / "assets" / "urdf"
PORT = 8013

# The "_tip" frames sit at the far end of each distal mesh -- the pad that
# actually touches a key, ~30 mm past the distal link's own origin. Targeting
# the origin makes reachable keys look 30 mm out of reach.
FINGERTIPS = ["robot0_thdistal_tip", "robot0_ffdistal_tip", "robot0_mfdistal_tip",
              "robot0_rfdistal_tip", "robot0_lfdistal_tip"]
FINGER_LABELS = ["thumb", "index", "middle", "ring", "little"]
FINGER_COLORS = [(255, 205, 80), (90, 170, 255), (120, 235, 130),
                 (235, 130, 235), (255, 120, 110)]


# ------------------------------------------------------------------ pose I/O
def read_pose_fields() -> dict:
    """The locked base/piano placement, read straight out of piano_env_cfg.py."""
    src = CFG.read_text()
    fields = ("piano_pos", "piano_rot", "left_base_pos", "left_base_rot",
              "right_base_pos", "right_base_rot")
    out = {}
    for f in fields:
        m = re.search(rf"{f}\s*=\s*(\([^)]*\))", src)
        if m is None:
            raise SystemExit(f"could not find {f} in {CFG}")
        out[f] = np.array(eval(m.group(1), {"__builtins__": {}}, {}), dtype=float)  # noqa: S307
    return out


def read_ready_pose(side: str) -> dict:
    src = CFG.read_text()
    m = re.search(rf"{side}_ready_pose\s*=\s*\{{(.*?)\}}", src, re.DOTALL)
    if m is None:
        return {}
    return {n: float(v) for n, v in re.findall(r'"([^"]+)"\s*:\s*([-\d.eE+]+)', m.group(1))}


# ------------------------------------------------------------ quaternion math
def quat_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_mul(a, b) -> np.ndarray:
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                     w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                     w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                     w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1])


def quat_conj(q) -> np.ndarray:
    return np.asarray(q, float) * np.array([1.0, -1.0, -1.0, -1.0])


# ------------------------------------------------------------------ IK solver
@jdc.jit
def _solve(robot: pk.Robot, target_wxyz: jnp.ndarray, target_pos: jnp.ndarray,
           link_indices: jnp.ndarray, rest: jnp.ndarray,
           pos_weight: jnp.ndarray, ori_weight: jnp.ndarray,
           rest_weight: jnp.ndarray) -> jnp.ndarray:
    """Least-squares IK: fingertips to targets, biased toward the ready pose."""
    JointVar = robot.joint_var_cls
    target = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_wxyz), target_pos)
    costs = [
        pk.costs.pose_cost_analytic_jac(
            jax_tree_broadcast(robot), JointVar(jnp.full(target.get_batch_axes(), 0)),
            target, link_indices, pos_weight=pos_weight, ori_weight=ori_weight),
        # Fingers are highly redundant against 5 tip targets; without a rest bias
        # the solver happily curls them into anatomically silly configurations.
        pk.costs.rest_cost(JointVar(0), rest_pose=rest, weight=rest_weight),
        pk.costs.limit_cost(robot, JointVar(0), weight=100.0),
        pk.costs.limit_constraint(robot, JointVar(0)),
    ]
    return (jaxls.LeastSquaresProblem(costs=costs, variables=[JointVar(0)])
            .analyze()
            .solve(verbose=False, linear_solver="dense_cholesky",
                   trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0))
            )[JointVar(0)]


def jax_tree_broadcast(robot):
    import jax
    return jax.tree.map(lambda x: x[None], robot)


class Hand:
    """One rail-mounted Shadow Hand: URDF, pyroki model, viser visual, targets."""

    def __init__(self, server: viser.ViserServer, side: str, base_pos, base_quat,
                 show_gizmos: bool):
        urdf_path = URDF_DIR / f"shadow_hand_{side}" / f"shadow_hand_{side}.urdf"
        if not urdf_path.exists():
            raise SystemExit(f"missing {urdf_path}\nrun: source env.sh && "
                             "python scripts/tools/export_hand_urdf.py")
        self.side = side
        self.urdf = yourdfpy.URDF.load(str(urdf_path))
        self.robot = pk.Robot.from_urdf(self.urdf)
        self.base_pos = np.asarray(base_pos, float)
        self.base_quat = np.asarray(base_quat, float)

        self.root = f"/{side}_hand"
        server.scene.add_frame(self.root, position=tuple(self.base_pos),
                               wxyz=tuple(self.base_quat), show_axes=False)
        self.vis = ViserUrdf(server, self.urdf, root_node_name=self.root)

        # start from the sim's ready pose so the demo opens in a familiar shape
        ready = read_ready_pose(side)
        self.rest = np.array([ready.get(n, 0.0) for n in self.robot.joints.actuated_names],
                             dtype=float)
        self.cfg = self.rest.copy()
        self.vis.update_cfg(self._cfg_dict())

        self.link_indices = np.array([self.robot.links.names.index(n) for n in FINGERTIPS],
                                     dtype=np.int32)
        self.targets = []
        for tip, label, color in zip(FINGERTIPS, FINGER_LABELS, FINGER_COLORS):
            pos_w, quat_w = self.tip_world(tip)
            tc = server.scene.add_transform_controls(
                f"{self.root}_target/{label}", scale=0.045, position=tuple(pos_w),
                wxyz=tuple(quat_w), visible=show_gizmos, opacity=0.7)
            server.scene.add_icosphere(f"{self.root}_target/{label}/dot", radius=0.006,
                                       color=color)
            self.targets.append(tc)

    # -- frames ------------------------------------------------------------
    def _cfg_dict(self) -> dict:
        return dict(zip(self.robot.joints.actuated_names, self.cfg))

    def tip_world(self, link: str) -> tuple[np.ndarray, np.ndarray]:
        """Fingertip pose in WORLD, from the current configuration."""
        self.urdf.update_cfg(self._cfg_dict())
        local = self.urdf.get_transform(link, self.urdf.base_link)
        R = quat_matrix(self.base_quat)
        pos = self.base_pos + R @ local[:3, 3]
        return pos, quat_mul(self.base_quat, matrix_quat(local[:3, :3]))

    def solve(self, pos_weight: float, ori_weight: float, rest_weight: float) -> float:
        """Pull world-frame gizmo poses into the root frame, solve, report the miss.

        Returns the largest fingertip-to-target distance in millimetres. That
        number is the honest reachability signal: these hands have no arm, so a
        gizmo dragged off the rail's plane simply cannot be reached and the
        residual stays large no matter how long the solver runs.
        """
        Rt = quat_matrix(self.base_quat).T
        pos = np.stack([Rt @ (np.asarray(t.position) - self.base_pos) for t in self.targets])
        wxyz = np.stack([quat_mul(quat_conj(self.base_quat), np.asarray(t.wxyz))
                         for t in self.targets])
        self.cfg = np.asarray(_solve(
            self.robot, jnp.array(wxyz), jnp.array(pos), jnp.array(self.link_indices),
            jnp.array(self.rest), jnp.array(pos_weight), jnp.array(ori_weight),
            jnp.array(rest_weight)))
        self.vis.update_cfg(self._cfg_dict())
        self.urdf.update_cfg(self._cfg_dict())
        miss = [np.linalg.norm(self.urdf.get_transform(tip, self.urdf.base_link)[:3, 3] - want)
                for tip, want in zip(FINGERTIPS, pos)]
        return float(max(miss) * 1000.0)

    def snap_to_key(self, finger: int, key: int, depth: float) -> None:
        """Park one fingertip just above (or into) a key's top face.

        X (toward/away from the player) is deliberately left alone: these hands
        ride a Y rail with no arm behind them, so the depth into the keyboard is
        whatever the base placement gives. Snapping X to the key's centre would
        just hand the solver an unreachable target ~80 mm away.
        """
        top = geometry.key_local_top_positions()[key]
        world = PIANO_R @ np.asarray(top, float) + PIANO_T
        tc = self.targets[finger]
        tc.position = (float(tc.position[0]), float(world[1]), float(world[2] + depth))

    def reset(self) -> None:
        self.cfg = self.rest.copy()
        self.vis.update_cfg(self._cfg_dict())
        for tip, tc in zip(FINGERTIPS, self.targets):
            pos_w, quat_w = self.tip_world(tip)
            tc.position = tuple(float(v) for v in pos_w)
            tc.wxyz = tuple(float(v) for v in quat_w)


def matrix_quat(m: np.ndarray) -> np.ndarray:
    """3x3 rotation -> (w, x, y, z)."""
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                         (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                         (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                         0.25 * s, (m[1, 2] + m[2, 1]) / s])
    s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    return np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                     (m[1, 2] + m[2, 1]) / s, 0.25 * s])


PIANO_R = np.eye(3)
PIANO_T = np.zeros(3)


def build_piano(server: viser.ViserServer, pos, quat) -> None:
    """The 88-key board, drawn from the same geometry module the sim uses."""
    global PIANO_R, PIANO_T
    PIANO_R, PIANO_T = quat_matrix(quat), np.asarray(pos, float)
    server.scene.add_frame("/piano", position=tuple(pos), wxyz=tuple(quat),
                           show_axes=False)
    tops = geometry.key_local_top_positions()
    for i, (cx, cy, cz_top) in enumerate(tops):
        black = bool(geometry.KEY_IS_BLACK[i])
        half_h = float(geometry.KEY_HALF_H[i])
        server.scene.add_box(
            f"/piano/key_{i:02d}",
            color=(35, 35, 38) if black else (238, 238, 233),
            dimensions=(geometry.BLACK_L if black else geometry.WHITE_L,
                        geometry.BLACK_W if black else geometry.WHITE_W,
                        2 * half_h),
            position=(float(cx), float(cy), float(cz_top) - half_h))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=PORT)
    a = p.parse_args()

    vals = read_pose_fields()
    server = viser.ViserServer(host="0.0.0.0", port=a.port, label="dexsim piano IK")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=4.0, height=4.0, cell_size=0.1, plane="xy",
                          position=(0.6, 0.0, 0.0))
    server.scene.add_box("/table", color=(150, 130, 110), dimensions=(0.6, 1.5, 0.02),
                         position=(0.6, 0.0, 0.71))
    build_piano(server, vals["piano_pos"], vals["piano_rot"])

    gui_show = server.gui.add_checkbox("Show fingertip gizmos", True)
    gui_solve = server.gui.add_checkbox("Solve IK", True)
    gui_pos_w = server.gui.add_slider("Position weight", 1.0, 200.0, 1.0, 50.0)
    gui_ori_w = server.gui.add_slider("Orientation weight", 0.0, 20.0, 0.1, 1.0)
    gui_rest_w = server.gui.add_slider("Rest-pose bias", 0.0, 1.0, 0.01, 0.05)
    gui_ms = server.gui.add_number("Solve (ms)", 0.0, disabled=True)
    gui_miss = server.gui.add_number("Worst tip miss (mm)", 0.0, disabled=True)

    hands = {side: Hand(server, side, vals[f"{side}_base_pos"], vals[f"{side}_base_rot"],
                        gui_show.value)
             for side in ("left", "right")}

    with server.gui.add_folder("Snap a finger to a key"):
        gui_hand = server.gui.add_dropdown("Hand", ("left", "right"))
        gui_finger = server.gui.add_dropdown("Finger", tuple(FINGER_LABELS))
        gui_key = server.gui.add_slider("Key (0-87)", 0, 87, 1, 44)
        gui_depth = server.gui.add_slider("Height above key (m)", -0.01, 0.06, 0.002,
                                          float(geometry.HOVER_CLEARANCE))
        gui_snap = server.gui.add_button("Snap")
    gui_reset = server.gui.add_button("Reset to ready pose")

    @gui_snap.on_click
    def _(_) -> None:
        hands[gui_hand.value].snap_to_key(FINGER_LABELS.index(gui_finger.value),
                                          int(gui_key.value), float(gui_depth.value))

    @gui_reset.on_click
    def _(_) -> None:
        for hand in hands.values():
            hand.reset()

    @gui_show.on_update
    def _(_) -> None:
        for hand in hands.values():
            for tc in hand.targets:
                tc.visible = gui_show.value

    print(f"[piano-ik] http://localhost:{a.port}  "
          f"({sum(h.robot.joints.num_actuated_joints for h in hands.values())} DoF total)")
    while True:
        if gui_solve.value:
            start = time.time()
            miss = max(hand.solve(gui_pos_w.value, gui_ori_w.value, gui_rest_w.value)
                       for hand in hands.values())
            gui_ms.value = round(0.9 * gui_ms.value + 0.1 * (time.time() - start) * 1e3, 2)
            gui_miss.value = round(miss, 2)
        else:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
