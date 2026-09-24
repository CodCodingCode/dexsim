"""Bimanual MacBook-keyboard scene: two Shadow Hands over a 16-inch MacBook
Pro (M3 Max) keyboard, for watching / scripting how the robot types.

Differences from the piano scene (``dexsim.mjcf.scene``), on purpose:

  * The hands ride a 3-axis GANTRY (world X/Y/Z slide joints with position
    servos, ``{L,R}_gantry_{x,y,z}``) instead of the piano's single Y rail.
    A keyboard needs front-back reach (6 rows over 11 cm) and a vertical
    press stroke; the gantry gives a scripted typist all three without any
    hand IK. The hand bodies are gravity-compensated so the Z servo holds
    a hover height without sag.
  * Home position: the same 🔒 pose G finger/wrist pose as the piano
    (imported from ``PianoMjEnvCfg`` -- not redefined here) with the mount
    calibrated numerically so the LEFT middle fingertip hovers over ``d``
    and the RIGHT middle fingertip over ``k``, ``cfg.hover`` above the caps.
    Long-finger tips then land ~1 key apart, i.e. roughly on a/s/d/f and
    j/k/l/; -- the touch-typing home row.

The keyboard/laptop geometry lives in :mod:`dexsim.mjcf.keyboard`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .keyboard import KEY_NAMES, BASE_H, BODY_D, add_macbook
from .scene import MOUNT_QUAT, _palm_offset, _add_lookat_camera
from .shadow_hand import load_hand_spec, MENAGERIE_DIR, FINGERTIP_SITES

_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KEYBOARD_SCENE_XML = _ROOT / "assets" / "mj" / "keyboard_scene.xml"

TABLE_TOP_Z = 0.712                 # same table as the piano scene
HAND_PREFIXES = ("L_", "R_")
FINGERS = ("th", "ff", "mf", "rf", "lf")


def _default_pose(side: str) -> dict:
    """The 🔒 locked pose G finger/wrist pose, single-sourced from the env cfg."""
    from dexsim.tasks.piano_mj.piano_mj_env_cfg import PianoMjEnvCfg
    cfg = PianoMjEnvCfg()
    return dict(cfg.left_ready_pose if side == "left" else cfg.right_ready_pose)


@dataclass
class KeyboardSceneCfg:
    sim_dt: float = 0.002               # scissor keys are stiff+light: 500 Hz
    # laptop: centre of the palm-rest deck top, in world; +X toward the typist
    laptop_pos: tuple = (0.62, 0.0, TABLE_TOP_Z + BASE_H)
    laptop_quat: tuple = (1.0, 0.0, 0.0, 0.0)
    hover: float = 0.020                # fingertip height above cap tops at home
    left_home_key: str = "d"            # under the LEFT middle fingertip
    right_home_key: str = "k"           # under the RIGHT middle fingertip
    gantry_xy_limit: float = 0.20       # m, each way
    gantry_z_range: tuple = (-0.08, 0.06)
    gantry_stiffness: float = 3000.0
    gantry_damping: float = 200.0
    gantry_force: float = 800.0
    left_ready_pose: dict = field(default_factory=lambda: _default_pose("left"))
    right_ready_pose: dict = field(default_factory=lambda: _default_pose("right"))


def ready_qpos(model: mujoco.MjModel, cfg: KeyboardSceneCfg) -> np.ndarray:
    """qpos holding the ready pose (same first-matching-regex rule as the
    piano env's ``_build_ready_state``). Gantry joints stay at 0."""
    q = np.zeros(model.nq)
    poses = {"L_": cfg.left_ready_pose, "R_": cfg.right_ready_pose}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        pose = poses.get(name[:2])
        if pose is None or "gantry" in name:
            continue
        suffix = name[2:]
        for pattern, value in pose.items():
            if re.fullmatch(pattern, suffix):
                q[model.jnt_qposadr[j]] = float(value)
                break
    return q


def ready_ctrl(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    """Actuator ctrl that holds ``qpos`` (transmission lengths, clipped)."""
    d = mujoco.MjData(model)
    d.qpos[:] = qpos
    mujoco.mj_forward(model, d)
    return np.clip(d.actuator_length.copy(), model.actuator_ctrlrange[:, 0],
                   model.actuator_ctrlrange[:, 1])


def build_keyboard_scene_spec(cfg: KeyboardSceneCfg | None = None) -> mujoco.MjSpec:
    """Two-pass self-calibrating build (like the piano scene): probe once,
    measure where the middle fingertips land at the ready pose, then rebuild
    with the mounts shifted so they sit ``cfg.hover`` above the home keys."""
    cfg = cfg or KeyboardSceneCfg()
    zero = {p: np.zeros(3) for p in HAND_PREFIXES}
    spec = _build(cfg, mount_shift=zero)
    model = spec.copy().compile()
    data = mujoco.MjData(model)
    data.qpos[:] = ready_qpos(model, cfg)
    mujoco.mj_forward(model, data)
    shift = {}
    for p, home in (("L_", cfg.left_home_key), ("R_", cfg.right_home_key)):
        tip = data.site_xpos[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, p + "robot0_mftip")]
        key = data.site_xpos[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"key_site_{home}")]
        target = key + np.array([0.0, 0.0, cfg.hover])
        shift[p] = target - tip
    return _build(cfg, mount_shift=shift)


def _build(cfg: KeyboardSceneCfg, mount_shift: dict) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.modelname = "dexsim_macbook_keyboard_bimanual"
    spec.meshdir = str(MENAGERIE_DIR / "assets")
    spec.option.timestep = cfg.sim_dt
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.map.znear = 0.005          # close-up cameras

    # --- world dressing -----------------------------------------------------
    spec.worldbody.add_light(pos=[1.0, -0.6, 2.2], dir=[-0.25, 0.25, -1.0],
                             diffuse=[0.85, 0.85, 0.85], castshadow=True)
    # key light straight over the keyboard so the printed legends read
    spec.worldbody.add_light(pos=[cfg.laptop_pos[0] + 0.1, 0.0, cfg.laptop_pos[2] + 0.9],
                             dir=[-0.1, 0.0, -1.0], diffuse=[0.6, 0.6, 0.6],
                             specular=[0.1, 0.1, 0.1], castshadow=False)
    spec.worldbody.add_light(pos=[-0.3, 0.8, 1.8], dir=[0.4, -0.5, -1.0],
                             diffuse=[0.35, 0.35, 0.4], castshadow=False)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[5.0, 5.0, 0.1], rgba=[0.28, 0.29, 0.31, 1.0])
    # desk: the laptop sits 1 cm from its FRONT edge so the Shadow Hand's
    # fat forearm (13 cm dia., level with the palm) hangs over the edge
    # instead of resting on the desk top and lifting the hand.
    front_edge = cfg.laptop_pos[0] + BODY_D / 2 + 0.01
    spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
                            pos=[front_edge - 0.4, 0.0, TABLE_TOP_Z / 2],
                            size=[0.4, 0.8, TABLE_TOP_Z / 2],
                            rgba=[0.35, 0.27, 0.22, 1.0])

    # --- MacBook Pro --------------------------------------------------------
    add_macbook(spec, cfg.laptop_pos, cfg.laptop_quat)

    # --- two gantry-mounted hands -------------------------------------------
    # nominal mount: palm 8 cm above the deck, 12 cm apart, before calibration
    lp = np.asarray(cfg.laptop_pos)
    base_z = lp[2] + 0.08
    for side, prefix, y0 in (("left", "L_", -0.10), ("right", "R_", 0.10)):
        base = np.array([lp[0] + 0.05, lp[1] + y0, base_z]) + mount_shift[prefix]
        mount = spec.worldbody.add_body(name=f"{prefix}mount", pos=base.tolist())
        mount.add_geom(name=f"{prefix}carriage", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[0.03, 0.04, 0.015], rgba=[0.2, 0.2, 0.22, 1.0],
                       mass=0.5, contype=0, conaffinity=0)
        lim = cfg.gantry_xy_limit
        for axis, tag, rng in (((1, 0, 0), "x", (-lim, lim)),
                               ((0, 1, 0), "y", (-lim, lim)),
                               ((0, 0, 1), "z", cfg.gantry_z_range)):
            jname = f"{prefix}gantry_{tag}"
            mount.add_joint(name=jname, type=mujoco.mjtJoint.mjJNT_SLIDE,
                            axis=list(map(float, axis)), range=list(rng),
                            damping=2.0, armature=0.01, limited=True)
            act = spec.add_actuator(name=f"{prefix}A_gantry_{tag}",
                                    trntype=mujoco.mjtTrn.mjTRN_JOINT,
                                    target=jname)
            act.gainprm[0] = cfg.gantry_stiffness
            act.biasprm[1] = -cfg.gantry_stiffness
            act.biasprm[2] = -cfg.gantry_damping
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.ctrlrange = list(rng)
            act.forcerange = [-cfg.gantry_force, cfg.gantry_force]
            act.ctrllimited = True

        hand = load_hand_spec(side)
        for b in hand.bodies:                 # gantry-held: no gravity sag
            b.gravcomp = 1.0
            if b.name == "robot0_forearm":
                # The E3M5 forearm is a 13 cm cylinder; with the palms only
                # ~10 cm apart over the home row the two forearms would jam
                # against each other (and the desk). It never touches keys,
                # so it is visual-only here (a real typist's forearms angle
                # outward and are far thinner).
                for g in b.geoms:
                    g.contype = 0
                    g.conaffinity = 0
        off = _palm_offset(hand)
        frame = mount.add_frame(pos=[off[0], -off[1], off[2]],
                                quat=list(MOUNT_QUAT))
        spec.attach(hand, prefix=prefix, frame=frame)

    # --- cameras ------------------------------------------------------------
    kb = lp + np.array([-0.045, 0.0, 0.0])          # keyboard well centre
    _add_lookat_camera(spec, "main", eye=kb + (0.75, -0.55, 0.55), target=kb + (0, 0, 0.04))
    _add_lookat_camera(spec, "front", eye=kb + (0.55, 0.0, 0.30), target=kb + (0, 0, 0.03))
    _add_lookat_camera(spec, "top", eye=kb + (0.0, 0.0, 0.75), target=kb,
                       up_hint=(-1.0, 0.0, 0.0))
    # "close": front-left, high enough that the near carriage doesn't occlude
    _add_lookat_camera(spec, "close", eye=kb + (0.34, -0.36, 0.30), target=kb + (0.02, 0.0, 0.02))
    # "keys": tight on the keyboard from the typist's left, legends readable
    _add_lookat_camera(spec, "keys", eye=kb + (0.20, -0.16, 0.19), target=kb + (-0.02, 0.0, 0.01))
    return spec


def compile_keyboard_scene(cfg: KeyboardSceneCfg | None = None) -> mujoco.MjModel:
    return build_keyboard_scene_spec(cfg).compile()


def save_keyboard_scene_xml(cfg: KeyboardSceneCfg | None = None,
                            out_path: str | Path = DEFAULT_KEYBOARD_SCENE_XML) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    spec = build_keyboard_scene_spec(cfg)
    spec.compile()
    out.write_text(spec.to_xml())
    return out


# --- convenience lookups for demo / env code --------------------------------
def key_site_ids(model: mujoco.MjModel) -> np.ndarray:
    return np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                       f"key_site_{n}") for n in KEY_NAMES])


def key_qpos_adr(model: mujoco.MjModel) -> np.ndarray:
    return np.array([model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{n}")] for n in KEY_NAMES])


def fingertip_site_ids(model: mujoco.MjModel) -> dict[tuple[str, str], int]:
    """{("L"|"R", finger): site id} for the 10 fingertips."""
    out = {}
    for p in HAND_PREFIXES:
        for f, s in zip(FINGERS, FINGERTIP_SITES):
            out[(p[0], f)] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, p + s)
    return out
