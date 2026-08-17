"""Compose the full bimanual piano scene (MuJoCo port).

Layout is identical to the Isaac env (``PianoMjEnvCfg`` carries the same
constants): the piano at ``piano_pos`` rotated 180° about Z, and two Shadow
Hands, each riding a world-Y prismatic rail (``railJoint``) -- the same
embodiment the Isaac slider USDs implement. The rail carriage sits at the
cfg base position and the hand is attached so that its PALM body lands
exactly on that point, palm-down with the fingers reaching toward the keys
(-X); the 🔒 locked ready pose (WRJ0=0.45 wrist tilt) then drops the
fingertips to hover a few cm above the key tops, pointing down.

The mount rotation is solved numerically from the hand model itself: at
qpos=0 the Menagerie hand lies flat with fingers along +X and palm facing +Z,
so a 180° rotation about Y gives fingers -X / palm down for both chiralities.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .piano import add_piano
from .shadow_hand import load_hand_spec, MENAGERIE_DIR

_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENE_XML = _ROOT / "assets" / "mj" / "piano_scene.xml"

# 180° about Y: hand-frame fingers (+X) -> world -X, palm (+Z) -> world down.
MOUNT_QUAT = (0.0, 0.0, 1.0, 0.0)
# Isaac rail: build_shadow_hand_sliders.py RAIL_LIMIT_M / PIANO_SHADOW_HAND cfg
RAIL_LIMIT = 0.12
RAIL_STIFFNESS = 1200.0
RAIL_DAMPING = 120.0
RAIL_FORCE = 500.0


def _palm_offset(hand: mujoco.MjSpec) -> np.ndarray:
    """Hand-frame position of the palm body at qpos=0 (via FK on a copy), so
    the attach frame can place the PALM at the cfg base position."""
    model = hand.copy().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_palm")
    if pid < 0:
        raise RuntimeError("hand spec has no robot0_palm body")
    return data.xpos[pid].copy()


def build_scene_spec(cfg) -> mujoco.MjSpec:
    """Build the composed scene from a ``PianoMjEnvCfg``-like object.

    Two-pass, self-calibrating mount: after a probe build, the hands are
    shifted along world X so that (at the 🔒 locked ready pose) the long
    fingertips land over the KEY PRESS LINE -- the center of the white keys'
    playable surface, ~50% leverage on the hinge. Without this the Menagerie
    hand (whose palm->tip reach differs from the Isaac USD) presses too close
    to the hinge and can never rotate a key past the sound angle.
    """
    spec = _build_scene_spec(cfg, tip_shift=0.0)
    shift = _measure_tip_shift(spec, cfg)
    if abs(shift) > 1e-4:
        spec = _build_scene_spec(cfg, tip_shift=shift)
    return spec


def _measure_tip_shift(spec: mujoco.MjSpec, cfg) -> float:
    """World-X offset from the (ready-pose) ff/mf/rf fingertips to the white
    keys' press line. Positive = hands must move toward the player (-X…) --
    the returned value is subtracted from the attach frame X."""
    model = spec.copy().compile()
    data = mujoco.MjData(model)
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if name.endswith("robot0_WRJ0"):
            data.qpos[model.jnt_qposadr[j]] = 0.45     # locked ready tilt
        elif name.endswith("robot0_WRJ1"):
            data.qpos[model.jnt_qposadr[j]] = 0.13
    mujoco.mj_forward(model, data)
    tips_x = []
    for p in ("L_", "R_"):
        for s in ("robot0_fftip", "robot0_mftip", "robot0_rftip"):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, p + s)
            tips_x.append(data.site_xpos[i][0])
    key0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "key_site_0")
    return float(np.mean(tips_x) - data.site_xpos[key0][0])


def _build_scene_spec(cfg, tip_shift: float) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.modelname = "dexsim_piano_bimanual"
    spec.meshdir = str(MENAGERIE_DIR / "assets")
    spec.option.timestep = cfg.sim_dt
    # the Menagerie hand is tuned with the elliptic cone + high impratio
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    # implicitfast handles the stiff, damped key springs + tendons robustly
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # offscreen framebuffer big enough for 1080p video renders
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080

    # --- world dressing -----------------------------------------------------
    spec.worldbody.add_light(pos=[1.2, -0.8, 2.5], dir=[-0.3, 0.3, -1.0],
                             diffuse=[0.9, 0.9, 0.9], castshadow=True)
    spec.worldbody.add_light(pos=[-0.5, 1.0, 2.0], dir=[0.4, -0.5, -1.0],
                             diffuse=[0.35, 0.35, 0.4], castshadow=False)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[5.0, 5.0, 0.1], rgba=[0.28, 0.29, 0.31, 1.0])
    # table under the piano (top ~0.712, the "table top ~0.72" of the Isaac rig)
    spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
                            pos=[0.7, 0.0, 0.356], size=[0.4, 0.8, 0.356],
                            rgba=[0.35, 0.27, 0.22, 1.0])

    # --- piano --------------------------------------------------------------
    add_piano(spec, cfg.piano_pos, cfg.piano_rot,
              key_damping=getattr(cfg, "key_damping", 0.0))

    # --- two rail-mounted hands --------------------------------------------
    for side, prefix, base in (("left", "L_", cfg.left_base_pos),
                               ("right", "R_", cfg.right_base_pos)):
        base = (base[0], base[1], cfg.hand_fixed_z)
        mount = spec.worldbody.add_body(name=f"{prefix}mount", pos=list(base))
        # rail carriage: small visual block, gives the mount its inertia
        mount.add_geom(name=f"{prefix}carriage",
                       type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[0.03, 0.05, 0.02], rgba=[0.2, 0.2, 0.22, 1.0],
                       mass=0.5, contype=0, conaffinity=0)
        mount.add_joint(name=f"{prefix}railJoint",
                        type=mujoco.mjtJoint.mjJNT_SLIDE,
                        axis=[0.0, 1.0, 0.0],
                        range=[-RAIL_LIMIT, RAIL_LIMIT],
                        damping=1.0, armature=0.01, limited=True)

        hand = load_hand_spec(side)
        off = _palm_offset(hand)
        # attach so the palm body sits at the mount origin (minus the measured
        # fingertip->press-line calibration shift along world X):
        # palm_world = mount + p + R·palm_hand  =>  p = -R·palm_hand,
        # with R = 180° about Y  =>  p = (palm_x, -palm_y, palm_z).
        frame = mount.add_frame(pos=[off[0] - tip_shift, -off[1], off[2]],
                                quat=list(MOUNT_QUAT))
        spec.attach(hand, prefix=prefix, frame=frame)

        # rail position servo (matches the Isaac slider actuator gains)
        act = spec.add_actuator(name=f"{prefix}A_rail",
                                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                                target=f"{prefix}railJoint")
        act.gainprm[0] = RAIL_STIFFNESS
        act.biasprm[1] = -RAIL_STIFFNESS
        act.biasprm[2] = -RAIL_DAMPING
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.ctrlrange = [-RAIL_LIMIT, RAIL_LIMIT]
        act.forcerange = [-RAIL_FORCE, RAIL_FORCE]
        act.ctrllimited = True

    # --- cameras ------------------------------------------------------------
    _add_lookat_camera(spec, "main", eye=(2.2, -1.5, 1.8),
                       target=(0.45, 0.0, 0.78))
    _add_lookat_camera(spec, "front", eye=(-0.4, 0.0, 1.35),
                       target=(0.62, 0.0, 0.80))
    _add_lookat_camera(spec, "top", eye=(0.62, 0.0, 1.9),
                       target=(0.62, 0.0, 0.76), up_hint=(-1.0, 0.0, 0.0))
    return spec


def _add_lookat_camera(spec, name, eye, target, up_hint=(0.0, 0.0, 1.0)):
    """Fixed camera at ``eye`` looking at ``target`` (MuJoCo cams look -Z)."""
    eye = np.asarray(eye, dtype=float)
    fwd = np.asarray(target, dtype=float) - eye
    fwd /= np.linalg.norm(fwd)
    up = np.asarray(up_hint, dtype=float)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rot = np.column_stack([right, up, -fwd])       # camera axes as columns
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    spec.worldbody.add_camera(name=name, pos=eye.tolist(), quat=quat.tolist(),
                              fovy=45.0)


def compile_scene(cfg) -> mujoco.MjModel:
    """Build + compile in one go (used by the env)."""
    return build_scene_spec(cfg).compile()


def save_scene_xml(cfg, out_path: str | Path = DEFAULT_SCENE_XML) -> Path:
    """Write the composed scene XML (mesh paths resolved absolute) for
    inspection or for loading with plain ``MjModel.from_xml_path``."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    spec = build_scene_spec(cfg)
    spec.compile()                                  # resolves + validates
    out.write_text(spec.to_xml())
    return out
