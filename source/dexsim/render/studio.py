"""Single source of truth for the bimanual-piano render scene.

The cold one-shot scripts (``render_scene.py``, ``render_rollout.py``) AND the
warm ``render_server.py`` all build the scene through these helpers, so a CACHED
(warm) render is byte-for-byte identical to a cold render -- the whole point of
caching is that the fast path looks the same as the slow path.

USAGE ORDER: these functions touch ``isaaclab.sim`` / ``isaaclab.assets`` /
``isaaclab.sensors``, imported at module top. Isaac Sim requires those imports to
happen AFTER ``AppLauncher(...).app`` has started, so import this module only
after the app is launched.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg

NUM_KEYS = 88
_BLACK_IN_OCTAVE = {1, 3, 6, 8, 10}        # which semitones are black keys


# --------------------------------------------------------------------------- #
# RTX / path-tracing settings (OptiX denoiser weights are absent on this box, so
# we disable the denoiser and rely on a high total-spp accumulation instead).
# --------------------------------------------------------------------------- #
def apply_pathtrace_settings(total_spp, *, max_bounces=6, tonemap=True, spp_per_frame=8):
    import carb
    s = carb.settings.get_settings()
    s.set("/rtx/rendermode", "PathTracing")
    s.set("/rtx/pathtracing/spp", spp_per_frame)
    s.set("/rtx/pathtracing/totalSpp", int(total_spp))
    s.set("/rtx/pathtracing/optixDenoiser/enabled", False)
    s.set("/rtx/pathtracing/maxBounces", max_bounces)
    if tonemap:
        s.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", max_bounces)
        s.set("/rtx/post/tonemap/op", 1)                 # filmic-ish tonemap
        s.set("/rtx/post/histogram/enabled", True)
    s.set("/app/asyncRendering", False)


def set_total_spp(total_spp):
    """Per-job override of the accumulation depth without rebuilding the scene."""
    import carb
    carb.settings.get_settings().set("/rtx/pathtracing/totalSpp", int(total_spp))


# --------------------------------------------------------------------------- #
# scene construction
# --------------------------------------------------------------------------- #
def _quat_aim(eye, tgt):
    """quaternion (w,x,y,z) aiming a light's -Z from eye toward tgt."""
    d = np.array(tgt, float) - np.array(eye, float); d /= (np.linalg.norm(d) + 1e-9)
    z = -d; up = np.array([0, 0, 1.0]); x = np.cross(up, z); x /= (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x); R = np.stack([x, y, z], 1)
    w = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w < 1e-6:
        return (1.0, 0.0, 0.0, 0.0)
    return (float(w), float((R[2, 1] - R[1, 2]) / (4 * w)),
            float((R[0, 2] - R[2, 0]) / (4 * w)), float((R[1, 0] - R[0, 1]) / (4 * w)))


def _spawn_studio_floor_lights():
    """Clean studio floor + soft 3-point-ish lighting (the render_rollout look)."""
    _fm = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.10, 0.12), roughness=0.55)
    sim_utils.CuboidCfg(size=(40.0, 40.0, 0.04), visual_material=_fm).func(
        "/World/Floor", sim_utils.CuboidCfg(size=(40.0, 40.0, 0.04), visual_material=_fm),
        translation=(0.0, 0.0, -0.02))
    # SUPPORTS so nothing floats: a table under the piano + two pedestals out front.
    _wood = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.13, 0.08), roughness=0.7)
    _ped = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.12, 0.14), roughness=0.5)
    # real STATIC collider (collision_props, no rigid_props) so hands rest ON the
    # table instead of phasing through it during the settle steps.
    _table_cfg = sim_utils.CuboidCfg(
        size=(0.45, 1.55, 0.72), visual_material=_wood,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True))
    _table_cfg.func("/World/PianoTable", _table_cfg,
                    translation=(0.60, 0.0, 0.36))             # under the piano (center x=0.60)
    # (base pedestals are spawned by _spawn_base_pedestals(), auto-sized to the base z;
    #  the old hardcoded 1.05-tall block at x=1.25 was a stale leftover -- removed.)
    _ = _ped
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


def _spawn_simple_floor_lights():
    """Plain ground + bright dome (the render_scene look)."""
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=3000.0))


def _spawn_base_pedestals(cfg):
    """Cosmetic boxes under each world-fixed base so the arms don't float.
    Box runs ground -> base z, read live from the resolved cfg."""
    for _nm, _rc in (("LeftPedestal", cfg.left_robot_cfg),
                     ("RightPedestal", cfg.right_robot_cfg)):
        _bx, _by, _bz = _rc.init_state.pos
        if _bz and _bz > 0.02:
            _pc = sim_utils.CuboidCfg(
                size=(0.26, 0.26, float(_bz)),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.22, 0.22, 0.25), metallic=0.1, roughness=0.5))
            _pc.func(f"/World/{_nm}", _pc, translation=(_bx, _by, float(_bz) / 2.0))


def build_scene(cfg, *, style="studio"):
    """Spawn floor/lights/supports + piano and both arms. Returns (piano, left, right).

    Articulations are re-prim'd to single-env ``/World/...`` paths so the render
    matches what training sees (base pose + piano-ready joint pose are already
    baked into ``PianoEnvCfg.__post_init__``). Call BEFORE ``sim.reset()``.
    """
    if style == "studio":
        _spawn_studio_floor_lights()
    else:
        _spawn_simple_floor_lights()

    piano = Articulation(cfg.piano_cfg.replace(prim_path="/World/Piano"))
    left = Articulation(cfg.left_robot_cfg.replace(prim_path="/World/LeftRobot"))
    right = Articulation(cfg.right_robot_cfg.replace(prim_path="/World/RightRobot"))
    _spawn_base_pedestals(cfg)
    return piano, left, right


def make_camera(width=1280, height=720):
    return Camera(CameraCfg(
        prim_path="/World/cam", update_period=0, height=height, width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955,
                                         clipping_range=(0.05, 1e6)),
    ))


def aim_camera(cam, eye, target, device):
    """Re-aim the camera. eye/target are 3-element (x,y,z) sequences."""
    eye_t = torch.tensor([[float(v) for v in eye]], device=device)
    tgt_t = torch.tensor([[float(v) for v in target]], device=device)
    cam.set_world_poses_from_view(eye_t, tgt_t)


# --------------------------------------------------------------------------- #
# kinematic driving
# --------------------------------------------------------------------------- #
def set_state(art, q):
    """Kinematically drive an articulation to joint positions q (1, ndof)."""
    art.write_joint_state_to_sim(q, torch.zeros_like(q))
    art.set_joint_position_target(q)
    art.write_data_to_sim()


# --------------------------------------------------------------------------- #
# note-compare UI overlay (used by rollout videos)
# --------------------------------------------------------------------------- #
def _is_black(k: int) -> bool:
    return (k % 12) in _BLACK_IN_OCTAVE


def _font(sz):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def draw_note_ui(rgb: np.ndarray, goal_row, sound_row) -> Image.Image:
    """Composite a rendered frame with a piano-keyboard strip comparing GROUND
    TRUTH (what should play) to what is PLAYING.
      blue = should play, not played -> MISSED ; green = correct ; red = wrong
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
