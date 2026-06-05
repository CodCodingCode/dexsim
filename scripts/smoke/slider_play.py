"""Slider embodiment in-env test: mount ONE Shadow+slider hand over the piano,
drive the slider to place a finger on each melody note, and measure key-press F1.
This validates that the slider (0mm placement) actually sounds the right keys in the
piano env -- the thing the heavy arm could not (placement wall). Prints a geometry
calibration block first (fingertip vs key world positions) so the hand pose can be
dialed in, then plays the melody and reports recall/precision/F1.

  python scripts/smoke/slider_play.py --headless [--inspect]
"""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--midi", default="data/midi/twinkle.mid")
p.add_argument("--inspect", action="store_true", help="print geometry calibration and exit")
p.add_argument("--hand_x", type=float, default=0.50)
p.add_argument("--hand_z", type=float, default=0.95)
p.add_argument("--rotw", type=float, default=0.707)
p.add_argument("--rotx", type=float, default=0.707)
p.add_argument("--roty", type=float, default=0.0)
p.add_argument("--rotz", type=float, default=0.0)
p.add_argument("--press_finger", type=int, default=1, help="0=th 1=ff 2=mf 3=rf 4=lf")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import torch, numpy as np
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from dexsim.assets import SHADOW_SLIDER_CFG, PIANO_CFG, KEY_SOUND_ANGLE
from dexsim.piano import FINGERTIP_BODIES, load_song
from dexsim.piano import geometry

sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))
spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

PIANO_POS = (0.569, -0.606, 0.746)
hand_cfg = SHADOW_SLIDER_CFG.replace(prim_path="/World/Hand")
hand_cfg.init_state.pos = (args.hand_x, PIANO_POS[1], args.hand_z)
hand_cfg.init_state.rot = (args.rotw, args.rotx, args.roty, args.rotz)
piano_cfg = PIANO_CFG.replace(prim_path="/World/Piano")
piano_cfg.init_state.pos = PIANO_POS

# faster slider so it travels between keys within a control step (less lag)
hand_cfg.actuators["slider"].stiffness = 30000.0
hand_cfg.actuators["slider"].damping = 600.0
hand_cfg.actuators["slider"].effort_limit = 30000.0
# faster key RETURN so a released key springs back promptly (raw-angle metric counts
# any key below sound angle; the main env's damping=4 is slow-return for its latch metric)
piano_cfg.actuators["keys"].damping = 0.8
hand = Articulation(hand_cfg)
piano = Articulation(piano_cfg)
sim.reset()

names = list(hand.joint_names)
iy, iz = names.index("slider_y"), names.index("slider_z")
fb = FINGERTIP_BODIES[args.press_finger]
tip_ids, _ = hand.find_bodies([fb], preserve_order=True)
tip_id = tip_ids[0]
all_tip_ids = torch.tensor([hand.find_bodies([b], preserve_order=True)[0][0]
                            for b in FINGERTIP_BODIES], device="cuda:0")
key_ids, _ = piano.find_bodies([f"key_{i}" for i in range(88)], preserve_order=True)
key_ids = torch.tensor(key_ids, device=sim.device)

def settle(n=30):
    for _ in range(n):
        hand.set_joint_position_target(hand.data.joint_pos)
        hand.write_data_to_sim(); piano.write_data_to_sim()
        sim.step(); hand.update(1/120.); piano.update(1/120.)

settle(60)
# piano REST-STATE check: with no finger contact, NO key should be sounding. If keys
# read sounding here, they sag/stick (the known gravity/spring bug) -> phantom presses.
_ang0 = piano.data.joint_pos[0]
_phantom = torch.nonzero(_ang0 < KEY_SOUND_ANGLE).flatten().tolist()
print(f"[piano rest] sounding-at-rest (should be []): {_phantom}  "
      f"min_angle={_ang0.min():.4f} max={_ang0.max():.4f}")
tip = hand.data.body_pos_w[0, tip_id]
keyw = piano.data.body_pos_w[0, key_ids]            # (88,3)
print("\n========== SLIDER GEOMETRY CALIBRATION ==========")
print(f"hand pos=({args.hand_x},{PIANO_POS[1]},{args.hand_z}) rot=({args.rotw},{args.rotx},{args.roty},{args.rotz})")
print(f"press-finger {fb} tip world = ({tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f})")
print(f"key_20 top world = ({keyw[20,0]:.3f},{keyw[20,1]:.3f},{keyw[20,2]:.3f})")
print(f"key Y range [{keyw[:,1].min():.3f},{keyw[:,1].max():.3f}]  key Z(top)~{keyw[:,2].max():.3f}")
print(f"slider_y range {hand.data.joint_pos_limits[0,iy].tolist()}  slider_z {hand.data.joint_pos_limits[0,iz].tolist()}")
print(f"tip-to-key20 offset: dX={tip[0]-keyw[20,0]:.3f} dY={tip[1]-keyw[20,1]:.3f} dZ={tip[2]-keyw[20,2]:.3f}")
if args.inspect:
    print("==================================================\n")
    app.close(); raise SystemExit

# ---- scripted melody play: slider_y places the finger at each note's key Y ----
song = load_song(args.midi, control_dt=0.05)
from dexsim.piano.midi import fold_into_reach
act, ons = fold_into_reach(song.key_activation, song.onsets,
                           left_window=(19, 26), right_window=(53, 60))
goal = torch.as_tensor(act, dtype=torch.bool, device=sim.device)      # (T,88)
T = goal.shape[0]
# index flex joints for the press finger (robot0_{FF}J1/2/3)
ftag = ["TH","FF","MF","RF","LF"][args.press_finger]
flex_cols = [i for i,n in enumerate(names) if f"robot0_{ftag}J" in n]
# --- CALIBRATE slider->tip maps (the 180-deg hand rot INVERTS the slider axes) ---
def _drive(sy, sz, n=120):
    cmd = torch.zeros_like(hand.data.joint_pos)
    cmd[:, iy] = sy; cmd[:, iz] = sz
    for _ in range(n):
        hand.set_joint_position_target(cmd)
        hand.write_data_to_sim(); piano.write_data_to_sim()
        sim.step(); hand.update(1/120.); piano.update(1/120.)
    return hand.data.body_pos_w[0, tip_id].clone()

# slider_y -> tipY (hold sz=0)
yp, ym = _drive(0.3, 0.0), _drive(-0.3, 0.0)
by = (yp[1] - ym[1]).item() / 0.6                 # d tipY / d slider_y
ay = (ym[1].item() + 0.3 * by)                    # tipY at slider_y=0
# slider_z -> tipZ (hold sy=0)
zp, z0 = _drive(0.0, 0.08), _drive(0.0, 0.0)
bz = (zp[2] - z0[2]).item() / 0.08                # d tipZ / d slider_z
az = z0[2].item()                                 # tipZ at slider_z=0
print(f"[calib] tipY = {ay:.3f} + ({by:.3f})*slider_y   tipZ = {az:.3f} + ({bz:.3f})*slider_z")
key_top_z = float(keyw[:, 2].max())               # ~0.762

def sy_for(key_y):  return float(np.clip((key_y - ay) / by, -0.6, 0.6))
def sz_for(tip_z):  return float(np.clip((tip_z - az) / bz, -0.05, 0.10))
import sys; sys.stdout.flush()
# curl the NON-press fingers up & clear; keep the press finger STRAIGHT (curling a
# downward finger lifts its tip toward the palm = off the key). slider_z presses.
other_flex = [i for i,n in enumerate(names)
              if n.startswith("robot0_") and "J" in n and f"robot0_{ftag}J" not in n
              and any(n.rstrip("12345").endswith(t) for t in ("J",))]
# curl only the TIP joints (J1/J2) of the other fingers, NOT the MCP (J3): curling the
# MCP swings the whole finger and DROPS the middle knuckle onto keys (false presses);
# tip-only curl tucks the fingertip up while the knuckle stays high.
other_flex = [i for i,n in enumerate(names)
              if n.startswith("robot0_") and f"robot0_{ftag}" not in n and n[-1] in "12"]
# FIXED base pose (not the drifting current pose) so the wrist/fingers don't swing
base_pose = torch.zeros_like(hand.data.joint_pos)
only_left = True   # single hand: only play the LEFT window keys (<=26)
press_z = key_top_z - 0.004        # LIGHT press (4mm) -> sound without slamming the hand
lift_z  = key_top_z + 0.045        # lift 45mm clear when traversing/idle
OTHER_CURL = 1.5                   # curl non-press fingers HARD so they stay clear
sy_hold = 0.0
tp = fp = fn = 0
for t in range(T):
    g = goal[t].clone()
    if only_left:
        g[27:] = False                            # this one hand covers the left window
    active = torch.nonzero(g).flatten()
    cmd = base_pose.clone()
    for c in other_flex: cmd[:, c] = OTHER_CURL  # curl OTHER fingers up & clear of keys
    for c in flex_cols:  cmd[:, c] = 0.0         # press finger STRAIGHT (down)
    if len(active):
        k = int(active[0])                       # melody: first active key
        target_key_y = keyw[k, 1].item()
        cmd[:, iy] = sy_for(target_key_y)        # feed-forward seed
    else:
        target_key_y = None
    # CLOSED-LOOP placement + align-then-strike: each substep, measure the tip's
    # actual Y and nudge slider_y to drive the error to ~0 (kills the calibration
    # residual), and only press once aligned (<8mm) so the finger doesn't ring
    # neighbours while traversing.
    for s in range(12):
        tip_now_y = hand.data.body_pos_w[0, tip_id, 1].item()
        if target_key_y is not None:
            err_y = target_key_y - tip_now_y
            cmd[:, iy] = float(np.clip(cmd[0, iy].item() + err_y / by, -0.6, 0.6))
            aligned = abs(err_y) < 0.008
            cmd[:, iz] = sz_for(press_z if aligned else lift_z)
        else:
            cmd[:, iz] = sz_for(lift_z)
        hand.set_joint_position_target(cmd)
        hand.write_data_to_sim(); piano.write_data_to_sim()
        sim.step(); hand.update(1/120.); piano.update(1/120.)
    ang = piano.data.joint_pos[0]               # (88,) key angles
    sounding = ang < KEY_SOUND_ANGLE
    if len(active) and t < 40 and t % 8 == 0:
        kk = int(active[0])
        snd = torch.nonzero(sounding).flatten().tolist()
        allz = hand.data.body_pos_w[0, all_tip_ids, 2].tolist()
        print(f"  t={t} target_key={kk} sounding={snd}")
        print(f"      fingertip Z [th,ff,mf,rf,lf]={[round(z,3) for z in allz]} keytop={key_top_z:.3f}")
    want = g
    tp += int((sounding & want).sum()); fp += int((sounding & ~want).sum()); fn += int((~sounding & want).sum())

eps = 1e-9
rec = tp/(tp+fn+eps); prec = tp/(tp+fp+eps); f1 = 2*rec*prec/(rec+prec+eps)
print("========== SLIDER SCRIPTED PLAY (twinkle, 1 hand) ==========")
print(f"recall={rec:.3f} precision={prec:.3f} F1={f1:.3f}  (tp={tp} fp={fp} fn={fn})")
print("============================================================\n")
app.close()
