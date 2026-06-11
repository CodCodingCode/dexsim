"""Phase 1 of filming: roll out the policy (or the pure IK reference with --zero)
on the song and record arm + piano joint trajectories AND the per-step goal vs
sounding notes, to an npz. Phase 2 (render_rollout.py) replays it with a camera
and a note-comparison UI overlay.

  # pure reference arms (no policy needed) -- "are the arms positioned right?"
  python scripts/record_rollout.py --headless --zero --midi data/midi/song.mid
  # a trained policy
  python scripts/record_rollout.py --headless --checkpoint <model.pt>
"""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default=None)
p.add_argument("--zero", action="store_true", help="zero residual = pure IK reference")
p.add_argument("--midi", default="data/midi/song.mid")
p.add_argument("--out", default="logs/rollout.npz")
p.add_argument("--arm_ik_follow", action="store_true",
               help="online WristPoseIK servos the arms to the fingering centroid")
p.add_argument("--planar_ik", action="store_true",
               help="weighted DLS: hold constant Z + orientation, slide in XY")
p.add_argument("--planar_pin_x", action="store_true",
               help="also pin depth (world X) -> arm slides only laterally (Y), like the slider")
p.add_argument("--arm_z_constant", action="store_true",
               help="pin both arms to ONE constant hover height (aligned, no Z motion); X/Y track notes")
p.add_argument("--freeze_last_dof", action="store_true",
               help="pin the UR10e wrist_3 joint (last DoF) at its init value")
p.add_argument("--freeze_wrist", action="store_true",
               help="pin all 3 wrist joints -> only pan(turn)+lift(lean)[+elbow] move")
p.add_argument("--freeze_elbow", action="store_true",
               help="also pin the elbow -> pure 2-DoF turn+lean arm")
p.add_argument("--arm_ik_pos_only", action="store_true",
               help="position-only IK (drop orientation) -> smooth, no wrist fling")
p.add_argument("--arm_ik_hover", type=float, default=None,
               help="override hover height (m) of the servoed palm above the keys")
p.add_argument("--hand_tilt", type=float, default=None,
               help="rotate the IK orientation target (rad about world X) so the fingers "
                    "point DOWN at the keys instead of sideways; ~-1.22 = -70 deg")
p.add_argument("--phase0", action="store_true",
               help="PHASE 0: reproduce the gross-positioning env (2/3-DoF arm, fingers frozen, "
                    "RL drives the arm) so a phase0 checkpoint's actions map as trained")
p.add_argument("--phase0_joints", default=None,
               help="comma-separated arm-joint substrings the policy moves (must match training)")
p.add_argument("--phase0_arm_scale", type=float, default=None,
               help="residual scale of the live phase0 arm joints (must match training)")
p.add_argument("--no_fold", action="store_true", help="real key positions (must match training)")
p.add_argument("--no_mute", action="store_true", help="both arms active (must match training)")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app

import numpy as np, torch, gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import dexsim.tasks  # noqa
from dexsim.tasks.piano import PianoEnvCfg
from dexsim.tasks.piano.agents.rsl_rl_ppo_cfg import PianoPPORunnerCfg

cfg = PianoEnvCfg(); cfg.scene.num_envs = 1; cfg.midi_path = a.midi
if a.phase0:
    # mirror train_piano.py's --phase0 preset so a phase0 checkpoint maps as trained:
    # RL drives a reduced-DoF arm, fingers frozen, real key positions, both arms active.
    cfg.phase0_arm_positioning = True
    cfg.freeze_arms = False
    cfg.arm_ik_follow = False
    cfg.mute_right_hand = False
    cfg.fold_to_reach = False
    if a.phase0_joints is not None:
        cfg.phase0_arm_joints = tuple(j.strip() for j in a.phase0_joints.split(",") if j.strip())
    if a.phase0_arm_scale is not None:
        cfg.phase0_arm_scale = a.phase0_arm_scale
if a.no_fold:
    cfg.fold_to_reach = False
if a.no_mute:
    cfg.mute_right_hand = False
if a.arm_ik_follow:
    cfg.arm_ik_follow = True; cfg.freeze_arms = False
    if a.planar_ik:
        cfg.planar_ik = True
    if a.planar_pin_x:
        cfg.planar_pin_x = True
    if a.arm_z_constant:
        cfg.arm_z_constant = True
    if a.freeze_last_dof:
        cfg.freeze_last_dof = True
    if a.freeze_wrist:
        cfg.freeze_wrist = True
    if a.freeze_elbow:
        cfg.freeze_elbow = True
    if a.arm_ik_pos_only:
        cfg.arm_ik_pos_only = True
if a.arm_ik_hover is not None:
    cfg.arm_ik_hover = a.arm_ik_hover
if a.hand_tilt is not None:
    cfg.hand_tilt = a.hand_tilt; cfg.hand_tilt_axis = 0    # rotate about world X -> fingers down
env = gym.make("Dexsim-Piano-Bimanual-v0", cfg=cfg, render_mode=None)
le = env.unwrapped
wrapped = RslRlVecEnvWrapper(env)

if a.zero:
    def policy(obs):                              # pure reference (zero residual)
        return torch.zeros((le.num_envs, le.cfg.action_space), device=le.device)
    print("[record_rollout] ZERO residual (pure IK reference)")
else:
    from rsl_rl.runners import OnPolicyRunner
    _ad = PianoPPORunnerCfg().to_dict(); _ad.setdefault("policy", {})["noise_std_type"] = "log"
    runner = OnPolicyRunner(wrapped, _ad, log_dir=None, device=le.device)
    runner.load(a.checkpoint)
    policy = runner.get_inference_policy(device=le.device)

obs, _ = wrapped.get_observations()
L, R, K, GOAL, SOUND, PALM, TGT, TACT = [], [], [], [], [], [], [], []
for _ in range(le.song_len):
    goal_t = le._goal_now()[0].cpu().numpy().copy()      # which keys SHOULD sound now
    # gross-positioning target the arm is chasing (centroid of each hand's upcoming keys)
    cen, cact = le._hand_note_centroids()                # (1,2,3),(1,2)
    with torch.inference_mode():
        obs, _, _, _ = wrapped.step(policy(obs))
    L.append(le.left_robot.data.joint_pos[0].cpu().numpy())
    R.append(le.right_robot.data.joint_pos[0].cpu().numpy())
    K.append(le.piano.data.joint_pos[0].cpu().numpy())
    GOAL.append(goal_t)
    SOUND.append(le.key_sounding[0].cpu().numpy().copy())  # which keys the model SOUNDS
    PALM.append(le._palms_world()[0].cpu().numpy())        # (2,3) actual hand-base positions
    TGT.append(cen[0].cpu().numpy())                       # (2,3) where each hand SHOULD be
    TACT.append(cact[0].cpu().numpy())                     # (2,) which hand is being asked to reach
np.savez(a.out, left=np.array(L), right=np.array(R), keys=np.array(K),
         goal=np.array(GOAL).astype(np.uint8), sound=np.array(SOUND).astype(np.uint8),
         palm=np.array(PALM), target=np.array(TGT), target_active=np.array(TACT).astype(np.uint8),
         joint_names_left=le.left_robot.data.joint_names,
         piano_pos=np.array(cfg.piano_pos),
         left_base=np.array(cfg.left_base_pos), right_base=np.array(cfg.right_base_pos),
         control_dt=cfg.control_dt)
# reach-gap summary: how far each ACTIVE hand's palm is (laterally) from its target
_P=np.array(PALM); _T=np.array(TGT); _A=np.array(TACT).astype(bool)
for h,name in ((0,"LEFT"),(1,"RIGHT")):
    m=_A[:,h]
    if m.any():
        gap=np.linalg.norm(_P[m,h,:]-_T[m,h,:],axis=-1)
        gy=np.abs(_P[m,h,1]-_T[m,h,1])   # lateral (Y) component = the reach axis
        print(f"[reach] {name} hand: mean gap {gap.mean()*100:.1f}cm  max {gap.max()*100:.1f}cm  "
              f"(lateral max {gy.max()*100:.1f}cm) over {m.sum()} active steps")
# quick text summary so we know the rollout is sane before rendering
g = np.array(GOAL); s = np.array(SOUND)
tp = (g.astype(bool) & s.astype(bool)).sum()
print(f"[record_rollout] saved {len(L)} frames -> {a.out}  "
      f"goal_notes={int(g.sum())} sounded={int(s.sum())} correct={int(tp)}")
app.close()
