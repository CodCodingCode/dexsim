"""Train the bimanual piano policy (rsl_rl PPO).

  python scripts/train_piano.py --headless --num_envs 1024 --midi data/midi/twinkle.mid
  python scripts/train_piano.py --headless --num_envs 2048 --max_iterations 5000
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train bimanual piano policy.")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--midi", default=None, help="path to the song .mid (default: cfg's)")
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--save_interval", type=int, default=None, help="save a checkpoint every N iterations (default 10)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--freeze_arms", action="store_true", help="fixed-hands mode: drive fingers only (arms held)")
parser.add_argument("--planar_ik", action="store_true", help="weighted+iterated planar IK (gantry)")
parser.add_argument("--planar_pin_x", action="store_true", help="pin depth (world X) too -> lateral-only gantry (best in rollout_f1 A/B)")
parser.add_argument("--freeze_last_dof", action="store_true", help="freeze wrist_3 (note: over-constrains -> under-presses in eval)")
parser.add_argument("--arm_ik_pos_only", action="store_true", help="position-only arm IK (drop orientation rows): smooth, no wrist fling (wrist swing 2.4->0.5 rad in rollout A/B)")
parser.add_argument("--arm_ik_follow", action="store_true", help="arms servoed online by WristPoseIK to the fingering centroid; policy drives only the 48 finger DoF")
parser.add_argument("--arm_ik_hover", type=float, default=None, help="override arm_ik_hover (m palm hovers above keys)")
parser.add_argument("--arm_smooth", type=float, default=None, help="EMA factor smoothing the IK arm motion (0=snappy, 0.6 default, ->1=smooth/laggy)")
parser.add_argument("--arm_traj", default=None, help="play back a baked zero-phase-smoothed arm trajectory (.npz from bake_arm_traj.py) instead of live IK (no jitter, no lag)")
parser.add_argument("--wrist1_cap", type=float, default=None, help="cap UR10e wrist_1 up-tilt at this value (more-negative=more up; locked -4.782, -3.40~30%); arm IK repositions to compensate")
parser.add_argument("--no_wrist1_cap", action="store_true", help="disable the wrist-tilt cap (wrist_1 free)")
parser.add_argument("--strike_vel", type=float, default=None, help="override key_strike_vel (rad/s gate for a key to sound)")
parser.add_argument("--idle_clear_weight", type=float, default=None, help="penalty weight for idle fingers hanging low (anti-mash)")
parser.add_argument("--arm_sep_weight", type=float, default=None, help="penalty weight for the two palms getting closer than --arm_sep_min (anti arm-arm collision; policy-driven arms only)")
parser.add_argument("--arm_sep_min", type=float, default=None, help="min palm-palm distance (m) before the separation penalty kicks in (default 0.18)")
parser.add_argument("--wrist_clear_weight", type=float, default=None, help="penalty weight for a wrist dipping below --wrist_clear_z (anti wrist-into-table; policy-driven arms only)")
parser.add_argument("--wrist_clear_z", type=float, default=None, help="min wrist height (m) before the clearance penalty kicks in (default 0.82; table top ~0.72)")
parser.add_argument("--idle_hover_weight", type=float, default=None, help="reward idle fingers for sitting AT their hover-home (gradient anti-mash, positive twin of --idle_clear_weight; suggest 0.2-0.3)")
parser.add_argument("--anneal_false_press", action="store_true", help="recall-gated curriculum: start the false-press penalty at --false_press_start (energy at 0) and ramp to --false_press_weight / cfg energy once recall EMA >= gate")
parser.add_argument("--false_press_start", type=float, default=None, help="anneal: starting false-press penalty (default 0.15)")
parser.add_argument("--anneal_recall_gate", type=float, default=None, help="anneal: ramp only while recall EMA >= this (default 0.5)")
parser.add_argument("--anneal_steps", type=int, default=None, help="anneal: reward steps to go start -> final once gated (default 2000)")
parser.add_argument("--arm_stiffness", type=float, default=None, help="arm actuator stiffness (higher=faster tracking=more recall; safe, policy doesnt drive arm)")
parser.add_argument("--arm_effort", type=float, default=None, help="arm actuator effort_limit")
parser.add_argument("--hand_stiffness", type=float, default=None, help="override Shadow hand actuator stiffness (finger authority)")
parser.add_argument("--hand_effort", type=float, default=None, help="override Shadow hand actuator effort_limit")
parser.add_argument("--key_stiffness", type=float, default=None, help="override piano key return-spring stiffness")
parser.add_argument("--false_press_weight", type=float, default=None, help="override false-press penalty weight")
parser.add_argument("--hand_action_scale", type=float, default=None, help="override finger residual scale (lower = less jitter/blowup)")
parser.add_argument("--init_noise", type=float, default=None, help="override PPO initial action-noise std")
parser.add_argument("--lr", type=float, default=None, help="override PPO learning rate (lower = less degradation)")
parser.add_argument("--hand_tilt", type=float, default=None, help="tilt the hand toward pianist posture (rad, X axis)")
parser.add_argument("--start_curl", type=float, default=None, help="curl ALL fingers this many rad in the reset+base pose (start RL curled-up off the keys; anti-mash)")
parser.add_argument("--idle_finger_curl", type=float, default=None, help="curl NON-assigned fingers up in the base pose (rad; lift idle fingers off neighbor keys)")
parser.add_argument("--lookahead", type=int, default=None, help="goal_lookahead steps (10*88=880 obs dims; cut to shrink obs so the critic can fit)")
parser.add_argument("--struck_frac", type=float, default=None, help="key sounds at this fraction of full press depth (lower=easier/more reliable registration)")
parser.add_argument("--no_norm", action="store_true", help="disable empirical_normalization (sparse-reward PPO degrades with it on)")
parser.add_argument("--solo_arm_dip", action="store_true", help="solo mode: also allow shoulder_lift (press by arm dip, no finger flex-arc)")
parser.add_argument("--solo_middle", action="store_true", help="mask action to ONLY the right middle finger (no other finger moves -> no mash; learns timing)")
parser.add_argument("--remap_thumb", action="store_true", help="remap thumb fingering -> middle finger (cleaner straight-down presser)")
parser.add_argument("--lift_between", type=float, default=None, help="dip-to-strike: lift hand this many m above hover between notes (0=constant hover)")
parser.add_argument("--wrist_up_delta", type=float, default=None, help="RUNTIME-ONLY: add this to wrist_1_joint in init_state (angle fingers forward for top-down strike); does NOT edit the locked ready-pose file")
parser.add_argument("--palm_down", action="store_true", help="arm_ik_follow holds hand PALM-DOWN/fingers-fwd so finger flexion presses top-down (runtime servo orientation; locked pose untouched)")
parser.add_argument("--pd_pan", type=float, default=None, help="override shoulder_pan in palm-down pose to center the fingertip on a key")
parser.add_argument("--pd_drop", type=float, default=0.0, help="lower the palm-down hand (add to shoulder_lift) so fingertips rest closer to keys")
parser.add_argument("--right_palmdown", action="store_true", help="set RIGHT arm to a fixed PALM-DOWN playable pose (solve_both_arms) so finger flexion presses; runtime-only, locked pose file untouched")
parser.add_argument("--songs_npz", default=None, help="MULTI-SONG: train one policy across all songs in this precomputed goal bundle (.npz)")
parser.add_argument("--max_songs", type=int, default=0, help="cap multi-song training to the first N songs (0=all)")
parser.add_argument("--key_press_weight", type=float, default=None, help="reward for sounding the right key")
parser.add_argument("--onset_weight", type=float, default=None, help="reward for sounding a key on its onset")
parser.add_argument("--fingering_weight", type=float, default=None, help="shaping: fingertip near assigned key (lower = less hovering)")
parser.add_argument("--tag", default=None, help="run label -> wandb run name + log subdir (for parallel A/B/C runs)")
parser.add_argument("--no_fold", action="store_true", help="disable fold_to_reach (use the song's real key positions, e.g. for RP1M)")
parser.add_argument("--no_mute", action="store_true", help="disable mute_right_hand (needed for two-handed songs)")
parser.add_argument("--phase0", action="store_true", help="PHASE 0 curriculum: RL learns gross arm positioning only -- 2-DoF arm (turn+lean) moves each hand over its keys; fingers frozen; reward = arm positioning alone (press/finger/onset weights zeroed)")
parser.add_argument("--arm_position_weight", type=float, default=None, help="override the Phase-0 gross-positioning reward weight (default 1.0 under --phase0)")
parser.add_argument("--phase0_arm_scale", type=float, default=None, help="override the residual scale of the live Phase-0 arm joints")
parser.add_argument("--phase0_joints", default=None, help="comma-separated arm-joint name substrings the Phase-0 policy may move (default 'shoulder_pan,shoulder_lift' = 2-DoF turn+lean; add ',elbow' for 3-DoF reach)")
parser.add_argument("--resume_from", default=None, help="checkpoint .pt to load (model+optimizer) before training, e.g. for song-curriculum stages")
parser.add_argument("--entropy_coef", type=float, default=None, help="override PPO entropy bonus (phase0: dead action dims inflate their noise std under entropy pressure)")
parser.add_argument("--desired_kl", type=float, default=None, help="override the KL-adaptive lr target (default 0.01); lower = smaller policy steps (damps update-to-update mode flip-flop at large batch)")
parser.add_argument("--reset_noise_std", type=float, default=None, help="on --resume_from, hard-reset the LIVE action dims' noise std to this (resumed runs inherit inflated stds; natural anneal takes ~7600 iters)")
parser.add_argument("--reset_dead_noise_std", type=float, default=0.2, help="std for the DEAD (joint_scale==0) action dims when --reset_noise_std is set")
parser.add_argument("--no_random_init", action="store_true", help="start every episode at song step 0 (arms at ready) instead of random mid-song spawns -- matches the eval protocol; random spawns average a cold-start chase into every episode's reward")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.io import dump_yaml

import dexsim.tasks  # noqa: F401
from dexsim.tasks.piano import PianoEnvCfg
from dexsim.tasks.piano.agents.rsl_rl_ppo_cfg import PianoPPORunnerCfg

TASK = "Dexsim-Piano-Bimanual-v0"


def main():
    env_cfg = PianoEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    if args.midi:
        env_cfg.midi_path = args.midi
    if args.freeze_arms:
        env_cfg.freeze_arms = True
    if args.arm_ik_follow:
        env_cfg.arm_ik_follow = True
        if args.planar_ik:
            env_cfg.planar_ik = True
        if args.planar_pin_x:
            env_cfg.planar_pin_x = True     # lateral-only gantry (pin depth X) -- best config in rollout_f1 A/B
        if args.freeze_last_dof:
            env_cfg.freeze_last_dof = True
        if args.arm_ik_pos_only:
            env_cfg.arm_ik_pos_only = True
        env_cfg.freeze_arms = False     # arms move (IK-driven), not held static
    if args.phase0:
        # PHASE 0 curriculum: the policy learns ONLY gross arm placement. A reduced
        # 2-DoF arm (shoulder_pan turn + shoulder_lift lean) drives each hand-base over
        # the centroid of the keys it must play; fingers & distal arm joints are frozen;
        # IK is off (RL, not math, positions the arms). Reward is the positioning term
        # alone -- every pressing/shaping weight is zeroed so nothing competes with it.
        env_cfg.phase0_arm_positioning = True
        env_cfg.freeze_arms = False        # arms MOVE (policy-driven), not held static
        env_cfg.arm_ik_follow = False      # math does NOT position -- the policy learns it
        env_cfg.mute_right_hand = False    # both hands position (works for any song)
        # CRITICAL: use the songs' REAL key positions. fold_to_reach squishes every note
        # into each hand's FIXED reachable window -> the centroid barely moves -> nothing
        # to learn. Phase 0 is exactly "move the arm to where the notes really are".
        env_cfg.fold_to_reach = False
        env_cfg.arm_position_weight = 1.0
        if args.phase0_joints is not None:
            env_cfg.phase0_arm_joints = tuple(
                j.strip() for j in args.phase0_joints.split(",") if j.strip())
        env_cfg.key_press_weight = 0.0
        env_cfg.fingering_weight = 0.0
        env_cfg.onset_weight = 0.0
        env_cfg.false_press_weight = 0.0
        env_cfg.energy_weight = 0.0
        if args.phase0_arm_scale is not None:
            env_cfg.phase0_arm_scale = args.phase0_arm_scale
        print("[train_piano] PHASE 0: gross arm positioning only (2-DoF arm; fingers "
              "frozen; reward = arm placement)")
    if args.arm_position_weight is not None:
        env_cfg.arm_position_weight = args.arm_position_weight
    if args.songs_npz:
        env_cfg.songs_npz = args.songs_npz
        env_cfg.max_songs = args.max_songs
    if args.arm_ik_hover is not None:
        env_cfg.arm_ik_hover = args.arm_ik_hover
    if args.arm_smooth is not None:
        env_cfg.arm_smooth = args.arm_smooth
    if args.arm_traj is not None:
        env_cfg.arm_traj_npz = args.arm_traj
    if args.no_wrist1_cap:
        env_cfg.wrist1_cap = None
    elif args.wrist1_cap is not None:
        env_cfg.wrist1_cap = args.wrist1_cap
    if args.strike_vel is not None:
        env_cfg.key_strike_vel = args.strike_vel
    if args.idle_clear_weight is not None:
        env_cfg.idle_clear_weight = args.idle_clear_weight
    if args.arm_sep_weight is not None:
        env_cfg.arm_sep_weight = args.arm_sep_weight
    if args.arm_sep_min is not None:
        env_cfg.arm_sep_min = args.arm_sep_min
    if args.wrist_clear_weight is not None:
        env_cfg.wrist_clear_weight = args.wrist_clear_weight
    if args.wrist_clear_z is not None:
        env_cfg.wrist_clear_z = args.wrist_clear_z
    if args.idle_hover_weight is not None:
        env_cfg.idle_hover_weight = args.idle_hover_weight
    if args.anneal_false_press:
        env_cfg.anneal_false_press = True
    if args.false_press_start is not None:
        env_cfg.false_press_start = args.false_press_start
    if args.anneal_recall_gate is not None:
        env_cfg.anneal_recall_gate = args.anneal_recall_gate
    if args.anneal_steps is not None:
        env_cfg.anneal_steps = args.anneal_steps
    if args.key_stiffness is not None:
        env_cfg.piano_cfg.actuators["keys"].stiffness = args.key_stiffness
    if args.false_press_weight is not None:
        env_cfg.false_press_weight = args.false_press_weight
    if args.hand_action_scale is not None:
        env_cfg.hand_action_scale = args.hand_action_scale
    if args.hand_tilt is not None:
        env_cfg.hand_tilt = args.hand_tilt; env_cfg.hand_tilt_axis = 0
    if args.idle_finger_curl is not None:
        env_cfg.idle_finger_curl = args.idle_finger_curl
    if args.start_curl is not None:
        env_cfg.start_finger_curl = args.start_curl
    if args.remap_thumb:
        env_cfg.remap_thumb_to_middle = True
    if args.solo_middle:
        env_cfg.solo_right_middle = True
    if args.solo_arm_dip:
        env_cfg.solo_arm_dip = True
    if args.lift_between is not None:
        env_cfg.lift_between_notes = args.lift_between
    if args.palm_down:
        env_cfg.palm_down_servo = True
    if args.right_palmdown:
        # RUNTIME-ONLY palm-down right-arm pose (from solve_both_arms.py) so finger flexion
        # presses keys top-down. The locked right_ready_pose file is NOT edited.
        _PD = {"shoulder_pan_joint": -0.729, "shoulder_lift_joint": -1.281, "elbow_joint": 2.201,
               "wrist_1_joint": -0.337, "wrist_2_joint": 0.491, "wrist_3_joint": -2.904,
               "robot0_WRJ0": 0.0, "robot0_WRJ1": 0.0}
        if args.pd_pan is not None:
            _PD["shoulder_pan_joint"] = args.pd_pan
        if args.pd_drop:   # lower the hand so resting fingertips sit closer to the keys (reliable strike)
            _PD["shoulder_lift_joint"] += args.pd_drop
        jp = dict(env_cfg.right_robot_cfg.init_state.joint_pos); jp.update(_PD)
        env_cfg.right_robot_cfg.init_state.joint_pos = jp
        print("[train_piano] RIGHT arm set to PALM-DOWN playable pose (locked pose file untouched)")
    if args.wrist_up_delta is not None:
        # RUNTIME ONLY: nudge wrist_1 in the in-memory init_state to angle the fingers
        # forward (so finger flexion strikes the key top-down). The locked ready-pose
        # dicts in piano_env_cfg.py are NOT modified -- the "perfect" file stays intact.
        for rc in (env_cfg.left_robot_cfg, env_cfg.right_robot_cfg):
            jp = dict(rc.init_state.joint_pos)
            jp["wrist_1_joint"] = jp.get("wrist_1_joint", -4.782) + args.wrist_up_delta
            rc.init_state.joint_pos = jp
        print(f"[train_piano] RUNTIME wrist_1 += {args.wrist_up_delta} (fingers angled forward; locked file untouched)")
    for _w in ("key_press_weight", "onset_weight", "fingering_weight"):
        _v = getattr(args, _w)
        if _v is not None:
            setattr(env_cfg, _w, _v)
    if args.hand_stiffness is not None or args.hand_effort is not None:
        # override the Shadow hand actuator authority on BOTH arms (the "hand" group
        # = robot0_.* joints). Weak fingers (stiffness 3) may be why the policy can't
        # lift one finger while pressing another -> tests if finger authority unlocks it.
        for rc in (env_cfg.left_robot_cfg, env_cfg.right_robot_cfg):
            hand_act = rc.actuators["hand"]
            if args.hand_stiffness is not None:
                hand_act.stiffness = args.hand_stiffness
                hand_act.damping = max(0.1, 0.05 * args.hand_stiffness)  # ~crit-ish damping
            if args.hand_effort is not None:
                hand_act.effort_limit = args.hand_effort
        print(f"[train_piano] hand actuator override: stiffness={args.hand_stiffness} "
              f"effort={args.hand_effort}")
    if args.arm_stiffness is not None:
        import math as _m
        for rc in (env_cfg.left_robot_cfg, env_cfg.right_robot_cfg):
            aa = rc.actuators["arm"]
            aa.stiffness = args.arm_stiffness
            aa.damping = 2.0 * _m.sqrt(args.arm_stiffness)
            if args.arm_effort is not None:
                aa.effort_limit = args.arm_effort
    if args.no_fold:
        env_cfg.fold_to_reach = False
    if args.no_mute:
        env_cfg.mute_right_hand = False

    agent_cfg = PianoPPORunnerCfg()
    if args.lookahead is not None:
        env_cfg.goal_lookahead = args.lookahead
    if args.struck_frac is not None:
        env_cfg.key_struck_frac = args.struck_frac
    if args.no_norm:
        agent_cfg.empirical_normalization = False
        env_cfg.energy_weight = 0.0
    agent_cfg.seed = args.seed
    if args.init_noise is not None:
        agent_cfg.policy.init_noise_std = args.init_noise
    if args.lr is not None:
        agent_cfg.algorithm.learning_rate = args.lr
    if args.entropy_coef is not None:
        agent_cfg.algorithm.entropy_coef = args.entropy_coef
    if args.desired_kl is not None:
        agent_cfg.algorithm.desired_kl = args.desired_kl
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.save_interval is not None:
        agent_cfg.save_interval = args.save_interval

    leaf = args.tag if args.tag else f"seed{args.seed}"
    log_dir = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, leaf)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    # log-parameterize the action-noise std so it can't go negative (the scalar
    # std drifted < 0 and crashed PPO at iter 117 with "normal expects std>=0").
    agent_dict = agent_cfg.to_dict()
    agent_dict.setdefault("policy", {})["noise_std_type"] = "log"
    runner = OnPolicyRunner(env, agent_dict, log_dir=log_dir, device=agent_cfg.device)
    if args.resume_from:
        runner.load(args.resume_from)
        print(f"[train_piano] resumed from {args.resume_from}")
        if args.reset_noise_std is not None:
            # resumed checkpoints carry entropy-inflated stds (live ~0.5-0.7, dead ~1.0)
            # that smear collection (~-0.18 on logged arm_pos) and anneal far too slowly.
            import math as _math
            js = env.unwrapped.joint_scale[0]                      # (per_arm,) residual scales
            live_arm = (js > 0).nonzero().squeeze(-1)
            live = torch.cat([live_arm, live_arm + js.shape[0]])   # action layout [L | R]
            with torch.no_grad():
                pol = runner.alg.policy
                pol.log_std.data.fill_(_math.log(args.reset_dead_noise_std))
                pol.log_std.data[live] = _math.log(args.reset_noise_std)
            print(f"[train_piano] noise std reset: {live.numel()} live dims -> "
                  f"{args.reset_noise_std}, dead -> {args.reset_dead_noise_std}")
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    print(f"[train_piano] task={TASK} num_envs={args.num_envs} "
          f"song={env_cfg.midi_path} log_dir={log_dir}")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                 init_at_random_ep_len=not args.no_random_init)
    env.close()


main()
simulation_app.close()
