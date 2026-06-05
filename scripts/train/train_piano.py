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
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--freeze_hands", action="store_true", help="curriculum phase 1: drive arms only (hands frozen)")
parser.add_argument("--freeze_arms", action="store_true", help="fixed-hands mode: drive fingers only (arms held)")
parser.add_argument("--planar_ik", action="store_true", help="weighted+iterated planar IK (gantry)")
parser.add_argument("--planar_pin_x", action="store_true", help="pin depth (world X) too -> lateral-only gantry (best in rollout_f1 A/B)")
parser.add_argument("--freeze_last_dof", action="store_true", help="freeze wrist_3 (note: over-constrains -> under-presses in eval)")
parser.add_argument("--arm_ik_follow", action="store_true", help="arms servoed online by WristPoseIK to the fingering centroid; policy drives only the 48 finger DoF")
parser.add_argument("--arm_ik_hover", type=float, default=None, help="override arm_ik_hover (m palm hovers above keys)")
parser.add_argument("--strike_vel", type=float, default=None, help="override key_strike_vel (rad/s gate for a key to sound)")
parser.add_argument("--idle_clear_weight", type=float, default=None, help="penalty weight for idle fingers hanging low (anti-mash)")
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
parser.add_argument("--idle_finger_curl", type=float, default=None, help="curl NON-assigned fingers up in the base pose (rad; lift idle fingers off neighbor keys)")
parser.add_argument("--songs_npz", default=None, help="MULTI-SONG: train one policy across all songs in this precomputed goal bundle (.npz)")
parser.add_argument("--max_songs", type=int, default=0, help="cap multi-song training to the first N songs (0=all)")
parser.add_argument("--use_slider", action="store_true", help="SLIDER embodiment (RP1M): 2-DoF prismatic hands, 0mm placement")
parser.add_argument("--arm_ftip_track", action="store_true", help="drive arm via pos-only IK on the primary fingertip (lands finger on key, not palm 90mm short)")
parser.add_argument("--ftip_max_step", type=float, default=None, help="per-step arm travel for fingertip tracking")
parser.add_argument("--key_press_weight", type=float, default=None, help="reward for sounding the right key")
parser.add_argument("--onset_weight", type=float, default=None, help="reward for sounding a key on its onset")
parser.add_argument("--fingering_weight", type=float, default=None, help="shaping: fingertip near assigned key (lower = less hovering)")
parser.add_argument("--arm_base_weight", type=float, default=None, help="shaping: arm over note centroid (lower = less hovering)")
parser.add_argument("--tag", default=None, help="run label -> wandb run name + log subdir (for parallel A/B/C runs)")
parser.add_argument("--no_fold", action="store_true", help="disable fold_to_reach (use the song's real key positions, e.g. for RP1M)")
parser.add_argument("--no_mute", action="store_true", help="disable mute_right_hand (needed for two-handed songs)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
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
    env_cfg.freeze_hands = args.freeze_hands
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
        env_cfg.freeze_arms = False     # arms move (IK-driven), not held static
    if args.songs_npz:
        env_cfg.songs_npz = args.songs_npz
        env_cfg.max_songs = args.max_songs
    if args.use_slider:
        env_cfg.use_slider = True
        env_cfg.__post_init__()   # re-apply: swap robots to slider + resize spaces
    if args.arm_ftip_track:
        env_cfg.arm_ftip_track = True
    if args.ftip_max_step is not None:
        env_cfg.ftip_max_step = args.ftip_max_step
    if args.arm_ik_hover is not None:
        env_cfg.arm_ik_hover = args.arm_ik_hover
    if args.strike_vel is not None:
        env_cfg.key_strike_vel = args.strike_vel
    if args.idle_clear_weight is not None:
        env_cfg.idle_clear_weight = args.idle_clear_weight
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
    for _w in ("key_press_weight", "onset_weight", "fingering_weight", "arm_base_weight"):
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
    agent_cfg.seed = args.seed
    if args.init_noise is not None:
        agent_cfg.policy.init_noise_std = args.init_noise
    if args.lr is not None:
        agent_cfg.algorithm.learning_rate = args.lr
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations

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
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    print(f"[train_piano] task={TASK} num_envs={args.num_envs} "
          f"song={env_cfg.midi_path} log_dir={log_dir}")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


main()
simulation_app.close()
