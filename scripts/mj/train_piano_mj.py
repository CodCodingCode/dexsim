"""Train the bimanual piano policy in MuJoCo (rsl_rl PPO, CPU-vectorized sim).

MuJoCo twin of scripts/train/train_piano.py -- same task recipe, same PPO
hyper-parameters (from the Isaac PianoPPORunnerCfg), no Isaac boot. The sim
runs CPU-threaded (30-core box: ~64-128 envs is the sweet spot); the policy
trains on the GPU.

  .venv/bin/python scripts/mj/train_piano_mj.py --num_envs 64 --midi data/midi/song.mid
  .venv/bin/python scripts/mj/train_piano_mj.py --num_envs 64 --songs_npz data/multisong/repertoire40.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))

parser = argparse.ArgumentParser(description="Train bimanual piano policy (MuJoCo).")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--midi", default=None, help="path to the song .mid (default: cfg's)")
parser.add_argument("--songs_npz", default=None, help="MULTI-SONG goal bundle (.npz)")
parser.add_argument("--max_songs", type=int, default=0)
parser.add_argument("--max_iterations", type=int, default=2000)
parser.add_argument("--save_interval", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", default="cuda", help="policy device (sim is CPU)")
parser.add_argument("--threads", type=int, default=None, help="sim worker threads (single-process env)")
parser.add_argument("--workers", type=int, default=1,
                    help=">1: shard envs across this many PROCESSES (true parallelism; "
                         "the threaded env serializes on the GIL at ~400 steps/s)")
parser.add_argument("--freeze_arms", action="store_true", help="rails held; fingers only")
parser.add_argument("--no_rail_follow", action="store_true",
                    help="policy drives the rails from scratch (default cfg: servo + residual)")
parser.add_argument("--rail_residual", type=float, default=None,
                    help="m; policy residual around the rail servo target (default 0.05)")
parser.add_argument("--rail_follow", action="store_true",
                    help="rails servoed analytically to the note centroid; policy drives fingers only")
parser.add_argument("--no_mute", action="store_true", help="disable mute_right_hand")
parser.add_argument("--mute_right", action="store_true", help="hold the right hand at ready")
parser.add_argument("--no_fold", action="store_true", help="disable fold_to_reach")
parser.add_argument("--obs_mode", default=None, choices=["ego", "global"],
                    help="observation layout (default cfg: ego)")
parser.add_argument("--ego_hand_vel", action="store_true",
                    help="ego obs: include hand joint velocities (default off)")
parser.add_argument("--no_ego_all_keys", action="store_true",
                    help="ego obs: drop the 88 absolute key angles")
parser.add_argument("--no_ego_piano_roll", action="store_true",
                    help="ego obs: drop the goal piano roll (goal_lookahead x 88)")
parser.add_argument("--legacy_ego", action="store_true",
                    help="pre-2026-09-12 ego obs (314 dims) -- to resume older checkpoints")
parser.add_argument("--ego_finger_obs", action="store_true",
                    help="re-enable the per-finger obs block (target-tip, press-now, countdowns; "
                         "the 2026-09-13 nettspend_rp1m_a100 layout, 1234 dims)")
parser.add_argument("--no_demand_unsounded", action="store_true",
                    help="fingering reward keeps demanding a finger on keys that already ring")
parser.add_argument("--legacy_reach", action="store_true",
                    help="pre-2026-09-10 setup: rails +/-0.12 m and fold_to_reach on "
                         "(needed to play checkpoints trained before then)")
parser.add_argument("--fingering", default=None, choices=["heuristic", "ot", "hand"],
                    help="fingering plan: 'ot' = nearest-finger (RP1M); default cfg = heuristic")
parser.add_argument("--hand_action_scale", type=float, default=None)
parser.add_argument("--key_press_weight", type=float, default=None)
parser.add_argument("--false_press_weight", type=float, default=None)
parser.add_argument("--fingering_weight", type=float, default=None)
parser.add_argument("--onset_weight", type=float, default=None)
parser.add_argument("--idle_hover_weight", type=float, default=None)
parser.add_argument("--strike_vel", type=float, default=None)
parser.add_argument("--anneal_false_press", action="store_true",
                    help="recall-gated curriculum: hold false-press at --false_press_start (energy 0) until recall EMA >= gate, then ramp to the cfg finals")
parser.add_argument("--false_press_start", type=float, default=None)
parser.add_argument("--anneal_recall_gate", type=float, default=None)
parser.add_argument("--anneal_steps", type=int, default=None)
parser.add_argument("--start_curl", type=float, default=None)
parser.add_argument("--idle_finger_curl", type=float, default=None)
parser.add_argument("--lookahead", type=int, default=None)
parser.add_argument("--episode_s", type=float, default=None,
                    help="episode length in seconds (default 0 = the whole song)")
parser.add_argument("--no_random_start", action="store_true",
                    help="reset every env at song step 0 (default: uniformly random step)")
parser.add_argument("--sounding_gate", default=None, choices=["position", "hammer"],
                    help="key sounds by depression only (default) or depression + strike speed")
parser.add_argument("--no_fingering_online", action="store_true",
                    help="fingering reward from the planned table instead of live matching")
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--entropy_coef", type=float, default=None)
parser.add_argument("--init_noise", type=float, default=None)
parser.add_argument("--desired_kl", type=float, default=None)
parser.add_argument("--no_norm", action="store_true", help="disable obs normalization")
parser.add_argument("--legacy_obs", action="store_true",
                    help="A/B baseline: the pre-2026-09-09 obs (all 88 keys, no key vel/sounding, "
                         "goal SDF on) with a symmetric critic")
parser.add_argument("--num_steps_per_env", type=int, default=32)
parser.add_argument("--resume_from", default=None, help="checkpoint .pt to load before training")
parser.add_argument("--logger", default="tensorboard", choices=["tensorboard", "wandb"])
parser.add_argument("--wandb_project", default="dexsim-piano-mj")
parser.add_argument("--tag", default=None, help="run label -> log subdir / wandb name")
args = parser.parse_args()

import torch  # noqa: E402

from dexsim.tasks.piano_mj import PianoMjEnvCfg, PianoMjVecEnv, make_rsl_rl_env  # noqa: E402
from dexsim.tasks.piano_mj.vec_env import PianoMjSubprocVecEnv  # noqa: E402


def build_env_cfg() -> PianoMjEnvCfg:
    cfg = PianoMjEnvCfg()
    cfg.seed = args.seed
    if args.midi:
        cfg.midi_path = args.midi
    if args.songs_npz:
        cfg.songs_npz = args.songs_npz
        cfg.max_songs = args.max_songs
    if args.freeze_arms:
        cfg.freeze_arms = True
    if args.rail_follow:
        cfg.rail_follow = True
    if args.no_rail_follow:
        cfg.rail_follow = False
    if args.rail_residual is not None:
        cfg.rail_residual = args.rail_residual
    if args.mute_right and not args.no_mute:
        cfg.mute_right_hand = True
    if args.no_fold:
        cfg.fold_to_reach = False
    if args.anneal_false_press:
        cfg.anneal_false_press = True
    if args.ego_hand_vel:
        cfg.ego_hand_vel = True
    if args.no_ego_all_keys:
        cfg.ego_all_keys = False
    if args.no_ego_piano_roll:
        cfg.ego_piano_roll = False
    if args.legacy_ego:
        cfg.ego_hand_vel, cfg.ego_all_keys, cfg.ego_piano_roll = True, False, False
    if args.no_random_start:
        cfg.random_song_start = False
    if args.sounding_gate:
        cfg.sounding_gate = args.sounding_gate
    if args.no_fingering_online:
        cfg.fingering_online = False
    if args.ego_finger_obs:
        cfg.ego_finger_obs = True
    if args.no_demand_unsounded:
        cfg.fingering_demand_unsounded = False
    for name, val in [
        ("hand_action_scale", args.hand_action_scale),
        ("key_press_weight", args.key_press_weight),
        ("false_press_weight", args.false_press_weight),
        ("fingering_weight", args.fingering_weight),
        ("onset_weight", args.onset_weight),
        ("idle_hover_weight", args.idle_hover_weight),
        ("key_strike_vel", args.strike_vel),
        ("start_finger_curl", args.start_curl),
        ("idle_finger_curl", args.idle_finger_curl),
        ("goal_lookahead", args.lookahead),
        ("fingering_method", args.fingering),
        ("obs_mode", args.obs_mode),
        ("episode_length_s", args.episode_s),
        ("false_press_start", args.false_press_start),
        ("anneal_recall_gate", args.anneal_recall_gate),
        ("anneal_steps", args.anneal_steps),
    ]:
        if val is not None:
            setattr(cfg, name, val)
    if args.legacy_reach:
        cfg.rail_limit = 0.12
        cfg.arm_action_scale = 0.12
        cfg.fold_to_reach = True
        cfg.obs_mode = "global"
        cfg.rail_follow = False
        cfg.sustain_pedal = False
        cfg.rail_stiffness, cfg.rail_damping, cfg.rail_force = 1200.0, 120.0, 500.0
    if args.legacy_obs:
        cfg.obs_mode = "global"
        cfg.obs_reachable_keys_only = False
        cfg.obs_key_vel = False
        cfg.obs_key_sounding = False
        cfg.obs_goal_sdf = True
        cfg.critic_obs = False
    cfg.__post_init__()          # recompute obs size after overrides
    return cfg


def build_train_cfg(env_cfg: PianoMjEnvCfg) -> dict:
    """rsl_rl >=5.x runner config; hyper-parameters == the Isaac PianoPPORunnerCfg."""
    from dexsim.tasks.piano_mj.ppo_cfg import piano_ppo_cfg

    # asymmetric critic when the env emits the privileged "critic_priv" group
    priv = env_cfg.critic_extra_dim() > 0
    return piano_ppo_cfg(
        obs_groups={"actor": ["policy"],
                    "critic": ["policy", "critic_priv"] if priv else ["policy"]},
        num_steps_per_env=args.num_steps_per_env,
        save_interval=args.save_interval,
        logger=args.logger,
        wandb_project=args.wandb_project,
        run_name=args.tag or "",
        learning_rate=args.lr,
        entropy_coef=args.entropy_coef,
        desired_kl=args.desired_kl,
        init_std=args.init_noise,
        obs_normalization=False if args.no_norm else None,
    )


def main():
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() or "cuda" not in args.device \
        else "cpu"
    if device != args.device:
        print(f"[train] CUDA unavailable -> policy on {device}")

    env_cfg = build_env_cfg()
    if args.workers > 1:
        venv = PianoMjSubprocVecEnv(env_cfg, num_envs=args.num_envs, workers=args.workers)
    else:
        venv = PianoMjVecEnv(env_cfg, num_envs=args.num_envs, threads=args.threads)
    env = make_rsl_rl_env(venv, device=device)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{stamp}" + (f"_{args.tag}" if args.tag else "")
    log_dir = ROOT / "logs" / "piano_mj" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] {args.num_envs} envs ({'%d worker processes' % args.workers if args.workers > 1 else 'threaded'}), "
          f"policy on {device}, logs -> {log_dir}")
    print(f"[train] obs: actor {env_cfg.observation_space} dims "
          f"(mode {env_cfg.obs_mode}), critic "
          f"{env_cfg.state_space} dims (+{env_cfg.critic_extra_dim()} privileged)")

    from rsl_rl.runners import OnPolicyRunner

    try:
        runner = OnPolicyRunner(env, build_train_cfg(env_cfg), log_dir=str(log_dir),
                                device=device)
    except RuntimeError as e:
        if "out of memory" not in str(e) or device == "cpu":
            raise
        # shared-GPU box: fall back to CPU (the policy is a small MLP)
        print(f"[train] CUDA OOM ({e}); falling back to --device cpu")
        device = "cpu"
        env.device = device
        env.episode_length_buf = env.episode_length_buf.cpu()
        runner = OnPolicyRunner(env, build_train_cfg(env_cfg), log_dir=str(log_dir),
                                device=device)
    if args.resume_from:
        print(f"[train] resuming from {args.resume_from}")
        runner.load(args.resume_from)
    runner.learn(num_learning_iterations=args.max_iterations)
    runner.save(os.path.join(str(log_dir), "model_final.pt"))
    venv.close()
    print(f"[train] done -> {log_dir}")


if __name__ == "__main__":
    main()
