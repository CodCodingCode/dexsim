"""Roll out a checkpoint over the full song (deterministic mean action) and break
recall/precision down per key, per hand, and by note onset timing."""
import sys, re, numpy as np, torch
sys.path.insert(0, "source")
from dexsim.tasks.piano_mj import PianoMjEnvCfg, PianoMjVecEnv, make_rsl_rl_env
ckpt, midi = sys.argv[1], sys.argv[2]; fing = sys.argv[3] if len(sys.argv) > 3 else "heuristic"
cfg = PianoMjEnvCfg(); cfg.midi_path = midi; cfg.episode_length_s = 65.0; cfg.fingering_method = fing; cfg.random_song_start = False   # diagnose the WHOLE song from step 0
if len(sys.argv) > 4 and sys.argv[4] == "legacy_reach": cfg.rail_limit = 0.12; cfg.arm_action_scale = 0.12; cfg.fold_to_reach = True; cfg.obs_mode = "global"; cfg.rail_follow = False; cfg.sustain_pedal = False; cfg.rail_stiffness, cfg.rail_damping, cfg.rail_force = 1200.0, 120.0, 500.0
if len(sys.argv) > 4 and sys.argv[4] == "global": cfg.obs_mode = "global"
cfg.__post_init__()
venv = PianoMjVecEnv(cfg, num_envs=1, threads=1); env = venv.envs[0]
rl_env = make_rsl_rl_env(venv, device="cpu")
from rsl_rl.runners import OnPolicyRunner
from dexsim.tasks.piano_mj.ppo_cfg import piano_ppo_cfg
tcfg = piano_ppo_cfg(obs_groups={"actor": ["policy"], "critic": ["policy", "critic_priv"]})
runner = OnPolicyRunner(rl_env, tcfg, log_dir=None, device="cpu"); runner.load(ckpt)
policy = runner.get_inference_policy(device="cpu")
T = int(venv.bank.song_lens[0])
obs = rl_env.get_observations()
goal, snd, fk, fa = [], [], [], []
for t in range(T):
    with torch.no_grad(): a = policy(obs)
    goal.append(env._goal_now().copy()); fk.append(env.bank.finger_key[0, env.song_step].copy()); fa.append(env.bank.finger_active[0, env.song_step].copy())
    obs, _, done, _ = rl_env.step(a)
    snd.append(env.key_sounding.copy())
    if bool(done[0]) and t < T - 1: print(f"[diag] episode ended early at step {t}"); break
G = np.stack(goal).astype(bool); S = np.stack(snd); FK = np.stack(fk); FA = np.stack(fa)
tp = (S & G).sum(); rec = tp / G.sum(); prec = tp / max(S.sum(), 1)
print(f"\n=== {ckpt.split('/')[-1]}  deterministic rollout, {len(G)} steps ===")
print(f"overall  F1 {2*rec*prec/(rec+prec):.3f}  recall {rec:.3f}  precision {prec:.3f}   goal key-steps {G.sum()}  sounded key-steps {S.sum()}")
# per hand: which hand is assigned each key by the fingering plan
half = 5
print("\nper key (goal keys only):")
print(f"{'key':>4} {'hand':>5} {'finger':>7} {'goal':>6} {'hit':>5} {'recall':>7} {'false':>6}  note: false = sounded when NOT goal")
fname = ["th","ff","mf","rf","lf"]
for k in np.nonzero(G.any(0))[0]:
    g = G[:, k]; s = S[:, k]
    # which finger is assigned to this key when it's a goal
    fs = [(h, f) for t in np.nonzero(g)[0] for h in range(2) for f in range(half) if FA[t, h*half+f] and FK[t, h*half+f] == k]
    hf = max(set(fs), key=fs.count) if fs else (-1, -1)
    print(f"{k:4d} {'L' if hf[0]==0 else 'R' if hf[0]==1 else '?':>5} {fname[hf[1]] if hf[1]>=0 else '?':>7} {g.sum():6d} {(g&s).sum():5d} {(g&s).sum()/g.sum():7.2f} {(s&~g).sum():6d}")
# per hand totals via key mask
for name, mask in (("LEFT", env.left_key_mask.astype(bool)), ("RIGHT", env.right_key_mask.astype(bool))):
    g = G[:, mask]; s = S[:, mask]; t_ = (g & s).sum()
    print(f"{name:5s}: recall {t_/max(g.sum(),1):.3f}  precision {t_/max(s.sum(),1):.3f}  goal {g.sum()}  sounded {s.sum()}")
# onset timing: for each note onset, first sounded step offset (or miss)
on = G & ~np.vstack([np.zeros((1,88),bool), G[:-1]])
offs = []; miss = 0
for t, k in zip(*np.nonzero(on)):
    end = t
    while end + 1 < len(G) and G[end+1, k]: end += 1
    hit = np.nonzero(S[t:end+1, k])[0]
    if hit.size: offs.append(hit[0])
    else: miss += 1
offs = np.array(offs)
print(f"\nnote onsets: {on.sum()}  hit {len(offs)}  missed entirely {miss}  ({miss/on.sum():.0%})")
if offs.size: print(f"latency of hits (steps @50ms): median {np.median(offs):.0f}  mean {offs.mean():.1f}  >=3 late: {(offs>=3).mean():.0%}")
# note durations of missed vs hit
durs_hit, durs_miss = [], []
for t, k in zip(*np.nonzero(on)):
    end = t
    while end + 1 < len(G) and G[end+1, k]: end += 1
    (durs_hit if S[t:end+1, k].any() else durs_miss).append(end - t + 1)
print(f"note length (steps): hit median {np.median(durs_hit):.0f}, missed median {np.median(durs_miss) if durs_miss else float('nan'):.0f}")
# chords: notes per step when goal present
npg = G.sum(1)[G.any(1)]
print(f"chord size when goal present: mean {npg.mean():.2f}, max {npg.max()};  steps with goal: {G.any(1).sum()}/{len(G)}")
