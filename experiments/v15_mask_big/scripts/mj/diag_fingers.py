"""Per-finger breakdown: planned onsets vs who actually struck, false strikes, hold usage."""
import sys, numpy as np, torch
sys.path.insert(0, "source")
from dexsim.tasks.piano_mj import PianoMjEnvCfg, PianoMjVecEnv, make_rsl_rl_env
ckpt, midi = sys.argv[1], sys.argv[2]; fing = sys.argv[3] if len(sys.argv) > 3 else "seq"
cfg = PianoMjEnvCfg(); cfg.midi_path = midi; cfg.episode_length_s = 65.0; cfg.fingering_method = fing; cfg.random_song_start = False
cfg.__post_init__()
venv = PianoMjVecEnv(cfg, num_envs=1, threads=1); env = venv.envs[0]
rl_env = make_rsl_rl_env(venv, device="cpu")
from rsl_rl.runners import OnPolicyRunner
from dexsim.tasks.piano_mj.ppo_cfg import piano_ppo_cfg
tcfg = piano_ppo_cfg(obs_groups={"actor": ["policy"], "critic": ["policy", "critic_priv"]})
runner = OnPolicyRunner(rl_env, tcfg, log_dir=None, device="cpu"); runner.load(ckpt)
policy = runner.get_inference_policy(device="cpu")
T = int(venv.bank.song_lens[0]); obs = rl_env.get_observations()
goal, snd, fk, fa, own, hcmd = [], [], [], [], [], []
for t in range(T):
    with torch.no_grad(): a = policy(obs)
    goal.append(env._goal_now().copy()); fk.append(env.bank.finger_key[0, env.song_step].copy()); fa.append(env.bank.finger_active[0, env.song_step].copy())
    obs, _, done, _ = rl_env.step(a)
    snd.append(env.key_sounding.copy()); own.append(env.hold_owner.copy()); hcmd.append(env.hold_cmd.copy())
    if bool(done[0]) and t < T - 1: print(f"[diag] episode ended early at step {t}"); break
G = np.stack(goal).astype(bool); S = np.stack(snd); FK = np.stack(fk); FA = np.stack(fa); OWN = np.stack(own); HC = np.stack(hcmd)
NF = OWN.shape[1]; names = [f"{h}{f}" for h in "LR" for f in ["th","ff","mf","rf","lf"]]
newS = S & ~np.vstack([np.zeros((1,88),bool), S[:-1]])
on = G & ~np.vstack([np.zeros((1,88),bool), G[:-1]])
prev_own = np.vstack([np.full((1,NF), -1), OWN[:-1]])
def striker(t, k):
    changed = np.nonzero((OWN[t] == k) & (prev_own[t] != k))[0]
    if changed.size: return int(changed[0])
    same = np.nonzero(OWN[t] == k)[0]
    return int(same[0]) if same.size else -1
strikes = np.zeros((NF+1, 2), int)
for t, k in zip(*np.nonzero(newS)):
    f = striker(t, k); strikes[f if f >= 0 else NF, 0 if G[t, k] else 1] += 1
plan = np.zeros((NF, 3), int); conf = np.zeros((NF, NF+1), int)
for t, k in zip(*np.nonzero(on)):
    end = t
    while end + 1 < len(G) and G[end+1, k]: end += 1
    pf = [f for f in range(NF) if FA[t, f] and FK[t, f] == k]
    if not pf: continue
    pf = pf[0]
    hits = [tt for tt in range(t, end+1) if newS[tt, k]]
    if not hits: plan[pf, 2] += 1; continue
    sf = striker(hits[0], k); plan[pf, 0 if sf == pf else 1] += 1; conf[pf, sf if sf >= 0 else NF] += 1
print(f"\n=== {ckpt.split('/')[-1]} per finger ===")
print(f"{'finger':>7} {'planned':>8} {'hit_self':>9} {'hit_other':>10} {'missed':>7} {'recall':>7} | {'strikes':>8} {'correct':>8} {'false':>6} {'prec':>5} | {'hold%':>6} {'hold_ok%':>8}")
for f in range(NF):
    p = plan[f]; tot = p.sum(); s = strikes[f]
    hc = HC[:, f]; o = OWN[:, f]
    ring = np.array([S[t, o[t]] if o[t] >= 0 else False for t in range(len(S))])
    holding = hc & ring
    ok = np.array([G[t, o[t]] if (holding[t] and o[t] >= 0) else False for t in range(len(S))])
    print(f"{names[f]:>7} {tot:8d} {p[0]:9d} {p[1]:10d} {p[2]:7d} {(p[0]+p[1])/max(tot,1):7.2f} | {s.sum():8d} {s[0]:8d} {s[1]:6d} {s[0]/max(s.sum(),1):5.2f} | {hc.mean()*100:6.1f} {ok.sum()/max(holding.sum(),1)*100:8.1f}")
s = strikes[NF]; print(f"{'none':>7} {'':>8} {'':>9} {'':>10} {'':>7} {'':>7} | {s.sum():8d} {s[0]:8d} {s[1]:6d} {s[0]/max(s.sum(),1):5.2f}   (no tip within 1.5 cm)")
print("\nplanned finger (rows) -> actual striker (cols):")
print(f"{'':>7} " + " ".join(f"{n:>4}" for n in names) + " none")
for f in range(NF):
    if conf[f].sum(): print(f"{names[f]:>7} " + " ".join(f"{c:4d}" for c in conf[f]))
