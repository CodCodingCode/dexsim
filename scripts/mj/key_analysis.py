"""Per-onset analysis of one key: where is the assigned finger at each onset, is the hand there, does it strike?"""
import sys, numpy as np, torch
sys.path.insert(0, "source")
from dexsim.tasks.piano_mj import PianoMjEnvCfg, PianoMjVecEnv, make_rsl_rl_env
ckpt, midi, KEY = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = PianoMjEnvCfg(); cfg.midi_path = midi; cfg.episode_length_s = 65.0; cfg.random_song_start = False; cfg.__post_init__()
venv = PianoMjVecEnv(cfg, num_envs=1, threads=1); env = venv.envs[0]; rl = make_rsl_rl_env(venv, device="cpu")
from rsl_rl.runners import OnPolicyRunner
from dexsim.tasks.piano_mj.ppo_cfg import piano_ppo_cfg
runner = OnPolicyRunner(rl, piano_ppo_cfg(obs_groups={"actor": ["policy"], "critic": ["policy", "critic_priv"]}), log_dir=None, device="cpu"); runner.load(ckpt)
pol = runner.get_inference_policy(device="cpu")
T = int(venv.bank.song_lens[0]); b = env.bank; obs = rl.get_observations()
rec = []
for t in range(T):
    with torch.no_grad(): a = pol(obs)
    fa, fk = b.finger_active[0, env.song_step], b.finger_key[0, env.song_step]
    f = next((i for i in range(10) if fa[i] and fk[i] == KEY), -1)
    tips = env._fingertips_world(); palm = env.data.xpos[env.palm_body, 1]
    rec.append(dict(t=t, f=f, tip_dy=(tips[f,1]-env.key_y[KEY]) if f>=0 else np.nan, tip_dz=(tips[f,2]-env._key_top_world()[KEY,2]) if f>=0 else np.nan,
                    palm_dy=palm[f//5]-env.key_y[KEY] if f>=0 else np.nan, key_frac=float(np.clip(env.data.qpos[env.key_qadr[KEY]]/ -0.012, 0, 2)),
                    sounding=bool(env.key_sounding[KEY]), pedal=bool(env.pedal_down), goal=bool(b.goal[0, env.song_step, KEY] > 0.5)))
    obs, _, done, _ = rl.step(a)
    if bool(done[0]) and t < T-1: break
G = np.array([r["goal"] for r in rec]); on = np.nonzero(G & ~np.concatenate([[False], G[:-1]]))[0]
fn = ["th","ff","mf","rf","lf"]
print(f"key {KEY}: {len(on)} onsets")
print(f"{'t(s)':>6} {'len':>4} {'finger':>6} {'palm dy@on':>10} {'tip dy@on':>9} {'tip dz@on':>9} {'min tip dz':>10} {'max frac':>8} {'sounded':>7} {'pedal@on':>8}")
hit = 0
for s in on:
    e = s
    while e+1 < len(rec) and G[e+1]: e += 1
    seg = rec[s:e+1]; r0 = rec[s]
    snd = any(x["sounding"] for x in seg); hit += snd
    print(f"{s*0.05:6.1f} {e-s+1:4d} {(('L' if r0['f']<5 else 'R')+'-'+fn[r0['f']%5]) if r0['f']>=0 else 'none':>6} {r0['palm_dy']*100:9.1f}cm {r0['tip_dy']*100:8.1f}cm {r0['tip_dz']*100:8.1f}cm {min(x['tip_dz'] for x in seg)*100:9.1f}cm {max(x['key_frac'] for x in seg):8.2f} {str(snd):>7} {str(r0['pedal']):>8}")
print(f"\nsounded on {hit}/{len(on)} onsets. tip dy = lateral offset of the assigned fingertip from the key (0 = over it); tip dz = height above key top; max frac = deepest press during the note (>=1.0 = sound angle)")
