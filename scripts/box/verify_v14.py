"""v14 checks: dims, score sustain, servo follows the plan, planned-finger reward."""
import sys; sys.path.insert(0, "source")
import numpy as np
from dexsim.tasks.piano_mj import PianoMjEnv, PianoMjEnvCfg
from dexsim.piano import geometry as geom
MIDI = "results/nettspend - we not like you (1).mid"
def mk(**kw):
    cfg = PianoMjEnvCfg(); cfg.midi_path = MIDI; cfg.fingering_method = "seq"; cfg.random_song_start = False
    for k, v in kw.items(): setattr(cfg, k, v)
    cfg.__post_init__(); return cfg, PianoMjEnv(cfg)
cfg, env = mk()
print(f"[dims] obs {cfg.observation_space} (v13: 1235) critic {cfg.state_space} actions {cfg.action_space} (v13: 43); sustain_pedal={cfg.sustain_pedal} score_sustain={cfg.score_sustain} fingering_online={cfg.fingering_online} rail_follow_plan={cfg.rail_follow_plan}")
# --- 1. score sustain: a sounding GOAL key that is physically up keeps ringing; a non-goal key stops ---
env.reset(); b = env.bank
goal = b.goal[0, env.song_step] > 0.5; kg = int(np.nonzero(goal)[0][0]); kn = int(np.nonzero(~goal)[0][0])
env.key_sounding[:] = False; env.key_sounding[kg] = True; env.key_sounding[kn] = True
assert (env.data.qpos[env.key_qadr] > -1e-4).all(), "keys not at rest"
env._update_strike_latch()
print(f"[sustain] goal key {kg} up but sounding -> stays {bool(env.key_sounding[kg])}; non-goal key {kn} -> stays {bool(env.key_sounding[kn])}")
# advance to a step where kg is no longer a goal
t_end = env.song_step
while b.goal[0, t_end, kg] > 0.5: t_end += 1
env.song_step = t_end; env._update_strike_latch()
print(f"[sustain] after goal for key {kg} ends (step {t_end}) -> sounding {bool(env.key_sounding[kg])}")
# --- 2. servo follows plan: zero-action rollout, palm world-Y vs planned palm ---
def track(follow_plan):
    cfg, env = mk(rail_follow_plan=follow_plan); env.reset()
    plan = env._plan_palm_world() if follow_plan else np.asarray(env.bank.seq_palm) + float(np.mean(env.key_y - geom.key_local_top_positions()[:, 1]))
    T = int(env.bank.song_lens[0]); err = np.zeros((T, 2)); still = np.zeros(T, bool)
    a = np.zeros(cfg.action_space)
    for t in range(T):
        env.step(a); tt = min(env.song_step, T - 1)
        err[t] = env.data.xpos[env.palm_body, 1] - plan[tt]
        still[t] = tt >= 3 and np.all(np.abs(plan[tt-3:tt+1] - plan[tt]) < 1e-9)
    e = np.abs(err[still]) * 100
    return f"palm-plan |err| when the plan is still (>=0.15 s): median L {np.median(e[:,0]):.1f} cm R {np.median(e[:,1]):.1f} cm, 90th pct L {np.percentile(e[:,0],90):.1f} R {np.percentile(e[:,1],90):.1f}, >2cm L {(e[:,0]>2).mean():.0%} R {(e[:,1]>2).mean():.0%}"
print("[servo] old (finger-offset average):", track(False))
print("[servo] v14 (planned palm):          ", track(True))
# --- 3. planned-finger reward path ---
cfg, env = mk(); env.reset()
_, r, _, _, logs = env.step(np.zeros(cfg.action_space))
print(f"[reward] step ok r={r:.3f}; finger term {logs['reward/finger']:.3f}; online-matching logs present: {any(k.startswith('finger/online') for k in logs)} (expect False); pedal logs present: {any('pedal' in k for k in logs)} (expect False)")
