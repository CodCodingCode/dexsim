"""Empirical per-(finger, key) reach check through the REAL env.step path.
For every finger and every key the song uses: park the hand so the finger's
resting tip sits over the key (rail commanded directly, servo off), then command
the policy's MAXIMUM flexion on that finger (action = +1 on its flex actuators)
for 20 steps. Pairs that still do not sound get a second chance: 40 random
settings of the rest of that hand's actuators (wrist, abduction, other fingers)
on top of the max flex. Writes a (10, 88) allowed mask (True = can sound)."""
import sys, numpy as np
sys.path.insert(0, "source")
from dexsim.tasks.piano_mj import PianoMjEnv, PianoMjEnvCfg
from dexsim.piano import geometry as geom
import dexsim.tasks.piano_mj.piano_mj_env as pe
MID = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else None
FN = ["th","ff","mf","rf","lf"]
cfg = PianoMjEnvCfg(); cfg.midi_path = MID; cfg.episode_length_s = 65; cfg.random_song_start = False
cfg.fingering_method = "seq"; cfg.rail_follow = False
cfg.__post_init__()
env = PianoMjEnv(cfg); env.reset()
b = env.bank; T = int(b.song_lens[0])
song_keys = np.nonzero(b.goal[0, :T].max(0) > 0.5)[0]
pos = {}
for h in range(2):
    for j, aid in enumerate(env.hand_acts[h]): pos[int(aid)] = h * env.n_act_per_hand + j
rail_idx = [pos[env.rail_act[h]] for h in range(2)]
hand_idx = [[pos[int(a)] for a in env.hand_acts[h] if int(a) != env.rail_act[h]] for h in range(2)]
flex_idx = [[[pos[int(a)] for a in env._finger_flex_acts[h][fi]] for fi in range(5)] for h in range(2)]
KSA = pe.KEY_SOUND_ANGLE
def settle(a, n):
    for _ in range(n): env.step(a)
def tips_y(): return env._fingertips_world()[:, 1]
# rail calibration per hand: tip y per unit rail action
gain, y0 = [], []
for h in range(2):
    env.reset(); a = np.zeros(cfg.action_space); settle(a, 15); ya = tips_y()[h*5:(h+1)*5].copy()
    env.reset(); a[rail_idx[h]] = 0.5; settle(a, 15); yb = tips_y()[h*5:(h+1)*5].copy()
    gain.append(float((yb - ya).mean() / 0.5)); y0.append(ya)
print(f"rail gain m/unit: L {gain[0]:.3f}  R {gain[1]:.3f}")
def trial(h, fi, k, extra=None):
    env.reset(); a = np.zeros(cfg.action_space)
    u = (env.key_y[k] - y0[h][fi]) / gain[h]
    if abs(u) > 1.0: return None
    a[rail_idx[h]] = u; settle(a, 15)
    yerr = float(tips_y()[h*5+fi] - env.key_y[k])
    if extra is not None:
        for idx, v in extra.items(): a[idx] = v
    for idx in flex_idx[h][fi]: a[idx] = 1.0
    frac = 0.0; snd = False; others = set(); zmin = 1e9
    for _ in range(20):
        env.step(a)
        f = float(np.clip(env.data.qpos[env.key_qadr[k]] / KSA, 0, 3)); frac = max(frac, f)
        snd |= bool(env.key_sounding[k]); others |= set(np.nonzero(env.key_sounding)[0].tolist()) - {int(k)}
        zmin = min(zmin, float(env._fingertips_world()[h*5+fi, 2] - env._key_top_world()[k, 2]))
    return dict(yerr=yerr, frac=frac, snd=snd, others=sorted(others), zmin=zmin)
mask = np.ones((10, 88), bool); rows = []
rng = np.random.default_rng(0)
print(f"\n{'finger':>7} " + " ".join(f"{k:>5d}{'b' if geom.KEY_IS_BLACK[k] else 'w'}" for k in song_keys))
for h in range(2):
    for fi in range(5):
        f = h*5+fi; cells = []
        for k in song_keys:
            r = trial(h, fi, int(k))
            if r is None: cells.append("  rail"); mask[f, k] = False; rows.append((f, int(k), "rail out of range", None)); continue
            if not r["snd"]:
                best = r
                for _ in range(40):
                    ex = {idx: float(v) for idx, v in zip(hand_idx[h], rng.uniform(-1, 1, len(hand_idx[h])))}
                    rr = trial(h, fi, int(k), ex)
                    if rr and rr["frac"] > best["frac"]: best = rr
                    if rr and rr["snd"]: break
                r = best
            tag = ("  YES" if r["snd"] else f"{r['frac']:5.2f}") + ("*" if r["others"] else " ")
            cells.append(tag)
            if not r["snd"]: mask[f, k] = False
            rows.append((f, int(k), r["snd"], r))
        print(f"{('L' if h==0 else 'R')+'-'+FN[fi]:>7} " + " ".join(f"{c:>6}" for c in cells))
print("\ncells: YES = key sounded under the policy's max press; number = deepest depression as a fraction of the sounding angle; * = a neighbour also sounded")
print("\nDEAD PAIRS (never sound, even with 40 random wrist/abduction/other-finger settings):")
for f, k, snd, r in rows:
    if snd is not True:
        extra = "" if r is None else f"depth {r['frac']:.2f}, tip {r['zmin']*100:+.1f} cm above key top, lateral err {r['yerr']*100:+.1f} cm"
        print(f"  {('L' if f<5 else 'R')+'-'+FN[f%5]:>5} key {k:2d} {'black' if geom.KEY_IS_BLACK[k] else 'white'}: {snd if r is None else extra}")
if OUT: np.save(OUT, mask); print(f"\nmask saved -> {OUT}  (allowed pairs among song keys: {mask[:, song_keys].sum()}/{10*len(song_keys)})")
