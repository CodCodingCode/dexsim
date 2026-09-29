"""v13 checks: seq planner vs hand planner on nettspend under pose G."""
import sys, time; sys.path.insert(0, "source")
import numpy as np
from dexsim.tasks.piano_mj import PianoMjEnv, PianoMjEnvCfg
from dexsim.piano import geometry as geom
from dexsim.tasks.piano_mj.piano_mj_env import measure_finger_offsets
names = ["L.th","L.ff","L.mf","L.rf","L.lf","R.th","R.ff","R.mf","R.rf","R.lf"]
MIDI = "results/nettspend - we not like you (1).mid"
def build(method):
    cfg = PianoMjEnvCfg(); cfg.midi_path = MIDI; cfg.fingering_method = method; cfg.__post_init__()
    t0 = time.time(); env = PianoMjEnv(cfg); dt = time.time() - t0
    return cfg, env, dt
def report(method, env, dt, show_phrase):
    b = env.bank; FA, FK = b.finger_active[0], b.finger_key[0]; T = int(b.song_lens[0]); key_y = env.key_y
    palm = getattr(b, "seq_palm", None)
    if palm is not None:
        palm = palm + float(np.mean(env.key_y - geom.key_local_top_positions()[:, 1]))
        mv = np.abs(np.diff(palm, axis=0)); print(f"planned palm: moves L {int((mv[:,0]>1e-6).sum())} (path {mv[:,0].sum():.1f} m), R {int((mv[:,1]>1e-6).sum())} (path {mv[:,1].sum():.1f} m); range L [{palm[:,0].min():+.2f},{palm[:,0].max():+.2f}] R [{palm[:,1].min():+.2f},{palm[:,1].max():+.2f}] (rails: L [-0.62,0], R [0,+0.62])")
    wide = [(t, h, sorted(FK[t, h*5:h*5+5][FA[t, h*5:h*5+5]].tolist())) for t in range(T) for h in range(2) if FA[t, h*5:h*5+5].sum() >= 2 and (t == 0 or not (FA[t-1, h*5:h*5+5] == FA[t, h*5:h*5+5]).all()) and key_y[FK[t, h*5:h*5+5][FA[t, h*5:h*5+5]]].max() - key_y[FK[t, h*5:h*5+5][FA[t, h*5:h*5+5]]].min() > 0.14]
    print(f"chords wider than 14 cm on one hand: {len(wide)}", wide[:8])
    G = b.goal[0, :T] > 0.5
    print(f"\n##### method={method}  (env+plan build {dt:.1f}s) #####")
    on = FA[:T] & ~np.vstack([np.zeros((1,10),bool), FA[:T-1]])
    print("note onsets per finger:", {n: int(on[:, i].sum()) for i, n in enumerate(names)})
    gon = G & ~np.vstack([np.zeros((1,88),bool), G[:-1]])
    covered = sum(1 for t, k in zip(*np.nonzero(gon)) if any(FA[t:t+5, f].any() and (FK[t:t+5, f] == k).any() for f in range(10)))
    print(f"goal onsets {gon.sum()}, assigned to a finger within 5 steps: {covered} ({covered/gon.sum():.0%})")
    for h, nm in ((0, "L"), (1, "R")):
        cents = [key_y[FK[t, h*5:h*5+5][FA[t, h*5:h*5+5]]].mean() for t in range(T) if FA[t, h*5:h*5+5].any()]
        d = np.abs(np.diff(cents)) * 100
        spans = [key_y[FK[t, h*5:h*5+5][FA[t, h*5:h*5+5]]].max()*100 - key_y[FK[t, h*5:h*5+5][FA[t, h*5:h*5+5]]].min()*100 for t in range(T) if FA[t, h*5:h*5+5].sum() >= 2]
        print(f"{nm} hand: centroid jumps >5cm {int((d>5).sum())}, >10cm {int((d>10).sum())}, path {d.sum()/100:.1f} m; chord span max {max(spans) if spans else 0:.1f} cm")
    # hand separation: both hands active -> min(R centroid - L centroid)
    seps = [key_y[FK[t,5:10][FA[t,5:10]]].mean() - key_y[FK[t,0:5][FA[t,0:5]]].mean() for t in range(T) if FA[t,0:5].any() and FA[t,5:10].any()]
    print(f"min R-L centroid separation when both active: {min(seps)*100:.1f} cm" if seps else "hands never both active")
    if show_phrase:
        for h, nm in ((1, "RIGHT"), (0, "LEFT")):
            print(f"  {nm} first 160 steps: step keys->finger | palm cm | palm move cm")
            prev = None
            for t in range(160):
                idx = [f for f in range(5) if FA[t, h*5+f]]
                if not idx: continue
                new = [f for f in idx if t == 0 or not FA[t-1, h*5+f] or FK[t-1, h*5+f] != FK[t, h*5+f]]
                if not new: continue
                c = (palm[t, h] if palm is not None else key_y[[FK[t, h*5+f] for f in idx]].mean())*100
                print(f"   {t:4d}  " + "  ".join(f"{FK[t,h*5+f]:2d}->{names[h*5+f][2:]}" for f in new).ljust(30) + f"{c:7.1f}  {'' if prev is None else f'{abs(c-prev):5.1f}'}")
                prev = c
cfg_h, env_h, dt_h = build("hand"); report("hand", env_h, dt_h, False)
cfg_s, env_s, dt_s = build("seq");  report("seq", env_s, dt_s, True)
# frame check: planned palms (key frame) -> world, inside the rail range?
plan_palm = getattr(env_s.bank, "_seq_palm", None)
loc = geom.key_local_top_positions()[:, 1]; off = float(np.mean(env_s.key_y - loc))
print(f"\nkey frame -> world offset {off:+.3f} m (std {np.std(env_s.key_y - loc):.4f})")
print("measured finger offsets (cm):", np.round(measure_finger_offsets(env_s.model)*100, 1))
print("obs dims", cfg_s.observation_space, "critic", cfg_s.state_space, "actions", cfg_s.action_space)
o = env_s.reset(); o2, r, te, tr, logs = env_s.step(np.zeros(cfg_s.action_space)); print("env step ok, reward", round(float(r), 3))
