"""v15 check: plan the song with the feasibility mask; per-finger planned onsets, drops, travel."""
import sys, numpy as np
sys.path.insert(0, "source")
from dexsim.tasks.piano_mj import PianoMjEnv, PianoMjEnvCfg
from dexsim.piano import geometry as geom
FN = ["th","ff","mf","rf","lf"]
cfg = PianoMjEnvCfg(); cfg.midi_path = sys.argv[1]; cfg.episode_length_s = 65; cfg.random_song_start = False
cfg.fingering_method = "seq"
if len(sys.argv) > 2 and sys.argv[2] == "nomask": cfg.seq_reach_mask = None; cfg.seq_exclude_thumbs = False
cfg.__post_init__()
env = PianoMjEnv(cfg); b = env.bank; T = int(b.song_lens[0])
G = b.goal[0, :T] > 0.5; FK = b.finger_key[0, :T]; FA = b.finger_active[0, :T]
on = G & ~np.vstack([np.zeros((1, 88), bool), G[:-1]])
mask = np.load(cfg.seq_reach_mask) if cfg.seq_reach_mask else np.ones((10, 88), bool)
per = np.zeros(10, int); bad = 0; unassigned = 0; keys = {}
for t, k in zip(*np.nonzero(on)):
    end = t
    while end + 1 < T and G[end + 1, k]: end += 1
    fs = [f for tt in range(t, min(end, t + 8) + 1) for f in range(10) if FA[tt, f] and FK[tt, f] == k]
    if not fs: unassigned += 1; keys[int(k)] = keys.get(int(k), 0) + 1; continue
    f = fs[0]; per[f] += 1
    if not mask[f, k] or (cfg.seq_exclude_thumbs and f in (0, 5)): bad += 1
print(f"onsets {on.sum()}  assigned {on.sum() - unassigned}  unassigned {unassigned} (by key: {dict(sorted(keys.items()))})  forbidden pairs used: {bad}")
print("planned onsets per finger: L " + "/".join(f"{FN[i]} {per[i]}" for i in range(5)) + "   R " + "/".join(f"{FN[i]} {per[5+i]}" for i in range(5)))
if hasattr(b, "seq_palm"):
    tr = np.abs(np.diff(b.seq_palm[:T], axis=0)).sum(0); print(f"palm travel L {tr[0]:.2f} m  R {tr[1]:.2f} m")
print(f"obs {cfg.obs_dim if hasattr(cfg,'obs_dim') else '?'}  actions {cfg.action_space}")
