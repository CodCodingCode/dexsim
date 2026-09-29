"""v15 patch (2026-09-28) on a copy of the v14 tree:
 1. seq planner: hard per-(finger,key) feasibility mask + finger exclusion.
 2. cfg: seq_reach_mask (npy path, (10,88) bool), seq_exclude_thumbs (default True).
 3. song_bank: pass both to plan_fingering_seq.
 4. train script: --hidden_dims A,B,C (actor+critic), --seq_reach_mask, --no_seq_exclude_thumbs.
"""
import re, sys, pathlib
root = pathlib.Path(__file__).resolve().parent
def sub(path, old, new, count=1):
    p = root / path; s = p.read_text()
    assert s.count(old) >= 1, f"{path}: pattern not found:\n{old[:200]}"
    if count == 1: assert s.count(old) == 1, f"{path}: pattern not unique ({s.count(old)}x):\n{old[:120]}"
    p.write_text(s.replace(old, new) if count != 1 else s.replace(old, new, 1)); print("patched", path)

# ---- 1. fingering_seq.py -------------------------------------------------
FS = "source/dexsim/piano/fingering_seq.py"
sub(FS, """def _hand_cost_grid(keys, off_sorted, P, key_y, is_black, thumb_mask,
                    reach_tol, reach_max, drop_cost, black_thumb_weight,
                    thumb_reach_max=None):""",
"""def _hand_cost_grid(keys, off_sorted, P, key_y, is_black, thumb_mask,
                    reach_tol, reach_max, drop_cost, black_thumb_weight,
                    thumb_reach_max=None, allowed=None):""")
sub(FS, """    c = np.where(d > rmax[None, None, :], _INF, c)
    c = c + black_thumb_weight * (is_black[keys][None, :, None] & thumb_mask[None, None, :])
    # f[i][j] = min cost for notes i.. using fingers j..""",
"""    c = np.where(d > rmax[None, None, :], _INF, c)
    if allowed is not None:                     # v15: measured (finger, key) feasibility, (F, K)
        c = np.where(~np.asarray(allowed, bool).T[None, :, :], _INF, c)
    c = c + black_thumb_weight * (is_black[keys][None, :, None] & thumb_mask[None, None, :])
    # f[i][j] = min cost for notes i.. using fingers j..""")
sub(FS, """                 last_key, last_step, t, hold, switch_penalty, switch_window,
                 thumb_reach_max=None):""",
"""                 last_key, last_step, t, hold, switch_penalty, switch_window,
                 thumb_reach_max=None, allowed=None):""")
sub(FS, """    cost = np.where(d > rmax[:, None], drop_cost * 4.0, cost)             # never beats dropping
    cost = cost + black_thumb_weight * (thumb[:, None] & is_black[keys][None, :])
    for r, f in enumerate(fingers_sorted):""",
"""    cost = np.where(d > rmax[:, None], drop_cost * 4.0, cost)             # never beats dropping
    if allowed is not None:                                                # v15 feasibility mask (F, K)
        cost = np.where(~np.asarray(allowed, bool), drop_cost * 4.0, cost)
    cost = cost + black_thumb_weight * (thumb[:, None] & is_black[keys][None, :])
    for r, f in enumerate(fingers_sorted):""")
sub(FS, """                       switch_penalty: float = 0.01, switch_window: int = 10,
                       verbose: bool = True) -> FingeringPlan:""",
"""                       switch_penalty: float = 0.01, switch_window: int = 10,
                       finger_key_mask=None, exclude_fingers=(),
                       verbose: bool = True) -> FingeringPlan:""")
sub(FS, """    thumb_mask = [np.isin(fingers_sorted[h], list(_THUMBS)) for h in range(2)]
    # palm grids in the key frame""",
"""    thumb_mask = [np.isin(fingers_sorted[h], list(_THUMBS)) for h in range(2)]
    # v15: hard feasibility -- measured (finger, key) reach mask and excluded fingers
    fk_mask = np.ones((NUM_FINGERS, NUM_KEYS), dtype=bool)
    if finger_key_mask is not None:
        fk_mask &= np.asarray(finger_key_mask, dtype=bool).reshape(NUM_FINGERS, NUM_KEYS)
    for f in exclude_fingers:
        fk_mask[int(f), :] = False
    def allowed_for(h, ks):
        return fk_mask[fingers_sorted[h]][:, np.asarray(ks, dtype=int)]      # (F, K)
    if verbose:
        n_ex = [int((~fk_mask[f]).sum()) for f in range(NUM_FINGERS)]
        print(f"[fingering:seq] feasibility mask: forbidden keys per finger {n_ex}"
              f"{' (excluded fingers ' + str(tuple(int(f) for f in exclude_fingers)) + ')' if len(exclude_fingers) else ''}")
    # palm grids in the key frame""")
# emission
sub(FS, """                e = _hand_cost_grid(ks, off_sorted[h], P[h], key_y, is_black, thumb_mask[h],
                                    reach_tol, reach_max, drop_cost, black_thumb_weight, thumb_reach_max)
                parts.append(e + drop_cost * extra)""",
"""                e = _hand_cost_grid(ks, off_sorted[h], P[h], key_y, is_black, thumb_mask[h],
                                    reach_tol, reach_max, drop_cost, black_thumb_weight, thumb_reach_max,
                                    allowed_for(h, ks))
                parts.append(e + drop_cost * extra)""")
# split_at
sub(FS, """                tot += float(_hand_cost_grid(ks, off_sorted[h], p, key_y, is_black, thumb_mask[h],
                                             reach_tol, reach_max, drop_cost, black_thumb_weight, thumb_reach_max)[0])""",
"""                tot += float(_hand_cost_grid(ks, off_sorted[h], p, key_y, is_black, thumb_mask[h],
                                             reach_tol, reach_max, drop_cost, black_thumb_weight, thumb_reach_max,
                                             allowed_for(h, ks))[0])""")
# re-issue pass
sub(FS, """                                                  t, {}, 0.0, 0, thumb_reach_max)}""",
"""                                                  t, {}, 0.0, 0, thumb_reach_max, allowed_for(h, sub))}""")
# output pass
sub(FS, """                                    switch_penalty, switch_window, thumb_reach_max)
            n_drop += len(group[h]) - len(assigned)""",
"""                                    switch_penalty, switch_window, thumb_reach_max,
                                    allowed_for(h, group[h]))
            n_drop += len(group[h]) - len(assigned)""")

# ---- 2. cfg -------------------------------------------------------------
CFG = "source/dexsim/tasks/piano_mj/piano_mj_env_cfg.py"
sub(CFG, """    seq_switch_window: int | None = None     # steps (10)
""",
"""    seq_switch_window: int | None = None     # steps (10)
    # v15 (2026-09-28): HARD feasibility. reach_check.py measured, through the real
    # env.step path at pose G, which (finger, key) pairs can sound under the policy's
    # maximum press: thumbs sound NOTHING (tip 2.5-6.5 cm above every key), long
    # fingers sound almost everything in rail range. The planner never assigns a
    # forbidden pair (the note is dropped/staggered or given to another finger/hand).
    seq_reach_mask: str | None = "reach_mask_poseG.npy"   # (10, 88) bool npy; cwd or tree root
    seq_exclude_thumbs: bool = True
""")

# ---- 3. song_bank ---------------------------------------------------------
SB = "source/dexsim/tasks/piano_mj/song_bank.py"
sub(SB, """                for k in ("reach_tol", "reach_max", "thumb_reach_max", "travel_weight", "move_cost",
                          "drop_cost", "min_hand_sep", "switch_penalty", "switch_window"):
                    v = getattr(cfg, "seq_" + k, None)
                    if v is not None:
                        ot_kw[k] = v
""",
"""                for k in ("reach_tol", "reach_max", "thumb_reach_max", "travel_weight", "move_cost",
                          "drop_cost", "min_hand_sep", "switch_penalty", "switch_window"):
                    v = getattr(cfg, "seq_" + k, None)
                    if v is not None:
                        ot_kw[k] = v
                # v15: measured feasibility mask + thumb exclusion
                mp = getattr(cfg, "seq_reach_mask", None)
                if mp:
                    import os
                    cands = [mp, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", mp)]
                    hit = next((c for c in cands if os.path.exists(c)), None)
                    if hit is None:
                        raise FileNotFoundError(f"seq_reach_mask {mp!r} not found (cwd or tree root)")
                    ot_kw["finger_key_mask"] = np.load(hit)
                if getattr(cfg, "seq_exclude_thumbs", False):
                    ot_kw["exclude_fingers"] = (L_THUMB, R_THUMB)
""")

# ---- 4. train script ------------------------------------------------------
TR = "scripts/mj/train_piano_mj.py"
sub(TR, """parser.add_argument("--desired_kl", type=float, default=None)
""",
"""parser.add_argument("--desired_kl", type=float, default=None)
parser.add_argument("--hidden_dims", default=None,
                    help="v15 A/B: actor+critic MLP widths, e.g. 1024,512,256 (default 512,256,128)")
parser.add_argument("--seq_reach_mask", default=None, help="v15: (10,88) bool npy of feasible (finger,key) pairs")
parser.add_argument("--no_seq_exclude_thumbs", action="store_true", help="v15: let the seq planner use thumbs")
""")
sub(TR, """    if args.no_fingering_online:
        cfg.fingering_online = False
""",
"""    if args.no_fingering_online:
        cfg.fingering_online = False
    if args.seq_reach_mask is not None:
        cfg.seq_reach_mask = args.seq_reach_mask
    if args.no_seq_exclude_thumbs:
        cfg.seq_exclude_thumbs = False
""")
sub(TR, """    return piano_ppo_cfg(
        obs_groups={"actor": ["policy"],
                    "critic": ["policy", "critic_priv"] if priv else ["policy"]},""",
"""    tcfg = piano_ppo_cfg(
        obs_groups={"actor": ["policy"],
                    "critic": ["policy", "critic_priv"] if priv else ["policy"]},""")
sub(TR, """        init_std=args.init_noise,
        obs_normalization=False if args.no_norm else None,
    )
""",
"""        init_std=args.init_noise,
        obs_normalization=False if args.no_norm else None,
    )
    if args.hidden_dims:
        dims = [int(x) for x in args.hidden_dims.split(",")]
        tcfg["actor"]["hidden_dims"] = dims
        tcfg["critic"]["hidden_dims"] = dims
        print(f"[train] hidden_dims -> {dims}")
    return tcfg
""")
print("v15 patch applied")
