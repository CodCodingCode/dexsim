p = "source/dexsim/tasks/piano_mj/piano_mj_env_cfg.py"; s = open(p).read()
old = "    hand_tol: float = 0.0\n"
new = """    hand_tol: float = 0.0
    # ARPEGGIATE: an onset group wider than the hand (e.g. the 44+56 octave the
    # right hand plays 14x in nettspend, 16.5 cm vs a 13.5 cm max span) is
    # split: the half nearer the hand plays at t, the other half this many
    # steps later with the pedal carrying the first. 2 steps (0.1 s) was at the
    # rail's physical limit (1.5 m/s + 0.1 s settle) and every staggered 56
    # onset went unstruck (2026-09-23); 4 steps = 0.2 s is reachable.
    stagger_steps: int = 4
"""
assert s.count(old) == 1; s = s.replace(old, new)
old2 = "    onset_window_steps: int = 2\n"
new2 = "    onset_window_steps: int = 4   # was 2; widened with stagger_steps so the rolled note is still paid\n"
assert s.count(old2) == 1; s = s.replace(old2, new2); open(p, "w").write(s)
p = "source/dexsim/tasks/piano_mj/song_bank.py"; s = open(p).read()
old = '                ot_kw["hand_tol"] = float(cfg.hand_tol)\n'
new = old + '            if method == "hand":\n                ot_kw["stagger_steps"] = int(getattr(cfg, "stagger_steps", 2))\n'
assert s.count(old) == 1; s = s.replace(old, new); open(p, "w").write(s); print("v10 patched")
