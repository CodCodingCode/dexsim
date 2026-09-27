import re
# 1) fingering.py: plan_fingering_hand gets split_key
p = "source/dexsim/piano/fingering.py"; s = open(p).read()
old = "                        hand_tol: float = 0.0,\n                        sticky_weight: float = 0.0) -> FingeringPlan:"
new = "                        hand_tol: float = 0.0,\n                        sticky_weight: float = 0.0,\n                        split_key: int | None = None) -> FingeringPlan:"
assert s.count(old) == 1; s = s.replace(old, new)
old = "        split = _balanced_split(active) if active.size else NUM_KEYS // 2\n"
new = ("        # hand split boundary: keys >= split go to the right hand. cfg.hand_split_key\n"
       "        # (default 44 = keyboard middle). 40 on nettspend: key 40 is assigned to\n"
       "        # the left thumb (cannot reach it) but is in fact hit by the RIGHT index\n"
       "        # finger 11/12 times in the 0.750 model -- give it to the hand that plays it.\n"
       "        split = int(split_key) if split_key is not None else (_balanced_split(active) if active.size else NUM_KEYS // 2)\n")
assert s.count(old) == 1; s = s.replace(old, new); open(p, "w").write(s)
# 2) song_bank.py: plumb split_key + pedal goal mid
p = "source/dexsim/tasks/piano_mj/song_bank.py"; s = open(p).read()
old = '            if method == "hand":\n                ot_kw["stagger_steps"] = int(getattr(cfg, "stagger_steps", 2))\n'
new = old + '            if method == "hand" and getattr(cfg, "hand_split_key", None) is not None:\n                ot_kw["split_key"] = int(cfg.hand_split_key)\n'
assert s.count(old) == 1; s = s.replace(old, new)
old = "        mid = NUM_KEYS // 2\n"
new = "        mid = int(getattr(cfg, \"hand_split_key\", None) or NUM_KEYS // 2)\n"
assert s.count(old) == 1; s = s.replace(old, new); open(p, "w").write(s)
# 3) cfg
p = "source/dexsim/tasks/piano_mj/piano_mj_env_cfg.py"; s = open(p).read()
old = "    pedal_goal_span: float = 0.14\n"
new = ("    # 0.07 (v11, 2026-09-25; was 0.14): with 0.14 the pedal was only requested for\n"
       "    # spans > 14 cm, but the little finger leaves key 28 for keys 6-9 cm away and\n"
       "    # 12 of its 22 onsets died 1-6 steps after the strike with the pedal UP\n"
       "    # (measured on v8 @1200). 0.07 requests the pedal on 87% of key-28 goal steps\n"
       "    # (was 38%); pedal-goal-on rises from 35% to 75% of the song.\n"
       "    pedal_goal_span: float = 0.07\n")
assert s.count(old) == 1; s = s.replace(old, new)
old = "    key_weight_floor: float = 0.25         # a mastered key keeps this fraction\n"
new = ("    # 0.10 (v11; was 0.25): weak-key curriculum -- a mastered key keeps only 10% of\n"
       "    # its static weight so the reward concentrates on the keys still being missed.\n"
       "    key_weight_floor: float = 0.10         # a mastered key keeps this fraction\n")
assert s.count(old) == 1; s = s.replace(old, new)
old = "    stagger_steps: int = 4\n"
new = old + ("    # Hand split boundary for the planner and the pedal goal: keys >= this go to\n"
             "    # the RIGHT hand. 44 = keyboard middle (old behaviour). 40 (v11, 2026-09-25):\n"
             "    # key 40 (C#4) was assigned to the left thumb, which cannot reach the keyboard,\n"
             "    # yet the 0.750 model's hits on it came from the RIGHT index finger (11/12).\n"
             "    hand_split_key: int = 40\n")
assert s.count(old) == 1; s = s.replace(old, new); open(p, "w").write(s)
# 4) env: metric masks follow the same split
p = "source/dexsim/tasks/piano_mj/piano_mj_env.py"; s = open(p).read()
i = s.index("        self.left_key_mask = (kidx <= split)")
j = s.rfind("split = ", 0, i); line_end = s.index("\n", j)
print("env split line was:", s[j:line_end])
s = s[:j] + "split = (int(getattr(cfg, \"hand_split_key\", None) or NUM_KEYS // 2) - 1)  # v11: keys > split are right-hand" + s[line_end:]
open(p, "w").write(s); print("patched")
