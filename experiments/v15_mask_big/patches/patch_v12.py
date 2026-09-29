# v12: leap curriculum -- scaffold (lift every finger of a travelling hand), annealed to 0,
# swapped for an extra penalty on false presses struck while that hand travels.
p = "source/dexsim/tasks/piano_mj/piano_mj_env_cfg.py"; s = open(p).read()
old = "    idle_finger_curl: float = 0.0     # rad: curl NON-assigned fingers up (rail_follow)\n"
new = old + '''    # --- LEAP CURRICULUM (v12, 2026-09-25) ---
    # Measured on the 0.750 model: 93% of false strikes happen while the hand's rail
    # is moving (median 0.65 m/s), 60% of them 3+ cm from any goal key -- fingers
    # dragged across keys in transit. Scaffold-then-penalise:
    #   * while a hand's rail speed > leap_move_speed, add leap_lift rad of curl to
    #     ALL of that hand's flex actuators (lifts the fingers clear), scaled by
    #     (1 - leap_s);
    #   * false key-steps struck under a travelling hand cost an extra
    #     leap_penalty_weight * leap_s each;
    #   * leap_s ramps 0 -> 1 over leap_anneal_steps once the recall EMA passes
    #     anneal_recall_gate (same gate as the false-press anneal).
    # The scaffold is TRAINING-ONLY (random_song_start envs); deterministic evals
    # and playback run without it, so the score is always the policy's own lift.
    leap_lift: float = 0.35
    leap_move_speed: float = 0.10
    leap_penalty_weight: float = 0.5
    leap_anneal_steps: int = 2000
'''
assert s.count(old) == 1; s = s.replace(old, new); open(p, "w").write(s)

p = "source/dexsim/tasks/piano_mj/piano_mj_env.py"; s = open(p).read()
# init state
old = "        self._anneal = bool(getattr(cfg, \"anneal_false_press\", False))\n"
new = ("        self._leap_s = 0.0                      # leap curriculum progress 0 -> 1\n"
       "        self._leap_moving = np.zeros(2, bool)   # per hand: rail travelling this step\n"
       "        self._leap_scaffold = bool(getattr(cfg, \"random_song_start\", True))  # training envs only\n"
       "        self._rail_dadr = None\n") + old
assert s.count(old) == 1; s = s.replace(old, new)
# scaffold in step(): after the idle-finger curl block, before the ctrl clip
old = "        np.clip(ctrl, self.ctrl_lo, self.ctrl_hi, out=ctrl)\n        self.data.ctrl[:] = ctrl\n"
new = '''        # LEAP CURRICULUM scaffold: lift every finger of a travelling hand
        if self._rail_dadr is None:
            self._rail_dadr = [int(self.model.jnt_dofadr[self._joint_of_qadr(q)]) for q in self.rail_qadr]
        rail_v = np.abs(self.data.qvel[self._rail_dadr])
        self._leap_moving = rail_v > float(getattr(cfg, "leap_move_speed", 0.10))
        lift = float(getattr(cfg, "leap_lift", 0.0)) * (1.0 - self._leap_s)
        if lift > 0.0 and self._leap_scaffold:
            for h in range(2):
                if self._leap_moving[h]:
                    for fi in range(5):
                        ctrl[self._finger_flex_acts[h][fi]] += lift
        np.clip(ctrl, self.ctrl_lo, self.ctrl_hi, out=ctrl)
        self.data.ctrl[:] = ctrl
'''
assert s.count(old) == 1; s = s.replace(old, new)
# penalty + schedule in the reward
old = "        g = lambda x: float(np.clip(np.nan_to_num(x), -10.0, 10.0))\n        reward = g(r_key) + g(r_finger) + g(r_onset) + g(r_hover) + g(r_jerk) + g(r_pedal)\n"
new = '''        # LEAP CURRICULUM penalty: false key-steps under a travelling hand
        split = int(getattr(cfg, "hand_split_key", None) or NUM_KEYS // 2)
        key_hand = (np.arange(NUM_KEYS) >= split).astype(int)
        moving_false = float(((pressed >= 0.5) & (goal < 0.5) & self._leap_moving[key_hand]).sum())
        r_leap = -float(getattr(cfg, "leap_penalty_weight", 0.0)) * self._leap_s * moving_false
        if self._anneal and self._anneal_ema >= float(self.cfg.anneal_recall_gate):
            self._leap_s = min(1.0, self._leap_s + 1.0 / max(1, int(getattr(cfg, "leap_anneal_steps", 2000))))
        g = lambda x: float(np.clip(np.nan_to_num(x), -10.0, 10.0))
        reward = g(r_key) + g(r_finger) + g(r_onset) + g(r_hover) + g(r_jerk) + g(r_pedal) + g(r_leap)
'''
assert s.count(old) == 1; s = s.replace(old, new)
old = '            "reward/jerk_pen": g(r_jerk),\n'
new = old + '            "reward/leap_pen": g(r_leap),\n            "curriculum/leap_s": float(self._leap_s),\n            "leap/moving_false": moving_false,\n            "leap/hands_moving": float(self._leap_moving.sum()),\n'
assert s.count(old) == 1; s = s.replace(old, new)
open(p, "w").write(s); print("v12 patched")
