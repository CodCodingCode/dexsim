"""v14 revision (2026-09-27, user request): the POLICY decides sustain, per finger.
Replaces score_sustain (automatic) with 10 hold actions: hold_f > threshold keeps
the key finger f most recently struck ringing after the finger lifts. Obs gains
per finger: holding-a-ringing-note, that note is a goal, steps until its goal
ends (/cap). Reward: hold_goal_weight * mean over ringing owned keys of
[hold == key is a goal]. Actions 42 -> 52, obs 1232 -> 1272."""
from pathlib import Path
def sub(path, old, new, count=1):
    p = Path(path); s = p.read_text(); assert s.count(old) == count, (path, old[:70], s.count(old)); p.write_text(s.replace(old, new)); print("edited", path)
cfg = "source/dexsim/tasks/piano_mj/piano_mj_env_cfg.py"
env = "source/dexsim/tasks/piano_mj/piano_mj_env.py"
# --- cfg ---
sub(cfg, "    score_sustain: bool = True\n",
    "    score_sustain: bool = False   # superseded by hold_per_finger (policy-decided sustain)\n"
    "    # v14 (user request 2026-09-27): sustain is a POLICY decision, per finger. The\n"
    "    # last NUM_FINGERS action dims are hold bits: while hold_f > hold_threshold,\n"
    "    # the key finger f most recently STRUCK keeps ringing after the finger lifts\n"
    "    # (any key, so a held wrong note still pays the false-press penalty). The\n"
    "    # striker is the fingertip within hold_owner_tol laterally and lowest over\n"
    "    # the key at the strike. Obs adds per finger: holding a ringing note, that\n"
    "    # note is a goal now, steps until its goal ends (/ego_time_cap).\n"
    "    hold_per_finger: bool = True\n"
    "    hold_threshold: float = 0.5\n"
    "    hold_owner_tol: float = 0.015     # m, lateral tolerance for striker identification\n"
    "    hold_goal_weight: float = 0.3     # reward: hold == (owned key is a goal), per ringing owned key\n")
sub(cfg, "        if self.sustain_pedal:\n            n += 2                          # pedal state, pedal goal (now)\n        if self.obs_prev_action:\n            n += self.action_space\n        return n\n",
    "        if self.sustain_pedal:\n            n += 2                          # pedal state, pedal goal (now)\n"
    "        if getattr(self, \"hold_per_finger\", False):\n            n += NUM_FINGERS * 3            # holding, owned key is goal, steps to its goal end\n"
    "        if self.obs_prev_action:\n            n += self.action_space\n        return n\n")
sub(cfg, "    def __post_init__(self):\n        self.action_space = 2 * PER_HAND_ACT + (1 if self.sustain_pedal else 0)\n",
    "    def __post_init__(self):\n        self.action_space = (2 * PER_HAND_ACT + (1 if self.sustain_pedal else 0)\n"
    "                             + (NUM_FINGERS if getattr(self, \"hold_per_finger\", False) else 0))\n")
# --- env: state init + reset ---
sub(env, "        self.prev_actions = np.zeros(cfg.action_space, dtype=np.float64)\n",
    "        self.prev_actions = np.zeros(cfg.action_space, dtype=np.float64)\n"
    "        self.hold_owner = np.full(NUM_FINGERS, -1, dtype=np.int64)   # key each finger last struck\n"
    "        self.hold_cmd = np.zeros(NUM_FINGERS, dtype=bool)             # hold bits from the action\n")
sub(env, "        self.pedal_down = False\n        self.prev_actions = a.copy()\n", "XX", 0)  # (no-op guard)
s = Path(env).read_text()
i = s.index("    def reset(self):"); j = s.index("        self.pedal_down = False\n", i)
s = s[:j] + "        self.hold_owner[:] = -1; self.hold_cmd[:] = False\n" + s[j:]
Path(env).write_text(s); print("edited reset")
# --- env: action parsing (holds are the LAST dims, stripped before the pedal) ---
sub(env, "        if getattr(cfg, \"sustain_pedal\", False):\n            self.pedal_down = bool(a[-1] > float(getattr(cfg, \"pedal_threshold\", 0.0)))\n            a = a[:-1]                                # actuators only from here\n",
    "        if getattr(cfg, \"hold_per_finger\", False):\n"
    "            self.hold_cmd = a[-NUM_FINGERS:] > float(getattr(cfg, \"hold_threshold\", 0.5))\n"
    "            a = a[:-NUM_FINGERS]\n"
    "        if getattr(cfg, \"sustain_pedal\", False):\n            self.pedal_down = bool(a[-1] > float(getattr(cfg, \"pedal_threshold\", 0.0)))\n            a = a[:-1]                                # actuators only from here\n")
# --- env: latch -- striker ownership + per-finger hold ---
sub(env, "        if getattr(cfg, \"score_sustain\", False):\n",
    "        if getattr(cfg, \"hold_per_finger\", False):\n"
    "            new = struck & ~self.key_sounding\n"
    "            if new.any():                          # who struck it: nearest tip, lowest over the key\n"
    "                tips = self._fingertips_world(); key_top = self._key_top_world()\n"
    "                tol = float(getattr(cfg, \"hold_owner_tol\", 0.015))\n"
    "                for k in np.nonzero(new)[0]:\n"
    "                    dy = np.abs(tips[:, 1] - key_top[k, 1]); dz = tips[:, 2] - key_top[k, 2]\n"
    "                    cand = np.nonzero(dy < tol)[0]\n"
    "                    if cand.size:\n"
    "                        self.hold_owner[int(cand[np.argmin(dz[cand])])] = int(k)\n"
    "            held = np.zeros(NUM_KEYS, dtype=bool)\n"
    "            own = self.hold_owner[self.hold_cmd & (self.hold_owner >= 0)]\n"
    "            held[own] = True\n"
    "            released = released & ~held\n"
    "        if getattr(cfg, \"score_sustain\", False):\n")
# --- env: obs (before prev action) ---
sub(env, "        if getattr(cfg, \"obs_prev_action\", False):\n            parts.append(self.prev_actions)\n",
    "        if getattr(cfg, \"hold_per_finger\", False):\n"
    "            own = self.hold_owner; has = own >= 0; safe = np.where(has, own, 0)\n"
    "            ringing = has & self.key_sounding[safe]\n"
    "            holding = ringing & self.hold_cmd\n"
    "            g_now = self.bank.goal[sid, t0] > 0.5\n"
    "            is_goal = has & g_now[safe]\n"
    "            capi = int(cap)\n"
    "            gwin = self.bank.goal[sid, t0:t0 + capi + 1, safe] > 0.5          # (<=cap+1, 10)\n"
    "            gwin = np.concatenate([gwin, np.zeros((1, NUM_FINGERS), bool)], 0)\n"
    "            t_end = np.argmin(gwin, axis=0).astype(np.float64)               # first non-goal step\n"
    "            t_end = np.where(is_goal, np.minimum(t_end, cap) / cap, 1.0)\n"
    "            parts += [holding.astype(np.float32), is_goal.astype(np.float32), t_end]\n"
    "        if getattr(cfg, \"obs_prev_action\", False):\n            parts.append(self.prev_actions)\n")
# --- env: reward ---
sub(env, "        reward = g(r_key) + g(r_finger) + g(r_onset) + g(r_hover) + g(r_jerk) + g(r_pedal) + g(r_leap)\n",
    "        r_hold = 0.0; n_hold = 0.0\n"
    "        if getattr(cfg, \"hold_per_finger\", False):\n"
    "            own = self.hold_owner; has = own >= 0; safe = np.where(has, own, 0)\n"
    "            ringing = has & self.key_sounding[safe]\n"
    "            n_hold = float((ringing & self.hold_cmd).sum())\n"
    "            if ringing.any():\n"
    "                want = goal[safe] > 0.5\n"
    "                r_hold = float(getattr(cfg, \"hold_goal_weight\", 0.0)) * float(\n"
    "                    (self.hold_cmd[ringing] == want[ringing]).mean())\n"
    "        reward = g(r_key) + g(r_finger) + g(r_onset) + g(r_hover) + g(r_jerk) + g(r_pedal) + g(r_leap) + g(r_hold)\n")
sub(env, "            \"reward/leap_pen\": g(r_leap),\n",
    "            \"reward/leap_pen\": g(r_leap),\n            \"reward/hold\": g(r_hold),\n            \"play/holds\": n_hold,\n")
print("patch_v14_hold applied")
