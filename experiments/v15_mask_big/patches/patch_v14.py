"""v14 = v13 (seq planner + pose G + leap curriculum) + three setup changes (2026-09-27):
 1. score sustain: a struck GOAL note keeps sounding until its goal ends; the pedal
    action is removed (43 -> 42 actions, obs loses the 2 pedal dims).
 2. rail servo follows the seq planner's palm (leave-early on the plan's next change).
 3. fingering reward uses the PLANNED finger (fingering_online False)."""
from pathlib import Path
def sub(path, old, new, count=1):
    p = Path(path); s = p.read_text(); assert s.count(old) == count, (path, old[:70], s.count(old)); p.write_text(s.replace(old, new)); print("edited", path)
cfg = "source/dexsim/tasks/piano_mj/piano_mj_env_cfg.py"
env = "source/dexsim/tasks/piano_mj/piano_mj_env.py"
# --- 1. cfg: pedal off, score sustain on ---
sub(cfg, "    sustain_pedal: bool = True\n",
    "    sustain_pedal: bool = False   # v14: replaced by score_sustain (per-note sustain from the score)\n"
    "    # v14 (2026-09-27): a struck GOAL note keeps ringing until its goal ends, even\n"
    "    # after the finger leaves; non-goal keys stop when released. Measured on v13\n"
    "    # @2300: the all-or-nothing pedal was lifted 102x/64 s as a 'clear' button\n"
    "    # and silenced 50 of the 108 early-cut notes; 18% of goal key-steps were\n"
    "    # lost inside struck notes. The score decides what is held, not the policy.\n"
    "    score_sustain: bool = True\n")
# --- 3. cfg: planned-finger reward ---
sub(cfg, "    fingering_online: bool = True\n",
    "    fingering_online: bool = False   # v14: shaping pulls the PLANNED finger (the thumb got 7 strikes/song under live matching)\n")
# --- 2. cfg: servo follows the plan ---
sub(cfg, "    rail_leave_early: bool = True\n",
    "    rail_leave_early: bool = True\n"
    "    # v14: with the seq planner, the servo targets the PLANNED palm (bank.seq_palm)\n"
    "    # instead of averaging finger offsets (which parked 1.6-2.8 cm short on\n"
    "    # stretched chords: the 44+56 octave was 0/14 at v13 @2300).\n"
    "    rail_follow_plan: bool = True\n")
# --- 1. env: score sustain in the latch ---
sub(env, "        released = frac < cfg.key_release_frac\n        if self.pedal_down:                           # sustain: nothing releases\n            released = np.zeros_like(released)\n",
    "        released = frac < cfg.key_release_frac\n        if self.pedal_down:                           # sustain: nothing releases\n            released = np.zeros_like(released)\n"
    "        if getattr(cfg, \"score_sustain\", False):\n"
    "            # per-note sustain from the score: a sounding GOAL key keeps ringing\n"
    "            # until its goal ends (v14; replaces the all-or-nothing pedal)\n"
    "            released = released & ~(self.key_sounding & (self._goal_now() > 0.5))\n")
# --- 2. env: servo follows the plan ---
sub(env, "    def _apply_rail_leave_early(self, ctrl):\n",
    "    def _plan_palm_world(self):\n"
    "        \"\"\"(T, 2) planned palm world-Y from the seq planner, or None. Also caches\n"
    "        per hand the index of the next step at which the plan's palm changes.\"\"\"\n"
    "        if getattr(self, \"_plan_palm_w\", None) is None:\n"
    "            plan = getattr(self.bank, \"seq_palm\", None)\n"
    "            if plan is None:\n"
    "                return None\n"
    "            off = float(np.mean(self.key_y - geometry.key_local_top_positions()[:, 1]))\n"
    "            pw = np.asarray(plan, dtype=np.float64) + off\n"
    "            T = pw.shape[0]\n"
    "            nxt = np.full((T, 2), T, dtype=np.int64)\n"
    "            for h in range(2):\n"
    "                last = T\n"
    "                for t in range(T - 2, -1, -1):\n"
    "                    if abs(pw[t + 1, h] - pw[t, h]) > 1e-9:\n"
    "                        last = t + 1\n"
    "                    nxt[t, h] = last\n"
    "            self._plan_palm_w, self._plan_next = pw, nxt\n"
    "        return self._plan_palm_w\n"
    "\n"
    "    def _apply_rail_follow_plan(self, ctrl, plan):\n"
    "        \"\"\"v14 servo: target the planned palm; leave for the plan's NEXT palm\n"
    "        when the time left is <= the travel time (same leave-early rule).\"\"\"\n"
    "        cfg = self.cfg\n"
    "        T = plan.shape[0]\n"
    "        t0 = min(self.song_step, T - 1)\n"
    "        palm_y = self.data.xpos[self.palm_body, 1]\n"
    "        mid = 0.5 * (float(cfg.left_base_pos[1]) + float(cfg.right_base_pos[1]))\n"
    "        for h in range(2):\n"
    "            base_y = float((cfg.left_base_pos, cfg.right_base_pos)[h][1])\n"
    "            y = float(plan[t0, h])\n"
    "            t_next = int(self._plan_next[t0, h])\n"
    "            if t_next < T:\n"
    "                y_next = float(plan[t_next, h])\n"
    "                travel = abs(y_next - palm_y[h]) / max(float(cfg.rail_speed), 1e-3) \\\n"
    "                         + float(cfg.rail_settle_s)\n"
    "                sl = slice(5 * h, 5 * h + 5)\n"
    "                tt = min(t0, self.bank.finger_key.shape[1] - 1)\n"
    "                fa_h = self.bank.finger_active[self.song_id, tt, sl]\n"
    "                age_h = self.bank.finger_onset_age[self.song_id, tt, sl]\n"
    "                dwell_ok = (not fa_h.any()) or (int(age_h[fa_h].min()) >= int(getattr(cfg, \"rail_min_dwell_steps\", 0)))\n"
    "                if (t_next - t0) * cfg.control_dt <= travel and dwell_ok:\n"
    "                    y = y_next\n"
    "            if getattr(cfg, \"lane_clamp\", True):\n"
    "                y = min(y, mid) if base_y <= mid else max(y, mid)\n"
    "            tgt = y - base_y\n"
    "            sm = float(getattr(cfg, \"arm_smooth\", 0.0))\n"
    "            self._rail_ema[h] = sm * self._rail_ema[h] + (1.0 - sm) * tgt\n"
    "            i_act = self.rail_act[h]\n"
    "            ctrl[i_act] = np.clip(self._rail_ema[h] + ctrl[i_act], self.ctrl_lo[i_act], self.ctrl_hi[i_act])\n"
    "\n"
    "    def _apply_rail_leave_early(self, ctrl):\n"
    "        if getattr(self.cfg, \"rail_follow_plan\", False):\n"
    "            plan = self._plan_palm_world()\n"
    "            if plan is not None:\n"
    "                return self._apply_rail_follow_plan(ctrl, plan)\n")
print("patch_v14 applied")
