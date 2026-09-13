"""MuJoCo env for two Shadow Hands sliding along independent Y rails.

Direct port of the Isaac ``PianoEnv`` recipe onto plain MuJoCo (CPU):

  * **Residual action** over the 🔒 locked ready pose: zero action holds the
    ready hover; the policy learns pressing as a residual on the 42 position
    actuators (1 rail + 20 hand channels per hand).
  * **Fingering shaping reward** (finger -> assigned key) -- the make-or-break
    term (RoboPianist: F1 = 0 without it).
  * **Composite reward**: key-press + false-press penalty + fingering + onset
    + energy + idle-finger shaping + action-jerk penalty.
  * **Velocity-gated ("hammer") sounding**: a key rings only when struck past
    the sound angle while moving down fast enough, and keeps ringing until it
    springs back -- a statically-resting hand rings nothing.
  * **Rich observation**: proprioception + key state + goal lookahead +
    fingertip positions + fingering targets + the analytic SDF goal encoding.

One instance = one env (its own ``MjData``); the compiled ``MjModel`` and the
:class:`SongBank` are shared read-only across instances (see ``vec_env``).
All math is numpy; the reward functions in ``dexsim.piano.reward`` are
backend-agnostic and used as-is.
"""

from __future__ import annotations

import re

import mujoco
import numpy as np

from dexsim.piano import geometry, NUM_FINGERS, NUM_KEYS
from dexsim.piano.fingering import FINGERTIP_BODIES  # noqa: F401 (doc parity)
from dexsim.piano.goal_encoding import nearest_active_distance_np
from dexsim.piano.reward import (
    PianoRewardCfg, piano_reward, fingering_reward, onset_reward,
    idle_hover_reward, press_accuracy,
)
from dexsim.mjcf import KEY_SOUND_ANGLE, compile_scene
from dexsim.mjcf.shadow_hand import FINGERTIP_SITES
from .song_bank import SongBank

try:  # online fingering (min-cost assignment); the "hand" planner needs it too
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:  # pragma: no cover
    _lsa = None

_HAND_PREFIXES = ("L_", "R_")
_FLEX_RE = re.compile(r"robot0_(FF|MF|RF|LF|TH)J[123]$")


def measure_finger_offsets(model: mujoco.MjModel) -> np.ndarray:
    """(10,) fingertip world-Y offset from each hand's palm at the model's
    initial (ready) pose, [L th..lf, R th..lf]. Feeds the hand-relative
    fingering planner so it assigns fingers by where they physically sit."""
    d = mujoco.MjData(model); mujoco.mj_forward(model, d)
    out = []
    for p in _HAND_PREFIXES:
        palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p + "robot0_palm")
        for s_ in FINGERTIP_SITES:
            site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, p + s_)
            out.append(float(d.site_xpos[site, 1] - d.xpos[palm, 1]))
    return np.array(out)


class PianoMjEnv:
    """Single bimanual piano env (numpy API; see PianoMjVecEnv for batching)."""

    def __init__(self, cfg, model: mujoco.MjModel | None = None,
                 bank: SongBank | None = None, song_id: int = 0,
                 env_index: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng([int(getattr(cfg, "seed", 0)), int(env_index)])
        self.model = model if model is not None else compile_scene(cfg)
        self.data = mujoco.MjData(self.model)
        self.bank = bank if bank is not None else SongBank(
            cfg, finger_offsets=measure_finger_offsets(self.model))
        self.song_id = int(song_id) % self.bank.num_songs

        m = self.model
        self._cache_indices()
        self._build_ready_state()

        # per-actuator residual scale: gentle rail, generous hand (== Isaac)
        scale = np.empty(m.nu, dtype=np.float64)
        for i in range(m.nu):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            scale[i] = (cfg.arm_action_scale if name.endswith("A_rail")
                        else cfg.hand_action_scale)
        self._rail_act_mask = np.array(
            [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i).endswith("A_rail")
             for i in range(m.nu)])
        if getattr(cfg, "freeze_arms", False):
            scale[self._rail_act_mask] = 0.0
        elif getattr(cfg, "rail_follow", False):
            # servo does the travel; the policy keeps a small residual on top
            scale[self._rail_act_mask] = float(getattr(cfg, "rail_residual", 0.0))
        self.act_scale = scale
        self.ctrl_lo = m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = m.actuator_ctrlrange[:, 1].copy()

        self.reward_cfg = PianoRewardCfg(
            press_threshold=0.5,
            key_press_weight=cfg.key_press_weight,
            false_press_weight=cfg.false_press_weight,
            energy_weight=cfg.energy_weight,
            fingering_weight=cfg.fingering_weight,
            onset_weight=cfg.onset_weight,
            idle_hover_weight=cfg.idle_hover_weight,
            idle_hover_close=cfg.idle_hover_close,
            idle_hover_margin_mult=cfg.idle_hover_margin_mult,
            idle_hover_z_only=cfg.idle_hover_z_only,
        )

        # RECALL-GATED ANNEALING (press-discovery curriculum, == Isaac PianoEnv):
        # hold false-press at false_press_start (energy at 0) until this env's
        # recall EMA >= the gate, then ramp both to their cfg finals.
        self._anneal = bool(getattr(cfg, "anneal_false_press", False))
        if self._anneal:
            self._fp_final = float(cfg.false_press_weight)
            self._en_final = float(cfg.energy_weight)
            # start clamped to the final so the anneal can only ever LOWER the
            # early penalty, never raise it.
            self.reward_cfg.false_press_weight = min(
                float(cfg.false_press_start), self._fp_final)
            self.reward_cfg.energy_weight = 0.0
            self._anneal_ema = 0.0
            _steps = max(1, int(cfg.anneal_steps))
            self._fp_rate = max(
                0.0, self._fp_final - self.reward_cfg.false_press_weight) / _steps
            self._en_rate = self._en_final / _steps

        # static left/right key split for the per-hand F1 diagnostic
        split = (cfg.left_key_window[1] + cfg.right_key_window[0]) / 2.0
        kidx = np.arange(NUM_KEYS)
        self.left_key_mask = (kidx <= split).astype(np.float32)
        self.right_key_mask = (kidx > split).astype(np.float32)

        ep_s = float(getattr(cfg, "episode_length_s", 0.0))
        self.max_episode_length = (int(round(ep_s / cfg.control_dt)) if ep_s > 0
                                   else int(self.bank.song_len))
        self.key_half_h = geometry.KEY_HALF_H.astype(np.float64)

        # episode state
        self.song_step = 0
        self.episode_step = 0
        self.key_sounding = np.zeros(NUM_KEYS, dtype=bool)
        self._just_struck = np.zeros(NUM_KEYS, dtype=bool)
        self.prev_actions = np.zeros(cfg.action_space, dtype=np.float64)
        # adaptive key weighting: per-key recall EMA (starts at 0 = every key
        # still "unlearned" = full weight). NOT reset per episode on purpose.
        self._key_recall_ema = np.zeros(NUM_KEYS, dtype=np.float64)
        self._key_weight_s = 0.0
        self.pedal_down = False
        self._action_jerk = 0.0
        self._rail_ema = np.zeros(2)
        self.reset()

    # ------------------------------------------------------------- indexing
    def _cache_indices(self):
        m = self.model

        def jid(name):
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if i < 0:
                raise KeyError(f"no joint {name!r}")
            return i

        # piano keys, ordered 0..87
        kj = [jid(f"joint_{i}") for i in range(NUM_KEYS)]
        self.key_qadr = np.array([m.jnt_qposadr[j] for j in kj])
        self.key_dadr = np.array([m.jnt_dofadr[j] for j in kj])
        self.key_site = np.array([
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"key_site_{i}")
            for i in range(NUM_KEYS)])

        # per-hand joints (model order: rail first, then the 24 hand joints),
        # fingertip sites [th,ff,mf,rf,lf], palm body
        self.hand_qadr, self.hand_dadr = [], []
        self.hand_joint_suffixes = []
        self.tip_sites = []
        self.palm_body = []
        for p in _HAND_PREFIXES:
            js = [j for j in range(m.njnt)
                  if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "")
                  .startswith(p)]
            names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in js]
            self.hand_joint_suffixes.append([n[len(p):] for n in names])
            self.hand_qadr.append(np.array([m.jnt_qposadr[j] for j in js]))
            self.hand_dadr.append(np.array([m.jnt_dofadr[j] for j in js]))
            self.tip_sites.append(np.array([
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, p + s)
                for s in FINGERTIP_SITES]))
            self.palm_body.append(mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_BODY, p + self.cfg.hand_base_body))
        # actuator ids per hand, model order
        self.hand_acts = []
        for p in _HAND_PREFIXES:
            self.hand_acts.append(np.array([
                i for i in range(m.nu)
                if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i).startswith(p)]))
        self.n_act_per_hand = len(self.hand_acts[0])
        self.rail_act = [int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                               p + "A_rail"))
                         for p in _HAND_PREFIXES]
        self.rail_qadr = [int(m.jnt_qposadr[jid(p + "railJoint")])
                          for p in _HAND_PREFIXES]
        # critic extras: the body each fingertip site sits on (10,) and the
        # kinematic root of each hand (its mount body), for contact bookkeeping
        self.tip_bodies = np.concatenate([m.site_bodyid[sites]
                                          for sites in self.tip_sites])
        self.hand_root = [int(m.body_rootid[self.palm_body[h]]) for h in range(2)]
        # keys every per-key OBSERVATION chunk is sliced to (reward/F1 use all 88)
        self.obs_key_ids = np.array(self.cfg.obs_key_indices(), dtype=int)
        # egocentric obs: world Y of every key (fixed; keys only travel in Z)
        d0 = mujoco.MjData(m); mujoco.mj_forward(m, d0)
        self.key_y = d0.site_xpos[self.key_site, 1].copy()               # (88,)
        # per-hand upcoming-note tables (onset steps, keys) for searchsorted
        self._hand_ev_t = [[np.array([e[0] for e in ev], dtype=np.int64)
                            for ev in song] for song in self.bank.hand_events]
        self._hand_ev_k = [[np.array([e[1] for e in ev], dtype=np.int64)
                            for ev in song] for song in self.bank.hand_events]

    # --------------------------------------------------------- ready state
    def _build_ready_state(self):
        """Resolve the 🔒 locked ready pose into qpos + actuator ctrl."""
        cfg = self.cfg
        m = self.model
        self.ready_qpos = np.zeros(m.nq)
        for h, pose in enumerate((cfg.left_ready_pose, cfg.right_ready_pose)):
            for suffix, qadr in zip(self.hand_joint_suffixes[h], self.hand_qadr[h]):
                for pattern, value in pose.items():
                    if re.fullmatch(pattern, suffix):
                        self.ready_qpos[qadr] = float(value)
                        break
        # START-CURLED: curl every finger flex joint in the ready pose
        sc = float(getattr(cfg, "start_finger_curl", 0.0))
        if sc != 0.0:
            for h in range(2):
                for suffix, qadr in zip(self.hand_joint_suffixes[h],
                                        self.hand_qadr[h]):
                    if _FLEX_RE.fullmatch(suffix):
                        j = self._joint_of_qadr(qadr)
                        lo, hi = m.jnt_range[j]
                        self.ready_qpos[qadr] = np.clip(
                            self.ready_qpos[qadr] + sc, lo, hi)
        # actuator ctrl that HOLDS the ready pose: transmission length at the
        # ready qpos (exact for both joint and tendon actuators)
        d = mujoco.MjData(m)
        d.qpos[:] = self.ready_qpos
        mujoco.mj_forward(m, d)
        self.ready_ctrl = np.clip(d.actuator_length.copy(),
                                  m.actuator_ctrlrange[:, 0],
                                  m.actuator_ctrlrange[:, 1])
        # per-finger flexion actuator columns (for idle_finger_curl), per hand,
        # finger order [th,ff,mf,rf,lf]; identified by actuator name
        self._finger_flex_acts = []
        for p in _HAND_PREFIXES:
            cols = []
            for tag in ("TH", "FF", "MF", "RF", "LF"):
                ids = []
                for i in range(m.nu):
                    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                    if n.startswith(p + "robot0_A_" + tag) and n[-1] in "012":
                        ids.append(i)
                cols.append(np.array(ids, dtype=int))
            self._finger_flex_acts.append(cols)

    def _joint_of_qadr(self, qadr):
        return int(np.nonzero(self.model.jnt_qposadr == qadr)[0][0])

    # ----------------------------------------------------------------- reset
    def reset(self):
        self.data.qpos[:] = self.ready_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.ready_ctrl
        mujoco.mj_forward(self.model, self.data)
        self.song_step = 0
        if getattr(self.cfg, "random_song_start", False):
            song_len = int(self.bank.song_lens[self.song_id])
            hi = max(1, song_len - int(getattr(self.cfg, "random_start_min_steps", 40)))
            self.song_step = int(self.rng.integers(0, hi))
        self.episode_step = 0
        self.key_sounding[:] = False
        self._just_struck[:] = False
        self._prev_sounding = np.zeros_like(self.key_sounding)
        self.prev_actions[:] = 0.0
        self._action_jerk = 0.0
        self._rail_ema[:] = 0.0
        self.pedal_down = False
        return self._get_obs()

    # ------------------------------------------------------------------ step
    def step(self, action: np.ndarray):
        cfg = self.cfg
        a = np.clip(np.nan_to_num(np.asarray(action, dtype=np.float64)),
                    -1.0, 1.0)
        self._action_jerk = float(np.abs(a - self.prev_actions).mean())
        self.prev_actions = a.copy()
        if getattr(cfg, "sustain_pedal", False):
            self.pedal_down = bool(a[-1] > 0.0)
            a = a[:-1]                                # actuators only from here

        if getattr(cfg, "mute_right_hand", False):
            a = a.copy()
            a[self.n_act_per_hand:] = 0.0

        ctrl = self.ready_ctrl.copy()
        for h in range(2):
            ctrl[self.hand_acts[h]] += (self.act_scale[self.hand_acts[h]]
                                        * a[h * self.n_act_per_hand:
                                            (h + 1) * self.n_act_per_hand])
        if getattr(cfg, "rail_follow", False):
            self._apply_rail_follow(ctrl)
        if getattr(cfg, "idle_finger_curl", 0.0) and getattr(cfg, "rail_follow", False):
            fa = self.bank.finger_active[self.song_id, self.song_step]
            for h in range(2):
                for fi in range(5):
                    if not fa[h * 5 + fi]:
                        cols = self._finger_flex_acts[h][fi]
                        ctrl[cols] += cfg.idle_finger_curl
        np.clip(ctrl, self.ctrl_lo, self.ctrl_hi, out=ctrl)
        self.data.ctrl[:] = ctrl

        self._prev_sounding = self.key_sounding.copy()
        for _ in range(cfg.decimation):
            mujoco.mj_step(self.model, self.data)
            if getattr(cfg, "substep_strike_detect", True):
                self._update_strike_latch()

        reward, logs = self._compute_reward_and_logs(a)
        obs = self._get_obs()

        # dones (== Isaac: song end / timeout -> truncation; blow-up -> termination)
        self.episode_step += 1
        song_len = int(self.bank.song_lens[self.song_id])
        song_done = self.song_step >= song_len - 1
        time_out = self.episode_step >= self.max_episode_length
        blown = not (np.isfinite(self.data.qpos).all()
                     and np.isfinite(self.data.qvel).all())
        terminated = bool(blown)
        truncated = bool(song_done or time_out)
        self.song_step = min(self.song_step + 1, song_len - 1)
        logs["debug/blown"] = float(blown)
        if getattr(cfg, "sustain_pedal", False):
            logs["play/pedal_down"] = float(self.pedal_down)
        if getattr(cfg, "key_weight_mode", "none") == "adaptive":
            used = self.bank.goal[self.song_id, :int(song_len)].max(0) > 0.5
            logs["keyw/min_recall_ema"] = float(self._key_recall_ema[used].min()) if used.any() else 0.0
            logs["keyw/mean_recall_ema"] = float(self._key_recall_ema[used].mean()) if used.any() else 0.0
            logs["keyw/ramp"] = float(getattr(self, "_key_weight_s", 0.0))
        return obs, reward, terminated, truncated, logs

    # ------------------------------------------------------- rail servo
    def _hand_target_y(self, h: int, t: int, key_y, tip_y, palm_y) -> float | None:
        """Desired palm world-Y so hand h's fingers assigned at song step t land
        on their keys (finger-offset compensated). None if the hand is idle at t."""
        T = self.bank.finger_key.shape[1]
        t = min(int(t), T - 1)
        sl = slice(5 * h, 5 * h + 5)
        fa = self.bank.finger_active[self.song_id, t, sl]
        if not fa.any():
            return None
        fk = self.bank.finger_key[self.song_id, t, sl][fa]
        fidx = np.arange(5 * h, 5 * h + 5)[fa]
        return float((key_y[fk] - (tip_y[fidx] - palm_y[h])).mean())

    def _apply_rail_follow(self, ctrl):
        if getattr(self.cfg, "rail_leave_early", False):
            return self._apply_rail_leave_early(ctrl)
        return self._apply_rail_centroid(ctrl)

    def _apply_rail_leave_early(self, ctrl):
        """Rail servo that leaves for the next onset early enough to arrive on
        time (see PianoMjEnvCfg.rail_leave_early)."""
        cfg = self.cfg
        t0, sid = self.song_step, self.song_id
        key_y = self.key_y
        tip_y = self._fingertips_world()[:, 1]
        palm_y = self.data.xpos[self.palm_body, 1]
        mid = 0.5 * (float(cfg.left_base_pos[1]) + float(cfg.right_base_pos[1]))
        for h in range(2):
            base_y = float((cfg.left_base_pos, cfg.right_base_pos)[h][1])
            y_cur = self._hand_target_y(h, t0, key_y, tip_y, palm_y)
            ev_t = self._hand_ev_t[sid][h]
            i = int(np.searchsorted(ev_t, t0 + 1))
            y_next, lead_ok = None, False
            if i < len(ev_t):
                t_next = int(ev_t[i])
                y_next = self._hand_target_y(h, t_next, key_y, tip_y, palm_y)
                if y_next is not None:
                    travel = abs(y_next - palm_y[h]) / max(float(cfg.rail_speed), 1e-3) \
                             + float(cfg.rail_settle_s)
                    lead_ok = (t_next - t0) * cfg.control_dt <= travel
            if y_cur is None:
                y = y_next if y_next is not None else base_y   # idle: pre-position
            elif lead_ok:
                y = y_next                                      # leave now or be late
            else:
                y = y_cur
            if getattr(cfg, "lane_clamp", True):
                y = min(y, mid) if base_y <= mid else max(y, mid)
            tgt = y - base_y
            sm = float(getattr(cfg, "arm_smooth", 0.0))
            self._rail_ema[h] = sm * self._rail_ema[h] + (1.0 - sm) * tgt
            i_act = self.rail_act[h]
            ctrl[i_act] = np.clip(self._rail_ema[h] + ctrl[i_act], self.ctrl_lo[i_act], self.ctrl_hi[i_act])

    def _apply_rail_centroid(self, ctrl):
        """Analytic 1-DoF twin of the Isaac WristPoseIK arm servo: slide each
        rail so the hand centers on the world-Y centroid of the keys it must
        play over the next ``arm_lookahead`` steps (EMA-smoothed, lane-clamped)."""
        cfg = self.cfg
        H = int(cfg.arm_lookahead)
        t0 = self.song_step
        idx = np.minimum(np.arange(t0, t0 + H), self.bank.finger_key.shape[1] - 1)
        fk = self.bank.finger_key[self.song_id, idx]        # (H,10)
        fa = self.bank.finger_active[self.song_id, idx]     # (H,10)
        key_y = self.data.site_xpos[self.key_site, 1]       # (88,) world Y
        mid = 0.5 * (float(cfg.left_base_pos[1]) + float(cfg.right_base_pos[1]))
        # finger-offset compensation: aim so the ASSIGNED FINGERTIP lands on its
        # key, not the palm (tips sit ~5 cm from the palm along the keyboard;
        # centering the palm left a 6 cm median finger-to-key error)
        tip_y = self._fingertips_world()[:, 1]                       # (10,)
        palm_y = self.data.xpos[self.palm_body, 1]                   # (2,)
        for h, sl in enumerate((slice(0, 5), slice(5, 10))):
            fa_h = fa[:, sl]
            if fa_h.any():
                keys = fk[:, sl][fa_h]                               # (n,)
                fidx = np.broadcast_to(np.arange(5 * h, 5 * h + 5), fa_h.shape)[fa_h]
                off = tip_y[fidx] - palm_y[h]                        # (n,) tip - palm
                y = float((key_y[keys] - off).mean())                # desired palm y
                if getattr(cfg, "lane_clamp", True):
                    base_y = float((cfg.left_base_pos, cfg.right_base_pos)[h][1])
                    y = min(y, mid) if base_y <= mid else max(y, mid)
                tgt = y - float((cfg.left_base_pos, cfg.right_base_pos)[h][1])
            else:
                tgt = 0.0                                    # idle -> rail home
            sm = float(getattr(cfg, "arm_smooth", 0.0))
            self._rail_ema[h] = sm * self._rail_ema[h] + (1.0 - sm) * tgt
            i = self.rail_act[h]
            # ctrl[i] currently holds the policy's residual (ready ctrl for the
            # rail is 0, so the step loop left scale * action there)
            ctrl[i] = np.clip(self._rail_ema[h] + ctrl[i], self.ctrl_lo[i], self.ctrl_hi[i])

    # --------------------------------------------------------- world helpers
    def _update_strike_latch(self):
        """Advance the velocity-gated ("hammer") sounding latch from the
        CURRENT instantaneous key state. Called per physics substep when
        cfg.substep_strike_detect (the strike's velocity spike lasts ~30ms --
        a 50ms control-rate snapshot misses most real presses)."""
        cfg = self.cfg
        angle = self.data.qpos[self.key_qadr]                # negative = pressed
        frac = np.clip(angle / KEY_SOUND_ANGLE, 0.0, 2.0)
        frac = np.nan_to_num(frac, nan=0.0, posinf=2.0, neginf=0.0)
        struck = frac >= cfg.key_struck_frac
        if getattr(cfg, "sounding_gate", "position") == "hammer":
            vel = np.nan_to_num(self.data.qvel[self.key_dadr])   # <0 = pressing down
            struck = struck & (vel < -cfg.key_strike_vel)
        released = frac < cfg.key_release_frac
        if self.pedal_down:                           # sustain: nothing releases
            released = np.zeros_like(released)
        self.key_sounding = (self.key_sounding | struck) & ~released

    def _key_pressed_fraction(self) -> np.ndarray:
        """(88,) velocity-gated sounding fraction at the control boundary.
        With substep_strike_detect the latch has already been advanced inside
        the decimation loop; otherwise this applies the Isaac control-rate
        semantics. Also computes the rising-edge onset diagnostic."""
        if not getattr(self.cfg, "substep_strike_detect", True):
            self._update_strike_latch()
        angle = self.data.qpos[self.key_qadr]
        frac = np.clip(angle / KEY_SOUND_ANGLE, 0.0, 2.0)
        frac = np.nan_to_num(frac, nan=0.0, posinf=2.0, neginf=0.0)
        # rising edge over the whole control step (prev snapshot taken in step())
        prev = getattr(self, "_prev_sounding", None)
        if prev is None:
            prev = np.zeros_like(self.key_sounding)
        self._just_struck = self.key_sounding & ~prev
        # a sustained key may be physically up (frac ~ 0) yet sounding: report
        # it as fully pressed so the press reward and the metric count it
        return np.where(self.key_sounding, np.maximum(frac, 1.0), 0.0).astype(np.float32)

    def _key_top_world(self) -> np.ndarray:
        """(88,3) measured world position of each key's press point."""
        top = self.data.site_xpos[self.key_site].copy()
        return np.nan_to_num(top, nan=0.0, posinf=10.0, neginf=-10.0)

    def _fingertips_world(self) -> np.ndarray:
        """(10,3) fingertip world positions, [L_th..L_lf, R_th..R_lf]."""
        tips = np.concatenate([self.data.site_xpos[self.tip_sites[0]],
                               self.data.site_xpos[self.tip_sites[1]]], axis=0)
        return np.nan_to_num(tips, nan=0.0, posinf=10.0, neginf=-10.0)

    def _online_fingering(self, tips: np.ndarray, key_top: np.ndarray, goal: np.ndarray):
        """RP1M-style live fingering: min-cost assignment of the 10 fingertips
        to the keys that are goals NOW, cost = euclidean tip->key-top distance.
        Returns (surface (10,3), active (10,) bool, matched_key (10,) int, -1 idle).
        With no goal keys nothing is active (fingering reward 0, as before)."""
        keys = np.nonzero(goal > 0.5)[0]
        surface = key_top[self.bank.finger_home].copy()
        active = np.zeros(NUM_FINGERS, dtype=bool)
        matched = np.full(NUM_FINGERS, -1, dtype=np.int64)
        if keys.size == 0:
            return surface, active, matched
        cost = np.linalg.norm(tips[:, None, :] - key_top[keys][None, :, :], axis=2)  # (10,K)
        rows, cols = _lsa(np.nan_to_num(cost, nan=1e3, posinf=1e3))
        for r, c in zip(rows, cols):
            surface[r] = key_top[keys[c]]
            active[r] = True
            matched[r] = keys[c]
        return surface, active, matched

    def _finger_targets_world(self, key_top: np.ndarray):
        """(surface (10,3), press (10,3), active (10,)) for the current step."""
        fk = self.bank.finger_key[self.song_id, self.song_step]
        fa = self.bank.finger_active[self.song_id, self.song_step]
        idx_safe = np.where(fa, fk, self.bank.finger_home)
        surface = key_top[idx_safe]
        press = surface.copy()
        press[:, 2] += np.where(fa, -geometry.PRESS_DEPTH, geometry.HOVER_CLEARANCE)
        return surface, press, fa

    # ----------------------------------------------------------- observations
    def _goal_now(self) -> np.ndarray:
        return self.bank.goal[self.song_id, self.song_step]

    def _onset_now(self) -> np.ndarray:
        return self.bank.onset[self.song_id, self.song_step]

    def _get_obs(self) -> np.ndarray:
        if getattr(self.cfg, "obs_mode", "global") == "ego":
            return self._get_obs_ego()
        return self._get_obs_global()

    def _get_obs_ego(self) -> np.ndarray:
        """Egocentric, position-invariant obs (layout documented in
        PianoMjEnvCfg.obs_mode). Everything key-related is expressed relative
        to the hand it belongs to, so the same input pattern means the same
        thing wherever the hand is on the keyboard."""
        cfg = self.cfg
        d = self.data
        K, U = int(cfg.ego_keys), int(cfg.ego_upcoming)
        cap = float(cfg.ego_time_cap)
        t0, sid = self.song_step, self.song_id
        fk = self.bank.finger_key[sid, t0]
        fa = self.bank.finger_active[sid, t0]
        parts = [d.qpos[self.hand_qadr[0]], d.qpos[self.hand_qadr[1]]]
        if getattr(cfg, "ego_hand_vel", False):
            parts += [d.qvel[self.hand_dadr[0]], d.qvel[self.hand_dadr[1]]]
        parts.append(d.qpos[self.rail_qadr])                                  # (2,) rail pos
        key_q, key_v = d.qpos[self.key_qadr], d.qvel[self.key_dadr]
        snd = self.key_sounding.astype(np.float32)
        if getattr(cfg, "ego_all_keys", False):
            parts.append(key_q)                                               # (88,) absolute
        if getattr(cfg, "ego_piano_roll", False):
            L = int(cfg.goal_lookahead)
            parts.append(self.bank.goal[sid, t0:t0 + L].reshape(-1))          # (L*88,) binary
        palm_y = d.xpos[self.palm_body, 1]                                    # (2,)
        for h in range(2):
            dy = self.key_y - palm_y[h]
            win = np.argsort(np.abs(dy))[:K]
            win = win[np.argsort(self.key_y[win])]                           # left -> right
            parts.append(np.stack([dy[win], key_q[win], key_v[win], snd[win]], 1).reshape(-1))
        # per hand: dy from palm to the hand's next assigned note (current if active)
        nxt = np.zeros(2)
        for h in range(2):
            sl = slice(5 * h, 5 * h + 5)
            if fa[sl].any():
                nxt[h] = self.key_y[fk[sl][fa[sl]]].mean() - palm_y[h]
            else:
                ev_t, ev_k = self._hand_ev_t[sid][h], self._hand_ev_k[sid][h]
                i = int(np.searchsorted(ev_t, t0))
                nxt[h] = (self.key_y[ev_k[i]] - palm_y[h]) if i < len(ev_t) else 0.0
        parts.append(nxt)
        # per finger: target - tip, press-now
        tips = self._fingertips_world()
        _, press, active = self._finger_targets_world(self._key_top_world())
        parts.append((press - tips).reshape(-1))
        parts.append(active.astype(np.float32))
        # per finger: steps to next onset, steps to release (clipped, /cap)
        parts.append(np.clip(self.bank.finger_next_onset[sid, t0], 0, cap) / cap)
        parts.append(np.clip(self.bank.finger_release[sid, t0], 0, cap) / cap)
        # per hand: next U upcoming notes (strictly after now): dy, steps
        for h in range(2):
            ev_t, ev_k = self._hand_ev_t[sid][h], self._hand_ev_k[sid][h]
            i = int(np.searchsorted(ev_t, t0 + 1))
            up = np.zeros((U, 2))
            for j in range(U):
                if i + j < len(ev_t):
                    up[j, 0] = self.key_y[ev_k[i + j]] - palm_y[h]
                    up[j, 1] = min(ev_t[i + j] - t0, cap) / cap
                else:
                    up[j] = (0.0, 1.0)
            parts.append(up.reshape(-1))
        if getattr(cfg, "sustain_pedal", False):
            parts.append(np.array([1.0 if self.pedal_down else 0.0]))
        if getattr(cfg, "obs_prev_action", False):
            parts.append(self.prev_actions)
        obs = np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1) for p in parts])
        return np.clip(np.nan_to_num(obs, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)

    def _get_obs_global(self) -> np.ndarray:
        """Per-key ("global") obs, in the order PianoMjEnvCfg.__post_init__ sizes it:
          hand qpos+qvel (100) | fingertip xyz (30) | key angles (K) |
          key vel (K) | key sounding (K) | goal lookahead (L*K) |
          target fingertip xyz (30) | [goal SDF (K)]
        K = len(self.obs_key_ids): per-key chunks are sliced to the observed
        keys (the 16 reachable ones by default; reward/F1 still use all 88)."""
        cfg = self.cfg
        L = cfg.goal_lookahead
        t0 = self.song_step
        kid = self.obs_key_ids
        look = self.bank.goal[self.song_id, t0:t0 + L][:, kid]          # (L,K)
        parts = [
            self.data.qpos[self.hand_qadr[0]], self.data.qvel[self.hand_dadr[0]],
            self.data.qpos[self.hand_qadr[1]], self.data.qvel[self.hand_dadr[1]],
        ]
        if cfg.obs_fingertip_pos:
            parts.append(self._fingertips_world().reshape(-1))
        parts.append(self.data.qpos[self.key_qadr[kid]])                # key angles
        if cfg.obs_key_vel:
            parts.append(self.data.qvel[self.key_dadr[kid]])            # <0 = down
        if cfg.obs_key_sounding:
            parts.append(self.key_sounding[kid].astype(np.float32))     # latch
        parts.append(look.reshape(-1))
        if cfg.obs_finger_targets:
            _, press, _ = self._finger_targets_world(self._key_top_world())
            parts.append(press.reshape(-1))
        if cfg.obs_goal_sdf:
            parts.append(nearest_active_distance_np(self._goal_now())[kid])
        obs = np.concatenate([np.asarray(p, dtype=np.float32) for p in parts])
        return np.clip(np.nan_to_num(obs, nan=0.0, posinf=50.0, neginf=-50.0),
                       -50.0, 50.0)

    def critic_extras(self) -> np.ndarray:
        """(cfg.critic_extra_dim(),) privileged features for the critic only,
        read from the CURRENT MjData (call right after step()/reset()):
          * fingertip |F| (10): summed contact force magnitude on each fingertip
            body / critic_tip_force_clip, clamped to [0, 1];
          * collided (1): any contact between a left-hand and a right-hand body.
        """
        cfg = self.cfg
        if not cfg.critic_obs:
            return np.zeros(0, dtype=np.float32)
        m, d = self.model, self.data
        tip_f = np.zeros(NUM_FINGERS, dtype=np.float64)
        collided = False
        f6 = np.zeros(6)
        want_f = bool(cfg.critic_obs_tip_forces)
        want_c = bool(cfg.critic_obs_collision)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if want_c and not collided:
                r1, r2 = m.body_rootid[b1], m.body_rootid[b2]
                if {r1, r2} == set(self.hand_root):
                    collided = True
            if want_f:
                hit = np.nonzero((self.tip_bodies == b1) | (self.tip_bodies == b2))[0]
                if hit.size:
                    mujoco.mj_contactForce(m, d, i, f6)
                    tip_f[hit] += float(np.linalg.norm(f6[:3]))
        parts = []
        if want_f:
            parts.append(np.clip(tip_f / max(float(cfg.critic_tip_force_clip), 1e-6),
                                 0.0, 1.0))
        if want_c:
            parts.append(np.array([float(collided)]))
        x = np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)
        return np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

    # ---------------------------------------------------------------- reward
    def _hand_f1(self, pressed, goal, key_mask) -> float:
        rec, prec = press_accuracy(pressed * key_mask, goal * key_mask)
        if (goal * key_mask).sum() <= 0:
            return 0.0
        return float(2 * rec * prec / (rec + prec + 1e-9))

    def _goal_weight_now(self, goal: np.ndarray, pressed: np.ndarray):
        """(88,) per-key reward weight for this step, or None (mode 'none').

        static  = bank.goal_w[song, step]  (inverse key frequency / note length)
        adaptive: x (floor + (1 - floor) * (1 - recall_ema_k)), where the EMA of
                  "is goal key k down" is updated every step the key is a goal.
        """
        cfg = self.cfg
        mode = getattr(cfg, "key_weight_mode", "none")
        if mode == "none":
            return None
        w = self.bank.goal_w[self.song_id, self.song_step]              # (88,)
        if mode == "adaptive":
            g = goal > 0.5
            if g.any():
                b = float(cfg.key_weight_beta)
                hit = (pressed >= self.reward_cfg.press_threshold).astype(np.float64)
                self._key_recall_ema[g] = b * self._key_recall_ema[g] + (1.0 - b) * hit[g]
            floor = float(cfg.key_weight_floor)
            w = w * (floor + (1.0 - floor) * (1.0 - self._key_recall_ema))
        if getattr(cfg, "key_weight_ramp", False):
            # blend from uniform (s=0) to full weighting (s=1) as recall rises;
            # uses the anneal's recall EMA when annealing, else the mean per-key EMA
            if self._anneal:
                s = self._anneal_ema / max(float(cfg.anneal_recall_gate), 1e-6)
            else:
                g = goal > 0.5
                s = float(self._key_recall_ema[g].mean()) if g.any() else 0.0
            s = float(np.clip(s, 0.0, 1.0))
            g = goal > 0.5
            w = np.where(g, 1.0 + (w - 1.0) * s, 0.0)
            self._key_weight_s = s
        return w.astype(np.float32)

    def _compute_reward_and_logs(self, action):
        cfg = self.cfg
        pressed = self._key_pressed_fraction()
        goal = self._goal_now()
        energy = float((action ** 2).mean())
        # DENSE GOAL-KEY PRESS: goal columns carry the raw depression fraction
        # (continuous reward as the key travels down, RoboPianist-style); the
        # velocity-latched `pressed` keeps governing non-goal columns (the
        # false-press term) and every metric below.
        if getattr(cfg, "dense_goal_press", True):
            raw = np.clip(self.data.qpos[self.key_qadr] / KEY_SOUND_ANGLE, 0.0, 2.0)
            raw = np.nan_to_num(raw, nan=0.0, posinf=2.0, neginf=0.0).astype(np.float32)
            raw = np.maximum(raw, pressed)            # sustained goal keys count as down
            reward_pressed = np.where(goal > 0.5, raw, pressed)
        else:
            reward_pressed = pressed
        goal_w = self._goal_weight_now(goal, pressed)
        r_key = float(piano_reward(reward_pressed, goal, self.reward_cfg,
                                   energy=energy, goal_weight=goal_w))

        key_top = self._key_top_world()
        surface, _press_tgt, active = self._finger_targets_world(key_top)
        tips = self._fingertips_world()
        online_logs = {}
        if getattr(cfg, "fingering_online", False):
            # live matching replaces the planned table for the REWARD only
            tbl_key = self.bank.finger_key[self.song_id, self.song_step]
            surface, active, matched = self._online_fingering(tips, key_top, goal)
            if active.any():
                d = np.linalg.norm(tips[active] - surface[active], axis=1)
                online_logs["finger/online_dist"] = float(d.mean())
                online_logs["finger/online_agree"] = float(
                    (matched[active] == tbl_key[active]).mean())
        r_finger = float(fingering_reward(tips, surface,
                                          active.astype(np.float32), self.reward_cfg))
        # hover targets for the NON-assigned fingers: home key top + clearance
        # (independent of the table so online/offline modes see the same target)
        press_tgt = key_top[self.bank.finger_home].copy()
        press_tgt[:, 2] += geometry.HOVER_CLEARANCE
        r_onset = float(onset_reward(pressed, self._onset_now(), self.reward_cfg,
                                     goal_weight=goal_w))

        # idle-finger hover shaping
        if self.reward_cfg.idle_hover_weight > 0.0:
            r_hover = float(idle_hover_reward(tips, press_tgt,
                                              active.astype(np.float32),
                                              self.reward_cfg))
        else:
            r_hover = 0.0

        r_jerk = -float(getattr(cfg, "jerk_weight", 0.0)) * self._action_jerk

        # metrics
        recall, precision = press_accuracy(pressed, goal)
        has_goal = goal.sum() > 0
        rec = float(recall) if has_goal else 0.0
        prec = float(precision) if has_goal else 0.0
        f1 = 2 * rec * prec / (rec + prec + 1e-9)

        # RECALL-GATED ANNEAL: track the recall EMA (only over steps that had
        # goal notes) and, while it sits above the gate, ramp the false-press
        # penalty and energy cost toward their finals. Affects the NEXT step's
        # r_key -- a one-step lag, irrelevant over a 2000-step ramp.
        if self._anneal:
            if has_goal:
                b = float(self.cfg.anneal_recall_beta)
                self._anneal_ema = b * self._anneal_ema + (1.0 - b) * rec
            if self._anneal_ema >= float(self.cfg.anneal_recall_gate):
                rc_ = self.reward_cfg
                rc_.false_press_weight = min(self._fp_final,
                                             rc_.false_press_weight + self._fp_rate)
                rc_.energy_weight = min(self._en_final,
                                        rc_.energy_weight + self._en_rate)
        played_on = self._just_struck.astype(np.float32)
        near = self.bank.onset_win[self.song_id, self.song_step]
        n_played = played_on.sum()
        on_timing = float((played_on * near).sum() / n_played) if n_played > 0 else 0.0

        g = lambda x: float(np.clip(np.nan_to_num(x), -10.0, 10.0))
        reward = g(r_key) + g(r_finger) + g(r_onset) + g(r_hover) + g(r_jerk)
        reward = float(np.clip(reward, -10.0, 10.0))

        logs = {
            "play/F1": f1,
            "play/recall": rec,
            "play/precision": prec,
            "play/keys_sounding": float((pressed >= 0.5).sum()),
            "play/onset_timing": on_timing,
            "play/F1_left": self._hand_f1(pressed, goal, self.left_key_mask),
            "play/F1_right": self._hand_f1(pressed, goal, self.right_key_mask),
            "play/has_goal": float(has_goal),
            "arm/action_jerk": self._action_jerk,
            "reward/key": g(r_key),
            "reward/finger": g(r_finger),
            "reward/onset": g(r_onset),
            "reward/idle_hover": g(r_hover),
            "reward/jerk_pen": g(r_jerk),
            "reward/total": reward,
        }
        logs.update(online_logs)
        if self._anneal:
            logs["curriculum/false_press_w"] = float(self.reward_cfg.false_press_weight)
            logs["curriculum/energy_w"] = float(self.reward_cfg.energy_weight)
            logs["curriculum/recall_ema"] = float(self._anneal_ema)
        return reward, logs
