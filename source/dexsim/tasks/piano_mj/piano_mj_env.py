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

_HAND_PREFIXES = ("L_", "R_")
_FLEX_RE = re.compile(r"robot0_(FF|MF|RF|LF|TH)J[123]$")


class PianoMjEnv:
    """Single bimanual piano env (numpy API; see PianoMjVecEnv for batching)."""

    def __init__(self, cfg, model: mujoco.MjModel | None = None,
                 bank: SongBank | None = None, song_id: int = 0):
        self.cfg = cfg
        self.model = model if model is not None else compile_scene(cfg)
        self.data = mujoco.MjData(self.model)
        self.bank = bank if bank is not None else SongBank(cfg)
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
        if getattr(cfg, "freeze_arms", False) or getattr(cfg, "rail_follow", False):
            scale[self._rail_act_mask] = 0.0
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

        self.max_episode_length = int(round(cfg.episode_length_s / cfg.control_dt))
        self.key_half_h = geometry.KEY_HALF_H.astype(np.float64)

        # episode state
        self.song_step = 0
        self.episode_step = 0
        self.key_sounding = np.zeros(NUM_KEYS, dtype=bool)
        self._just_struck = np.zeros(NUM_KEYS, dtype=bool)
        self.prev_actions = np.zeros(cfg.action_space, dtype=np.float64)
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
        self.episode_step = 0
        self.key_sounding[:] = False
        self._just_struck[:] = False
        self._prev_sounding = np.zeros_like(self.key_sounding)
        self.prev_actions[:] = 0.0
        self._action_jerk = 0.0
        self._rail_ema[:] = 0.0
        return self._get_obs()

    # ------------------------------------------------------------------ step
    def step(self, action: np.ndarray):
        cfg = self.cfg
        a = np.clip(np.nan_to_num(np.asarray(action, dtype=np.float64)),
                    -1.0, 1.0)
        self._action_jerk = float(np.abs(a - self.prev_actions).mean())
        self.prev_actions = a.copy()

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
        return obs, reward, terminated, truncated, logs

    # ------------------------------------------------------- rail servo
    def _apply_rail_follow(self, ctrl):
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
        for h, sl in enumerate((slice(0, 5), slice(5, 10))):
            fa_h = fa[:, sl]
            if fa_h.any():
                keys = fk[:, sl][fa_h]
                y = float(key_y[keys].mean())
                if getattr(cfg, "lane_clamp", True):
                    base_y = float((cfg.left_base_pos, cfg.right_base_pos)[h][1])
                    y = min(y, mid) if base_y <= mid else max(y, mid)
                tgt = y - float((cfg.left_base_pos, cfg.right_base_pos)[h][1])
            else:
                tgt = 0.0                                    # idle -> rail home
            sm = float(getattr(cfg, "arm_smooth", 0.0))
            self._rail_ema[h] = sm * self._rail_ema[h] + (1.0 - sm) * tgt
            i = self.rail_act[h]
            ctrl[i] = np.clip(self._rail_ema[h], self.ctrl_lo[i], self.ctrl_hi[i])

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
        vel = np.nan_to_num(self.data.qvel[self.key_dadr])   # <0 = pressing down
        struck = (frac >= cfg.key_struck_frac) & (vel < -cfg.key_strike_vel)
        released = frac < cfg.key_release_frac
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
        return np.where(self.key_sounding, frac, 0.0).astype(np.float32)

    def _key_top_world(self) -> np.ndarray:
        """(88,3) measured world position of each key's press point."""
        top = self.data.site_xpos[self.key_site].copy()
        return np.nan_to_num(top, nan=0.0, posinf=10.0, neginf=-10.0)

    def _fingertips_world(self) -> np.ndarray:
        """(10,3) fingertip world positions, [L_th..L_lf, R_th..R_lf]."""
        tips = np.concatenate([self.data.site_xpos[self.tip_sites[0]],
                               self.data.site_xpos[self.tip_sites[1]]], axis=0)
        return np.nan_to_num(tips, nan=0.0, posinf=10.0, neginf=-10.0)

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
        cfg = self.cfg
        L = cfg.goal_lookahead
        t0 = self.song_step
        look = self.bank.goal[self.song_id, t0:t0 + L]
        parts = [
            self.data.qpos[self.hand_qadr[0]], self.data.qvel[self.hand_dadr[0]],
            self.data.qpos[self.hand_qadr[1]], self.data.qvel[self.hand_dadr[1]],
            self.data.qpos[self.key_qadr],
            look.reshape(-1),
        ]
        if cfg.obs_fingertip_pos:
            parts.append(self._fingertips_world().reshape(-1))
        if cfg.obs_finger_targets:
            _, press, _ = self._finger_targets_world(self._key_top_world())
            parts.append(press.reshape(-1))
        if cfg.obs_goal_sdf:
            parts.append(nearest_active_distance_np(self._goal_now()))
        obs = np.concatenate([np.asarray(p, dtype=np.float32) for p in parts])
        return np.clip(np.nan_to_num(obs, nan=0.0, posinf=50.0, neginf=-50.0),
                       -50.0, 50.0)

    # ---------------------------------------------------------------- reward
    def _hand_f1(self, pressed, goal, key_mask) -> float:
        rec, prec = press_accuracy(pressed * key_mask, goal * key_mask)
        if (goal * key_mask).sum() <= 0:
            return 0.0
        return float(2 * rec * prec / (rec + prec + 1e-9))

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
            reward_pressed = np.where(goal > 0.5, raw, pressed)
        else:
            reward_pressed = pressed
        r_key = float(piano_reward(reward_pressed, goal, self.reward_cfg,
                                   energy=energy))

        key_top = self._key_top_world()
        surface, press_tgt, active = self._finger_targets_world(key_top)
        tips = self._fingertips_world()
        r_finger = float(fingering_reward(tips, surface,
                                          active.astype(np.float32), self.reward_cfg))
        r_onset = float(onset_reward(pressed, self._onset_now(), self.reward_cfg))

        # idle-finger clearance penalty
        icw = float(getattr(cfg, "idle_clear_weight", 0.0))
        if icw > 0.0:
            kb_top = key_top[:, 2].max()
            plane = kb_top + float(getattr(cfg, "idle_clear_margin", 0.02))
            below = np.clip(plane - tips[:, 2], 0.0, None)
            idle = (~active).astype(np.float64)
            r_idle = -icw * float((below * idle).sum() / max(idle.sum(), 1.0))
        else:
            r_idle = 0.0

        # idle-finger hover shaping (positive twin)
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
        reward = (g(r_key) + g(r_finger) + g(r_onset) + g(r_idle) + g(r_hover)
                  + g(r_jerk))
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
            "reward/idle_clear": g(r_idle),
            "reward/idle_hover": g(r_hover),
            "reward/jerk_pen": g(r_jerk),
            "reward/total": reward,
        }
        if self._anneal:
            logs["curriculum/false_press_w"] = float(self.reward_cfg.false_press_weight)
            logs["curriculum/energy_w"] = float(self.reward_cfg.energy_weight)
            logs["curriculum/recall_ema"] = float(self._anneal_ema)
        return reward, logs
