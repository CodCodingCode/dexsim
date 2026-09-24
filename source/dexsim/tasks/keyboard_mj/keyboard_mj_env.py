"""MuJoCo typing env: two gantry-mounted Shadow Hands type text on the
MacBook keyboard scene (``dexsim.mjcf.keyboard_scene``).

Keystroke semantics (== the scripted demo's keylog): a key REGISTERS on the
rising edge of ``qpos <= -KEY_ACTUATION_DEPTH`` and releases when it comes
back above ``RELEASE_DEPTH`` (debounced, checked every physics substep).
Each registration is compared with the next keystroke of the text:

  * correct key (and matching shift state)  -> +key_reward, cursor advances
  * anything else                           -> -stray_penalty

Dense shaping (``reach_weight`` / ``press_weight``) pulls the touch-typing
finger for the next key onto its cap and rewards depressing it; a per-step
``time_penalty`` makes finishing sooner worth more; ``complete_bonus`` ends
the episode when the text is done. Rewards use the same "registered or not"
logic as the metrics, so play/accuracy and play/cps are what is optimised.
"""

from __future__ import annotations

import mujoco
import numpy as np

from dexsim.mjcf import keyboard as kb
from dexsim.mjcf import keyboard_scene as ks
from .keyboard_mj_env_cfg import (KeyboardMjEnvCfg, PER_HAND_ACT, GANTRY_ACT,
                                  NUM_FINGERS)

RELEASE_DEPTH = 0.0002
_HANDS = ("L", "R")
_FINGERS = ("th", "ff", "mf", "rf", "lf")
_FINGER_INDEX = {(h, f): i * 5 + j for i, h in enumerate(_HANDS)
                 for j, f in enumerate(_FINGERS)}
_SHIFT_KEYS = (kb.KEY_INDEX["lshift"], kb.KEY_INDEX["rshift"])


def scene_cfg_from(cfg: KeyboardMjEnvCfg) -> ks.KeyboardSceneCfg:
    return ks.KeyboardSceneCfg(sim_dt=cfg.sim_dt, hover=cfg.hover,
                               gantry_xy_limit=cfg.gantry_xy_limit,
                               gantry_z_range=tuple(cfg.gantry_z_range))


def compile_keyboard_model(cfg: KeyboardMjEnvCfg) -> mujoco.MjModel:
    return ks.compile_keyboard_scene(scene_cfg_from(cfg))


def keystrokes(text: str) -> list[tuple[int, bool]]:
    """text -> [(key index, shift?), ...]"""
    return [(kb.KEY_INDEX[k], s) for k, s in kb.text_to_keys(text)]


class KeyboardMjEnv:
    """Single typing env (numpy API; see vec_env for batching)."""

    def __init__(self, cfg: KeyboardMjEnvCfg, model: mujoco.MjModel | None = None,
                 env_index: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng([int(cfg.seed), int(env_index)])
        self.scene_cfg = scene_cfg_from(cfg)
        self.model = model if model is not None else ks.compile_keyboard_scene(self.scene_cfg)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self._cache_indices()

        self.ready_qpos = ks.ready_qpos(m, self.scene_cfg)
        self.ready_ctrl = ks.ready_ctrl(m, self.ready_qpos)
        self.ctrl_lo = m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = m.actuator_ctrlrange[:, 1].copy()

        # action scale per actuator: hand joints residual, gantry absolute
        scale = np.full(m.nu, cfg.hand_action_scale, dtype=np.float64)
        for i in range(m.nu):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if n.endswith("A_gantry_x") or n.endswith("A_gantry_y"):
                scale[i] = cfg.gantry_xy_action_scale
            elif n.endswith("A_gantry_z"):
                scale[i] = cfg.gantry_z_action_scale
        for h, frozen in enumerate((cfg.freeze_left_hand, cfg.freeze_right_hand)):
            if frozen:
                scale[self.hand_acts[h]] = 0.0
        self.act_scale = scale

        self.key_travel = kb.KEY_TRAVEL
        self.registered = np.zeros(kb.NUM_KEYS, dtype=bool)
        self.prev_actions = np.zeros(cfg.action_space)
        self._new_presses: list[int] = []
        self.text = ""
        self.strokes: list[tuple[int, bool]] = []
        self.cursor = 0
        self.n_correct = 0
        self.n_stray = 0
        self.episode_step = 0
        self.max_episode_length = 1
        self._set_text(cfg.text or cfg.corpus[0])

    # ---------------------------------------------------------------- setup
    def _cache_indices(self):
        m = self.model
        self.key_sites = ks.key_site_ids(m)
        self.key_qadr = ks.key_qpos_adr(m)
        self.key_dadr = np.array([m.jnt_dofadr[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{n}")] for n in kb.KEY_NAMES])
        tips = ks.fingertip_site_ids(m)
        self.tip_sites = np.array([tips[(h, f)] for h in _HANDS for f in _FINGERS])
        self.palm_bodies = np.array([mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, f"{h}_robot0_palm") for h in _HANDS])
        # per-hand joint qpos / dof addresses (gantry + hand), in joint order
        self.hand_qadr, self.hand_dadr, self.hand_acts = [], [], []
        for h in _HANDS:
            q, d = [], []
            for j in range(m.njnt):
                n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
                if n.startswith(h + "_"):
                    q.append(m.jnt_qposadr[j]); d.append(m.jnt_dofadr[j])
            self.hand_qadr.append(np.array(q)); self.hand_dadr.append(np.array(d))
            self.hand_acts.append(np.array([
                i for i in range(m.nu)
                if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i).startswith(h + "_")]))
        assert all(len(a) == PER_HAND_ACT for a in self.hand_acts), \
            [len(a) for a in self.hand_acts]
        assert all(len(q) == 24 + GANTRY_ACT for q in self.hand_qadr), \
            [len(q) for q in self.hand_qadr]
        self.key_top_local = kb.key_local_top_positions()

    def _set_text(self, text: str):
        self.text = text
        self.strokes = keystrokes(text)
        n = max(1, len(self.strokes))
        self.max_episode_length = int(round(
            (self.cfg.episode_head_s + self.cfg.time_per_char_s * n) / self.cfg.control_dt))

    def _sample_text(self) -> str:
        if self.cfg.text:
            return self.cfg.text
        t = self.cfg.corpus[int(self.rng.integers(len(self.cfg.corpus)))]
        return t[:self.cfg.max_chars]

    # ---------------------------------------------------------------- api
    def reset(self):
        self._set_text(self._sample_text())
        self.data.qpos[:] = self.ready_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.ready_ctrl
        mujoco.mj_forward(self.model, self.data)
        self.registered[:] = False
        self.prev_actions[:] = 0.0
        self.cursor = 0
        self.n_correct = self.n_stray = 0
        self.episode_step = 0
        self._action_jerk = 0.0
        return self._get_obs()

    def step(self, action: np.ndarray):
        cfg = self.cfg
        a = np.clip(np.nan_to_num(np.asarray(action, dtype=np.float64)), -1.0, 1.0)
        self._action_jerk = float(np.abs(a - self.prev_actions).mean())
        self.prev_actions = a.copy()
        ctrl = self.ready_ctrl.copy()
        for h in range(2):
            idx = self.hand_acts[h]
            ctrl[idx] += self.act_scale[idx] * a[h * PER_HAND_ACT:(h + 1) * PER_HAND_ACT]
        np.clip(ctrl, self.ctrl_lo, self.ctrl_hi, out=ctrl)
        self.data.ctrl[:] = ctrl

        self._new_presses = []
        for _ in range(cfg.decimation):
            mujoco.mj_step(self.model, self.data)
            self._update_key_edges()

        reward, logs = self._reward_and_logs(a)
        obs = self._get_obs()
        self.episode_step += 1
        done_text = self.cursor >= len(self.strokes)
        time_out = self.episode_step >= self.max_episode_length
        blown = not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        terminated = bool(done_text or blown)
        truncated = bool(time_out and not terminated)
        if done_text or time_out:
            secs = self.episode_step * cfg.control_dt
            logs["play/cps"] = self.cursor / max(secs, 1e-6)
            logs["play/episode_done_frac"] = self.cursor / max(len(self.strokes), 1)
        logs["debug/blown"] = float(blown)
        return obs, reward, terminated, truncated, logs

    # ------------------------------------------------------------- physics
    def _update_key_edges(self):
        q = self.data.qpos[self.key_qadr]
        now = (q <= -kb.KEY_ACTUATION_DEPTH) | (self.registered & (q <= -RELEASE_DEPTH))
        rising = np.nonzero(now & ~self.registered)[0]
        for i in rising:
            if i not in _SHIFT_KEYS:
                self._new_presses.append(int(i))
        self.registered = now

    def _shift_held(self) -> bool:
        return bool(self.registered[_SHIFT_KEYS[0]] or self.registered[_SHIFT_KEYS[1]])

    def _key_top_world(self) -> np.ndarray:
        return self.data.site_xpos[self.key_sites]

    def _tips_world(self) -> np.ndarray:
        return self.data.site_xpos[self.tip_sites]

    def _target(self, k: int = 0):
        """(key index, shift, finger index) of the k-th upcoming keystroke,
        or None past the end of the text."""
        i = self.cursor + k
        if i >= len(self.strokes):
            return None
        key, shift = self.strokes[i]
        hand, finger = kb.TOUCH_TYPING_FINGER[kb.KEY_NAMES[key]]
        return key, shift, _FINGER_INDEX[(hand, finger)]

    # -------------------------------------------------------------- reward
    def _reward_and_logs(self, action):
        cfg = self.cfg
        r_key = r_stray = 0.0
        shift = self._shift_held()
        for k in self._new_presses:
            tgt = self._target()
            if tgt is not None and k == tgt[0] and shift == tgt[1]:
                r_key += cfg.key_reward
                self.cursor += 1
                self.n_correct += 1
            else:
                r_stray -= cfg.stray_penalty
                self.n_stray += 1
        tgt = self._target()
        held = self.registered.copy()
        if tgt is not None:
            held[tgt[0]] = False
            if tgt[1]:
                held[list(_SHIFT_KEYS)] = False
        r_hold = -cfg.hold_penalty * float(held.sum())

        r_reach = r_press = 0.0
        dist = 0.0
        if tgt is not None:
            key, _, fi = tgt
            cap = self._key_top_world()[key]
            tip = self._tips_world()[fi]
            dist = float(np.linalg.norm(tip - cap))
            r_reach = cfg.reach_weight * max(0.0, 1.0 - dist / cfg.reach_scale)
            depth = float(np.clip(-self.data.qpos[self.key_qadr[key]] / self.key_travel, 0.0, 1.0))
            r_press = cfg.press_weight * depth
        r_time = -cfg.time_penalty
        r_energy = -cfg.energy_weight * float((action ** 2).mean())
        r_jerk = -cfg.jerk_weight * self._action_jerk
        r_done = cfg.complete_bonus if self.cursor >= len(self.strokes) else 0.0
        reward = r_key + r_stray + r_hold + r_reach + r_press + r_time + r_energy + r_jerk + r_done
        n_press = self.n_correct + self.n_stray
        logs = {
            "reward/key": r_key, "reward/stray": r_stray, "reward/hold": r_hold,
            "reward/reach": r_reach, "reward/press": r_press, "reward/done": r_done,
            "reward/energy": r_energy,
            "play/accuracy": self.n_correct / n_press if n_press else 1.0,
            "play/done_frac": self.cursor / max(len(self.strokes), 1),
            "play/stray_per_step": float(len(self._new_presses) > 0 and r_stray < 0),
            "play/target_dist": dist,
        }
        return float(reward), logs

    # ----------------------------------------------------------------- obs
    def _get_obs(self) -> np.ndarray:
        cfg = self.cfg
        d = self.data
        parts = []
        for h in range(2):
            parts.append(d.qpos[self.hand_qadr[h]])
        for h in range(2):
            parts.append(np.clip(d.qvel[self.hand_dadr[h]] * 0.1, -5.0, 5.0))
        tgt = self._target()
        caps = self._key_top_world()
        tips = self._tips_world()
        palms = d.xpos[self.palm_bodies]
        ref = caps[tgt[0]] if tgt is not None else caps[kb.KEY_INDEX["g"]]
        parts.append((tips - ref).reshape(-1) * 10.0)          # dm units
        parts.append((palms - ref).reshape(-1) * 10.0)
        for k in range(cfg.goal_lookahead):
            t = self._target(k)
            blk = np.zeros(2 + 2 + 1 + NUM_FINGERS)
            if t is not None:
                c = caps[t[0]]
                blk[0:2] = (c[:2] - palms[0, :2]) * 10.0
                blk[2:4] = (c[:2] - palms[1, :2]) * 10.0
                blk[4] = float(t[1])
                blk[5 + t[2]] = 1.0
            parts.append(blk)
        parts.append([float(self._shift_held())])
        if cfg.obs_key_state:
            parts.append(np.clip(-d.qpos[self.key_qadr] / self.key_travel, 0.0, 1.5))
        if cfg.obs_prev_action:
            parts.append(self.prev_actions)
        obs = np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1) for p in parts])
        assert obs.shape[0] == cfg.observation_space, (obs.shape, cfg.observation_space)
        return np.nan_to_num(obs)

    def critic_extras(self) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)

    def typed_string(self) -> str:
        return self.text[:self.cursor]
