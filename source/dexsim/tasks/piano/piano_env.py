"""DirectRLEnv for the bimanual piano task (decoupled control, RoboPianist-style).

Two UR10e+Shadow arms (60 action DoF total) over an 88-key spring-loaded piano.
A MIDI song defines a per-step key-activation goal; an automatic fingering assigns
each note to a finger. The control is **decoupled**: pure-math IK (``WristPoseIK``)
servos the arms onto the upcoming-note centroid, while the RL policy learns finger
pressing as a residual on top of a static ready pose. The slider embodiment is the
same recipe with an analytic 1-DoF rail in place of the arm servo.

What this env implements from RoboPianist:
  * **Residual action** over a ready-pose base: arm columns are overwritten each
    step by IK; zero action already tracks a competent positioning, so the policy
    only learns the press.
  * **Fingering shaping reward** (finger -> assigned key): the term RoboPianist
    showed is make-or-break (F1 = 0 without it).
  * **Composite reward**: key-press (right keys down, none wrong) + fingering +
    onset (crisp attacks) + energy.
  * **Rich observation**: proprioception + key state + goal lookahead + fingertip
    positions + the fingering targets ("where fingers should go") + an analytic
    SDF goal encoding.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from dexsim.piano import (
    load_song, plan_fingering, geometry, FINGERTIP_BODIES, NUM_FINGERS, NUM_KEYS,
)
from dexsim.piano.reward import (
    piano_reward, fingering_reward, onset_reward, arm_position_reward,
    PianoRewardCfg,
)
from dexsim.piano.goal_encoding import nearest_active_distance
from dexsim.assets import KEY_SOUND_ANGLE
from .piano_env_cfg import PianoEnvCfg


class PianoEnv(DirectRLEnv):
    cfg: PianoEnvCfg

    def __init__(self, cfg: PianoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.per_arm_dof = self.left_robot.num_joints
        self.left_default = self.left_robot.data.default_joint_pos.clone()
        self.right_default = self.right_robot.data.default_joint_pos.clone()

        # Per-joint residual scale. The arm joints are stiff (stiffness 6000): a
        # large residual jerks them -> solver explodes -> NaN body poses. The hand
        # joints are weak (stiffness 3) and need a LARGE range to actually travel
        # between keys and press them. A single global scale can't serve both --
        # 0.5 blew up the arm; 0.15 was too small for the fingers to play (F1 stuck
        # ~0.02, fingers just resting on the reference keys). So scale arm joints
        # gently and hand joints generously.
        is_hand = torch.tensor(
            [1.0 if "robot0_" in n else 0.0 for n in self.left_robot.data.joint_names],
            device=self.device,
        )
        self.joint_scale = (
            self.cfg.arm_action_scale * (1.0 - is_hand)
            + self.cfg.hand_action_scale * is_hand
        ).unsqueeze(0)                                  # (1, 30) broadcast over envs
        # arm DoF (the non-"robot0_" joints): used by the arm-health diagnostics
        # (_arm_limit_margin) to score only the 6-per-arm UR10e joints, not fingers.
        self.arm_joint_mask = (is_hand < 0.5)           # (30,) bool, True for arm joints
        # curriculum phase 1: freeze the hand DoF (zero their residual) so the policy
        # drives ONLY the 12 arm DoF -> learns to position the hands before pressing.
        if getattr(self.cfg, "freeze_hands", False):
            self.joint_scale = self.joint_scale * (1.0 - is_hand).unsqueeze(0)
            print("[PianoEnv] curriculum phase 1: HANDS FROZEN, arms-only training")
        # FIXED-HANDS mode: freeze the 12 arm DoF so the arms hold a constant pose
        # hovering over the keyboard; the policy drives ONLY the 48 finger DoF to
        # press the keys (RoboPianist-style). Notes are folded into each hand's
        # reachable window so the fingers have notes to hit.
        if getattr(self.cfg, "freeze_arms", False):
            self.joint_scale = self.joint_scale * is_hand.unsqueeze(0)
            print("[PianoEnv] FIXED ARMS: hands held over keyboard, fingers-only training")
        # ARM-IK-FOLLOW mode: the policy drives ONLY the 48 finger DoF (zero arm
        # residual); the 12 arm DoF are servoed online by WristPoseIK to the per-hand
        # fingering centroid in _servo_arms(). Math positions the hands, RL presses.
        self._arm_ik_follow = bool(getattr(self.cfg, "arm_ik_follow", False))
        if self._arm_ik_follow:
            self.joint_scale = self.joint_scale * is_hand.unsqueeze(0)
            self._arm_dof_mask = (is_hand < 0.5)            # (30,) bool: the arm joints
            # per-finger flexion-joint columns (J1/J2/J3) for the idle-finger curl, in
            # per-hand finger order [th,ff,mf,rf,lf]; same layout on both robots.
            _names = self.left_robot.data.joint_names
            self._finger_flex_cols = [
                torch.tensor([i for i, n in enumerate(_names)
                              if f"robot0_{tag}J" in n and n[-1] in "123"],
                             device=self.device, dtype=torch.long)
                for tag in ("TH", "FF", "MF", "RF", "LF")
            ]
            print("[PianoEnv] ARM-IK-FOLLOW: WristPoseIK servos arms to the fingering "
                  "centroid; policy drives only the 48 finger DoF")

        # --- song(s) -> goal / onset / fingering tensors (STACKED over N songs) ---
        # Single-song training is just N=1. Multi-song training (cfg.songs_npz set)
        # stacks N real songs along a leading dim and assigns each env a song_id, so
        # one policy is trained across many songs -> generalization, not a per-song
        # specialist. Every per-song tensor is indexed [song_id, song_step].
        L = self.cfg.goal_lookahead
        songs = self._gather_songs()                                # list of (name, act(T,88), ons(T,88))
        self.song_names = [s[0] for s in songs]
        N = len(songs)
        lens = [s[1].shape[0] for s in songs]
        Tmax = max(lens)
        self.song_lens = torch.tensor(lens, device=self.device, dtype=torch.long)
        goals, onsets, fkeys, factives = [], [], [], []
        finger_home = None
        for name, act, ons in songs:
            T = act.shape[0]
            g = torch.as_tensor(act, dtype=torch.float32, device=self.device)
            o = torch.as_tensor(ons, dtype=torch.float32, device=self.device)
            # pad each song to Tmax + L so the stacked tensor is rectangular and
            # lookahead/clamped indexing past a short song's end is safe (zeros).
            gpad = torch.zeros((Tmax + L - T, NUM_KEYS), device=self.device)
            goals.append(torch.cat([g, gpad], 0))
            onsets.append(torch.cat([o, gpad.clone()], 0))
            plan = plan_fingering(act)
            fk = torch.as_tensor(plan.finger_key, device=self.device)        # (T,10)
            fa = torch.as_tensor(plan.finger_active, device=self.device)     # (T,10)
            fk = torch.cat([fk, fk[-1:].repeat(Tmax + L - T, 1)], 0)
            fa = torch.cat([fa, torch.zeros((Tmax + L - T, NUM_FINGERS),
                                            dtype=torch.bool, device=self.device)], 0)
            fkeys.append(fk); factives.append(fa)
            if finger_home is None:
                finger_home = torch.as_tensor(plan.home_key, device=self.device)
        self.goal_padded = torch.stack(goals, 0)                    # (N, Tmax+L, 88)
        self.onset_padded = torch.stack(onsets, 0)                  # (N, Tmax+L, 88)
        self.finger_key = torch.stack(fkeys, 0)                     # (N, Tmax+L, 10)
        self.finger_active = torch.stack(factives, 0)              # (N, Tmax+L, 10)
        self.finger_home = finger_home                             # (10,) shared
        self.song_len = int(Tmax)                                  # legacy: longest song
        # per-env song assignment: round-robin over songs (deterministic, balanced).
        self.song_id = (torch.arange(self.num_envs, device=self.device) % N).long()
        self.num_songs = N
        # Time-dilated onset target for the onset-timing metric. A finger takes a few
        # 20Hz steps to descend and trip the velocity-gated strike, so the played
        # rising edge lands a step or two AFTER the nominal onset -- exact-step
        # matching reads ~0. onset_win[t,k]=1 if a real onset for key k falls within
        # +/-W steps of t, so "struck near its onset" is measured with a tolerance.
        W = int(getattr(self.cfg, "onset_tol_steps", 3))            # ~+/-150ms @ 20Hz
        self.onset_tol_steps = W
        _base = self.onset_padded                                   # (N, Tmax+L, 88)
        _ow = _base.clone()
        for _s in range(1, W + 1):
            _fut = torch.zeros_like(_base); _fut[:, :-_s] = _base[:, _s:]
            _pst = torch.zeros_like(_base); _pst[:, _s:] = _base[:, :-_s]
            _ow = torch.maximum(_ow, torch.maximum(_fut, _pst))
        self.onset_win = _ow                                        # (N, Tmax+L, 88)
        self.song_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # velocity-gated sounding latch (see _key_pressed_fraction): which keys are
        # currently ringing (struck and not yet released).
        self.key_sounding = torch.zeros((self.num_envs, NUM_KEYS), dtype=torch.bool, device=self.device)
        # rising edge of key_sounding -> keys that STARTED sounding this step ("played
        # onsets"), set in _key_pressed_fraction. Used by the onset-F1 diagnostic.
        self._just_struck = torch.zeros_like(self.key_sounding)
        # static left/right key split for per-hand F1: every key assigned to the hand
        # whose reachable window it falls nearer to (split at the midpoint between the
        # two windows). Covers all 88 keys, disjoint, so per-hand F1 never drops notes.
        _split = (self.cfg.left_key_window[1] + self.cfg.right_key_window[0]) / 2.0
        _kidx = torch.arange(NUM_KEYS, device=self.device)
        self.left_key_mask = (_kidx <= _split).float()    # (88,) 1.0 for left-hand keys
        self.right_key_mask = (_kidx > _split).float()     # (88,) 1.0 for right-hand keys

        # --- body indices: piano keys (ordered 0..87) and fingertips per hand ---
        key_ids, _ = self.piano.find_bodies([f"key_{i}" for i in range(NUM_KEYS)],
                                            preserve_order=True)
        self.key_body_ids = torch.tensor(key_ids, device=self.device)
        self.key_half_h = torch.as_tensor(geometry.KEY_HALF_H, device=self.device)    # (88,)
        ltips, _ = self.left_robot.find_bodies(FINGERTIP_BODIES, preserve_order=True)
        rtips, _ = self.right_robot.find_bodies(FINGERTIP_BODIES, preserve_order=True)
        self.ltip_ids = torch.tensor(ltips, device=self.device)
        self.rtip_ids = torch.tensor(rtips, device=self.device)
        # hand-base (palm) bodies, for the arm gross-positioning reward
        lpalm, _ = self.left_robot.find_bodies([self.cfg.hand_base_body], preserve_order=True)
        rpalm, _ = self.right_robot.find_bodies([self.cfg.hand_base_body], preserve_order=True)
        self.lpalm_id = torch.tensor(lpalm, device=self.device)
        self.rpalm_id = torch.tensor(rpalm, device=self.device)

        # --- residual base pose: the static ready pose, broadcast over every song/step.
        # In arm_ik_follow / slider the arm columns are overwritten each control step by
        # WristPoseIK; the policy learns finger pressing as a residual on top.
        _Tpad = self.goal_padded.shape[1]                           # Tmax + L
        _ready = torch.stack([self.left_default[0], self.right_default[0]], dim=0)  # (2,30)
        self.base_pose = _ready[None, None].repeat(self.num_songs, _Tpad, 1, 1)  # (N,Tpad,2,30)

        # action buffer / targets
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        # previous-step action + its frame-to-frame change, for the action-jerk
        # diagnostic (policy smoothness: a twitchy policy slams the arm around even
        # while F1 looks fine). Updated each _pre_physics_step.
        self.prev_actions = torch.zeros_like(self.actions)
        self._action_jerk = torch.zeros(self.num_envs, device=self.device)
        self._left_target = self.left_default.clone()
        self._right_target = self.right_default.clone()

        self.reward_cfg = PianoRewardCfg(
            press_threshold=0.5,
            key_press_weight=self.cfg.key_press_weight,
            false_press_weight=self.cfg.false_press_weight,
            energy_weight=self.cfg.energy_weight,
            fingering_weight=self.cfg.fingering_weight,
            onset_weight=self.cfg.onset_weight,
            arm_base_weight=self.cfg.arm_base_weight,
            arm_close_enough=self.cfg.arm_close_enough,
            arm_margin_mult=self.cfg.arm_margin_mult,
        )
        # SPLIT REWARD by curriculum phase. Phase 1 (freeze_hands) is the ARM's job:
        # pure positioning (fingering + arm-position), with NO pressing terms -- the
        # frozen fingers would otherwise trigger false-press penalties and punish the
        # arm for exploring. Phase 2 (hands in) keeps the full pressing reward.
        if getattr(self.cfg, "freeze_hands", False):
            self.reward_cfg.key_press_weight = 0.0
            self.reward_cfg.onset_weight = 0.0
            self.reward_cfg.false_press_weight = 0.0
            print("[PianoEnv] phase-1 reward: positioning only "
                  "(fingering+arm; pressing terms OFF)")
        _names = ", ".join(self.song_names[:4]) + ("..." if self.num_songs > 4 else "")
        print(f"[PianoEnv] {self.num_songs} song(s) [{_names}]: longest {self.song_len} steps "
              f"@ {1/self.cfg.control_dt:.0f}Hz")

        self._use_slider = bool(getattr(self.cfg, "use_slider", False))
        if self._use_slider:
            self._setup_slider()

    # ------------------------------------------------------------- slider
    def _setup_slider(self):
        """Calibrate slider_y/z -> world press-fingertip Y/Z (linear) for each hand, so
        _slider_follow_base can place the striking finger ON the note. The 180deg
        fingers-down rot INVERTS the slider axes, so we measure rather than assume."""
        names = self.left_robot.data.joint_names
        self._sy_col = names.index("slider_y")
        self._sz_col = names.index("slider_z")
        pf = int(getattr(self.cfg, "slider_press_finger", 1))
        fb = FINGERTIP_BODIES[pf]
        self._ptip_l = self.left_robot.find_bodies([fb], preserve_order=True)[0][0]
        self._ptip_r = self.right_robot.find_bodies([fb], preserve_order=True)[0][0]
        ftags = ["TH", "FF", "MF", "RF", "LF"]; ptag = ftags[pf]
        # FINGER-STRIKE: angle the strike finger at the MCP (J3) and strike via PIP (J2)
        # so only that fingertip descends onto the key (hand at hover -> no mash).
        self._finger_strike = bool(getattr(self.cfg, "slider_finger_strike", False))
        if self._finger_strike:
            self._strike_mcp_col = names.index(f"robot0_{ptag}J3")
            self._strike_pip_col = names.index(f"robot0_{ptag}J2")
            self._strike_mcp = float(getattr(self.cfg, "slider_strike_mcp", 0.6))
            self._strike_pip = float(getattr(self.cfg, "slider_strike_pip", 0.9))
            self._strike_hover = float(getattr(self.cfg, "slider_strike_hover", 0.02))

        def _measure(sy, sz):
            for robot in (self.left_robot, self.right_robot):
                jp = robot.data.joint_pos.clone()
                jp[:, self._sy_col] = sy; jp[:, self._sz_col] = sz
                if self._finger_strike:
                    jp[:, self._strike_mcp_col] = self._strike_mcp   # measure the ANGLED tip
                robot.set_joint_position_target(jp); robot.write_data_to_sim()
            for _ in range(60):
                self.sim.step(render=False)
                self.left_robot.update(self.cfg.sim.dt); self.right_robot.update(self.cfg.sim.dt)
            o0 = self.scene.env_origins[0]                       # env-0 grid origin
            return (self.left_robot.data.body_pos_w[0, self._ptip_l].clone() - o0,
                    self.right_robot.data.body_pos_w[0, self._ptip_r].clone() - o0)

        # Y-cal with the hand LIFTED clear of the keys (slider_z=-0.04 -> up, no contact)
        # so finger-key collisions can't destabilise the measurement.
        lyp, ryp = _measure(0.3, -0.04); lym, rym = _measure(-0.3, -0.04)
        lzp, rzp = _measure(0.0, 0.06); lz0, rz0 = _measure(0.0, -0.04)
        print(f"[slider cal raw] L: y@+.3={lyp[1]:.3f} y@-.3={lym[1]:.3f} "
              f"z@.06={lzp[2]:.3f} z@-.04={lz0[2]:.3f}")
        # per-hand linear fits: tipY = ay + by*slider_y ; tipZ = az + bz*slider_z
        self._cal = {}
        for tag, yp, ym, zp, z0 in (("l", lyp, lym, lzp, lz0), ("r", ryp, rym, rzp, rz0)):
            by = (yp[1] - ym[1]).item() / 0.6
            ay = ym[1].item() + 0.3 * by
            bz = (zp[2] - z0[2]).item() / 0.10        # points at slider_z -0.04 and +0.06
            az = z0[2].item() + 0.04 * bz             # tipZ at slider_z=0
            self._cal[tag] = (ay, by, az, bz)
        self._key_top_z = float((self._key_top_world()[0, :, 2]
                                 - self.scene.env_origins[0, 2]).max())   # env-local
        # reset the sliders to neutral after the calibration sweep
        for robot in (self.left_robot, self.right_robot):
            jp = robot.data.joint_pos.clone()
            jp[:, self._sy_col] = 0.0; jp[:, self._sz_col] = 0.0
            robot.set_joint_position_target(jp); robot.write_data_to_sim()
        # let the policy NUDGE the slider (placement Y + press Z) on top of the open-loop
        # base -> it can correct the ~2cm calibration residual and modulate press depth.
        # (arm_ik_follow had zeroed these; the fingers keep their full residual scale.)
        _sres = float(getattr(self.cfg, "slider_residual", 0.05))
        self.joint_scale[:, self._sy_col] = _sres          # lateral correction (0 = pure IK)
        self.joint_scale[:, self._sz_col] = _sres * 0.8    # press/lift correction
        # TIP-only curl columns of the NON-press fingers (J1/J2): curling these tucks
        # idle fingertips UP & clear without dropping the knuckle onto keys (the harness
        # mash fix) -> a cleaner base for RL to refine -> higher precision.
        ftags = ["TH", "FF", "MF", "RF", "LF"]
        ptag = ftags[pf]
        self._slider_curl_cols = torch.tensor(
            [i for i, n in enumerate(names)
             if n.startswith("robot0_") and f"robot0_{ptag}" not in n and n[-1] in "12"],
            device=self.device, dtype=torch.long)
        self._slider_curl = float(getattr(self.cfg, "slider_idle_curl", 0.8))
        # MCP (J3) of the idle fingers: flexing the knuckle lifts the WHOLE idle finger
        # up & clear of the keys (vs tip-only curl which leaves the lower segments near
        # key level -> mash). Sign unknown a priori; sweep cfg.slider_idle_mcp.
        self._slider_mcp_cols = torch.tensor(
            [i for i, n in enumerate(names)
             if n.startswith("robot0_") and f"robot0_{ptag}" not in n and n[-1] == "3"],
            device=self.device, dtype=torch.long)
        self._slider_mcp = float(getattr(self.cfg, "slider_idle_mcp", 0.0))
        print(f"[PianoEnv] SLIDER calibrated: L={tuple(round(v,3) for v in self._cal['l'])} "
              f"R={tuple(round(v,3) for v in self._cal['r'])} key_top_z={self._key_top_z:.3f}"
              f"{' [FINGER-STRIKE]' if self._finger_strike else ''}")

    # ------------------------------------------------------------- songs
    def _gather_songs(self):
        """Return a list of (name, key_activation (T,88) bool, onsets (T,88) bool).
        One entry from cfg.midi_path normally; N entries from cfg.songs_npz (a
        precomputed multi-song goal bundle) when multi-song training is requested.
        Onsets for npz songs are the rising edge of the key activation."""
        from dexsim.piano.midi import fold_into_reach

        def _fold(act, ons):
            if self.cfg.fold_to_reach:
                act, ons = fold_into_reach(
                    act, ons, left_window=tuple(self.cfg.left_key_window),
                    right_window=tuple(self.cfg.right_key_window))
            return act, ons

        npz = getattr(self.cfg, "songs_npz", None)
        if npz:
            import numpy as np
            d = np.load(npz, allow_pickle=True)
            G, lens, names = d["goals"], d["lens"], d["names"]
            off = int(getattr(self.cfg, "song_offset", 0))
            cap = int(getattr(self.cfg, "max_songs", 0)) or (G.shape[0] - off)
            out = []
            for i in range(off, min(off + cap, G.shape[0])):
                T = int(lens[i])
                act = G[i, :T].astype(bool)
                ons = np.zeros_like(act)
                ons[0] = act[0]
                ons[1:] = act[1:] & ~act[:-1]          # rising edge = note onset
                act, ons = _fold(act, ons)
                out.append((str(names[i]), act, ons))
            print(f"[PianoEnv] MULTI-SONG: {len(out)} songs from {npz} "
                  f"(fold_to_reach={self.cfg.fold_to_reach})")
            return out

        song = load_song(self.cfg.midi_path, control_dt=self.cfg.control_dt)
        act, ons = _fold(song.key_activation, song.onsets)
        if self.cfg.fold_to_reach:
            print(f"[PianoEnv] folded '{song.name}' into reach: "
                  f"{int(act.any(0).sum())} distinct keys")
        return [(song.name, act, ons)]

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.left_robot = Articulation(self.cfg.left_robot_cfg)
        self.right_robot = Articulation(self.cfg.right_robot_cfg)
        self.piano = Articulation(self.cfg.piano_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)

        self.scene.articulations["left_robot"] = self.left_robot
        self.scene.articulations["right_robot"] = self.right_robot
        self.scene.articulations["piano"] = self.piano

        light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.98))
        light.func("/World/Light", light)

    # ------------------------------------------------------------------ step
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = torch.nan_to_num(actions, nan=0.0).clamp(-1.0, 1.0)
        # action jerk = how much the policy's output moved since last step (L1, mean
        # over action dims). Diagnostic only -- not fed back into reward.
        self._action_jerk = (self.actions - self.prev_actions).abs().mean(dim=-1)
        self.prev_actions = self.actions.clone()
        a_left = self.actions[:, : self.per_arm_dof]
        a_right = self.actions[:, self.per_arm_dof :]
        ref = self.base_pose[self.song_id, self.song_step]          # (E, 2, 30) ready pose
        # residual on all joints, but per-joint scaled (gentle arm / generous hand;
        # see self.joint_scale). Targets are clamped to joint limits and obs are
        # NaN-guarded, so a transient blow-up can't crash PPO.
        scale = self.joint_scale
        lo = self.left_robot.data.soft_joint_pos_limits[..., 0]
        hi = self.left_robot.data.soft_joint_pos_limits[..., 1]
        # mute_right_hand: for left-hand-only songs, hold the right arm at its
        # ref pose so it can't mash idle keys (kills the false-press noise).
        if getattr(self.cfg, "mute_right_hand", False):
            a_right = a_right * 0.0
        base_l, base_r = ref[:, 0], ref[:, 1]
        # ARM-IK-FOLLOW: replace the arm columns of the base with the WristPoseIK
        # servo toward the note centroid. Finger columns stay at the ready pose;
        # the policy learns pressing as a RESIDUAL on top -- a sustained action offset
        # the weak hand actuators (stiffness 3) integrate into motion.
        if getattr(self, "_arm_ik_follow", False):
            base_l, base_r = self._ik_follow_base(base_l, base_r)
        self._left_target = torch.clamp(base_l + scale * a_left, lo, hi)
        self._right_target = torch.clamp(base_r + scale * a_right, lo, hi)
        # SLIDER teleport-ONCE-per-control-step: snap the 2 slider DoF to their IK target
        # HERE (before the decimation substeps), then let the PD HOLD them while the finger
        # presses -> tight sub-key placement at onset WITHOUT resetting key contact every
        # substep (the every-substep teleport killed recall by clearing the contact).
        if getattr(self, "_use_slider", False) and getattr(self.cfg, "slider_teleport_once", False):
            cols = torch.tensor([self._sy_col, self._sz_col], device=self.device)
            for robot, tgt in ((self.left_robot, self._left_target),
                               (self.right_robot, self._right_target)):
                jp = robot.data.joint_pos.clone(); jv = robot.data.joint_vel.clone()
                jp[:, cols] = tgt[:, cols]; jv[:, cols] = 0.0
                robot.write_joint_state_to_sim(jp, jv)

    def _slider_follow_base(self, base_l, base_r):
        """SLIDER positioning: set each hand's slider_y so the striking finger lands on
        the upcoming-note centroid (calibrated map), and slider_z to press (note active)
        or lift (idle). Finger columns stay at ref; the policy presses as a residual."""
        centroid, active = self._hand_note_centroids()          # (E,2,3) WORLD, (E,2)
        # work in env-LOCAL frame (calibration was env-local): subtract each env origin
        oy = self.scene.env_origins[:, 1]                       # (E,)
        press_z = self._key_top_z - 0.004
        lift_z = self._key_top_z + 0.040
        outs = []
        for hand_i, (tag, base, robot, ptip) in enumerate((
                ("l", base_l, self.left_robot, self._ptip_l),
                ("r", base_r, self.right_robot, self._ptip_r))):
            ay, by, az, bz = self._cal[tag]
            note_y = centroid[:, hand_i, 1] - oy                # (E,) env-local Y
            has = active[:, hand_i]                              # (E,) bool
            # CLOSED-LOOP placement: correct slider_y from the MEASURED press-finger tip
            # (kills the ~2cm open-loop calibration residual -> finger lands ON the key,
            # like the harness that hit recall 0.58). Open-loop seed when idle.
            tip_y = robot.data.body_pos_w[:, ptip, 1] - oy      # (E,) env-local tip Y
            cur_sy = robot.data.joint_pos[:, self._sy_col]      # (E,) current slider_y
            ff_sy = (note_y - ay) / by                          # open-loop seed
            cl_sy = cur_sy + (note_y - tip_y) / by              # closed-loop correction
            sy = torch.where(has, cl_sy, ff_sy).clamp(-0.6, 0.6)
            b = base.clone()
            b[:, self._sy_col] = sy
            if getattr(self, "_finger_strike", False):
                # hand stays at HOVER; strike via the angled strike-finger's PIP flex so
                # ONLY that fingertip descends onto the key (no hand descent -> no mash).
                hover_z = self._key_top_z + self._strike_hover
                sz = torch.full_like(note_y, (hover_z - az) / bz).clamp(-0.05, 0.10)
                b[:, self._sz_col] = sz
                b[:, self._strike_mcp_col] = self._strike_mcp          # angle the finger
                b[:, self._strike_pip_col] = torch.where(             # strike when active
                    has, torch.full_like(note_y, self._strike_pip), torch.zeros_like(note_y))
                b[:, self._slider_curl_cols] = b[:, self._slider_curl_cols] + self._slider_curl
            else:
                tgt_z = torch.where(has, torch.full_like(note_y, press_z),
                                    torch.full_like(note_y, lift_z))
                sz = ((tgt_z - az) / bz).clamp(-0.05, 0.10)
                b[:, self._sz_col] = sz
                # tuck idle fingertips up (tip-only curl) so only the press finger strikes
                b[:, self._slider_curl_cols] = b[:, self._slider_curl_cols] + self._slider_curl
                if self._slider_mcp != 0.0:
                    b[:, self._slider_mcp_cols] = b[:, self._slider_mcp_cols] + self._slider_mcp
            outs.append(b)
        return outs[0], outs[1]

    def _ik_follow_base(self, base_l, base_r):
        """Base joint pose for ARM-IK-FOLLOW mode, (base_left, base_right) each (E,30).
        Arm columns = WristPoseIK servoing the palm to the upcoming-note centroid
        (hover above keys, ready-pose down quat). Finger columns are left as passed in
        (the ready pose); the policy residual is added on top by the caller."""
        if getattr(self, "_use_slider", False):
            return self._slider_follow_base(base_l, base_r)
        ftip_track = getattr(self.cfg, "arm_ftip_track", False)
        if not hasattr(self, "ik_left"):
            from dexsim.piano.ik import WristPoseIK
            _planar = bool(getattr(self.cfg, "planar_ik", False))
            _pw = float(getattr(self.cfg, "planar_weight", 25.0))
            _pi = int(getattr(self.cfg, "planar_iters", 6))
            _frz = ("wrist_3",) if bool(getattr(self.cfg, "freeze_last_dof", False)) else ()
            self.ik_left = WristPoseIK(self.left_robot, self.cfg.hand_base_body,
                                       planar=_planar, planar_weight=_pw, planar_iters=_pi,
                                       freeze_joints=_frz)
            self.ik_right = WristPoseIK(self.right_robot, self.cfg.hand_base_body,
                                        planar=_planar, planar_weight=_pw, planar_iters=_pi,
                                        freeze_joints=_frz)
            if _planar:
                print(f"[PianoEnv] PLANAR-IK: weighted DLS (w={_pw} on z+orientation) x{_pi} "
                      "iters -> arm holds the plane & slides in XY (gantry emulation)")
            if _frz:
                print(f"[PianoEnv] FREEZE last DoF: {_frz} held at init value "
                      "(excluded from the arm IK solve)")
            if ftip_track:
                # POSITION-ONLY arm IK on the primary FINGERTIP: drives the arm so the
                # striking finger's TIP (not the palm) lands on the key -- closes the
                # ~1-finger-length (90mm) palm-vs-tip gap. diag_posik proved this
                # converges to ~18mm under PD (vs 93mm for palm-centroid). max_step
                # raised so the arm tracks fast-changing note targets in fewer steps.
                pf = int(getattr(self.cfg, "primary_finger", 1))
                fb = FINGERTIP_BODIES[pf]
                self.aftik_left = WristPoseIK(self.left_robot, fb,
                                              max_step=float(getattr(self.cfg, "ftip_max_step", 0.12)),
                                              pos_only=True)
                self.aftik_right = WristPoseIK(self.right_robot, fb,
                                               max_step=float(getattr(self.cfg, "ftip_max_step", 0.12)),
                                               pos_only=True)
            # hold the ready-pose palm orientation (fingers down) as the servo target
            _, self._arm_quat_l = self.ik_left.pose_w()
            _, self._arm_quat_r = self.ik_right.pose_w()
            self._arm_quat_l = self._arm_quat_l.clone()
            self._arm_quat_r = self._arm_quat_r.clone()
            # HAND-TILT: rotate the servo target orientation away from palm-straight-down
            # toward a pianist posture, so a finger curl strikes the key (vs lowering all).
            tilt = float(getattr(self.cfg, "hand_tilt", 0.0))
            if tilt != 0.0:
                ax = int(getattr(self.cfg, "hand_tilt_axis", 1))
                a = torch.zeros(3, device=self.device); a[ax] = 1.0
                half = tilt * 0.5
                qrot = torch.cat([torch.cos(torch.tensor([half], device=self.device)),
                                  a * torch.sin(torch.tensor(half, device=self.device))])  # (4,) wxyz
                self._arm_quat_l = self._quat_mul(qrot.unsqueeze(0), self._arm_quat_l)
                self._arm_quat_r = self._quat_mul(qrot.unsqueeze(0), self._arm_quat_r)
        if getattr(self.cfg, "single_finger", False):
            return self._single_finger_base(base_l, base_r)
        # --- arm servo to the upcoming-note centroid ---
        centroid, active = self._hand_note_centroids()      # (E,2,3), (E,2)
        pl, _ = self.ik_left.pose_w()
        pr, _ = self.ik_right.pose_w()                       # current palm positions
        tgt_l = centroid[:, 0].clone(); tgt_l[:, 2] += self.cfg.arm_ik_hover
        tgt_r = centroid[:, 1].clone(); tgt_r[:, 2] += self.cfg.arm_ik_hover
        # a hand with NO upcoming notes RETRACTS up off the keyboard (keeping its xy)
        # so its resting fingers stop ringing keys -- e.g. the muted right hand on a
        # left-only song was holding station AT key level, ringing ~5-7 false keys and
        # masking every left-hand tweak. Retract height = keyboard top + idle_hand_retract.
        kb_z = self._key_top_world()[..., 2].amax(dim=-1, keepdim=True)   # (E,1)
        retract_z = kb_z + getattr(self.cfg, "idle_hand_retract", 0.20)
        hi_l = pl.clone(); hi_l[:, 2:3] = retract_z
        hi_r = pr.clone(); hi_r[:, 2:3] = retract_z
        tgt_l = torch.where(active[:, 0:1], tgt_l, hi_l)
        tgt_r = torch.where(active[:, 1:2], tgt_r, hi_r)
        if ftip_track:
            # drive the PRIMARY FINGERTIP (not the palm) to the note centroid via
            # position-only IK -> the striking finger lands on the key, not 90mm short.
            # idle hand still retracts (target = current tip lifted to retract height).
            tl, _ = self.aftik_left.pose_w(); tr, _ = self.aftik_right.pose_w()
            il = tl.clone(); il[:, 2:3] = retract_z
            ir = tr.clone(); ir[:, 2:3] = retract_z
            ft_l = torch.where(active[:, 0:1], tgt_l, il)
            ft_r = torch.where(active[:, 1:2], tgt_r, ir)
            arm_l = self.aftik_left.solve(ft_l, self._arm_quat_l)
            arm_r = self.aftik_right.solve(ft_r, self._arm_quat_r)
        else:
            arm_l = self.ik_left.solve(tgt_l, self._arm_quat_l)  # (E,30); only arm cols move
            arm_r = self.ik_right.solve(tgt_r, self._arm_quat_r)
        m = self._arm_dof_mask                               # (30,) True = arm joint
        # finger columns stay at the ready-pose base; the policy presses as a residual.
        base_l = torch.where(m, arm_l, base_l)               # arm cols servo, finger cols base
        base_r = torch.where(m, arm_r, base_r)
        # IDLE-FINGER CURL: lift fingers with no note THIS step up into the palm so the
        # hand stops mashing its ~8-key footprint; the active finger stays straight.
        curl = getattr(self.cfg, "idle_finger_curl", 0.0)
        if curl != 0.0:
            fa = self.finger_active[self.song_id, self.song_step]  # (E,10) global finger order
            for hand_i, base in enumerate((base_l, base_r)):
                for fi in range(5):
                    cols = self._finger_flex_cols[fi]
                    idle = (~fa[:, hand_i * 5 + fi].bool()).float().unsqueeze(-1)  # (E,1)
                    base[:, cols] = base[:, cols] + curl * idle
        return base_l, base_r

    @staticmethod
    def _quat_mul(a, b):
        """Hamilton product of wxyz quats (broadcast over leading dim)."""
        aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return torch.stack([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw], dim=-1)

    def _single_finger_base(self, base_l, base_r):
        """ONE-FINGER-PER-NOTE base pose. Aim the PRIMARY fingertip directly at the
        current note (not the palm at a window centroid), curl the other 4 fingers up,
        retract a hand with no current note. For a monophonic melody this places the
        striking finger on the right key -> high precision (vs ~13cm gross-centroid error)."""
        pf = int(self.cfg.primary_finger)
        # Drive the PRIMARY FINGERTIP body itself with WristPoseIK (arm-only). diag_wrist_ik
        # proved the arm places a chosen body to ~1cm; targeting the fingertip (not the palm
        # + offset) puts the fingertip ON its key -> precise placement, not the ~100mm gap.
        if not hasattr(self, "ftik_left"):
            from dexsim.piano.ik import WristPoseIK
            fb = FINGERTIP_BODIES[pf]
            self.ftik_left = WristPoseIK(self.left_robot, fb, max_step=0.06, pos_only=True)
            self.ftik_right = WristPoseIK(self.right_robot, fb, max_step=0.06, pos_only=True)
            _, self._ftq_l = self.ftik_left.pose_w(); self._ftq_l = self._ftq_l.clone()
            _, self._ftq_r = self.ftik_right.pose_w(); self._ftq_r = self._ftq_r.clone()
        key_top = self._key_top_world()                       # (E,88,3)
        goal = self.goal_padded[self.song_id, self.song_step] # (E,88) keys active NOW
        palms = self._palms_world()                           # (E,2,3) [L,R]
        tips = self._fingertips_world()                        # (E,10,3) global order
        kb_z = key_top[..., 2].amax(dim=-1, keepdim=True)      # (E,1) keyboard surface
        m = self._arm_dof_mask
        out = []
        for hand_i, (mask, ftik, quat, base) in enumerate((
                (self.left_key_mask, self.ftik_left, self._ftq_l, base_l),
                (self.right_key_mask, self.ftik_right, self._ftq_r, base_r))):
            ak = goal * mask                                   # (E,88) this hand's active keys
            has = ak.sum(-1, keepdim=True) > 0                 # (E,1)
            kpos = (key_top * ak.unsqueeze(-1)).sum(1) / ak.sum(-1, keepdim=True).clamp(min=1e-6)  # (E,3)
            ft = tips[:, hand_i * 5 + pf]                      # (E,3) primary fingertip (current)
            # DRIVE THE FINGERTIP ITSELF to the key (arm-only IK on the fingertip body) -> the
            # arm places the fingertip on the key to ~1cm. Dip to press depth only when xy-
            # aligned (else hover, so transitions don't drag the tip across keys).
            xy_dist = torch.linalg.norm(ft[:, :2] - kpos[:, :2], dim=-1, keepdim=True)  # (E,1)
            aligned = (xy_dist < getattr(self.cfg, "single_align_thresh", 0.015)).float()  # (E,1)
            z_off = aligned * self.cfg.single_press_z + (1.0 - aligned) * getattr(self.cfg, "single_hover", 0.012)
            ft_tgt = kpos.clone(); ft_tgt[:, 2:3] = kpos[:, 2:3] + z_off
            retract = ft.clone(); retract[:, 2] = (kb_z + self.cfg.idle_hand_retract).squeeze(-1)
            ft_tgt = torch.where(has, ft_tgt, retract)         # idle hand: lift the fingertip away
            arm = ftik.solve(ft_tgt, quat)                     # (E,30); arm cols drive the FINGERTIP
            b = torch.where(m, arm, base)
            idle_h = (~has).float()                            # (E,1) 1 where hand has no note
            for fi in range(5):
                cols = self._finger_flex_cols[fi]
                # non-primary fingers ALWAYS curl up clear; primary curls up only when idle
                w = idle_h if fi == pf else torch.ones_like(idle_h)
                b[:, cols] = b[:, cols] + self.cfg.single_curl * w
            out.append(b)
        return out[0], out[1]

    def _apply_action(self):
        self.left_robot.set_joint_position_target(self._left_target)
        self.right_robot.set_joint_position_target(self._right_target)
        # SLIDER: teleport the 2 slider DoF straight to their IK target (no PD lag) so the
        # strike finger sits EXACTLY on the key at onset -> tight sub-key placement (the
        # PD-lagged placement left the tip ~2-3cm off = over a neighbour = false presses).
        # The slider is math-positioned, not policy-driven, so a kinematic set is valid.
        if getattr(self, "_use_slider", False) and getattr(self.cfg, "slider_teleport", False):
            cols = torch.tensor([self._sy_col, self._sz_col], device=self.device)
            for robot, tgt in ((self.left_robot, self._left_target),
                               (self.right_robot, self._right_target)):
                jp = robot.data.joint_pos.clone()
                jv = robot.data.joint_vel.clone()
                jp[:, cols] = tgt[:, cols]
                jv[:, cols] = 0.0
                robot.write_joint_state_to_sim(jp, jv)
        # piano keys are passive (spring drives only) -- never commanded.

    # --------------------------------------------------------- world helpers
    def _key_pressed_fraction(self) -> torch.Tensor:
        key_angle = self.piano.data.joint_pos            # negative when pressed
        frac = (key_angle / KEY_SOUND_ANGLE).clamp(0.0, 2.0)
        # a physics blow-up can NaN the key joints; keep it finite so the key/onset
        # reward terms (and their logged means) don't get poisoned to NaN.
        frac = torch.nan_to_num(frac, nan=0.0, posinf=2.0, neginf=0.0)
        # VELOCITY-GATED sounding (real piano hammer). A key only STARTS sounding
        # when STRUCK -- depressed past threshold AND its joint moving DOWN faster
        # than key_strike_vel -- and keeps sounding while held, until it returns up.
        # A hand/forearm resting statically on keys depresses them with ~0 velocity,
        # so it never triggers a strike -> no false ring (was 52 keys sounding).
        # NOTE: called exactly once per step (in _get_rewards), so latching here is
        # safe. Called once per step from _get_rewards, so the latch advances once.
        vel = torch.nan_to_num(self.piano.data.joint_vel, nan=0.0)   # <0 = pressing down
        # frac=1.0 IS the sound angle (frac = angle / KEY_SOUND_ANGLE). A key only
        # SOUNDS when pressed PAST it (frac>=1). The old 0.5/0.25 thresholds latched
        # keys merely BRUSHED to half the sound depth (never actually sounding) and
        # never released them -> keys_sounding inflated ~8 with nothing on the keys,
        # cratering precision. Align the latch to the real sound angle.
        struck = (frac >= getattr(self.cfg, "key_struck_frac", 1.0)) & (vel < -self.cfg.key_strike_vel)
        released = frac < getattr(self.cfg, "key_release_frac", 0.8)
        prev_sounding = self.key_sounding
        self.key_sounding = (self.key_sounding | struck) & ~released
        # rising edge: keys that went silent->sounding this step = the onsets the
        # hands actually played (vs onset_now() = the onsets they SHOULD have played).
        self._just_struck = self.key_sounding & ~prev_sounding
        return torch.where(self.key_sounding, frac, torch.zeros_like(frac))

    def _key_top_world(self) -> torch.Tensor:
        """(E, 88, 3) world position of each key's top surface."""
        centers = self.piano.data.body_pos_w[:, self.key_body_ids, :]   # (E,88,3)
        top = centers.clone()
        top[..., 2] += self.key_half_h                                  # + half thickness
        return torch.nan_to_num(top, nan=0.0, posinf=10.0, neginf=-10.0)

    def _fingertips_world(self) -> torch.Tensor:
        """(E, 10, 3) fingertip world positions in global finger order
        [L_th,L_ff,L_mf,L_rf,L_lf, R_th,R_ff,R_mf,R_rf,R_lf]."""
        l = self.left_robot.data.body_pos_w[:, self.ltip_ids, :]        # (E,5,3)
        r = self.right_robot.data.body_pos_w[:, self.rtip_ids, :]
        tips = torch.cat([l, r], dim=1)                                 # (E,10,3)
        # finite-guard: a blown-up env must not NaN-poison the fingering reward.
        return torch.nan_to_num(tips, nan=0.0, posinf=10.0, neginf=-10.0)

    def _finger_targets_world(self, key_top: torch.Tensor):
        """Return (target_surface, target_press, active) for the current step.
        target_surface: (E,10,3) key-top point used for the *fingering reward*.
        target_press:   (E,10,3) press/hover point used for *observation*.
        active:         (E,10) bool, which fingers are assigned a key now."""
        fk = self.finger_key[self.song_id, self.song_step]              # (E,10)
        fa = self.finger_active[self.song_id, self.song_step]           # (E,10)
        idx_safe = torch.where(fa, fk, self.finger_home.unsqueeze(0))   # valid indices
        gather_idx = idx_safe.unsqueeze(-1).expand(-1, -1, 3)           # (E,10,3)
        surface = torch.gather(key_top, 1, gather_idx)                  # (E,10,3)
        press = surface.clone()
        dz = torch.where(fa, torch.full_like(fa, -geometry.PRESS_DEPTH, dtype=torch.float32),
                         torch.full_like(fa, geometry.HOVER_CLEARANCE, dtype=torch.float32))
        press[..., 2] += dz
        return surface, press, fa

    def _palms_world(self) -> torch.Tensor:
        """(E, 2, 3) world position of each hand base [left, right]."""
        l = self.left_robot.data.body_pos_w[:, self.lpalm_id, :]        # (E,1,3)
        r = self.right_robot.data.body_pos_w[:, self.rpalm_id, :]
        palms = torch.cat([l, r], dim=1)                                # (E,2,3)
        # finite-guard: a blown-up env must not NaN-poison the arm reward.
        return torch.nan_to_num(palms, nan=0.0, posinf=10.0, neginf=-10.0)

    def _hand_note_centroids(self):
        """Per-hand gross-positioning target for the arm reward.

        Returns (centroid, active):
          centroid (E,2,3): world centroid of the keys each hand [left,right] must
                            play over the next ``arm_lookahead`` steps.
          active   (E,2)   : whether that hand has any upcoming notes in the window
                            (a hand with none contributes nothing to the reward).
        """
        H = self.cfg.arm_lookahead
        key_top = self._key_top_world()                                 # (E,88,3)
        idx = self.song_step.unsqueeze(1) + torch.arange(H, device=self.device).unsqueeze(0)
        idx = idx.clamp(max=self.finger_key.shape[1] - 1)              # (E,H) time dim
        sid = self.song_id.unsqueeze(1)                                 # (E,1) broadcast
        fk = self.finger_key[sid, idx]                                  # (E,H,10)
        fa = self.finger_active[sid, idx]                               # (E,H,10)
        half = NUM_FINGERS // 2
        centroids, actives = [], []
        for sl in (slice(0, half), slice(half, NUM_FINGERS)):          # left, then right
            fk_h = fk[..., sl].clamp(min=0)        # (E,H,5); idle=-1 -> 0 (masked out below)
            fa_h = fa[..., sl]                     # (E,H,5)
            flat = fk_h.reshape(self.num_envs, -1)                      # (E,H*5)
            gidx = flat.unsqueeze(-1).expand(-1, -1, 3)                # (E,H*5,3)
            pos = torch.gather(key_top, 1, gidx).reshape(self.num_envs, H, half, 3)
            m = fa_h.float().unsqueeze(-1)                              # (E,H,5,1)
            num = (pos * m).sum(dim=(1, 2))                             # (E,3)
            den = m.sum(dim=(1, 2)).clamp(min=1e-6)                     # (E,1)
            centroids.append(num / den)                                # (E,3)
            actives.append(fa_h.reshape(self.num_envs, -1).any(dim=1)) # (E,)
        return torch.stack(centroids, dim=1), torch.stack(actives, dim=1)

    # ----------------------------------------------------------- observations
    def _goal_lookahead(self) -> torch.Tensor:
        L = self.cfg.goal_lookahead
        idx = self.song_step.unsqueeze(1) + torch.arange(L, device=self.device).unsqueeze(0)
        return self.goal_padded[self.song_id.unsqueeze(1), idx]   # (E, L, 88)

    def _goal_now(self) -> torch.Tensor:
        return self.goal_padded[self.song_id, self.song_step]   # (E, 88)

    def _onset_now(self) -> torch.Tensor:
        return self.onset_padded[self.song_id, self.song_step]  # (E, 88)

    def _get_observations(self) -> dict:
        origin = self.scene.env_origins.unsqueeze(1)     # (E,1,3) for rel. positions
        parts = [
            self.left_robot.data.joint_pos,
            self.left_robot.data.joint_vel,
            self.right_robot.data.joint_pos,
            self.right_robot.data.joint_vel,
            self.piano.data.joint_pos,                   # (E,88) key angles
            self._goal_lookahead().reshape(self.num_envs, -1),
        ]
        if self.cfg.obs_fingertip_pos:
            tips = self._fingertips_world() - origin
            parts.append(tips.reshape(self.num_envs, -1))
        if self.cfg.obs_finger_targets:
            _, press, _ = self._finger_targets_world(self._key_top_world())
            parts.append((press - origin).reshape(self.num_envs, -1))
        if self.cfg.obs_goal_sdf:
            parts.append(nearest_active_distance(self._goal_now()))
        # guard: replace any NaN/inf (from a transient physics blow-up) and clamp,
        # so the policy never sees garbage -> no "std>=0" PPO crash.
        obs = torch.nan_to_num(torch.cat(parts, dim=-1), nan=0.0, posinf=50.0, neginf=-50.0)
        return {"policy": obs.clamp(-50.0, 50.0)}

    def _arm_limit_margin(self) -> torch.Tensor:
        """(E,) normalized distance of the WORST arm joint to its nearest joint limit.
        1.0 = mid-range (healthy), 0.0 = pinned against a limit (contorting / about to
        fail, won't transfer to hardware). Scored over the 6-per-arm UR10e DoF only
        (fingers excluded via arm_joint_mask), across both arms."""
        margins = []
        for robot in (self.left_robot, self.right_robot):
            q = robot.data.joint_pos
            lo = robot.data.soft_joint_pos_limits[..., 0]
            hi = robot.data.soft_joint_pos_limits[..., 1]
            frac = (q - lo) / (hi - lo).clamp(min=1e-6)             # 0 at lo, 1 at hi
            m = 2.0 * torch.minimum(frac, 1.0 - frac)              # 1 mid-range, 0 at a limit
            margins.append(m[:, self.arm_joint_mask])              # (E, n_arm)
        m = torch.cat(margins, dim=1).clamp(0.0, 1.0)              # (E, 2*n_arm)
        m = torch.nan_to_num(m, nan=0.0, posinf=1.0, neginf=0.0)
        return m.min(dim=1).values                                 # (E,) worst joint

    def _hand_f1(self, pressed: torch.Tensor, goal: torch.Tensor,
                 key_mask: torch.Tensor) -> torch.Tensor:
        """Per-step F1 restricted to one hand's keys (key_mask: (88,) 1.0/0.0).
        Masking both pressed and goal to the hand's keys reuses press_accuracy as-is:
        keys outside the mask read as not-sounding and not-wanted, so they drop out of
        TP/precision/recall. Averaged over envs that have a note for this hand now."""
        from dexsim.piano.reward import press_accuracy
        rec, prec = press_accuracy(pressed * key_mask, goal * key_mask)
        has = (goal * key_mask).sum(-1) > 0
        if not has.any():
            return torch.zeros((), device=self.device)
        r = rec[has].mean(); p = prec[has].mean()
        return 2 * r * p / (r + p + 1e-9)

    # ---------------------------------------------------------------- reward
    def _get_rewards(self) -> torch.Tensor:
        pressed = self._key_pressed_fraction()
        goal = self._goal_now()
        energy = (self.actions ** 2).mean(dim=-1)
        r_key = piano_reward(pressed, goal, self.reward_cfg, energy=energy)

        key_top = self._key_top_world()
        surface, _, active = self._finger_targets_world(key_top)
        tips = self._fingertips_world()
        r_finger = fingering_reward(tips, surface, active, self.reward_cfg)
        r_onset = onset_reward(pressed, self._onset_now(), self.reward_cfg)

        # IDLE-FINGER CLEARANCE: penalize fingers NOT assigned a note for hanging at
        # or below the key surface (where they mash). Lifts the 4 idle fingers out of
        # the way so pressing one finger doesn't ring the whole ~12-key hand footprint.
        icw = getattr(self.cfg, "idle_clear_weight", 0.0)
        if icw > 0.0:
            kb_top = key_top[..., 2].amax(dim=-1, keepdim=True)      # (E,1) keyboard surface z
            clear_plane = kb_top + getattr(self.cfg, "idle_clear_margin", 0.02)
            below = (clear_plane - tips[..., 2]).clamp(min=0.0)      # (E,10) how far below the plane
            idle = (~active.bool()).float()                          # (E,10) fingers with no note now
            n_idle = idle.sum(-1).clamp(min=1.0)
            r_idle = -icw * (below * idle).sum(-1) / n_idle          # mean dip over idle fingers
        else:
            r_idle = torch.zeros_like(r_key)

        # arm gross-positioning: aim each hand base over its upcoming notes (the
        # 60-DoF extra over RoboPianist's slider-mounted hands). Skip the work when
        # the term is off.
        if self.reward_cfg.arm_base_weight > 0.0:
            centroid, hand_active = self._hand_note_centroids()
            r_arm = arm_position_reward(self._palms_world(), centroid, hand_active,
                                        self.reward_cfg)
        else:
            r_arm = torch.zeros_like(r_key)

        # --- log the REAL metric to wandb: is it actually sounding notes (F1)? ---
        # (reward can be high while F1=0; F1 is the truth.)
        from dexsim.piano.reward import press_accuracy
        recall, precision = press_accuracy(pressed, goal)
        has_goal = goal.sum(-1) > 0
        if has_goal.any():
            rec = recall[has_goal].mean()
            prec = precision[has_goal].mean()
            f1 = 2 * rec * prec / (rec + prec + 1e-9)
        else:
            rec = prec = f1 = torch.zeros((), device=self.device)
        # --- onset timing: of the keys the hands STRUCK this step (rising edge of
        # sounding), how many landed near a real note onset (within +/-onset_tol_steps,
        # via onset_win)? Catches mis-timed strikes / mashing that the held-note F1
        # above hides. Online-computable precision-flavored proxy; eval_reference.py
        # computes the full windowed onset P/R/F1 offline. ---
        played_on = self._just_struck.float()
        near_onset = self.onset_win[self.song_id, self.song_step]   # (E,88)
        n_played = played_on.sum(-1)
        has_played = n_played > 0
        if has_played.any():
            on_timing = ((played_on * near_onset).sum(-1) / (n_played + 1e-9))[has_played].mean()
        else:
            on_timing = torch.zeros((), device=self.device)
        # --- per-hand F1: is one arm carrying the song while the other fails? ---
        f1_left = self._hand_f1(pressed, goal, self.left_key_mask)
        f1_right = self._hand_f1(pressed, goal, self.right_key_mask)
        # --- arm health (motion quality, invisible to F1) ---
        limit_margin = self._arm_limit_margin().mean()
        action_jerk = self._action_jerk.mean()
        # Per-component finite guard. The world-pos/key accessors are already
        # nan-guarded at source, but guard each term here too so (a) one blown-up
        # env can never NaN-poison the *summed* reward (which nan_to_num would then
        # zero wholesale, silently deleting the key/onset signal the env earned),
        # and (b) the logged means below are a single bad env away from reading nan.
        g = lambda x: torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        r_key, r_finger, r_onset, r_arm = g(r_key), g(r_finger), g(r_onset), g(r_arm)
        r_idle = g(r_idle)

        if not hasattr(self, "extras") or self.extras is None:
            self.extras = {}
        self.extras["log"] = {
            "play/F1": float(f1),
            "play/recall": float(rec),
            "play/precision": float(prec),
            "play/keys_sounding": float((pressed >= 0.5).float().sum(-1).mean()),
            "play/onset_timing": float(on_timing),
            "play/F1_left": float(f1_left),
            "play/F1_right": float(f1_right),
            "arm/limit_margin": float(limit_margin),
            "arm/action_jerk": float(action_jerk),
            "reward/key": float(r_key.mean()),
            "reward/finger": float(r_finger.mean()),
            "reward/onset": float(r_onset.mean()),
            "reward/arm": float(r_arm.mean()),
            "reward/idle_clear": float(r_idle.mean()),
        }
        # final band-clamp: a transient blow-up that slips past the per-term guards
        # still can't push a NaN/inf advantage -> NaN log_std -> PPO crash.
        reward = r_key + r_finger + r_onset + r_arm + r_idle
        return torch.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

    # ----------------------------------------------------------------- dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        song_len_e = self.song_lens[self.song_id]                   # (E,) per-env song length
        song_done = self.song_step >= (song_len_e - 1)
        self.song_step = torch.minimum(self.song_step + 1, song_len_e - 1)
        terminated = torch.zeros_like(time_out)
        truncated = time_out | song_done
        return terminated, truncated

    # ----------------------------------------------------------------- reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.left_robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # start each arm at the ready pose (the residual base's first frame)
        q0 = self.base_pose[0, 0]                         # (2, 30) song 0, step 0
        for robot, ref in ((self.left_robot, q0[0]), (self.right_robot, q0[1])):
            jp = ref.unsqueeze(0).repeat(len(env_ids), 1)
            jv = torch.zeros_like(jp)
            robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)

        kp = self.piano.data.default_joint_pos[env_ids]
        kv = torch.zeros_like(kp)
        self.piano.write_joint_state_to_sim(kp, kv, env_ids=env_ids)

        self.song_step[env_ids] = 0
        self.key_sounding[env_ids] = False
