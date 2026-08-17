"""DirectRLEnv for two physical Shadow Hands sliding along independent Y rails.

Each hand has one prismatic base joint plus its 24 physical hand joints.  The
rail locks world X/Z and all base rotations; disjoint travel lanes prevent the
two hands from crossing while finger/key and hand/hand contacts use PhysX.
A MIDI song defines a per-step key-activation goal; an automatic fingering assigns
each note to a finger. The RL policy drives everything: the rail slide positions
the hand along the keyboard and the fingers press, as a residual on top of a
static ready pose (RoboPianist's floating-forearm recipe with a 1-DoF base).

What this env implements from RoboPianist:
  * **Residual action** over a ready-pose base: zero action holds the ready
    pose over the keys, so the policy only learns to slide and press.
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
    idle_hover_reward, PianoRewardCfg, RP1MRewardCfg, rp1m_reward,
    hands_in_proximity,
)
from dexsim.piano.ot import ot_finger_cost
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

        # START-CURLED: curl every finger flex joint (J1/J2/J3) in the default pose so RL
        # resets with fingers curled UP off the keys (anti-mash) and learns to EXTEND only
        # the assigned finger down. Clamped to each joint's limit.
        _sc = float(getattr(self.cfg, "start_finger_curl", 0.0))
        if _sc != 0.0:
            _names = self.left_robot.data.joint_names
            _flex = torch.tensor(
                [i for i, n in enumerate(_names)
                 if any(f"robot0_{t}J" in n for t in ("FF", "MF", "RF", "LF", "TH")) and n[-1] in "123"],
                device=self.device, dtype=torch.long)
            lo_l = self.left_robot.data.soft_joint_pos_limits[..., 0]
            hi_l = self.left_robot.data.soft_joint_pos_limits[..., 1]
            lo_r = self.right_robot.data.soft_joint_pos_limits[..., 0]
            hi_r = self.right_robot.data.soft_joint_pos_limits[..., 1]
            self.left_default[:, _flex] = torch.clamp(self.left_default[:, _flex] + _sc,
                                                      lo_l[:, _flex], hi_l[:, _flex])
            self.right_default[:, _flex] = torch.clamp(self.right_default[:, _flex] + _sc,
                                                       lo_r[:, _flex], hi_r[:, _flex])

        # The sole non-robot0 joint is the physical Y rail. All other joints belong
        # to the Shadow Hand finger/wrist chain.
        is_hand = torch.tensor(
            [1.0 if "robot0_" in n else 0.0 for n in self.left_robot.data.joint_names],
            device=self.device,
        )
        self.joint_scale = (
            self.cfg.arm_action_scale * (1.0 - is_hand)
            + self.cfg.hand_action_scale * is_hand
        ).unsqueeze(0)                                  # (1, 25) broadcast over envs
        self.arm_joint_mask = (is_hand < 0.5)           # railJoint only
        # FIXED-HANDS mode: zero the arm columns of joint_scale so the arms hold the
        # ready pose; the policy drives only the 48 finger DoF.
        if getattr(self.cfg, "freeze_arms", False):
            self.joint_scale = self.joint_scale * is_hand.unsqueeze(0)
            print("[PianoEnv] FIXED RAILS: policy drives fingers only")

        # --- WRIST LOCK: keep both hands upright as a property of the mechanism ---
        # Three parts, and all three are needed:
        #   1. zero the policy's residual on WRJ0/WRJ1, so the commanded target is
        #      permanently the locked ready-pose tilt;
        #   2. collapse those joints' POSITION LIMITS to a hairline band around
        #      that tilt, which makes "upright" a solver constraint rather than
        #      something a PD gain has to win;
        #   3. stiffen the joints (and lift their effort ceiling) so they sit at
        #      the centre of the band instead of resting on its edge.
        # Part 2 is the one that actually does the work. With the stock setup the
        # wrist target IS the ready pose and the joint still ends up ~0.6 rad off
        # it, because the load through the wrist saturates the actuator's ~27 N*m
        # ceiling -- past saturation, raising stiffness changes nothing. A limit
        # is enforced by the constraint solver and cannot saturate.
        self.wrist_joint_ids = torch.tensor(
            [i for i, n in enumerate(self.left_robot.data.joint_names) if "WRJ" in n],
            device=self.device, dtype=torch.long)
        if getattr(self.cfg, "wrist_lock", False) and len(self.wrist_joint_ids):
            self.joint_scale = self.joint_scale.clone()
            self.joint_scale[:, self.wrist_joint_ids] = 0.0
            k = float(self.cfg.wrist_lock_stiffness)
            d = float(self.cfg.wrist_lock_damping)
            eff = float(self.cfg.wrist_lock_effort)
            slack = float(self.cfg.wrist_lock_slack)
            wj = self.wrist_joint_ids
            for robot, ref in ((self.left_robot, self.left_default),
                               (self.right_robot, self.right_default)):
                n_env, n_w = robot.num_instances, len(wj)
                hold = ref[:, wj]                                  # (E, n_w) locked tilt
                lim = torch.stack([hold - slack, hold + slack], dim=-1)   # (E, n_w, 2)
                robot.write_joint_position_limit_to_sim(lim, joint_ids=wj)
                robot.write_joint_stiffness_to_sim(
                    torch.full((n_env, n_w), k, device=self.device), joint_ids=wj)
                robot.write_joint_damping_to_sim(
                    torch.full((n_env, n_w), d, device=self.device), joint_ids=wj)
                robot.write_joint_effort_limit_to_sim(
                    torch.full((n_env, n_w), eff, device=self.device), joint_ids=wj)
            names = [self.left_robot.data.joint_names[i] for i in wj]
            print(f"[PianoEnv] WRIST LOCK: {names} constrained to +/-{slack} rad about the "
                  f"ready-pose tilt (stiffness {k}, damping {d}, effort {eff}) -- hands held "
                  f"upright, {2 * len(names)} action DoF inert")

        # --- song(s) -> goal / onset / fingering tensors, stacked over N songs ---
        # Single-song is N=1; multi-song (cfg.songs_npz) stacks N songs and assigns each
        # env a song_id. Every per-song tensor is indexed [song_id, song_step].
        L = self.cfg.goal_lookahead
        songs = self._gather_songs()                                # list of (name, act(T,88), ons(T,88))
        self.song_names = [s[0] for s in songs]
        N = len(songs)
        lens = [s[1].shape[0] for s in songs]
        Tmax = max(lens)
        self.song_lens = torch.tensor(lens, device=self.device, dtype=torch.long)
        # GEOMETRY GUARDRAIL: route each keyboard half to the hand physically on that
        # side (derived from real geometry, so a flipped/relocated piano self-corrects).
        self._swap_hands = self._compute_swap_hands()
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
            plan = plan_fingering(act, method=getattr(self.cfg, "fingering_method", "heuristic"),
                                  swap_hands=self._swap_hands)
            pfk = plan.finger_key.copy(); pfa = plan.finger_active.copy()
            if getattr(self.cfg, "remap_thumb_to_middle", False):
                # the heuristic hands sparse notes to the THUMB -- a poor straight-down
                # presser, offset from the palm and pointing sideways. Move thumb
                # assignments to the MIDDLE finger (centered under the palm, presses
                # DOWN cleanly onto the key the arm targets). (L_th->L_mf, R_th->R_mf)
                for th, mid in ((0, 2), (5, 7)):
                    mv = pfa[:, th] & ~pfa[:, mid]       # thumb active, middle free
                    pfk[mv, mid] = pfk[mv, th]; pfa[mv, mid] = True
                    pfa[mv, th] = False
            fk = torch.as_tensor(pfk, device=self.device)                    # (T,10)
            fa = torch.as_tensor(pfa, device=self.device)                    # (T,10)
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
        # Time-dilated onset target for the onset-timing metric: onset_win[t,k]=1 if a
        # real onset for key k falls within +/-W steps of t (the played strike lands a
        # step or two after the nominal onset, so exact-step matching reads ~0).
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
        # The policy's action is a residual on top of this pose (rail + fingers).
        _Tpad = self.goal_padded.shape[1]                           # Tmax + L
        _ready = torch.stack([self.left_default[0], self.right_default[0]], dim=0)  # (2,25)
        self.base_pose = _ready[None, None].repeat(self.num_songs, _Tpad, 1, 1)  # (N,Tpad,2,25)

        # action buffer / targets
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        # previous-step action + its frame-to-frame change, for the action-jerk
        # (smoothness) diagnostic. Updated each _pre_physics_step.
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
            idle_hover_weight=self.cfg.idle_hover_weight,
            idle_hover_close=self.cfg.idle_hover_close,
            idle_hover_margin_mult=self.cfg.idle_hover_margin_mult,
            idle_hover_z_only=self.cfg.idle_hover_z_only,
            arm_position_weight=self.cfg.arm_position_weight,
            arm_position_close=self.cfg.arm_position_close,
            arm_position_margin_mult=self.cfg.arm_position_margin_mult,
        )
        # RP1M reward mode: r_OT + r_Press + a1*r_Collision - a2*r_Energy, with the
        # fingering re-solved online by optimal transport each step (dexsim.piano.ot).
        self.rp1m_mode = getattr(self.cfg, "reward_mode", "dexsim") == "rp1m"
        self.rp1m_cfg = RP1MRewardCfg(
            ot_weight=self.cfg.rp1m_ot_weight,
            ot_close=self.cfg.rp1m_ot_close,
            ot_margin_mult=self.cfg.rp1m_ot_margin_mult,
            ot_reduce=self.cfg.rp1m_ot_reduce,
            ot_eps=self.cfg.rp1m_ot_eps,
            ot_iters=self.cfg.rp1m_ot_iters,
            ot_side_weight=self.cfg.rp1m_ot_side_weight,
            press_weight=self.cfg.rp1m_press_weight,
            press_close=self.cfg.rp1m_press_close,
            press_margin_mult=self.cfg.rp1m_press_margin_mult,
            press_false_soft=self.cfg.rp1m_press_false_soft,
            collision_weight=self.cfg.rp1m_collision_weight,
            collision_dist=self.cfg.rp1m_collision_dist,
            energy_weight=self.cfg.rp1m_energy_weight,
        )
        if self.rp1m_mode:
            print(f"[PianoEnv] REWARD=RP1M: r_OT(w={self.rp1m_cfg.ot_weight}, "
                  f"reduce={self.rp1m_cfg.ot_reduce}) + r_Press(w={self.rp1m_cfg.press_weight}, "
                  f"hard_fp={not self.rp1m_cfg.press_false_soft}) + "
                  f"{self.rp1m_cfg.collision_weight}*r_Collision - "
                  f"{self.rp1m_cfg.energy_weight}*r_Energy  [no sustain pedal in this piano]")

        # RECALL-GATED ANNEALING (press-discovery curriculum, see piano_env_cfg): hold
        # false-press at false_press_start (energy at 0) until recall EMA >= the gate,
        # then ramp both to their cfg finals over anneal_steps. Monotonic; updated per step.
        self._anneal = bool(getattr(self.cfg, "anneal_false_press", False))
        if self._anneal:
            self._fp_final = float(self.cfg.false_press_weight)
            self._en_final = float(self.cfg.energy_weight)
            # start clamped to the final so the anneal can only ever LOWER the early
            # penalty, never raise it (e.g. phase0 sets the final to 0 -> no-op).
            self.reward_cfg.false_press_weight = min(float(self.cfg.false_press_start),
                                                     self._fp_final)
            self.reward_cfg.energy_weight = 0.0
            self._anneal_recall_ema = 0.0
            _steps = max(1, int(self.cfg.anneal_steps))
            self._fp_rate = max(0.0, self._fp_final - self.reward_cfg.false_press_weight) / _steps
            self._en_rate = self._en_final / _steps
            print(f"[PianoEnv] ANNEAL: false_press {self.reward_cfg.false_press_weight} -> "
                  f"{self._fp_final}, energy 0 -> {self._en_final} over {_steps} steps, "
                  f"gated on recall EMA >= {self.cfg.anneal_recall_gate}")
        _names = ", ".join(self.song_names[:4]) + ("..." if self.num_songs > 4 else "")
        print(f"[PianoEnv] {self.num_songs} song(s) [{_names}]: longest {self.song_len} steps "
              f"@ {1/self.cfg.control_dt:.0f}Hz")

    # ------------------------------------------------------------- songs
    def _compute_swap_hands(self) -> bool:
        """GEOMETRY GUARDRAIL flag for plan_fingering. Returns True when low-pitch keys
        are physically nearer the RIGHT robot than the LEFT (e.g. a 180deg-flipped
        piano) -- so the default 'low-pitch -> left finger group' would route each hand
        ACROSS the body and the two arms jam in the middle. Computed from the real
        piano transform + base positions, so it self-corrects for any layout."""
        loc = geometry.key_local_top_positions()                 # (88,3) piano-local
        q = np.asarray(self.cfg.piano_rot, dtype=float)          # wxyz
        p = np.asarray(self.cfg.piano_pos, dtype=float)
        u = q[1:]
        # world Y of each key = piano_pos + quat-rotated local pos (v' = v + w t + u x t, t = 2 u x v)
        def world_y(v):
            t = 2.0 * np.cross(u, v)
            return float((p + v + q[0] * t + np.cross(u, t))[1])
        wy = np.array([world_y(v) for v in loc])                 # (88,)
        split = NUM_KEYS // 2
        low_y = float(wy[:split].mean())                         # mean world-Y of low-pitch half
        lb = float(self.cfg.left_base_pos[1]); rb = float(self.cfg.right_base_pos[1])
        swap = abs(low_y - rb) < abs(low_y - lb)                 # low-pitch nearer the right robot?
        print(f"[PianoEnv] hand-side guardrail: low-pitch keys mean worldY={low_y:+.2f}, "
              f"left_robot Y={lb:+.2f}, right_robot Y={rb:+.2f} -> swap_hands={swap}")
        return bool(swap)

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

        # --- contact sensors for RP1M's r_Collision ---------------------------
        # The hand USDs already spawn with activate_contact_sensors=True, which
        # puts PhysxContactReportAPI on every body but reports nothing on its own;
        # these sensors are what actually read it.
        #
        # BOTH sides of the pair must resolve to exactly ONE prim per env:
        #   * one sensor per left-hand body -- Isaac Lab documents filtered
        #     reporting as one-to-many, and a prim_path matching several bodies
        #     silently returns a wrong force matrix;
        #   * one filter PATTERN per right-hand body -- a wildcard like
        #     `RightRobot/.*` makes PhysX log "Filter pattern ... did not match the
        #     correct number of entries (expected 4, found 112)" and collapse to a
        #     single meaningless channel. So the "many" in one-to-many is the
        #     length of the pattern LIST, not the breadth of one pattern.
        # Net result: len(contact_bodies)^2 body-pair force channels.
        #
        # Only `force_matrix_w` is usable here -- `net_forces_w` cannot tell a key
        # press from a hand collision, and every fingertip is pressing keys.
        # Registered after clone_environments so the per-env prims exist; the
        # sensors themselves initialize lazily when the sim starts playing.
        self.hand_contacts = []
        if self._wants_contact_sensors():
            from isaaclab.sensors import ContactSensor, ContactSensorCfg
            filters = [f"/World/envs/env_.*/RightRobot/{b}" for b in self.cfg.contact_bodies]
            for body in self.cfg.contact_bodies:
                sensor = ContactSensor(ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/LeftRobot/{body}",
                    filter_prim_paths_expr=filters,
                    update_period=0.0,      # every sim step
                    history_length=0,       # >0 forces a recompute every substep
                    track_air_time=False,
                ))
                self.scene.sensors[f"contact_{body}"] = sensor
                self.hand_contacts.append(sensor)
            print(f"[PianoEnv] CONTACT SENSORS: {len(self.hand_contacts)} left-hand bodies x "
                  f"{len(filters)} right-hand filters {list(self.cfg.contact_bodies)}")

    def _wants_contact_sensors(self) -> bool:
        return (getattr(self.cfg, "reward_mode", "dexsim") == "rp1m"
                and getattr(self.cfg, "rp1m_collision_contacts", False)
                and float(getattr(self.cfg, "rp1m_collision_weight", 0.0)) > 0.0)

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
        ref = self.base_pose[self.song_id, self.song_step]          # (E, 2, 25) ready pose
        # residual on all joints, but per-joint scaled (gentle rail / generous hand;
        # see self.joint_scale). Targets are clamped to joint limits and obs are
        # NaN-guarded, so a transient blow-up can't crash PPO.
        scale = self.joint_scale
        lo = self.left_robot.data.soft_joint_pos_limits[..., 0]
        hi = self.left_robot.data.soft_joint_pos_limits[..., 1]
        # mute_right_hand: for left-hand-only songs, hold the right hand at its
        # ref pose so it can't mash idle keys (kills the false-press noise).
        if getattr(self.cfg, "mute_right_hand", False):
            a_right = a_right * 0.0
        # SOLO the right MIDDLE finger: zero every action except its flex DoFs so no
        # other finger can move (no neighbour mash); the policy only learns WHEN to strike.
        if getattr(self.cfg, "solo_right_middle", False):
            if not hasattr(self, "_solo_mask"):
                _names = self.right_robot.data.joint_names
                _m = torch.zeros(self.per_arm_dof, device=self.device)
                for _jn in ["robot0_MFJ3", "robot0_MFJ2", "robot0_MFJ1", "robot0_MFJ0"]:
                    if _jn in _names:
                        _m[_names.index(_jn)] = 1.0
                self._solo_mask = _m
            a_right = a_right * self._solo_mask
            a_left = a_left * 0.0
        base_l, base_r = ref[:, 0], ref[:, 1]
        self._left_target = torch.clamp(base_l + scale * a_left, lo, hi)
        self._right_target = torch.clamp(base_r + scale * a_right, lo, hi)
    def _apply_action(self):
        self.left_robot.set_joint_position_target(self._left_target)
        self.right_robot.set_joint_position_target(self._right_target)
        # piano keys are passive (spring drives only) -- never commanded.

    # --------------------------------------------------------- world helpers
    def _key_pressed_fraction(self) -> torch.Tensor:
        key_angle = self.piano.data.joint_pos            # negative when pressed
        frac = (key_angle / KEY_SOUND_ANGLE).clamp(0.0, 2.0)
        # a physics blow-up can NaN the key joints; keep it finite so the key/onset
        # reward terms (and their logged means) don't get poisoned to NaN.
        frac = torch.nan_to_num(frac, nan=0.0, posinf=2.0, neginf=0.0)
        # RAW depression, before the velocity gate below. RP1M's press reward shapes
        # on this: a key on its way down must already earn partial credit, which the
        # gated value (0 until a strike latches) cannot provide.
        self._key_frac_raw = frac
        # VELOCITY-GATED sounding (piano hammer): a key STARTS sounding only when struck
        # -- pressed past key_struck_frac AND moving down faster than key_strike_vel --
        # and rings until it springs back above key_release_frac. A statically-resting
        # hand (~0 velocity) never triggers a strike. Called once per step (_get_rewards).
        vel = torch.nan_to_num(self.piano.data.joint_vel, nan=0.0)   # <0 = pressing down
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
        centroid = torch.stack(centroids, dim=1)                       # (E,2,3) [left,right]
        active = torch.stack(actives, dim=1)                           # (E,2)
        # HARD LANE CLAMP (the guardrail): keep each hand's target in its OWN half so
        # the two arms can never reach across the midline into each other -> no jam.
        # Midline = halfway between the two bases; the robot with the smaller base-Y
        # owns Y<=mid, the other owns Y>=mid. Derived from geometry, not hardcoded.
        if getattr(self.cfg, "lane_clamp", True):
            lb = float(self.cfg.left_base_pos[1]); rb = float(self.cfg.right_base_pos[1])
            # cfg base positions are ENV-LOCAL but the centroid is WORLD-frame, so the
            # midline MUST be offset by each env's grid origin (a local scalar drags one
            # hand's target to the global y=0 line for every env with origin y != 0).
            mid = 0.5 * (lb + rb) + self.scene.env_origins[:, 1]             # (E,) per env
            lo_idx, hi_idx = (0, 1) if lb <= rb else (1, 0)  # index of the -Y / +Y hand
            centroid[:, lo_idx, 1] = torch.minimum(centroid[:, lo_idx, 1], mid)  # -Y hand stays Y<=mid
            centroid[:, hi_idx, 1] = torch.maximum(centroid[:, hi_idx, 1], mid)  # +Y hand stays Y>=mid
        return centroid, active

    def _hand_home_targets(self) -> torch.Tensor:
        """(E,2,3) per-hand 'home' rest point: centroid of each hand's idle home keys
        (self.finger_home, already split per hand and swap-corrected by plan_fingering),
        at key-top height. An idle hand parks here -- over its OWN half -- so it stays
        ready near the keys instead of being ignored by the positioning reward."""
        key_top = self._key_top_world()                               # (E,88,3)
        half = NUM_FINGERS // 2
        outs = []
        for sl in (slice(0, half), slice(half, NUM_FINGERS)):         # left, then right
            idx = self.finger_home[sl].clamp(min=0).long()            # (5,) home key per finger
            gidx = idx.view(1, -1, 1).expand(self.num_envs, -1, 3)    # (E,5,3)
            outs.append(torch.gather(key_top, 1, gidx).mean(dim=1))   # (E,3) hand home centroid
        return torch.stack(outs, dim=1)                               # (E,2,3) [left,right]

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
        """(E,) normalized distance of the WORST base joint to its nearest limit.
        1.0 = mid-range (healthy), 0.0 = pinned against a limit. Scored over each
        hand's rail joint only (fingers excluded via arm_joint_mask), both hands."""
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

    # ------------------------------------------------------------ RP1M reward
    def _ot_side_cost(self, key_top: torch.Tensor) -> torch.Tensor | None:
        """(E,10,88) optional penalty for a finger being matched to a key in the
        OTHER hand's half. RP1M has no such term -- its hands share a workspace --
        but ours ride disjoint rails, so without it the transport happily assigns a
        key to a finger that physically cannot reach it. 0 weight = pure RP1M."""
        w = float(self.rp1m_cfg.ot_side_weight)
        if w <= 0.0:
            return None
        lb = float(self.cfg.left_base_pos[1]); rb = float(self.cfg.right_base_pos[1])
        mid = 0.5 * (lb + rb) + self.scene.env_origins[:, 1]        # (E,) per-env midline
        ky = key_top[..., 1]                                        # (E,88) key world Y
        over = ky - mid.unsqueeze(-1)                               # +ve = high-Y half
        half = NUM_FINGERS // 2
        pen = torch.zeros(self.num_envs, NUM_FINGERS, NUM_KEYS, device=self.device)
        lo_first = lb <= rb          # is the LEFT robot the low-Y one?
        lo_sl = slice(0, half) if lo_first else slice(half, NUM_FINGERS)
        hi_sl = slice(half, NUM_FINGERS) if lo_first else slice(0, half)
        pen[:, lo_sl, :] = over.clamp(min=0.0).unsqueeze(1)         # low hand reaching up
        pen[:, hi_sl, :] = (-over).clamp(min=0.0).unsqueeze(1)      # high hand reaching down
        return w * pen

    def _rp1m_terms(self, pressed: torch.Tensor, goal: torch.Tensor):
        """The RP1M composite for this step. Returns ``(total (E,), parts dict)``.

        The fingering is NOT read from the precomputed plan: the finger->key
        matching is re-solved here, every step, from the live fingertip positions
        by optimal transport -- the whole point of the paper.
        """
        key_top = self._key_top_world()                             # (E,88,3)
        tips = self._fingertips_world()                             # (E,10,3)
        # Re-centre on each env's own origin before measuring: world coordinates
        # out on the env-spacing grid are large enough to cost real precision in
        # the distance matrix, and only relative geometry matters here.
        origin = self.scene.env_origins.unsqueeze(1)                # (E,1,3)
        tips_rel = tips - origin
        keys_rel = key_top - origin
        if getattr(self.cfg, "rp1m_ot_ignore_x", False):
            # X (depth into the keyboard) is UNACTUATED on the rail hands -- no
            # arm, so no action can reduce a finger's X miss. Including X in the
            # transport cost adds an irreducible floor that keeps r_OT's shaped
            # exponential pinned in its flat tail (measured: stuck at ~0.065 for
            # 5k iterations). Zero both X columns so the cost only measures the
            # axes the policy can actually move (rail Y + curl Z).
            tips_rel = tips_rel.clone(); keys_rel = keys_rel.clone()
            tips_rel[..., 0] = 0.0
            keys_rel[..., 0] = 0.0
        ot_dist, n_active = ot_finger_cost(
            tips_rel, keys_rel, goal, eps=self.rp1m_cfg.ot_eps,
            iters=self.rp1m_cfg.ot_iters, side_cost=self._ot_side_cost(key_top))

        collided = self._hands_collided(tips)                       # (E,) bool

        # r_Energy = |tau|^T |v| over every actuated joint of both hands. Real
        # mechanical power, not the action^2 proxy the dexsim reward uses.
        energy = torch.zeros(self.num_envs, device=self.device)
        for robot in (self.left_robot, self.right_robot):
            tau = torch.nan_to_num(robot.data.applied_torque, nan=0.0)
            vel = torch.nan_to_num(robot.data.joint_vel, nan=0.0)
            energy = energy + (tau.abs() * vel.abs()).sum(-1)

        return rp1m_reward(
            ot_dist, n_active, self._key_frac_raw, self.key_sounding.float(), goal,
            collided=collided, energy=energy, cfg=self.rp1m_cfg)

    def _hands_collided(self, tips: torch.Tensor) -> torch.Tensor:
        """(E,) bool -- are the two hands touching this step?

        Real PhysX contacts when the sensors are up (``rp1m_collision_contacts``),
        otherwise the geometric proximity stand-in. Touching ``sensor.data`` is
        what triggers the buffer read: Isaac Lab's ``scene.update()`` only marks
        the sensor outdated, the PhysX fetch happens lazily right here.
        """
        if self.hand_contacts:
            hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            thresh = float(self.cfg.rp1m_collision_force)
            for sensor in self.hand_contacts:
                f = sensor.data.force_matrix_w              # (E, 1, M, 3)
                if f is None:                              # not initialized yet
                    continue
                hit |= (torch.nan_to_num(f, nan=0.0).norm(dim=-1) > thresh).flatten(1).any(dim=1)
            self._collided_src = "contact"
            return hit
        self._collided_src = "proximity"
        half = NUM_FINGERS // 2
        palms = self._palms_world()                                 # (E,2,3)
        left_pts = torch.cat([tips[:, :half], palms[:, 0:1]], dim=1)
        right_pts = torch.cat([tips[:, half:], palms[:, 1:2]], dim=1)
        return hands_in_proximity(left_pts, right_pts, self.rp1m_cfg)

    # ---------------------------------------------------------------- reward
    def _get_rewards(self) -> torch.Tensor:
        pressed = self._key_pressed_fraction()
        goal = self._goal_now()
        energy = (self.actions ** 2).mean(dim=-1)
        r_key = piano_reward(pressed, goal, self.reward_cfg, energy=energy)

        key_top = self._key_top_world()
        surface, press_tgt, active = self._finger_targets_world(key_top)
        tips = self._fingertips_world()
        r_finger = fingering_reward(tips, surface, active, self.reward_cfg)
        r_onset = onset_reward(pressed, self._onset_now(), self.reward_cfg)

        # PHASE 0: gross arm positioning -- reward each hand-base for reaching the centroid
        # of the keys it must play over the lookahead window (the hand "covers" them). The
        # target is that centroid lifted to the playing hover height, so a correctly placed
        # hand sits over (not in) the keys. Off (weight 0) in Phases 1/2.
        if self.reward_cfg.arm_position_weight > 0.0:
            centroid, arm_active = self._hand_note_centroids()        # (E,2,3), (E,2)
            if getattr(self.cfg, "arm_home_idle", True):
                # idle hand -> rest at its home hover (over its own half); active hand
                # -> track its note centroid. Reward BOTH so both hands stay parked.
                home = self._hand_home_targets()                      # (E,2,3)
                arm_tgt = torch.where(arm_active.unsqueeze(-1), centroid, home).clone()
                arm_hands = torch.ones_like(arm_active)               # always position both hands
            else:
                arm_tgt = centroid.clone()
                arm_hands = arm_active                                # only active hands count
            arm_tgt[..., 2] += self.cfg.arm_ik_hover                  # aim above the keys
            r_arm = arm_position_reward(self._palms_world(), arm_tgt, arm_hands, self.reward_cfg)
            # debug: raw palm-target gaps so the training log itself localizes any
            # discrepancy between logged arm_pos and offline deterministic replays
            _gap = ((self._palms_world() - arm_tgt) ** 2).sum(-1).sqrt()  # (E,2) m
            self._dbg_gap = (float(_gap[:, 0].mean()), float(_gap[:, 1].mean()),
                             float(_gap.median()), float(_gap.max()))
        else:
            r_arm = torch.zeros_like(r_key)

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

        # IDLE-FINGER HOVER (positive twin of the clear penalty above): pull each
        # NON-assigned finger toward its hover-home -- press_tgt already holds that
        # point for idle fingers (home key top + HOVER_CLEARANCE, the same targets
        # the observation exposes). Continuous gradient on idle fingers vs the
        # penalty's flat-then-cliff, so "one finger down, the rest up" is shaped.
        if self.reward_cfg.idle_hover_weight > 0.0:
            r_hover = idle_hover_reward(tips, press_tgt, active, self.reward_cfg)
        else:
            r_hover = torch.zeros_like(r_key)

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
        # RECALL-GATED ANNEAL: track a recall EMA (only over steps that had goal
        # notes) and, while it sits above the gate, ramp the false-press penalty and
        # energy cost toward their final values. Affects the NEXT step's r_key (the
        # weights were already consumed above) -- a one-step lag that is irrelevant
        # at a 2000-step ramp. Monotonic: a recall dip pauses the ramp, never undoes it.
        if self._anneal:
            if has_goal.any():
                b = float(self.cfg.anneal_recall_beta)
                self._anneal_recall_ema = b * self._anneal_recall_ema + (1.0 - b) * float(rec)
            if self._anneal_recall_ema >= float(self.cfg.anneal_recall_gate):
                rc_ = self.reward_cfg
                rc_.false_press_weight = min(self._fp_final, rc_.false_press_weight + self._fp_rate)
                rc_.energy_weight = min(self._en_final, rc_.energy_weight + self._en_rate)
        # --- onset timing: of the keys struck this step, how many landed near a real
        # onset (within +/-onset_tol_steps via onset_win)? Catches mis-timed strikes the
        # held-note F1 hides. (eval_reference.py computes the full offline onset P/R/F1.) ---
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
        margin = self._arm_limit_margin()                           # (E,) 1=healthy, 0=at a limit
        jerk = self._action_jerk                                    # (E,) action thrash this step
        limit_margin = margin.mean()
        action_jerk = jerk.mean()
        # MOTION-HEALTH PENALTY: damp flailing (jerk) and contortion to joint limits.
        w_jerk = getattr(self.cfg, "jerk_weight", 0.0)
        w_limit = getattr(self.cfg, "limit_weight", 0.0)
        r_jerk = -w_jerk * jerk                                     # (E,) <= 0
        r_limit = -w_limit * (1.0 - margin)                        # (E,) <= 0, 0 when mid-range
        # Per-component finite guard. The world-pos/key accessors are already
        # nan-guarded at source, but guard each term here too so (a) one blown-up
        # env can never NaN-poison the *summed* reward (which nan_to_num would then
        # zero wholesale, silently deleting the key/onset signal the env earned),
        # and (b) the logged means below are a single bad env away from reading nan.
        g = lambda x: torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        r_key, r_finger, r_onset = g(r_key), g(r_finger), g(r_onset)
        r_idle, r_hover = g(r_idle), g(r_hover)
        r_arm = g(r_arm)
        r_jerk, r_limit = g(r_jerk), g(r_limit)

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
            "reward/idle_clear": float(r_idle.mean()),
            "reward/idle_hover": float(r_hover.mean()),
            "reward/arm_pos": float(r_arm.mean()),
            "reward/jerk_pen": float(r_jerk.mean()),
            "reward/limit_pen": float(r_limit.mean()),
        }
        if self._anneal:
            self.extras["log"].update({
                "curriculum/false_press_w": float(self.reward_cfg.false_press_weight),
                "curriculum/energy_w": float(self.reward_cfg.energy_weight),
                "curriculum/recall_ema": float(self._anneal_recall_ema),
            })
        if getattr(self, "_dbg_gap", None) is not None:
            gl, gr, gmed, gmax = self._dbg_gap
            self.extras["log"].update({"debug/gap_left": gl, "debug/gap_right": gr,
                                       "debug/gap_median": gmed, "debug/gap_max": gmax})
        if getattr(self, "_dbg_blown", None) is not None:
            self.extras["log"]["debug/blown_frac"] = self._dbg_blown
        # RP1M MODE: the paper's composite replaces the dexsim terms outright. The
        # dexsim terms above are still computed and logged (they cost a handful of
        # elementwise ops and make the two recipes directly comparable in wandb),
        # they just don't enter the returned reward.
        if self.rp1m_mode:
            rp1m_total, rp1m_parts = self._rp1m_terms(pressed, goal)
            for k, v in rp1m_parts.items():
                self.extras["log"][f"rp1m/{k}"] = float(g(v).mean())
            # r_Collision reads 0.5 when clean and 0 when colliding, so log the
            # raw rate too -- it's the number you actually want to watch.
            if "collision" in rp1m_parts and self.rp1m_cfg.collision_weight > 0:
                self.extras["log"]["rp1m/collided_frac"] = float(
                    1.0 - rp1m_parts["collision"].mean() / self.rp1m_cfg.collision_weight)
            reward = g(rp1m_total)
        else:
            reward = (r_key + r_finger + r_onset + r_idle + r_hover + r_arm
                      + r_jerk + r_limit)
        # F1 BONUS (both reward modes): the eval metric itself, added as a weighted
        # per-env, per-step term ON TOP of shaping -- never a replacement (pure F1
        # is gradient-dead at F1=0, the exact failure the shaping exists to avoid).
        # It is naturally self-annealing: pays ~0 until the policy earns nonzero
        # F1, then aligns the objective with the metric we actually score.
        w_f1 = float(getattr(self.cfg, "f1_weight", 0.0))
        if w_f1 > 0.0:
            f1_env = 2 * recall * precision / (recall + precision + 1e-9)  # (E,)
            r_f1 = w_f1 * g(f1_env * has_goal.float())
            reward = reward + r_f1
            self.extras["log"]["reward/f1_bonus"] = float(r_f1.mean())
        # final band-clamp: a transient blow-up that slips past the per-term guards
        # still can't push a NaN/inf advantage -> NaN log_std -> PPO crash.
        reward = torch.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        self.extras["log"]["reward/total"] = float(reward.mean())
        return reward

    # ----------------------------------------------------------------- dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        song_len_e = self.song_lens[self.song_id]                   # (E,) per-env song length
        song_done = self.song_step >= (song_len_e - 1)
        self.song_step = torch.minimum(self.song_step + 1, song_len_e - 1)
        # BLOWN-UP env detection: a physics explosion leaves NaN/inf body poses; the
        # NaN-guards clamp them so the env would otherwise sit scoring ~0 and drag every
        # logged mean. Terminate so it resets immediately.
        blown = ~(torch.isfinite(self.left_robot.data.body_pos_w).all(dim=(1, 2))
                  & torch.isfinite(self.right_robot.data.body_pos_w).all(dim=(1, 2)))
        self._dbg_blown = float(blown.float().mean())   # logged in _get_rewards' extras
        terminated = blown
        truncated = time_out | song_done
        return terminated, truncated

    # ----------------------------------------------------------------- reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.left_robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # start each arm at the ready pose (the residual base's first frame)
        q0 = self.base_pose[0, 0]                         # (2, 25) song 0, step 0
        for robot, ref in ((self.left_robot, q0[0]), (self.right_robot, q0[1])):
            jp = ref.unsqueeze(0).repeat(len(env_ids), 1)
            jv = torch.zeros_like(jp)
            robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)

        kp = self.piano.data.default_joint_pos[env_ids]
        kv = torch.zeros_like(kp)
        self.piano.write_joint_state_to_sim(kp, kv, env_ids=env_ids)

        self.song_step[env_ids] = 0
        self.key_sounding[env_ids] = False
