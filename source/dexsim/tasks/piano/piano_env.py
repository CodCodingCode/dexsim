"""DirectRLEnv for the bimanual piano task (PianoMime-style).

Two UR10e+Shadow arms (60 action DoF total) over an 88-key spring-loaded piano.
A MIDI song defines a per-step key-activation goal; an automatic fingering assigns
each note to a finger; a precomputed IK reference trajectory positions the arms so
the policy only has to learn a **residual** on top.

What this env implements from PianoMime / RoboPianist:
  * **Residual action** over an IK reference (``q_ref``): action = q_ref + scale*a.
    Zero action already tracks the reference, so the policy starts competent.
  * **Fingering shaping reward** (finger -> assigned key): the term RoboPianist
    showed is make-or-break (F1 = 0 without it).
  * **Composite reward**: key-press (right keys down, none wrong) + fingering +
    onset (crisp attacks) + energy.
  * **Rich observation**: proprioception + key state + goal lookahead + fingertip
    positions + the reference fingertip targets ("where fingers should go") + an
    analytic SDF goal encoding.

If no reference exists yet, ``q_ref`` falls back to the static ready pose, so the
env still builds and runs (that's how ``build_reference.py`` bootstraps one).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

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
from dexsim import DATA_DIR
from .piano_env_cfg import PianoEnvCfg


class PianoEnv(DirectRLEnv):
    cfg: PianoEnvCfg

    def __init__(self, cfg: PianoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.per_arm_dof = self.left_robot.num_joints
        self.left_default = self.left_robot.data.default_joint_pos.clone()
        self.right_default = self.right_robot.data.default_joint_pos.clone()

        # The policy only drives the HAND (finger) joints; the arm rigidly holds
        # its ready pose. Driving the arm with the residual blew it up (NaN obs ->
        # NaN policy -> "std>=0" crash) and is wrong for piano anyway (the arm
        # positions the hand, the fingers play). Mask: 1.0 on robot0_* joints.
        hand_mask = torch.tensor(
            [1.0 if "robot0_" in n else 0.0 for n in self.left_robot.data.joint_names],
            device=self.device,
        )
        self.hand_mask = hand_mask.unsqueeze(0)        # (1, 30)

        # --- song -> goal / onset tensors ---
        song = load_song(self.cfg.midi_path, control_dt=self.cfg.control_dt)
        self.song_len = song.num_steps
        L = self.cfg.goal_lookahead
        goal = torch.as_tensor(song.key_activation, dtype=torch.float32, device=self.device)
        onset = torch.as_tensor(song.onsets, dtype=torch.float32, device=self.device)
        pad = torch.zeros((L, NUM_KEYS), device=self.device)
        self.goal_padded = torch.cat([goal, pad], dim=0)            # (T+L, 88)
        self.onset_padded = torch.cat([onset, pad], dim=0)          # (T+L, 88)
        self.song_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # --- fingering plan -> per-step finger->key + active mask ---
        plan = plan_fingering(song.key_activation)
        self.finger_key = torch.as_tensor(plan.finger_key, device=self.device)        # (T,10)
        self.finger_active = torch.as_tensor(plan.finger_active, device=self.device)  # (T,10)
        self.finger_home = torch.as_tensor(plan.home_key, device=self.device)         # (10,)
        # pad to T+L so lookahead/clamped indexing is safe
        self.finger_key = torch.cat([self.finger_key, self.finger_key[-1:].repeat(L, 1)], 0)
        self.finger_active = torch.cat(
            [self.finger_active, torch.zeros((L, NUM_FINGERS), dtype=torch.bool, device=self.device)], 0)

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

        # --- reference trajectory (residual base) ---
        self.q_ref = self._load_reference()                         # (T+L, 2, 30)

        # action buffer / targets
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
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
        print(f"[PianoEnv] song '{song.name}': {self.song_len} steps "
              f"({song.duration_s:.1f}s @ {1/self.cfg.control_dt:.0f}Hz); "
              f"reference={'loaded' if self._has_reference else 'FALLBACK(ready pose)'}")

    # ------------------------------------------------------------- reference
    def _reference_file(self) -> Path:
        if self.cfg.reference_path:
            return Path(self.cfg.reference_path)
        return DATA_DIR / "reference" / (Path(self.cfg.midi_path).stem + ".npz")

    def _load_reference(self) -> torch.Tensor:
        """(T+L, 2, 30) joint reference, padded with its last frame for lookahead.
        Falls back to the static ready pose if no reference file is present."""
        L = self.cfg.goal_lookahead
        self._has_reference = False
        path = self._reference_file()
        if self.cfg.use_reference and path.exists():
            data = np.load(path)
            q = torch.as_tensor(data["q_ref"], dtype=torch.float32, device=self.device)
            # (T,2,30); align length to the song
            if q.shape[0] < self.song_len:
                q = torch.cat([q, q[-1:].repeat(self.song_len - q.shape[0], 1, 1)], 0)
            q = q[: self.song_len]
            self._has_reference = True
        else:
            ref_l = self.left_default[0]
            ref_r = self.right_default[0]
            q = torch.stack([ref_l, ref_r], dim=0)[None].repeat(self.song_len, 1, 1)
        return torch.cat([q, q[-1:].repeat(L, 1, 1)], dim=0)        # (T+L, 2, 30)

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
        a_left = self.actions[:, : self.per_arm_dof]
        a_right = self.actions[:, self.per_arm_dof :]
        ref = self.q_ref[self.song_step]                            # (E, 2, 30)
        # full residual on all joints (the arm repositions the hand per note ->
        # this is what reached F1=0.29). Targets are clamped to joint limits and
        # obs are NaN-guarded, so a transient blow-up can't crash PPO.
        scale = self.cfg.action_scale
        lo = self.left_robot.data.soft_joint_pos_limits[..., 0]
        hi = self.left_robot.data.soft_joint_pos_limits[..., 1]
        # mute_right_hand: for left-hand-only songs, hold the right arm at its
        # ref pose so it can't mash idle keys (kills the false-press noise).
        if getattr(self.cfg, "mute_right_hand", False):
            a_right = a_right * 0.0
        self._left_target = torch.clamp(ref[:, 0] + scale * a_left, lo, hi)
        self._right_target = torch.clamp(ref[:, 1] + scale * a_right, lo, hi)

    def _apply_action(self):
        self.left_robot.set_joint_position_target(self._left_target)
        self.right_robot.set_joint_position_target(self._right_target)
        # piano keys are passive (spring drives only) -- never commanded.

    # --------------------------------------------------------- world helpers
    def _key_pressed_fraction(self) -> torch.Tensor:
        key_angle = self.piano.data.joint_pos            # negative when pressed
        return (key_angle / KEY_SOUND_ANGLE).clamp(0.0, 2.0)

    def _key_top_world(self) -> torch.Tensor:
        """(E, 88, 3) world position of each key's top surface."""
        centers = self.piano.data.body_pos_w[:, self.key_body_ids, :]   # (E,88,3)
        top = centers.clone()
        top[..., 2] += self.key_half_h                                  # + half thickness
        return top

    def _fingertips_world(self) -> torch.Tensor:
        """(E, 10, 3) fingertip world positions in global finger order
        [L_th,L_ff,L_mf,L_rf,L_lf, R_th,R_ff,R_mf,R_rf,R_lf]."""
        l = self.left_robot.data.body_pos_w[:, self.ltip_ids, :]        # (E,5,3)
        r = self.right_robot.data.body_pos_w[:, self.rtip_ids, :]
        return torch.cat([l, r], dim=1)                                 # (E,10,3)

    def _finger_targets_world(self, key_top: torch.Tensor):
        """Return (target_surface, target_press, active) for the current step.
        target_surface: (E,10,3) key-top point used for the *fingering reward*.
        target_press:   (E,10,3) press/hover point used for *observation*.
        active:         (E,10) bool, which fingers are assigned a key now."""
        fk = self.finger_key[self.song_step]                            # (E,10)
        fa = self.finger_active[self.song_step]                         # (E,10)
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
        return torch.cat([l, r], dim=1)                                 # (E,2,3)

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
        idx = idx.clamp(max=self.finger_key.shape[0] - 1)              # (E,H)
        fk = self.finger_key[idx]                                       # (E,H,10)
        fa = self.finger_active[idx]                                    # (E,H,10)
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
        return self.goal_padded[idx]                     # (E, L, 88)

    def _goal_now(self) -> torch.Tensor:
        return self.goal_padded[self.song_step]          # (E, 88)

    def _onset_now(self) -> torch.Tensor:
        return self.onset_padded[self.song_step]         # (E, 88)

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
        if not hasattr(self, "extras") or self.extras is None:
            self.extras = {}
        self.extras["log"] = {
            "play/F1": float(f1),
            "play/recall": float(rec),
            "play/precision": float(prec),
            "play/keys_sounding": float((pressed >= 0.5).float().sum(-1).mean()),
            "reward/key": float(r_key.mean()),
            "reward/finger": float(r_finger.mean()),
            "reward/onset": float(r_onset.mean()),
            "reward/arm": float(r_arm.mean()),
        }
        return r_key + r_finger + r_onset + r_arm

    # ----------------------------------------------------------------- dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        song_done = self.song_step >= (self.song_len - 1)
        self.song_step = torch.clamp(self.song_step + 1, max=self.song_len - 1)
        terminated = torch.zeros_like(time_out)
        truncated = time_out | song_done
        return terminated, truncated

    # ----------------------------------------------------------------- reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.left_robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # start each arm at the reference's first frame (== ready pose if no ref)
        q0 = self.q_ref[0]                                # (2, 30)
        for robot, ref in ((self.left_robot, q0[0]), (self.right_robot, q0[1])):
            jp = ref.unsqueeze(0).repeat(len(env_ids), 1)
            jv = torch.zeros_like(jp)
            robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)

        kp = self.piano.data.default_joint_pos[env_ids]
        kv = torch.zeros_like(kp)
        self.piano.write_joint_state_to_sim(kp, kv, env_ids=env_ids)

        self.song_step[env_ids] = 0
