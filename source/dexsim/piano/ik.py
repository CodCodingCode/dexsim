"""Multi-fingertip damped-least-squares IK over a combined arm+hand articulation.

This is the "autopilot" from the PianoMime recipe: instead of asking RL to
explore all 30 joints of an arm+hand, we let IK convert *desired fingertip
positions* into joint targets. The arm's redundancy is absorbed here; RL only has
to learn a residual on top (see ``PianoEnv`` and ``build_reference.py``).

We drive all five fingertips at once: stack their 3-row position Jacobians into a
(3*5, num_dof) matrix and solve one damped least-squares step

    dq = Jᵀ (J Jᵀ + λ²I)⁻¹ e ,   e = clip(target − fingertip, ±max_step)

per control tick. Iterating this (the reference builder does several ticks of
physics between song steps) walks the fingertips onto their keys while the arm
follows. Fixed-base Jacobian indexing is detected from the physx view shape.

The "fingertip" point is the distal-link body origin (``robot0_*distal``) — the
*same* point the reward measures distance to, so IK and reward are consistent.
"""

from __future__ import annotations

import torch

from .fingering import FINGERTIP_BODIES


class FingertipIK:
    def __init__(self, articulation, damping: float = 0.05, max_step: float = 0.05):
        self.robot = articulation
        self.damping = damping
        self.max_step = max_step

        # fingertip body indices (order = FINGERTIP_BODIES = [th, ff, mf, rf, lf])
        self.tip_ids = []
        for name in FINGERTIP_BODIES:
            ids, names = articulation.find_bodies(name)
            if not ids:
                raise RuntimeError(f"fingertip body '{name}' not found; have: "
                                   f"{articulation.body_names}")
            self.tip_ids.append(ids[0])
        self.tip_ids_t = torch.tensor(self.tip_ids, device=articulation.device)
        self.num_dof = articulation.num_joints

        # detect fixed-base Jacobian layout once
        jac = self.robot.root_physx_view.get_jacobians()
        self.num_bodies = len(articulation.body_names)
        self._fixed_base = jac.shape[1] == (self.num_bodies - 1)
        self._dof_offset = 0 if self._fixed_base else 6
        self._jac_body_shift = 1 if self._fixed_base else 0

    # ------------------------------------------------------------------ reads
    def fingertips_w(self) -> torch.Tensor:
        """(E, 5, 3) current fingertip world positions."""
        return self.robot.data.body_pos_w[:, self.tip_ids_t, :]

    def _position_jacobians(self) -> torch.Tensor:
        """(E, 5, 3, num_dof) stacked translational Jacobians of the 5 tips."""
        jac = self.robot.root_physx_view.get_jacobians()        # (E, B, 6, Dj)
        cols = slice(self._dof_offset, self._dof_offset + self.num_dof)
        out = []
        for bid in self.tip_ids:
            jrow = jac[:, bid - self._jac_body_shift, 0:3, cols]  # (E, 3, num_dof)
            out.append(jrow)
        return torch.stack(out, dim=1)                          # (E, 5, 3, D)

    # ------------------------------------------------------------------ solve
    def solve(self, targets_w: torch.Tensor) -> torch.Tensor:
        """One DLS step. ``targets_w`` (E, 5, 3) desired fingertip world pos.
        Returns joint-position targets (E, num_dof) = current q + dq (clamped to
        joint limits)."""
        cur = self.fingertips_w()                               # (E, 5, 3)
        err = (targets_w - cur).clamp(-self.max_step, self.max_step)
        E = err.shape[0]
        e = err.reshape(E, -1)                                  # (E, 15)

        Jp = self._position_jacobians().reshape(E, -1, self.num_dof)  # (E, 15, D)
        Jt = Jp.transpose(-1, -2)                               # (E, D, 15)
        JJt = Jp @ Jt                                           # (E, 15, 15)
        eye = torch.eye(JJt.shape[-1], device=JJt.device).expand_as(JJt)
        A = JJt + (self.damping ** 2) * eye
        sol = torch.linalg.solve(A, e.unsqueeze(-1))            # (E, 15, 1)
        dq = (Jt @ sol).squeeze(-1)                             # (E, D)

        q = self.robot.data.joint_pos + dq
        lower = self.robot.data.soft_joint_pos_limits[..., 0]
        upper = self.robot.data.soft_joint_pos_limits[..., 1]
        return torch.clamp(q, lower, upper)


class WristPoseIK:
    """Damped-least-squares IK for a SINGLE body's 6-DoF pose (position + orientation).

    This is the "arm-as-servo" solver: drive one frame (the palm / wrist) to a
    target pose using the arm. Unlike :class:`FingertipIK` (which over-constrains a
    6-DoF arm with 5 fingertip targets -> blurry ~45 mm residual and occasional
    divergence), a single 6-DoF target on a 6-DoF arm is *well posed*, so it
    converges tightly and stably. The fingers are then free to splay onto the
    individual keys (driven by the policy / an RP1M reference), not by this solver.

    Solves, per tick, the 6-row spatial-Jacobian DLS step
        dq = Jᵀ (J Jᵀ + λ²I)⁻¹ [e_pos ; e_rot]
    with e_pos clipped to ``max_step`` and e_rot (axis-angle) clipped to
    ``max_ang_step``. Iterating walks the body onto the target pose.
    """

    def __init__(self, articulation, body_name: str, damping: float = 0.05,
                 max_step: float = 0.05, max_ang_step: float = 0.2,
                 arm_only: bool = True, arm_prefixes=("shoulder", "elbow", "wrist_")):
        self.robot = articulation
        self.damping = damping
        self.max_step = max_step
        self.max_ang_step = max_ang_step

        ids, _ = articulation.find_bodies(body_name)
        if not ids:
            raise RuntimeError(f"body '{body_name}' not found; have: {articulation.body_names}")
        self.body_id = ids[0]
        self.num_dof = articulation.num_joints

        # optionally restrict the solution to the arm joints (so the fingers are
        # NOT recruited to move the wrist — they're the policy's job).
        if arm_only:
            self.dof_mask = torch.tensor(
                [any(p in n for p in arm_prefixes) for n in articulation.data.joint_names],
                device=articulation.device)
        else:
            self.dof_mask = torch.ones(self.num_dof, dtype=torch.bool,
                                       device=articulation.device)

        jac = self.robot.root_physx_view.get_jacobians()
        self.num_bodies = len(articulation.body_names)
        self._fixed_base = jac.shape[1] == (self.num_bodies - 1)
        self._dof_offset = 0 if self._fixed_base else 6
        self._jac_body_shift = 1 if self._fixed_base else 0

    def pose_w(self):
        """(pos (E,3), quat (E,4) wxyz) current world pose of the controlled body."""
        return (self.robot.data.body_pos_w[:, self.body_id, :],
                self.robot.data.body_quat_w[:, self.body_id, :])

    @staticmethod
    def _orientation_error(q_des: torch.Tensor, q_cur: torch.Tensor) -> torch.Tensor:
        """Axis-angle (E,3) world-frame rotation taking q_cur -> q_des (small-angle)."""
        # quaternion product err = q_des ⊗ conj(q_cur), Isaac wxyz convention
        wc, xc, yc, zc = q_cur[:, 0], -q_cur[:, 1], -q_cur[:, 2], -q_cur[:, 3]
        wd, xd, yd, zd = q_des[:, 0], q_des[:, 1], q_des[:, 2], q_des[:, 3]
        w = wd * wc - xd * xc - yd * yc - zd * zc
        x = wd * xc + xd * wc + yd * zc - zd * yc
        y = wd * yc - xd * zc + yd * wc + zd * xc
        z = wd * zc + xd * yc - yd * xc + zd * wc
        v = torch.stack([x, y, z], dim=-1)
        # 2*vec part is the small-angle rotation vector; flip so we take the short way
        return 2.0 * torch.sign(w).unsqueeze(-1) * v

    def _spatial_jacobian(self) -> torch.Tensor:
        jac = self.robot.root_physx_view.get_jacobians()           # (E, B, 6, Dj)
        cols = slice(self._dof_offset, self._dof_offset + self.num_dof)
        return jac[:, self.body_id - self._jac_body_shift, 0:6, cols]  # (E, 6, D)

    def solve(self, target_pos: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
        """One DLS step toward (target_pos (E,3), target_quat (E,4) wxyz).
        Returns joint-position targets (E, num_dof), clamped to limits."""
        pos, quat = self.pose_w()
        e_pos = (target_pos - pos).clamp(-self.max_step, self.max_step)
        e_rot = self._orientation_error(target_quat, quat).clamp(-self.max_ang_step,
                                                                 self.max_ang_step)
        e = torch.cat([e_pos, e_rot], dim=-1)                      # (E, 6)

        J = self._spatial_jacobian().clone()                       # (E, 6, D)
        J[:, :, ~self.dof_mask] = 0.0                              # freeze non-arm dofs
        Jt = J.transpose(-1, -2)                                   # (E, D, 6)
        JJt = J @ Jt                                               # (E, 6, 6)
        eye = torch.eye(6, device=J.device).expand_as(JJt)
        sol = torch.linalg.solve(JJt + (self.damping ** 2) * eye, e.unsqueeze(-1))
        dq = (Jt @ sol).squeeze(-1)                                # (E, D)

        q = self.robot.data.joint_pos + dq
        lower = self.robot.data.soft_joint_pos_limits[..., 0]
        upper = self.robot.data.soft_joint_pos_limits[..., 1]
        return torch.clamp(q, lower, upper)
