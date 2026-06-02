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
