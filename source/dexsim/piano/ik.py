"""Single-frame damped-least-squares IK for the arm of a combined arm+hand articulation.

This is the "arm-as-servo" solver behind the decoupled control design: pure math
drives the arm so a chosen body (the palm, or one fingertip) reaches a target
pose, while an RL policy learns finger pressing on top. Driving a single 6-DoF
(or 3-DoF position-only) target on a 6-DoF arm is *well posed*, so it converges
tightly and stably across the whole keyboard (``diag_posik.py``: ~1 cm), and the
policy never touches the stiff arm joints (no PhysX blow-up).

Per control tick it solves the damped least-squares step

    dq = Jᵀ (J Jᵀ + λ²I)⁻¹ e ,   e = [e_pos ; e_rot]

with e_pos clipped to ``max_step`` and e_rot (axis-angle) to ``max_ang_step``.
Iterating walks the body onto the target. Fixed-base Jacobian indexing is detected
once from the physx view shape.
"""

from __future__ import annotations

import torch


class WristPoseIK:
    """Damped-least-squares IK for a SINGLE body's 6-DoF pose (position + orientation).

    Drive one frame (the palm / wrist, or a fingertip) to a target pose using the
    arm. A single 6-DoF target on a 6-DoF arm is *well posed*, so it converges
    tightly and stably; the fingers are then free to splay onto the individual keys
    (driven by the policy), not by this solver.

    Solves, per tick, the 6-row spatial-Jacobian DLS step
        dq = Jᵀ (J Jᵀ + λ²I)⁻¹ [e_pos ; e_rot]
    with e_pos clipped to ``max_step`` and e_rot (axis-angle) clipped to
    ``max_ang_step``. Iterating walks the body onto the target pose.
    """

    def __init__(self, articulation, body_name: str, damping: float = 0.05,
                 max_step: float = 0.05, max_ang_step: float = 0.2,
                 arm_only: bool = True, pos_only: bool = False,
                 planar: bool = False, planar_weight: float = 25.0, planar_iters: int = 6,
                 arm_prefixes=("shoulder", "elbow", "wrist_"), freeze_joints=()):
        self.robot = articulation
        self.damping = damping
        self.max_step = max_step
        self.max_ang_step = max_ang_step
        # PLANAR mode: emulate an XY gantry on the redundant arm. A plain 1-step DLS treats
        # all 6 pose components as equal soft objectives, so a large XY error makes it trade
        # away Z + orientation -> the hand tilts and SAGS onto the keys (toppling). Planar
        # mode (a) WEIGHTS the 4 pinned components (z, roll, pitch, yaw) heavily so the solver
        # refuses to sacrifice height/fingers-down posture, and (b) ITERATES the (fixed-
        # Jacobian) Gauss-Newton step so the XY target is actually reached per control step
        # (no mid-move sag). Result: the arm slides flat in XY -- exactly the slider's plane.
        self.planar = planar
        self.planar_weight = planar_weight    # weight on z + 3 orientation rows (xy stay 1.0)
        self.planar_iters = max(1, int(planar_iters))
        # pos_only: solve a 3-DoF POSITION target on the 6-DoF arm (drop orientation).
        # A 6-DoF pose target on a fingertip OVER-constrains (the held orientation fights
        # the position -> ~250mm residual); a 3-DoF position target is well-posed/under-
        # constrained, so the arm walks the fingertip ONTO the key to ~1cm (the diag proof
        # that the arm places any rigid body to 1cm is a POSITION result).
        self.pos_only = pos_only

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

        # freeze_joints: PIN specific arm joints (matched by name substring) at their
        # init/ready angle -- e.g. ("wrist_3",) locks the UR10e's last DoF (final wrist
        # roll) so the wrist orientation about its own axis stays constant while the
        # other 5 joints still servo. Masking the Jacobian alone only makes dq=0, which
        # sets the PD target to the CURRENT angle (zero restoring force) -> the joint
        # drifts; so solve() rewrites these columns to the captured constant every tick.
        self.freeze_joints = tuple(freeze_joints)
        self._frozen_idx = None
        self._frozen_target = None
        if self.freeze_joints:
            frozen = torch.tensor(
                [any(f in n for f in self.freeze_joints)
                 for n in articulation.data.joint_names],
                device=articulation.device)
            self.dof_mask = self.dof_mask & ~frozen
            self._frozen_idx = torch.nonzero(frozen, as_tuple=False).flatten()
            self._frozen_target = articulation.data.joint_pos[:, self._frozen_idx].clone()

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
        J = self._spatial_jacobian().clone()                       # (E, 6, D)
        J[:, :, ~self.dof_mask] = 0.0                              # freeze non-arm dofs
        if self.planar and not self.pos_only:
            dq = self._solve_planar(pos, quat, target_pos, target_quat, J)
        else:
            e_pos = (target_pos - pos).clamp(-self.max_step, self.max_step)
            if self.pos_only:
                e = e_pos                                          # (E, 3) position only
                Jr = J[:, 0:3, :]                                  # (E, 3, D) translational
            else:
                e_rot = self._orientation_error(target_quat, quat).clamp(
                    -self.max_ang_step, self.max_ang_step)
                e = torch.cat([e_pos, e_rot], dim=-1)              # (E, 6)
                Jr = J
            Jt = Jr.transpose(-1, -2)                              # (E, D, r)
            JJt = Jr @ Jt                                          # (E, r, r)
            eye = torch.eye(JJt.shape[-1], device=J.device).expand_as(JJt)
            sol = torch.linalg.solve(JJt + (self.damping ** 2) * eye, e.unsqueeze(-1))
            dq = (Jt @ sol).squeeze(-1)                            # (E, D)

        q = self.robot.data.joint_pos + dq
        if self._frozen_idx is not None:
            q[:, self._frozen_idx] = self._frozen_target   # hold pinned joints at init
        lower = self.robot.data.soft_joint_pos_limits[..., 0]
        upper = self.robot.data.soft_joint_pos_limits[..., 1]
        return torch.clamp(q, lower, upper)

    def _solve_planar(self, pos, quat, target_pos, target_quat, J):
        """Weighted + iterated DLS: hold Z + orientation (heavy weight) and SLIDE in XY
        (iterate the fixed-Jacobian Gauss-Newton step so the XY target is reached this
        control step). Returns accumulated dq -> emulates an XY gantry on the redundant arm."""
        E = pos.shape[0]
        W = self.planar_weight
        # task weights: x,y normal(1); z + roll/pitch/yaw heavy -> never traded away for XY.
        sw = torch.tensor([1.0, 1.0, W, W, W, W], device=J.device).sqrt()       # (6,)
        Jw = J * sw.view(1, 6, 1)                                                # weight rows
        Jt = Jw.transpose(-1, -2)                                               # (E, D, 6)
        JJt = Jw @ Jt                                                           # (E, 6, 6)
        eye = torch.eye(6, device=J.device).expand_as(JJt)
        A = JJt + (self.damping ** 2) * eye
        Jp, Jr = J[:, 0:3, :], J[:, 3:6, :]                                     # pos / rot rows
        e_rot0 = self._orientation_error(target_quat, quat).clamp(
            -self.max_ang_step, self.max_ang_step)
        dq_acc = torch.zeros(E, self.num_dof, device=J.device)
        pos_p = pos                                   # predicted controlled-body position
        rot_acc = torch.zeros(E, 3, device=J.device)  # rotation realized by dq_acc so far
        for _ in range(self.planar_iters):
            e_pos = (target_pos - pos_p).clamp(-self.max_step, self.max_step)
            e_rot = (e_rot0 - rot_acc).clamp(-self.max_ang_step, self.max_ang_step)
            e = torch.cat([e_pos, e_rot], dim=-1) * sw.view(1, 6)              # weighted err
            sol = torch.linalg.solve(A, e.unsqueeze(-1))
            dq = (Jt @ sol).squeeze(-1)
            dq_acc = dq_acc + dq
            pos_p = pos_p + (Jp @ dq.unsqueeze(-1)).squeeze(-1)                # predict pose
            rot_acc = rot_acc + (Jr @ dq.unsqueeze(-1)).squeeze(-1)
        return dq_acc
