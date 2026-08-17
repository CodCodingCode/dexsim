"""The PyRoki least-squares IK call, kept apart from the scripting surface.

Only the position of each fingertip is constrained: a finger's orientation on a
key is not something the task cares about, and constraining it just makes
reachable presses look unreachable.

The jitted solve is defined ONCE at module level. Defining it inside the call
would hand JAX a brand-new function object every time and re-trace the whole
problem on each solve, which turns a millisecond search into minutes.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk


@jdc.jit
def _run(robot, target_wxyz, target_pos, link_indices, rest, pos_weight,
         rest_weight):
    JointVar = robot.joint_var_cls
    target = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_wxyz), target_pos)
    costs = [
        pk.costs.pose_cost_analytic_jac(
            jax.tree.map(lambda x: x[None], robot),
            JointVar(jnp.full(target.get_batch_axes(), 0)),
            target, link_indices, pos_weight=pos_weight, ori_weight=0.0),
        pk.costs.rest_cost(JointVar(0), rest_pose=rest, weight=rest_weight),
        pk.costs.limit_cost(robot, JointVar(0), weight=100.0),
        pk.costs.limit_constraint(robot, JointVar(0)),
    ]
    return (jaxls.LeastSquaresProblem(costs=costs, variables=[JointVar(0)])
            .analyze()
            .solve(verbose=False, linear_solver="dense_cholesky",
                   trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0))
            )[JointVar(0)]


def solve_targets(side, targets: dict, pos_weight: float = 50.0,
                  rest_weight: float = 0.02) -> np.ndarray:
    """targets: {finger: world xyz} -> the hand's 25 joint values.

    Fingers with no target are held near the ready pose by the rest bias rather
    than being left free, which keeps the unused fingers looking like a hand.
    """
    from .piano_hands import FINGERTIPS

    fingers = sorted(targets)
    link_names = list(side.robot.links.names)
    indices = np.array([link_names.index(FINGERTIPS[f]) for f in fingers], np.int32)

    # world -> the articulation root frame pyroki solves in
    inv = side.base_quat.T
    local = np.stack([inv @ (np.asarray(targets[f], float) - side.base_pos)
                      for f in fingers])
    wxyz = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(fingers), 1))

    return np.asarray(_run(side.robot, jnp.array(wxyz), jnp.array(local),
                           jnp.array(indices), jnp.array(side.rest),
                           jnp.array(float(pos_weight)),
                           jnp.array(float(rest_weight))))
