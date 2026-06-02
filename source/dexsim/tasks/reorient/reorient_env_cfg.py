"""Env config for Dexsim cube reorientation.

Inherits the upstream Shadow Hand direct-RL env config and overrides only the
robot articulation so the task runs on dexsim's embodiment. Everything else
(observations, actions, reward, goal resampling, domain randomization) is the
upstream tuned setup.
"""

from __future__ import annotations

from isaaclab.utils import configclass

# Upstream Shadow Hand reorientation env config (ships with isaaclab_tasks).
from isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg


@configclass
class DexsimReorientCubeEnvCfg(ShadowHandEnvCfg):
    """Shadow Hand cube reorientation under a dexsim task id.

    Inherits the upstream, tuned ShadowHandEnvCfg unchanged -- its ``robot_cfg``
    already uses the canonical instanceable Shadow Hand with all 24 joints
    actuated and the cube/goal setup the reward expects. (dexsim's own
    SHADOW_HAND_CFG is API-identical; we keep the upstream init_state here so the
    cube-on-palm task is unperturbed.)
    """


@configclass
class DexsimReorientCubeEnvCfg_PLAY(DexsimReorientCubeEnvCfg):
    """Lightweight variant for visualization / policy playback."""

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.scene.num_envs = 16
