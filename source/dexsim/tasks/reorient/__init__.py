"""In-hand cube reorientation (RL-from-scratch) on the Shadow Hand.

This wraps Isaac Lab's built-in, well-tuned Shadow Hand reorientation env and
simply swaps in dexsim's own ``SHADOW_HAND_CFG`` so the embodiment is owned by
this project. Domain randomization, reward shaping and the goal-pose machinery
all come from the upstream env -- it's turnkey, no dataset required.
"""

import gymnasium as gym

from . import reorient_env_cfg

gym.register(
    id="Dexsim-Reorient-Cube-Shadow-v0",
    # the in-hand manipulation env drives the upstream Shadow reorientation task
    entry_point="isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env:InHandManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": reorient_env_cfg.DexsimReorientCubeEnvCfg,
        # rsl_rl PPO config shipped by Isaac Lab for the Shadow hand.
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.direct.shadow_hand.agents.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg"
        ),
    },
)

