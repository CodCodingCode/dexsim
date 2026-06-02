"""Bimanual piano task registration."""

import gymnasium as gym

from . import piano_env_cfg
from .piano_env import PianoEnv
from .agents.rsl_rl_ppo_cfg import PianoPPORunnerCfg

gym.register(
    id="Dexsim-Piano-Bimanual-v0",
    entry_point="dexsim.tasks.piano.piano_env:PianoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": piano_env_cfg.PianoEnvCfg,
        "rsl_rl_cfg_entry_point": PianoPPORunnerCfg,
    },
)

__all__ = ["PianoEnv", "PianoEnvCfg"]
PianoEnvCfg = piano_env_cfg.PianoEnvCfg
