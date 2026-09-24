"""MuJoCo typing task: Shadow Hands on the MacBook keyboard.

Import this package directly (``from dexsim.tasks.keyboard_mj import ...``).
"""

from .keyboard_mj_env_cfg import KeyboardMjEnvCfg
from .keyboard_mj_env import KeyboardMjEnv, keystrokes
from .vec_env import KeyboardMjVecEnv, KeyboardMjSubprocVecEnv

__all__ = ["KeyboardMjEnvCfg", "KeyboardMjEnv", "keystrokes",
           "KeyboardMjVecEnv", "KeyboardMjSubprocVecEnv"]
