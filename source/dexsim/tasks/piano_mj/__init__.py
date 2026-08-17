"""MuJoCo port of the bimanual piano task (Isaac-free).

Import this package directly (``from dexsim.tasks.piano_mj import ...``); it
is not auto-imported by ``dexsim.tasks`` so the Isaac venv never needs mujoco.
"""

from .piano_mj_env_cfg import PianoMjEnvCfg
from .piano_mj_env import PianoMjEnv
from .vec_env import PianoMjVecEnv, make_rsl_rl_env
from .song_bank import SongBank

__all__ = [
    "PianoMjEnvCfg",
    "PianoMjEnv",
    "PianoMjVecEnv",
    "make_rsl_rl_env",
    "SongBank",
]
