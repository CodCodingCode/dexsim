"""Gym task registrations for dexsim.

Importing this package registers the dexsim environments with Gymnasium so they
can be created by id (e.g. ``gym.make("Dexsim-Reorient-Cube-Shadow-v0")``).
"""

try:
    from . import reorient  # noqa: F401  (registers RL reorientation envs)
    from . import grasp     # noqa: F401  (registers grasp / imitation envs)
    from . import piano     # noqa: F401  (registers the bimanual piano env)
except ModuleNotFoundError as _e:  # Isaac not installed (MuJoCo-only venv)
    if "isaac" not in str(_e).lower():
        raise
# The MuJoCo piano task (dexsim.tasks.piano_mj) is Isaac-free; import it
# directly -- it is intentionally NOT auto-imported here so the Isaac venv
# doesn't need mujoco installed either.
