"""Gym task registrations for dexsim.

Importing this package registers the dexsim environments with Gymnasium so they
can be created by id (e.g. ``gym.make("Dexsim-Reorient-Cube-Shadow-v0")``).
"""

from . import reorient  # noqa: F401  (registers RL reorientation envs)
from . import grasp     # noqa: F401  (registers grasp / imitation envs)
from . import piano     # noqa: F401  (registers the bimanual piano env)
