"""Shared rendering helpers for the bimanual-piano scene.

IMPORTANT: the submodules import heavy ``isaaclab`` modules at top level, which
Isaac Sim requires to happen AFTER ``AppLauncher(...).app`` has started. So only
`from dexsim.render import studio` AFTER the app is launched (same rule the
one-shot render scripts already follow for their ``isaaclab.sim`` imports).
"""
