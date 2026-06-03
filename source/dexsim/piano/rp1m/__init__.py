"""RP1M / RoboPianist warm-start pipeline (decode -> remap -> retarget -> assemble).

Turns a RoboPianist/RP1M *action* trajectory (recorded for two MuJoCo Shadow
hands on sliding forearms) into a joint-space **reference** for our Isaac
UR10e+Shadow rig, so it can seed behaviour cloning / PPO warm-start. This is the
"fine-tune-not-replay" path: the Shadow *hand* pose transfers (same morphology),
the *arm* must be retargeted (their sliding forearm -> our UR10e), and the press
itself is left for RL to refine in our physics.

Submodules:
  * :mod:`decode`   — split + un-normalise the action vector into Isaac robot0_*
    hand-joint trajectories + forearm (tx, ty) + sustain  (Parts 3a).
  * :mod:`retarget` — forearm (tx, ty) -> wrist target in our piano frame, and
    assembling a name-keyed reference -> ``q_ref`` (T, 2, ndof)  (Parts 3b/3c).
"""

from . import decode, retarget  # noqa: F401
