"""Spatially-smooth goal encoding (PianoMime's SDF goal representation).

PianoMime conditions its policy not on the raw 88-D binary "which keys" vector
but on a learned 16-D latent from an autoencoder trained to reconstruct a
*signed distance field* of the goal, which makes the representation spatially
consistent (nearby keys -> nearby features). That smoothness matters: it tells
the policy "the nearest key to press is just to your right," which a flat binary
vector hides.

Training that autoencoder is a separate offline job over ~10K MIDIs. As a
faithful, dependency-free stand-in we compute the *analytic* version of the same
quantity: for each key, the (normalized) distance along the keyboard to the
nearest key that should be active. This is exactly the 1-D distance transform the
learned SDF approximates, and it drops straight into the observation. The learned
encoder is provided as an optional drop-in (``GoalSDFAutoencoder``) for the
generalist stage.
"""

from __future__ import annotations

import torch

from . import geometry as geom


# pairwise |y_i - y_j| between all keys, in meters (constant; built once).
_KEY_Y = torch.tensor(geom.KEY_Y, dtype=torch.float32)            # (88,)
_PAIR_DY = (_KEY_Y[:, None] - _KEY_Y[None, :]).abs()              # (88, 88)
_SPAN = float(geom.KEYBOARD_SPAN_Y)
_BIG = _SPAN * 4.0


def nearest_active_distance(goal: torch.Tensor) -> torch.Tensor:
    """(E, 88) normalized distance from each key to the nearest *active* key.

    0 at keys that are themselves active; grows with keyboard distance to the
    nearest note; == 1 (clamped) when there are no active keys. This is the
    analytic SDF feature fed to the policy alongside the raw goal.
    """
    pair = _PAIR_DY.to(goal.device)                               # (88, 88)
    inactive_pen = (1.0 - goal.float()) * _BIG                    # (E, 88) over j
    # dist[e,i] = min_j ( pair[i,j] + penalty_if_j_inactive )
    cost = pair[None] + inactive_pen[:, None, :]                  # (E, 88, 88)
    dist = cost.min(dim=-1).values                                # (E, 88)
    return (dist / _SPAN).clamp(0.0, 1.0)


class GoalSDFAutoencoder(torch.nn.Module):
    """Optional learned encoder matching PianoMime (88 -> latent via SDF recon).

    Encodes the goal into a low-dim latent; trained offline to reconstruct
    ``nearest_active_distance``. Use the analytic feature directly for single-song
    RL; train this for the multi-song generalist where a compact conditioning
    code helps a diffusion-BC policy generalize. Provided so the generalist
    scaffold has a concrete, swappable encoder.
    """

    def __init__(self, num_keys: int = 88, latent_dim: int = 16):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(num_keys, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 64), torch.nn.ELU(),
            torch.nn.Linear(64, latent_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 64), torch.nn.ELU(),
            torch.nn.Linear(64, 128), torch.nn.ELU(),
            torch.nn.Linear(128, num_keys),
        )

    def encode(self, goal: torch.Tensor) -> torch.Tensor:
        return self.encoder(goal.float())

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(goal))

    def recon_loss(self, goal: torch.Tensor) -> torch.Tensor:
        target = nearest_active_distance(goal)
        return torch.nn.functional.mse_loss(self.forward(goal), target)
