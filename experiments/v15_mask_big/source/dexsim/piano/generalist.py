"""PianoMime generalist scaffold: hierarchy + diffusion-BC distillation.

The single-song pipeline (residual RL over an IK reference, in the env) produces
*specialist* policies — one per song, like RoboPianist. PianoMime's contribution
on top is a **generalist** that plays unseen songs, built in two moves:

  1. **Hierarchy** — split the policy into
       * high-level: goal (which keys, over a horizon) -> fingertip trajectory.
         Cheap to supervise (PianoMime gets it from internet video; we get it
         from our analytic fingering -> :func:`dexsim.piano.fingering`).
       * low-level: fingertip trajectory + proprioception -> joint targets.
         Expensive (needs the RL/IK solve), so trained on fewer rollouts.
  2. **Diffusion BC distillation** — clone many song-specific experts into ONE
     conditional diffusion policy (DDPM), conditioned on a compact goal latent
     (see :class:`dexsim.piano.goal_encoding.GoalSDFAutoencoder`). Diffusion is
     used because the expert action distribution is multimodal.

This module provides working, minimal versions of those components plus a small
conditional DDPM so ``scripts/distill_generalist.py`` has a concrete model to
train once you have multi-song rollouts. It is the documented on-ramp to the
generalist, not the single-song critical path.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def mlp(sizes: list[int], act=nn.ELU, out_act=None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
        elif out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)


class HighLevelPolicy(nn.Module):
    """goal-horizon -> fingertip-target trajectory (10 fingers x 3, over `horizon`).

    Input is the goal lookahead (flattened) plus the SDF goal latent; output is
    the desired fingertip targets the low level should track. Supervised by the
    analytic fingering targets (``finger_targets_local``), which is the cheap,
    internet-video-free version of PianoMime's high level.
    """

    def __init__(self, goal_dim: int, latent_dim: int = 16, horizon: int = 1,
                 hidden=(256, 256)):
        super().__init__()
        self.horizon = horizon
        self.net = mlp([goal_dim + latent_dim, *hidden, 10 * 3 * horizon])

    def forward(self, goal_feat: torch.Tensor, goal_latent: torch.Tensor) -> torch.Tensor:
        x = torch.cat([goal_feat, goal_latent], dim=-1)
        return self.net(x).reshape(-1, self.horizon, 10, 3)


class ConditionalDDPM(nn.Module):
    """Minimal conditional DDPM over actions (PianoMime's policy class).

    A small MLP denoiser eps(a_t, t, cond) trained with the standard DDPM noise-
    prediction loss; ``sample`` runs the reverse chain. Used to distill the
    multimodal expert action distribution into one generalist policy.
    """

    def __init__(self, action_dim: int, cond_dim: int, steps: int = 50, hidden=(512, 512)):
        super().__init__()
        self.steps = steps
        self.net = mlp([action_dim + cond_dim + 1, *hidden, action_dim])
        betas = torch.linspace(1e-4, 0.02, steps)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

    def _eps(self, a_t, t, cond):
        t_emb = (t.float() / self.steps).unsqueeze(-1)
        return self.net(torch.cat([a_t, cond, t_emb], dim=-1))

    def loss(self, action, cond):
        b = action.shape[0]
        t = torch.randint(0, self.steps, (b,), device=action.device)
        ac = self.alphas_cumprod[t].unsqueeze(-1)
        noise = torch.randn_like(action)
        a_t = ac.sqrt() * action + (1 - ac).sqrt() * noise
        return torch.nn.functional.mse_loss(self._eps(a_t, t, cond), noise)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, action_dim: int) -> torch.Tensor:
        a = torch.randn(cond.shape[0], action_dim, device=cond.device)
        for i in reversed(range(self.steps)):
            t = torch.full((cond.shape[0],), i, device=cond.device, dtype=torch.long)
            eps = self._eps(a, t, cond)
            ac = self.alphas_cumprod[t].unsqueeze(-1)
            beta = self.betas[t].unsqueeze(-1)
            alpha = 1 - beta
            mean = (a - beta / (1 - ac).sqrt() * eps) / alpha.sqrt()
            a = mean + (beta.sqrt() * torch.randn_like(a) if i > 0 else 0.0)
        return a
