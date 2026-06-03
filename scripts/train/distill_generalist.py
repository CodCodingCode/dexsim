"""Distill per-song experts into one generalist diffusion policy (PianoMime).

This is the generalist stage: pool (observation, action) datasets collected from
several song-specific experts (``scripts/bc_pretrain.py --dump <song>.npz``, one
per song) and behavior-clone them into a single conditional DDPM policy
conditioned on the observation (which already carries the goal lookahead, the SDF
goal encoding, and the fingertip targets). Because it trains on pooled,
pre-collected data, it needs no simulator — run it after you have a few songs'
datasets.

  # 1) collect a dataset per song (sim):
  python scripts/bc_pretrain.py --midi data/midi/twinkle.mid --dump data/distill/twinkle.npz --headless
  python scripts/bc_pretrain.py --midi data/midi/song.mid    --dump data/distill/song.npz    --headless
  # 2) distill (no sim):
  python scripts/distill_generalist.py data/distill/*.npz --epochs 300 --out logs/generalist.pt

The result is PianoMime's generalist: one policy that plays unseen songs by
conditioning on their goal. Diffusion is used because the expert action
distribution across songs/contexts is multimodal.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from dexsim.piano.generalist import ConditionalDDPM


def main():
    ap = argparse.ArgumentParser(description="Distill experts into a diffusion generalist.")
    ap.add_argument("datasets", nargs="+", help="*.npz from bc_pretrain --dump (globs ok)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--steps", type=int, default=50, help="DDPM diffusion steps")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--out", default="logs/generalist.pt")
    args = ap.parse_args()

    paths = [p for pat in args.datasets for p in glob.glob(pat)]
    if not paths:
        raise SystemExit(f"no datasets matched {args.datasets}")
    obs, act = [], []
    for p in paths:
        d = np.load(p)
        obs.append(d["obs"]); act.append(d["action"])
        print(f"  loaded {p}: obs{d['obs'].shape}")
    X = torch.as_tensor(np.concatenate(obs, 0), dtype=torch.float32)
    Y = torch.as_tensor(np.concatenate(act, 0), dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, Y = X.to(device), Y.to(device)
    print(f"[distill] pooled {len(paths)} songs -> obs{tuple(X.shape)} act{tuple(Y.shape)}")

    model = ConditionalDDPM(action_dim=Y.shape[1], cond_dim=X.shape[1], steps=args.steps).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    n = X.shape[0]
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            loss = model.loss(Y[idx], X[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * idx.numel()
        if ep % max(1, args.epochs // 10) == 0:
            print(f"[distill] epoch {ep:4d}  ddpm_loss={tot / n:.5f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "obs_dim": X.shape[1], "action_dim": Y.shape[1], "steps": args.steps}, args.out)
    print(f"[distill] saved generalist -> {args.out}")


main()
