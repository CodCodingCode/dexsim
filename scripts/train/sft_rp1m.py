"""Offline SFT (behavior cloning) on the RP1M dataset — RoboPianist-native space.

This is the REAL "pretrain/SFT on RP1M": supervised learning directly on RP1M's
recorded (observation, action) pairs. No simulator, no embodiment gap — the
policy is trained in RoboPianist's own 39-d hands-on-sliding-forearms action
space, which is exactly where the data lives. (Deploying it on the UR10e is a
separate step: IK the forearm output -> wrist target, then residual RL for
dynamics. This script only does the SFT.)

Observation  (per step): hand_joints(46) + piano_states(89) + goal lookahead
                         (H future goal frames, 89 each)  ->  46+89+89*H
Action       (per step): RP1M's 39-d action  (right hand | left hand | sustain)

Per song we keep the best `--per_song` rollouts (by F1 of piano_states vs goals)
so we train on high-quality expert data, not all 500 noisy seeds.

  # prove on the 3-song toy (seconds):
  python scripts/sft_rp1m.py --zarr data/rp1m/rp1m_toy.zarr --epochs 50 --out logs/sft/toy.pt
  # scale to the 300-song set:
  python scripts/sft_rp1m.py --zarr data/rp1m/rp1m_repertoire.zarr --per_song 40 \
      --epochs 30 --out logs/sft/rp1m300.pt
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import zarr


def f1_per_traj(goals: np.ndarray, piano: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Proxy F1 of achieved vs goal key activation, per trajectory (N,)."""
    kg = goals[..., :88].astype(bool)
    pr = piano[..., :88] > thr
    tp = (pr & kg).sum((1, 2)); fp = (pr & ~kg).sum((1, 2)); fn = (~pr & kg).sum((1, 2))
    prec = tp / np.maximum(tp + fp, 1); rec = tp / np.maximum(tp + fn, 1)
    return 2 * prec * rec / np.maximum(prec + rec, 1e-9)


def build_song(g, per_song: int, horizon: int):
    """-> (obs (M,D), act (M,39)) for the best `per_song` rollouts of one song.

    obs = hand_joints(46) + piano_states(89) + goal-lookahead(89*H) + PREV action(39).
    The previous action is the single biggest BC lever: a closed-loop expert's
    action is strongly autocorrelated, so giving the policy a_{t-1} lets it predict
    the small delta instead of the absolute pose -> far lower, non-mode-averaged MSE.
    a_{-1} := 0 at each rollout's first step.
    """
    goals = g["goals"][:]            # (N,T,89) bool
    piano = g["piano_states"][:]     # (N,T,89)
    hand = g["hand_joints"][:]       # (N,T,46)
    act = g["actions"][:].astype(np.float32)   # (N,T,39)
    N, T, _ = goals.shape
    f1 = f1_per_traj(goals, piano)
    keep = np.argsort(-f1)[:min(per_song, N)]
    gf = goals.astype(np.float32)
    idx = np.minimum(np.arange(T)[:, None] + np.arange(horizon)[None, :], T - 1)  # (T,H)
    look = gf[:, idx, :].reshape(N, T, horizon * goals.shape[-1])                 # (N,T,89H)
    prev = np.concatenate([np.zeros_like(act[:, :1]), act[:, :-1]], axis=1)       # (N,T,39)
    obs = np.concatenate([hand.astype(np.float32), piano.astype(np.float32),
                          look, prev], axis=-1)
    obs = obs[keep].reshape(-1, obs.shape[-1])
    a = act[keep].reshape(-1, act.shape[-1])
    return obs, a, float(f1[keep].mean())


class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=(512, 256, 128)):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ELU()]; d = h
        layers += [nn.Linear(d, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_or_build(args):
    """Build the (X, Y) dataset from zarr, or load a cached .npz to skip the
    ~11-min rebuild while iterating on the model."""
    if args.cache and os.path.exists(args.cache):
        print(f"[sft] loading cached dataset {args.cache}")
        d = np.load(args.cache)
        return d["X"], d["Y"], float(d["kept_f1"])
    root = zarr.open(args.zarr, mode="r")
    songs = list(root.group_keys())
    if args.songs:
        songs = songs[:args.songs]
    print(f"[sft] {len(songs)} songs from {args.zarr}; building (best {args.per_song}"
          f"/song, horizon {args.horizon}, +prev-action)...")
    t0 = time.time()
    obs_list, act_list, f1s = [], [], []
    for i, name in enumerate(songs):
        o, a, mf1 = build_song(root[name], args.per_song, args.horizon)
        obs_list.append(o); act_list.append(a); f1s.append(mf1)
        if i % max(1, len(songs) // 10) == 0:
            print(f"  [{i+1}/{len(songs)}] kept-F1={mf1:.3f} "
                  f"rows={sum(x.shape[0] for x in obs_list):,}")
    X = np.concatenate(obs_list); Y = np.concatenate(act_list)
    kept = float(np.mean(f1s))
    print(f"[sft] dataset obs{X.shape} act{Y.shape} kept-F1={kept:.3f} ({time.time()-t0:.0f}s)")
    if args.cache:
        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        np.savez(args.cache, X=X, Y=Y, kept_f1=kept)
        print(f"[sft] cached -> {args.cache}")
    return X, Y, kept


def main():
    ap = argparse.ArgumentParser(description="Offline SFT (BC) on RP1M.")
    ap.add_argument("--zarr", default=None)
    ap.add_argument("--songs", type=int, default=0, help="limit #songs (0=all)")
    ap.add_argument("--per_song", type=int, default=50, help="best rollouts per song")
    ap.add_argument("--horizon", type=int, default=10, help="goal lookahead frames")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", default="1024,1024,512,256", help="MLP widths")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--cache", default=None, help=".npz dataset cache (build once, reuse)")
    ap.add_argument("--out", default="logs/sft/rp1m.pt")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xnp, Ynp, kept = load_or_build(args)
    X = torch.from_numpy(Xnp).to(device); Y = torch.from_numpy(Ynp).to(device)
    del Xnp, Ynp
    Y = torch.nan_to_num(Y).clamp(-1, 1)
    d_in, d_out = X.shape[1], Y.shape[1]
    mu = X.mean(0, keepdim=True); sd = X.std(0, keepdim=True).clamp_min(1e-6)
    X.sub_(mu).div_(sd)   # standardize IN PLACE — avoids a second 28GB tensor (OOM)
    Xn = X

    # train/val split (a low VAL mse is what "well-trained" actually means)
    n = Xn.shape[0]
    g = torch.Generator(device=device).manual_seed(0)
    perm = torch.randperm(n, generator=g, device=device)
    nval = int(n * args.val_frac)
    vi, ti = perm[:nval], perm[nval:]
    hidden = tuple(int(h) for h in args.hidden.split(","))
    model = MLP(d_in, d_out, hidden).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    print(f"[sft] model {hidden} ({nparams/1e6:.1f}M params)  train={len(ti):,} val={len(vi):,}")

    best_val = 1e9
    for ep in range(args.epochs):
        model.train(); p = ti[torch.randperm(len(ti), device=device)]; tot = 0.0
        for i in range(0, len(p), args.batch):
            idx = p[i:i + args.batch]
            loss = nn.functional.mse_loss(model(Xn[idx]), Y[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * idx.numel()
        sched.step()
        if ep % max(1, args.epochs // 20) == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                vs = 0.0
                for j in range(0, len(vi), args.batch):
                    b = vi[j:j + args.batch]
                    vs += nn.functional.mse_loss(model(Xn[b]), Y[b]).item() * len(b)
                vmse = vs / len(vi)
            best_val = min(best_val, vmse)
            print(f"[sft] epoch {ep:4d}  train_mse={tot/len(ti):.5f}  val_mse={vmse:.5f}")

    if any(not torch.isfinite(pp).all() for pp in model.parameters()):
        raise RuntimeError("[sft] model diverged to NaN; aborting save")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "obs_mean": mu.cpu(),
                "obs_std": sd.cpu(), "obs_dim": d_in, "act_dim": d_out,
                "horizon": args.horizon, "hidden": hidden, "kept_f1": kept,
                "best_val_mse": best_val}, args.out)
    print(f"[sft] saved -> {args.out}  best_val_mse={best_val:.5f}")


if __name__ == "__main__":
    main()
