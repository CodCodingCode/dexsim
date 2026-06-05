"""Parse the piano training log and render a one-glance health 'picture'.

Reads the rsl_rl iteration tables from a training log (the per-iteration block
prints play/F1, play/precision, play/recall and the reward terms), builds time
series, plots them to a PNG, and prints a terse verdict on whether the run is
actually learning to play the piano (F1 trending up).

  python scripts/snapshot_run.py --log logs/rl_run.log --out logs/progress.png
"""
from __future__ import annotations

import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--log", default="logs/rl_run.log")
p.add_argument("--out", default="logs/progress.png")
args = p.parse_args()

# rsl_rl strips ANSI but the iteration header has bold codes; match the number.
ITER_RE = re.compile(r"Learning iteration\s+(\d+)/")
METRICS = {
    "F1": re.compile(r"play/F1:\s*([-\d.eE]+)"),
    "precision": re.compile(r"play/precision:\s*([-\d.eE]+)"),
    "recall": re.compile(r"play/recall:\s*([-\d.eE]+)"),
    "keys_sounding": re.compile(r"play/keys_sounding:\s*([-\d.eE]+)"),
    "r_key": re.compile(r"reward/key:\s*([-\d.eE]+)"),
    "r_finger": re.compile(r"reward/finger:\s*([-\d.eE]+)"),
    "r_onset": re.compile(r"reward/onset:\s*([-\d.eE]+)"),
    "mean_reward": re.compile(r"Mean reward:\s*([-\d.eE]+)"),
}

with open(args.log, errors="ignore") as f:
    text = f.read()

# Split the log into per-iteration chunks keyed by the iteration header.
chunks = []  # (iter_num, chunk_text)
parts = re.split(r"(Learning iteration\s+\d+/)", text)
for i in range(1, len(parts), 2):
    m = ITER_RE.search(parts[i])
    if not m:
        continue
    chunks.append((int(m.group(1)), parts[i] + parts[i + 1] if i + 1 < len(parts) else parts[i]))

series = {k: [] for k in METRICS}
iters = []
for it, chunk in chunks:
    row = {}
    ok = True
    for k, rx in METRICS.items():
        mm = rx.search(chunk)
        if mm:
            row[k] = float(mm.group(1))
        else:
            ok = False
            row[k] = float("nan")
    iters.append(it)
    for k in METRICS:
        series[k].append(row[k])

if not iters:
    print("VERDICT: no iterations parsed yet (env still booting or first iter not done).")
    # still emit a placeholder so the loop has something to show
    raise SystemExit(0)

n = len(iters)


def ffill(xs):
    """Forward-fill NaNs: the per-step reward MEANS log as NaN whenever any single
    env had a transient blow-up that step (training reward itself is guarded), so
    carry the last finite value to keep the composition plot readable."""
    out, last = [], float("nan")
    for v in xs:
        if v == v:
            last = v
        out.append(last)
    return out


for k in ("r_key", "r_finger", "r_onset"):
    series[k] = ffill(series[k])
last = {k: series[k][-1] for k in METRICS}

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].plot(iters, series["F1"], "-o", ms=3, color="tab:green")
ax[0, 0].set_title("play/F1  (THE signal: correct notes sounding)")
ax[0, 0].set_xlabel("iteration"); ax[0, 0].set_ylabel("F1"); ax[0, 0].grid(alpha=.3)

ax[0, 1].plot(iters, series["precision"], "-o", ms=3, label="precision")
ax[0, 1].plot(iters, series["recall"], "-o", ms=3, label="recall")
ax[0, 1].set_title("precision vs recall"); ax[0, 1].set_xlabel("iteration")
ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)

for k, c in [("r_key", "tab:red"), ("r_finger", "tab:blue"),
             ("r_onset", "tab:orange")]:
    ax[1, 0].plot(iters, series[k], "-o", ms=2, label=k, color=c)
ax[1, 0].set_title("reward terms"); ax[1, 0].set_xlabel("iteration")
ax[1, 0].legend(); ax[1, 0].grid(alpha=.3)

ax[1, 1].plot(iters, series["mean_reward"], "-o", ms=3, color="black", label="mean reward")
ax[1, 1].plot(iters, series["keys_sounding"], "-o", ms=2, color="tab:gray", label="keys sounding")
ax[1, 1].set_title("mean reward / keys sounding"); ax[1, 1].set_xlabel("iteration")
ax[1, 1].legend(); ax[1, 1].grid(alpha=.3)

fig.suptitle(f"piano run @ iter {iters[-1]}  |  F1={last['F1']:.3f}  "
             f"prec={last['precision']:.3f}  rec={last['recall']:.3f}", fontsize=13)
fig.tight_layout()
fig.savefig(args.out, dpi=90)

# --- terse verdict (trend over the last ~half of the run) ---
half = max(1, n // 2)
f1_early = sum(series["F1"][:half]) / half
f1_late = sum(series["F1"][half:]) / (n - half) if n > half else series["F1"][-1]
trend = f1_late - f1_early
direction = "RISING" if trend > 0.005 else ("FLAT" if abs(trend) <= 0.005 else "FALLING")
print(f"iters_parsed={n}  last_iter={iters[-1]}")
print(f"F1: last={last['F1']:.3f}  early_mean={f1_early:.3f}  late_mean={f1_late:.3f}  trend={trend:+.3f} ({direction})")
print(f"precision={last['precision']:.3f}  recall={last['recall']:.3f}  "
      f"keys_sounding={last['keys_sounding']:.2f}  mean_reward={last['mean_reward']:.2f}")
print(f"rewards: key={last['r_key']:.3f} finger={last['r_finger']:.3f} "
      f"onset={last['r_onset']:.3f}")
if n < 40:
    verdict = f"TOO EARLY ({n} iters; piano RL needs 100s-1000s — just watch it's stable)"
elif direction == "RISING":
    verdict = "LEARNING (F1 climbing)"
elif direction == "FLAT":
    verdict = "STALLED (F1 flat over a real window — needs intervention)"
else:
    verdict = "REGRESSING (F1 falling)"
print(f"VERDICT: {verdict}")
