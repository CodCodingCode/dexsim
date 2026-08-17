"""Offline check of the RP1M reward port (Zhao et al., CoRL 2024).

Verifies the batched optimal-transport finger->key solver against exact scipy
Jonker-Volgenant, and pins the shape of every reward term against the paper's
equations (full credit inside the 1 cm band, the all-or-nothing false-press
clause, and so on).

Runs in seconds and needs NO Isaac -- ``dexsim.piano.ot`` and
``dexsim.piano.reward`` are pure torch/numpy:

    source env.sh && python scripts/smoke/check_rp1m_reward.py

The end-to-end counterpart (env builds, steps, and logs the terms) is
``scripts/smoke/piano_env_smoke.py --reward_mode rp1m``.
"""
import numpy as np
import torch

from dexsim.piano.ot import ot_finger_cost, hungarian_cost
from dexsim.piano.reward import (
    RP1MRewardCfg, rp1m_ot_reward, rp1m_press_reward, rp1m_collision_reward,
    rp1m_energy_cost, rp1m_reward, hands_in_proximity,
)

torch.manual_seed(0)
np.random.seed(0)
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


# ---------------------------------------------------------------- OT solver
print("\n== OT solver: batched Sinkhorn vs exact Jonker-Volgenant (scipy) ==")
E, F, K = 64, 10, 88
# keyboard-like geometry: keys on a line 1.2 m long, fingertips scattered above
key_pos = torch.zeros(E, K, 3)
key_pos[..., 1] = torch.linspace(-0.6, 0.6, K)
key_pos[..., 2] = 0.75
tips = torch.zeros(E, F, 3)
tips[..., 1] = torch.linspace(-0.5, 0.5, F).unsqueeze(0) + 0.05 * torch.randn(E, F)
tips[..., 2] = 0.80 + 0.03 * torch.randn(E, F)
tips[..., 0] = 0.02 * torch.randn(E, F)

goal = torch.zeros(E, K)
n_notes = torch.randint(0, 7, (E,))
for e in range(E):
    if n_notes[e] > 0:
        idx = torch.randperm(K)[: n_notes[e]]
        goal[e, idx] = 1.0

d_sink, n_act = ot_finger_cost(tips, key_pos, goal, eps=0.01, iters=50)
d_exact, n_exact = hungarian_cost(tips.numpy(), key_pos.numpy(), goal.numpy())
d_exact = torch.tensor(d_exact, dtype=torch.float32)

check("n_active matches", torch.allclose(n_act, torch.tensor(n_exact, dtype=torch.float32)))
gap = d_sink - d_exact
# The returned assignment is a feasible matching, so it can never beat the LP.
check("never below the LP optimum", float(gap.min()) > -1e-5, f"(min {float(gap.min())*1000:.4f} mm)")
check("mean gap to exact well under the 1 cm reward band", float(gap.mean()) < 0.001,
      f"(mean {float(gap.mean())*1000:.4f} mm, max {float(gap.max())*1000:.3f} mm)")
n_opt = int((gap.abs() < 1e-5).sum())
print(f"       -> hit the exact LP optimum on {n_opt}/{E} envs")
# Sinkhorn ranking must be at least as good as the default nearest-first ranking
d_sk, _ = ot_finger_cost(tips, key_pos, goal, iters=50)
check("Sinkhorn ranking >= default quality", float((d_sk - d_exact).mean()) <= float(gap.mean()) + 1e-6,
      f"({float((d_sk-d_exact).mean())*1000:.4f} mm vs {float(gap.mean())*1000:.4f} mm)")
check("rest steps cost 0", float(d_sink[n_act == 0].abs().max() if (n_act == 0).any() else 0.0) == 0.0)

# the returned assignment must be a valid matching: <=1 key per finger, and every
# active key matched exactly once (up to the 10-finger ceiling)
d_asg, _, asg = ot_finger_cost(tips, key_pos, goal, return_plan=True)
check("<=1 key per finger", bool((asg.sum(-1) <= 1.0 + 1e-6).all()))
check("every active key matched once", bool(torch.allclose(asg.sum(1), goal)))
C = torch.cdist(tips, key_pos, compute_mode="donot_use_mm_for_euclid_dist")
check("assignment cost == returned dist",
      torch.allclose((asg * C).sum((-2, -1)), d_asg, atol=1e-6))

# a finger sitting exactly on its key must cost ~0
tips2 = tips.clone()
goal2 = torch.zeros(E, K); goal2[:, 40] = 1.0
tips2[:, 5] = key_pos[:, 40]
d2, n2 = ot_finger_cost(tips2, key_pos, goal2)
check("finger on its key -> ~0 cost", d2.max() < 1e-3, f"(max {float(d2.max()):.2e})")

# >10-note polyphony stays finite and feasible
goal3 = torch.zeros(E, K); goal3[:, 20:35] = 1.0
d3, n3 = ot_finger_cost(tips, key_pos, goal3)
check(">10-key polyphony finite", torch.isfinite(d3).all() and float(n3[0]) == 15.0)

# ------------------------------------------------------------- press reward
print("\n== r_Press (RP1M eq. 4) ==")
cfg = RP1MRewardCfg()
g_ = torch.zeros(4, 88); g_[0, 10] = 1; g_[1, 10] = 1; g_[2, 10] = 1  # row 3 = rest
ks = torch.zeros(4, 88)
snd = torch.zeros(4, 88)
ks[0, 10] = 1.0; snd[0, 10] = 1.0                       # perfect
ks[1, 10] = 1.0; snd[1, 10] = 1.0; ks[1, 50] = 1.0; snd[1, 50] = 1.0   # + a wrong key
ks[2, 10] = 0.0                                          # wanted, not pressed
r = rp1m_press_reward(ks, snd, g_, cfg)
check("perfect press = 1.0", abs(float(r[0]) - 1.0) < 1e-5, f"({float(r[0]):.4f})")
check("one wrong key forfeits the whole 0.5 half", abs(float(r[1]) - 0.5) < 1e-5, f"({float(r[1]):.4f})")
check("wanted key untouched -> only the clean half", float(r[2]) < 0.55, f"({float(r[2]):.4f})")
check("rest step with clean hands = 0.5", abs(float(r[3]) - 0.5) < 1e-5, f"({float(r[3]):.4f})")
# with 2 wanted keys down and 1 stray, the hard rule still forfeits the whole
# half while the soft rule charges 1/2 of it
g2 = torch.zeros(1, 88); g2[0, 10] = 1; g2[0, 11] = 1
k2 = torch.zeros(1, 88); k2[0, 10] = 1.0; k2[0, 11] = 1.0; k2[0, 50] = 1.0
s2 = (k2 > 0.5).float()
r_hard = rp1m_press_reward(k2, s2, g2, cfg)
r_soft = rp1m_press_reward(k2, s2, g2, RP1MRewardCfg(press_false_soft=True))
check("soft false-press is gentler than hard", float(r_soft) > float(r_hard) + 1e-4,
      f"(soft {float(r_soft):.3f} vs hard {float(r_hard):.3f})")

# half-pressed key earns partial credit (the gradient the raw frac provides)
ks_half = torch.zeros(1, 88); ks_half[0, 10] = 0.6
g_half = torch.zeros(1, 88); g_half[0, 10] = 1
r_half = rp1m_press_reward(ks_half, torch.zeros(1, 88), g_half, cfg)
r_zero = rp1m_press_reward(torch.zeros(1, 88), torch.zeros(1, 88), g_half, cfg)
check("partial depression > none", float(r_half) > float(r_zero) + 1e-4,
      f"({float(r_half):.4f} > {float(r_zero):.4f})")

# ---------------------------------------------------------------- OT reward
print("\n== r_OT (RP1M eq. 3) ==")
# margin reaches value_at_margin=0.1 at (close + close*margin_mult) = 0.11 m
d = torch.tensor([0.0, 0.005, 0.11, 0.5, 0.0])
n = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
r = rp1m_ot_reward(d, n, cfg)
check("d=0 -> 1.0", abs(float(r[0]) - 1.0) < 1e-6, f"({float(r[0]):.4f})")
check("inside the 1 cm band -> 1.0", abs(float(r[1]) - 1.0) < 1e-6, f"({float(r[1]):.4f})")
check("~0.1 at the margin (11 cm)", abs(float(r[2]) - 0.1) < 0.01, f"({float(r[2]):.4f})")
check("far away -> ~0", float(r[3]) < 1e-3, f"({float(r[3]):.6f})")
check("rest step -> 0", float(r[4]) == 0.0)
check("monotone decreasing in distance", bool((r[:4].diff() <= 0).all()))

# ---------------------------------------------------- collision / energy
print("\n== r_Collision, r_Energy ==")
rc = rp1m_collision_reward(torch.tensor([False, True]), cfg)
check("clean -> 0.5 (a1*1)", abs(float(rc[0]) - 0.5) < 1e-6, f"({float(rc[0]):.3f})")
check("colliding -> 0", abs(float(rc[1])) < 1e-6, f"({float(rc[1]):.3f})")
# the geometric stand-in used when contact sensors are off
lp = torch.zeros(2, 6, 3); rp = torch.zeros(2, 6, 3)
rp[0, :, 1] = 0.5          # far apart
rp[1, :, 1] = 0.005        # overlapping
prox = hands_in_proximity(lp, rp, cfg)
check("proximity stand-in: apart -> False, overlapping -> True",
      bool(~prox[0] and prox[1]), f"({prox.tolist()})")

tau = torch.ones(2, 25); vel = torch.full((2, 25), 2.0)
en = rp1m_energy_cost(tau, vel)
check("energy = sum|tau||v| = 50", abs(float(en[0]) - 50.0) < 1e-4, f"({float(en[0]):.2f})")

# ------------------------------------------------------------- composite
print("\n== composite ==")
total, parts = rp1m_reward(d[:1], n[:1], ks[:1], snd[:1], g_[:1],
                           collided=prox[:1], energy=en[:1], cfg=cfg)
expect = float(parts["ot"] + parts["press"] + parts["collision"] + parts["energy"])
check("parts sum to total", abs(float(total) - expect) < 1e-5,
      f"(total {float(total):.4f} = {{{', '.join(f'{k}:{float(v):.3f}' for k, v in parts.items())}}})")
check("all four terms present", set(parts) == {"ot", "press", "collision", "energy"})

print("\n" + ("ALL CHECKS PASSED" if ok else "*** FAILURES ***"))
raise SystemExit(0 if ok else 1)
