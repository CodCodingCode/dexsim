#!/usr/bin/env python
"""Sim-free self-check for the RoboPianist/RP1M pre-training pipeline.

Exercises all three pieces set up from the RoboPianist analysis, end to end, with
no Isaac dependency:

  1. MIDI corpus ingestion   (dexsim.piano.corpus)
  2. OT auto-fingering        (dexsim.piano.fingering, method="ot")
  3. RP1M warm-start prep     (dexsim.piano.rp1m: decode -> remap -> merge)

Run:  source env.sh && python scripts/test_pretrain_pipeline.py
Exits non-zero on the first failed assertion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source"))

from dexsim.piano.corpus import MidiCorpus
from dexsim.piano.midi import load_song
from dexsim.piano import fingering as fg
from dexsim.piano.rp1m import decode as dc, retarget as rt

TWINKLE = REPO / "data/midi/twinkle.mid"
ACTIONS = REPO / "data/robopianist_ref/examples/twinkle_twinkle_actions.npy"
ROUSSEAU = REPO / "data/robopianist_ref/robopianist/music/data"

PASS, FAIL = "  [PASS]", "  [FAIL]"
_failures = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _failures
    print(f"{PASS if cond else FAIL} {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        _failures += 1


def part1_corpus() -> None:
    print("\n[1] MIDI corpus ingestion")
    corpus = MidiCorpus.scan([REPO / "data/midi", ROUSSEAU], verbose=False)
    check("scans >= 4 songs", len(corpus) >= 4, f"{len(corpus)} songs")
    cur = corpus.curriculum()
    check("curriculum is easy->hard", all(
        cur[i].difficulty() <= cur[i + 1].difficulty() + 1e-9 for i in range(len(cur) - 1)))
    # manifest round-trips
    mpath = REPO / "data/corpus/_selftest.json"
    corpus.write_manifest(mpath)
    rt2 = MidiCorpus.from_manifest(mpath)
    check("manifest round-trips", rt2.names() == corpus.names())
    mpath.unlink()
    # filtering + sampling
    easy = corpus.filter(max_difficulty=0.4)
    check("filter keeps a subset", 0 < len(easy) <= len(corpus), f"{len(easy)} easy")
    rng = np.random.default_rng(0)
    s = corpus.sample(rng, by_difficulty=True)
    check("sampler returns a real song", s.name in corpus.names())


def part2_ot_fingering() -> None:
    print("\n[2] OT auto-fingering")
    song = load_song(TWINKLE)
    heu = fg.plan_fingering(song.key_activation, method="heuristic")
    ot = fg.plan_fingering(song.key_activation, method="ot")

    def coverage(plan):
        want = np.minimum(song.key_activation.sum(1), 10)
        got = plan.finger_active.sum(1)
        return float(np.clip(got, 0, want).sum() / max(want.sum(), 1))

    def travel(plan):
        tl = fg.finger_targets_local(plan)
        return float(np.linalg.norm(np.diff(tl, axis=0), axis=2).sum())

    check("OT covers all needed notes", coverage(ot) >= 0.999, f"cov={coverage(ot):.3f}")
    check("OT lowers fingertip travel vs heuristic",
          travel(ot) < travel(heu), f"ot={travel(ot):.1f} < heu={travel(heu):.1f}")
    check("OT one key/active-finger consistency",
          bool(((ot.finger_key >= 0) == ot.finger_active).all()))


def part3_rp1m() -> None:
    print("\n[3] RP1M warm-start prep")
    a = np.load(ACTIONS)
    d = dc.decode_actions(a)
    check("decode shape (full 45-d)", (not d.reduced) and d.hand_q["right"].shape == (158, 24),
          f"right {d.hand_q['right'].shape}")
    # joint limits
    oor = 0
    for q in d.hand_q.values():
        for j, n in enumerate(d.hand_joint_names):
            lo, hi = dc.ISAAC_JOINT_LIMITS[n]
            if q[:, j].min() < lo - 1e-4 or q[:, j].max() > hi + 1e-4:
                oor += 1
    check("all hand joints within limits", oor == 0, f"{oor} out of range")

    # reduced (39-d) path
    full = [x[0] for x in dc._FULL_ACTUATORS]
    keep = [i for i, n in enumerate(full) if n not in dc._REDUCED_EXCLUDED]
    cols = keep + [20, 21]
    a39 = np.concatenate([a[:, :22][:, cols], a[:, 22:44][:, cols], a[:, 44:45]], axis=1)
    d39 = dc.decode_actions(a39)
    check("reduced 39-d auto-detected", d39.reduced and a39.shape[1] == 39)
    excluded_rest = all(np.allclose(d39.hand_q["right"][:, dc.ISAAC_HAND_JOINTS.index(j)], 0)
                        for j in ("robot0_THJ4", "robot0_THJ0", "robot0_LFJ4"))
    check("reduced excluded joints rest at 0", excluded_rest)

    # forearm -> wrist target lands inside our keyboard span
    wt = rt.forearm_to_wrist_target(d, "right")
    from dexsim.piano import geometry as geom
    check("wrist target Y within keyboard",
          float(geom.KEY_Y.min()) <= wt[:, 1].min() and wt[:, 1].max() <= float(geom.KEY_Y.max()))

    # merge into an existing reference (synthetic joint order) keeps arm, swaps hand
    jn = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"] + list(dc.ISAAC_HAND_JOINTS)
    q_ref = np.zeros((480, 2, len(jn)), dtype=np.float32)
    q_ref[:, :, :6] = 0.5
    merged = rt.merge_hand_into_reference(q_ref, jn, d)
    arm = slice(0, 6)
    check("merge keeps arm columns", np.allclose(merged[:, :, arm], q_ref[:, :, arm]))
    check("merge overwrites hand columns", np.any(merged[:, :, 6:] != 0))
    check("merge output shape == reference", merged.shape == q_ref.shape)


def main() -> None:
    print("=" * 60 + "\nPre-training pipeline self-check (sim-free)\n" + "=" * 60)
    part1_corpus()
    part2_ot_fingering()
    part3_rp1m()
    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED: {_failures} check(s) failed")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
