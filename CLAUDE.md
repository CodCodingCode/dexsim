# dexsim — Claude instructions

## Embodiment: two rail-mounted Shadow Hands — NO UR10e arms

The piano task uses **two armless Shadow Hands**, each on a world-anchored
prismatic Y rail (RoboPianist-style floating hands; 25 DoF per hand = 1 rail +
24 hand joints). **The UR10e arms were removed from the piano pipeline at the
user's explicit request (2026-08-13)** — do not reintroduce arm kinematics, arm
IK (`WristPoseIK` is deleted), or arm-pose machinery into the piano task. The
UR10e asset configs remain in `source/dexsim/assets/ur10e_shadow.py` only for
the grasp/reorient tasks.

🔒 Still locked: the hand ready pose in `piano_env_cfg.py`
(`robot0_WRJ0 = 0.45` / `robot0_WRJ1 = 0.13`, fingers at 0, rail at 0) — the
wrist tilt carried over from the approved 2026-06-08 baseline. Do not edit it
unless the user explicitly asks.

`wrist_lock` (default **on**, 2026-08-13) enforces that tilt structurally: the
policy's residual on `robot0_WRJ0/WRJ1` is zeroed and the joints are stiffened,
so both hands stay upright no matter what the policy or a contact load does.
It costs 4 of the 50 action DoF. Set `wrist_lock=False` to give the wrists back.

## 🔒 LOCKED: the hand START placement (approved 2026-08-13)

The per-hand base poses in `piano_env_cfg.py` are **the approved episode-start
placement** — hands side-by-side mid-keyboard, forearms horizontal from behind,
fingertips resting on the keys (RoboPianist-style fixed "home" pose; every
episode resets to it):

**Lowered 30 mm and moved 20 mm toward the keys on 2026-08-15 at the user's
explicit request** — the previous placement parked the fingertips ~40 mm above
the key tops and behind their front edge, so no finger could touch a key (see
`scripts/tools/fit_base_placement.py`, which measures this and writes the
result). The rotations were NOT touched. Read the live values from
`piano_env_cfg.py`; as of that change:

```python
left_base_pos  = (0.2013,  0.1278, 0.7857)
right_base_pos = (0.1940, -0.1786, 0.7838)
```

Why 30 mm down / 20 mm forward specifically: at −30 mm the index tip can reach
11.7 mm below the key top (a press needs `PRESS_DEPTH` = 8 mm) while the middle
phalanges still clear the keys by 10.5 mm at the ready pose, so nothing rests on
a key at reset. Going deeper (−38 mm) cuts that clearance to 2.5 mm. The +20 mm
forward shift matters because without it the only reachable spot is the key's
front lip (x ≈ 0.538 against a 0.5375–0.6825 key) — the fingertip lands on a
sliver and slides off; with it the tip lands at x ≈ 0.587, mid-key.
Re-measure any time with `fit_base_placement.py` (dry-run by default).

Reference visual: `results/two_shadow_hands_still_v7.rrd` (single-frame scene
snapshot; older physics recordings were removed as superseded). Do NOT change
these poses without an explicit user request. To re-place interactively, use
`scripts/tools/place_scene_viser.py` (browser GUI, port 8012) — its 💾 button
writes these fields; then regenerate the still rrd via the warm render server
(`render.py rerun --out ...`, no `--physics_demo`, ~10 s).

## Kinematics outside Isaac: URDF export + PyRoki IK

PyRoki / yourdfpy / viser need **URDF**; the embodiment is authored in **USD**.
Rather than substituting some other Shadow Hand model, export the actual asset:

```bash
source env.sh && python scripts/tools/export_hand_urdf.py   # ~1 s, no Isaac boot
```

writes `assets/urdf/shadow_hand_{left,right}/` (27 links, 25 DoF = rail + 24).
Two traps this hits, both already handled — do not "fix" them back:

* the USDs declare `metersPerUnit = 0.01` but are authored in **meters**
  (forearm = 0.256 m), so the export runs at `--scale 1.0`;
* USD gives a joint frame in both bodies; URDF wants the child frame to BE the
  joint frame. The exporter keeps the USD child frame (so meshes need no
  rebaking) via `origin = (localPos0, localRot0 · conj(localRot1))` and
  `axis_child = R(localRot1) · axis_joint`.

Verify after ANY asset change — it compares URDF FK against PhysX body poses and
currently agrees to **0.000 mm / 0.000 deg** on all 27 bodies of both hands:

```bash
.venv-pyroki/bin/python scripts/smoke/check_urdf_fk.py   # needs the warm server
```

The browser IK demo (pyroki venv, no Isaac, no GPU) — both hands at the locked
base poses over the real 88-key board, fingertip gizmos, snap-a-finger-to-a-key:

```bash
.venv-pyroki/bin/python scripts/tools/piano_ik_viser.py   # http://localhost:8013
```

Reachability is real: these hands have **no arm**, so X (into the keyboard) is
unactuated and only the Y rail + finger curl can move a tip. The GUI's "worst
tip miss (mm)" is the honest signal — a gizmo dragged off that manifold simply
will not be reached. `pyroki` itself lives at `/home/ubuntu/pyroki` (editable
install in `.venv-pyroki`).

To script presses instead of dragging (`dexsim.kinematics`, pyroki venv):

```bash
.venv-pyroki/bin/python scripts/tools/press_notes.py left:index:C4 right:thumb:F4
.venv-pyroki/bin/python scripts/tools/press_notes.py --reach left   # what it CAN touch
```

Target the **`*_tip` frames**, not the `*distal` links: the distal link ORIGIN
sits ~30 mm short of the pad that touches a key (the sim's own fingertip reward
measures that origin, which is fine for the reward but wrong for IK). The
exporter emits `robot0_<f>distal_tip` at the far end of each distal mesh.

## 🚨 THE HAND HAD NO DRIVES UNTIL 2026-08-17 — how it was fixed, how to not regress

The vendored hand USDs (`nvidia_shadow_right/shadow_hand_right.usd`,
`shadow_hand_left.usd`) author **no `PhysicsDriveAPI` on any joint**. PhysX only
creates a drive for a joint that has the API at parse time, so on BOTH Isaac
stacks every gain/target write from Isaac Lab's actuators silently no-oped: the
hands were pure rag dolls (fingers gravity-flopped onto the keys, thumbs pinned
at their THJ3 limit = the "inverted thumbs", rail drifting). All of it while
`get_dof_stiffnesses()` read back the configured 45.0 — **the getter reads a
cache, not the solver; never trust it as proof of actuation.** Every dynamic
"verified press" before this date was a rag-doll artifact; the entire 250M-step
`rp1m_12k_v3` run trained on driveless hands (F1 0.0008).

**The fix lives in the slider wrappers** `assets/shadow_hand_{left,right}_slider.usda`:
`over` blocks apply `PhysicsDriveAPI:angular` to all 24 hand joints (+ `:linear`
on `railJoint`). The authored numbers are per-degree placeholders; the real
gains still come from the actuator cfgs at init (those writes bind now that
drives exist). If a hand ever goes limp again, FIRST check those overs survived
whatever touched the assets. The honest actuation check is behavioral:
`scratchpad diag — free-air tracking` (lift bases, command the ready pose, joints
must track within ~0.03 rad; driveless joints blast to their limits in <50 ms).

The `fk` query's `keys_down`/`key_travel_max` (with a real `--settle`, e.g. 240)
remains the press ground truth. Two more geometry facts measured 2026-08-17:
* **Key indexing runs low-pitch = low world Y** (key 0 at y≈−0.60, key 87 at
  +0.60; `piano.data.joint_pos` IS in key order). Key 33 sits at y=−0.14 —
  the **RIGHT** hand's lane (right FF tip y=−0.13). The left hand's rail reach
  is roughly keys 37–64. `solo_left_middle` + `one_key.mid` (key 33) was a
  wrong-hand pairing; the scripted ceiling now picks hand+finger by geometry.
* **`key_strike_vel <= 0` selects RoboPianist/RP1M sounding** (key active while
  depressed past `key_struck_frac`, hysteresis release). The default hammer gate
  samples at 20 Hz and provably misses crisp strikes that bottom out and rebound
  within one 50 ms step — use `--strike_vel 0` for anything RP1M-shaped.

## Reward: two selectable recipes (`reward_mode`)

* `"dexsim"` (default) — the composite grown in this repo: key press + fingering
  (from the **precomputed** plan) + onset + idle-hover + arm/jerk penalties.
* `"rp1m"` — a port of RP1M (Zhao et al., CoRL 2024):
  `r_OT + r_Press + 0.5·r_Collision − 5e-3·r_Energy`. Its defining feature is
  that fingering is **not** read from the precomputed plan — it is re-solved
  every step by optimal transport from the live fingertip positions
  (`dexsim/piano/ot.py`), which is what lets RP1M drop human fingering labels.
  No `r_Sustain`: our piano articulation has no pedal joint, and the user
  decided (2026-08-13) they don't want one. Do NOT add a pedal DoF or the
  sustain term unless explicitly asked — the omission is intentional, not a gap.

`fingering_method` (`"heuristic"` | `"ot"`) separately selects how the *offline*
plan (observation targets, dexsim reward) is built.

`r_Collision` reads **real PhysX contacts** (`rp1m_collision_contacts=True`): one
`ContactSensor` per left-hand body in `cfg.contact_bodies`, each filtered against
the matching right-hand bodies. Both sides of a filtered contact pair must
resolve to exactly ONE prim per env — a sensor `prim_path` matching several
bodies, or a filter pattern like `RightRobot/.*`, makes PhysX log *"did not match
the correct number of entries"* and silently collapse to one junk channel. The
"many" in Isaac Lab's one-to-many filtering is the **length of the pattern list**,
not the breadth of one pattern. Set the flag False to fall back to a sensor-free
proximity check.

Verify reward changes with:

```bash
python scripts/smoke/check_rp1m_reward.py                                  # seconds, no Isaac
python scripts/smoke/piano_env_smoke.py --reward_mode rp1m                 # end-to-end
python scripts/smoke/piano_env_smoke.py --reward_mode rp1m --force_collision  # contacts really fire
```

The `--force_collision` run is the one that matters for `r_Collision`: with no
collision happening, a dead sensor and a working one look identical (both report
zero). It parks the hands inside each other and asserts the term actually trips.

## Rendering & geometry measurement: ALWAYS use the warm render server

Every cold render/diagnostic script (`render_scene.py`, `render_rollout.py`,
`diag_*.py`, `verify_palm.py`) boots the **entire** Isaac Sim app (~30 s, longer
under GPU contention) and rebuilds the scene from scratch on every run. A warm
server caches that boot + built scene so each render/measurement takes seconds.

**For ANY rendering, video, or geometry/measurement task, use the warm server —
do NOT cold-boot a render/diag script, and do NOT write a new one-shot
`AppLauncher` script for it.**

1. Check if the server is up: `logs/render_jobs/server.ready` exists AND its `pid` is alive.
2. If not up, boot it ONCE (wait for `READY` in the log, ~30 s):
   ```bash
   source env.sh
   python scripts/render/render_server.py --headless > logs/render_server.log 2>&1 &
   ```
3. Submit jobs with the thin client (returns in seconds, no Isaac boot):
   ```bash
   python scripts/render/render.py scene   --eye 2.2,-1.5,1.8 --target 0.45,0,0.78 --spp 160 --out logs/x.png
   python scripts/render/render.py rollout --rollout logs/rollout.npz --out results/v.mp4 --spp 96
   python scripts/render/render.py query   --kind layout|orient|palm|bodies [--rollout r.npz] --out logs/q.json
   ```
   - `query` kinds subsume the old diagnostics: `layout`←diag_layout, `orient`←diag_hand_orient,
     `palm`←verify_palm, `bodies`←diag_arm_links (pass `--left_joints`/`--right_joints`/`--bodies`).
   - Lower `--spp` for faster preview stills; raise it for final quality.
4. Leave the server running for iteration; `python scripts/render/render.py shutdown` to free its GPU memory.

Shared scene builders are in `source/dexsim/render/studio.py` (single source of
truth → a warm render matches a cold render). The cold scripts still work
standalone, but the server is the default path. If a render need isn't covered by
an existing job type, ADD a handler to `render_server.py` rather than reintroducing
a cold-boot script.

## Two Isaac stacks side-by-side (migration 2026-08-15)

* **OLD (default)**: `source env.sh` → `.venv` (py3.10, Isaac Sim 4.5, Isaac Lab
  v2.1 in `IsaacLab/`). The render server + all existing tooling run here.
* **NEW**: `source env51.sh` → `.venv-isaac51` (py3.11, Isaac Sim 5.1.0, Isaac
  Lab v2.3.2 in `IsaacLab51/`, rsl-rl-lib 3.1.2). `piano_env_smoke.py` passes
  UNCHANGED on it; `PianoPPORunnerCfg.__post_init__` version-gates the 2.3-only
  fields (obs_groups, per-net obs normalization) so ONE cfg drives both stacks.
  Source exactly ONE env file per shell.
* Livestream semantics differ: 4.5: `--livestream 2`=WebRTC; 5.1: `1`=WebRTC
  public (advertises `$PUBLIC_IP`), `2`=private/LAN. `live_scene.py` picks by
  isaaclab version. 5.1 media rides FIXED UDP 47998 (+TCP 49100 signaling);
  matching Mac client: **WebRTC Streaming Client 1.1.5** (arm64 dmg exists).
* Gotchas hit during install: `flatdict==4.0.1` needs `setuptools<81` +
  `--no-build-isolation`; keep isaacsim pins `packaging==23.0`,
  `psutil==5.9.8`, `typing_extensions==4.12.2` when adding packages.

## General

- `source env.sh` before anything (venv + Omniverse EULA + the staged Vulkan driver + PYTHONPATH).
- Isaac-only embodiment work: UR10e + Shadow Hand. No MuJoCo/RoboPianist routing.
- Heavy `isaaclab` imports (and `dexsim.render.studio`) must come AFTER `AppLauncher(...).app`.
- `logs/` is gitignored (the render job-queue lives in `logs/render_jobs/`).
