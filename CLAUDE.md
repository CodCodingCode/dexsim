# dexsim — Claude instructions

## 🔒 LOCKED: the constant static arm+hand pose — DO NOT EDIT

`left_ready_pose` / `right_ready_pose` in
`source/dexsim/tasks/piano/piano_env_cfg.py` are the **constant static config the
arm + hands must ALWAYS have** (arm reach joints, `wrist_3_joint = 3.14159` (π), and
the hand wrist tilt `robot0_WRJ0 = 0.45` / `robot0_WRJ1 = 0.13`).
**Do NOT edit these poses** — not the wrist flip, not the tilt, not the arm joints.
They are deliberately fixed; treat them as frozen unless the user explicitly says
otherwise in a new request.

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

## General

- `source env.sh` before anything (venv + Omniverse EULA + the staged Vulkan driver + PYTHONPATH).
- Isaac-only embodiment work: UR10e + Shadow Hand. No MuJoCo/RoboPianist routing.
- Heavy `isaaclab` imports (and `dexsim.render.studio`) must come AFTER `AppLauncher(...).app`.
- `logs/` is gitignored (the render job-queue lives in `logs/render_jobs/`).
