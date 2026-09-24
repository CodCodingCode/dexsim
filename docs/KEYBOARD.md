# MacBook keyboard scene + typing task (2026-09-23)

A computer-keyboard twin of the piano scene: the model, a bimanual scene, a
scripted typist, and an RL typing env (`dexsim.tasks.keyboard_mj`) trained
with the same PPO recipe as the piano.

```
source env.sh
python scripts/mj/keyboard_demo.py --text "hello world"                 # scripted typist, keylog check
python scripts/mj/keyboard_demo.py --text "Hi, Nathan!" --video logs/kb.mp4
python scripts/mj/keyboard_demo.py --xml                                # assets/mj/keyboard_scene.xml
python -m mujoco.viewer --mjcf=assets/mj/keyboard_scene.xml

# RL: train to type (Mac: --device cpu --workers 8; A100 box: --workers 28)
python scripts/mj/train_keyboard_mj.py --num_envs 64 --workers 8 --device cpu \
    --text "hello world" --right_only --max_iterations 2000 --tag hello_r
python scripts/mj/play_keyboard_mj.py --checkpoint logs/keyboard_mj/<run>/model_final.pt \
    --text "hello world" --right_only --video logs/typing.mp4
```

## Typing env (`source/dexsim/tasks/keyboard_mj/`)

- **Goal**: a text -> keystroke sequence (`keystrokes()`), cursor advances on
  each correct registration. Default: random sentence from `DEFAULT_CORPUS`
  (lowercase, no shift needed); `--text` fixes one. `--right_only` freezes
  the left hand (first curriculum).
- **Actions (46)**: per hand 20 hand actuators (residual ±0.8 rad around
  pose G, as the piano) + gantry x/y (absolute, ±0.15 m) + z (±0.04 m about
  the 2 cm hover).
- **Obs (314)**: hand joint pos/vel incl. gantry, 10 fingertips + 2 palms
  relative to the target cap, next 3 keystrokes (cap xy relative to each
  palm, shift flag, touch-typing finger one-hot), shift held, all 78 key
  depressions, previous action.
- **Reward**: +1 per correct keystroke, -0.5 per stray, -0.02/step per wrong
  key held, dense reach (assigned fingertip -> cap, 0.05 max) and press
  (target depression, 0.05 max), -0.005/step time penalty (type faster),
  +5 on finishing (terminal). Registration = rising edge past 0.5 mm with
  0.2 mm release hysteresis, checked every physics substep.
- **Metrics**: `play/accuracy` (correct / all registrations),
  `play/done_frac`, `play/cps` (chars per second at episode end).
- Control 50 Hz (sim 500 Hz x 10). ~1.8 ms/step single env on the M-series
  Mac; 32 envs x 8 workers train at ~0.4 s/iteration.
- Smoke run (40 iters, right hand, "hello"): mean reward -154 -> -96, i.e.
  the policy first learns to stop mashing keys (random actions register ~1.7
  strays/step). Expect the same press-discovery plateau as the piano; the
  piano's answer was the recall-gated false-press anneal -- port it
  (`anneal_false_press`) if accuracy stays near 0 after ~500 iterations.

## Key legends

Caps carry printed legends (Apple style: letters, shifted symbol above the
base on number/punctuation keys, word labels on modifiers). They are PNG
textures generated with PIL into `assets/mj/keycaps/` on first build
(`keyboard.render_legends`), one material per key. MuJoCo maps a 2D texture's
u axis onto a box's local X, so the typist-view image is rotated 90° before
saving. `add_keyboard(..., legends=False)` skips them.

## Files

- `source/dexsim/mjcf/keyboard.py` -- procedural 16-inch MacBook Pro (M3 Max)
  body + Apple Magic Keyboard (US ANSI, 78 keys). 19.05 mm x 18.6 mm pitch,
  16.5 x 16 mm caps, 1.0 mm scissor travel, registers at 0.5 mm, ~55 gf.
  Keys are passive slide joints (`joint_<name>`, negative = pressed) with a
  cap-centre site `key_site_<name>`; same conventions as the piano keys.
  Also: `text_to_keys`, `key_to_char`, and the touch-typing finger map.
- `source/dexsim/mjcf/keyboard_scene.py` -- laptop on the desk + two hands
  on 3-axis gantries (`{L,R}_gantry_{x,y,z}` position servos). Uses the 🔒
  pose G finger/wrist pose from `PianoMjEnvCfg` unchanged; the mounts are
  calibrated so the middle fingertips hover 2 cm over `d` and `k`, which
  puts the long fingers on a/s/d/f and j/k/l/; automatically.
- `scripts/mj/keyboard_demo.py` -- scripted typist: touch-typing finger,
  lift, curl the striking finger / extend the idle ones, XY servo on the
  fingertip site, Z lower until the key registers, home. Prints what the
  keyboard registered vs the requested text.

## Things learned building it (don't re-learn)

- **Curling a finger from pose G LOWERS its tip** (+0.2 rad MCP+PIP ->
  -14 mm, 12 mm toward the typist); extending raises it (-0.35/-0.40 ->
  +49 mm). The striking finger curls, the idle fingers straighten.
- **Change posture and travel with the hand raised (3 cm).** A finger curling
  at the 2 cm hover sweeps the row in front of the target (h -> n, m first).
- **The E3M5 forearms collide with each other** (13 cm cylinders, palms only
  10 cm apart on the home row) and with the desk. They are visual-only in
  this scene and the laptop sits at the desk's front edge.
- **Never park a hand over the other one.** The right thumb on the space
  bar's centre put the right palm above the left knuckles; the next left
  stroke was blocked and mashed a/s/d/f. Every stroke now returns home, and
  wide keys are hit at the end nearest the finger (`Typist.press_point`).
- Key debounce: register at 0.5 mm, release above 0.2 mm, or a held key
  chatters (`lllllll`).

Result: `hello world` and `Hi, Nathan! M3 Max: 2026?` both register exactly,
zero stray keys, ~1.8 s of sim per character.
