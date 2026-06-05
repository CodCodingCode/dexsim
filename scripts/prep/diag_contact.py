"""WHERE is the mash coming from? In a zero-residual arm_ik_follow rollout, report
each fingertip's height and the palm height relative to the keyboard top, plus how
many keys are depressed. Pinpoints whether ~8 keys sound from fingertips, the palm,
or the arm failing to hold hover -- so we stop guessing.

  python scripts/prep/diag_contact.py --headless
"""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--hover", type=float, default=None)
parser.add_argument("--retract", type=float, default=None)
parser.add_argument("--curl", type=float, default=None)
parser.add_argument("--key_stiffness", type=float, default=None)
parser.add_argument("--single_finger", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
import dexsim.tasks  # noqa
from dexsim.tasks.piano import PianoEnvCfg
from dexsim.tasks.piano.piano_env import PianoEnv
from dexsim.piano import FINGERTIP_BODIES, NUM_KEYS


def main():
    cfg = PianoEnvCfg()
    cfg.scene.num_envs = 4
    cfg.midi_path = "data/midi/easy.mid"
    cfg.arm_ik_follow = True
    cfg.freeze_arms = False
    if args.hover is not None:
        cfg.arm_ik_hover = args.hover
    if args.retract is not None:
        cfg.idle_hand_retract = args.retract
    if args.curl is not None:
        cfg.idle_finger_curl = args.curl
    if args.key_stiffness is not None:
        cfg.piano_cfg.actuators["keys"].stiffness = args.key_stiffness
    if args.single_finger:
        cfg.single_finger = True
        cfg.single_press_z = -0.015
        cfg.single_curl = 3.0
    env = PianoEnv(cfg, render_mode=None)
    obs, _ = env.reset()
    a = torch.zeros(env.num_envs, cfg.action_space, device=env.device)

    fnames = ["L_th", "L_ff", "L_mf", "L_rf", "L_lf", "R_th", "R_ff", "R_mf", "R_rf", "R_lf"]
    tip_h = torch.zeros(10, device=env.device); palm_h = torch.zeros(2, device=env.device)
    keys_snd = 0.0; n = 0
    snd_series = []
    for step in range(120):
        env.step(a)
        if step in (0, 1, 5, 30, 60, 119):
            ks = (env._key_pressed_fraction() > 1e-6)
            idx = torch.nonzero(ks[0]).flatten().tolist()
            snd_series.append((step, float(ks.float().sum(-1).mean()), idx[:12]))
        kt = env._key_top_world()[..., 2].amax(dim=-1)          # (E,) keyboard top z
        tips = env._fingertips_world()[..., 2]                   # (E,10)
        palms = env._palms_world()[..., 2]                       # (E,2)
        tip_h += (tips - kt.unsqueeze(-1)).mean(0)               # mean over envs, per finger
        palm_h += (palms - kt.unsqueeze(-1)).mean(0)
        keys_snd += float((env._key_pressed_fraction() > 1e-6).float().sum(-1).mean())
        n += 1
    tip_h /= n; palm_h /= n; keys_snd /= n
    print("[contact] mean height ABOVE keyboard top (mm); negative => pressing INTO keys")
    print("[contact] PALM  L=%.0f  R=%.0f" % (palm_h[0]*1000, palm_h[1]*1000))
    for i, nm in enumerate(fnames):
        print("[contact] tip %-5s %.0f mm" % (nm, tip_h[i]*1000))
    print("[contact] keys_sounding (avg) = %.2f  (easy.mid has ~1-3 active)" % keys_snd)
    for (st, ks, idx) in snd_series:
        print("[contact] step %3d  keys_sounding=%.1f  sounding_keys(env0)=%s" % (st, ks, idx))
    # also report the raw key angle of the worst keys vs the sound angle
    ka = env.piano.data.joint_pos[0]                         # (88,)
    worst = torch.argsort(ka)[:6].tolist()                   # most-depressed keys
    print("[contact] most-depressed keys (env0): " +
          ", ".join("k%d=%.4frad(frac%.2f)" % (k, ka[k].item(), (ka[k]/(-0.012)).item()) for k in worst))
    env.close(); app.close()


main()
