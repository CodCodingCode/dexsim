"""v14 per-finger hold check: strike with the right middle finger, lift with hold on/off."""
import sys; sys.path.insert(0, "source")
import numpy as np, mujoco
from dexsim.tasks.piano_mj import PianoMjEnv, PianoMjEnvCfg
cfg = PianoMjEnvCfg(); cfg.midi_path = "results/nettspend - we not like you (1).mid"; cfg.fingering_method = "seq"; cfg.random_song_start = False; cfg.__post_init__()
env = PianoMjEnv(cfg); print(f"[dims] obs {cfg.observation_space} (expect 1272) actions {cfg.action_space} (expect 52)")
m, d = env.model, env.data
aid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
mf0, mf2, wr0 = aid["R_robot0_A_MFJ0"], aid["R_robot0_A_MFJ2"], aid["R_robot0_A_WRJ0"]
def run(steps, press, hold):
    env.hold_cmd[:] = False; env.hold_cmd[7] = hold
    for _ in range(steps):
        d.ctrl[mf2] = env.ready_ctrl[mf2] + (0.80 if press else 0.0); d.ctrl[mf0] = env.ready_ctrl[mf0] + (0.40 if press else 0.0); d.ctrl[wr0] = env.ready_ctrl[wr0] + (0.20 if press else 0.0)
        for _ in range(cfg.decimation): mujoco.mj_step(m, d); env._update_strike_latch()
env.reset(); run(30, True, False); k = int(np.argmax(env.key_sounding))
print(f"[strike] key {k} sounding, owner R.mf -> {env.hold_owner[7]}")
run(20, False, True); a = bool(env.key_sounding[k]); run(5, False, False); b = bool(env.key_sounding[k])
print(f"[hold ON, finger lifted 1 s] still ringing: {a} (expect True)   [hold OFF 0.25 s] ringing: {b} (expect False)")
assert env.hold_owner[7] == k and a and not b; print("HOLD OK")
