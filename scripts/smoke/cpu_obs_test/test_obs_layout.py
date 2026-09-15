import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "source"))
import isaaclab  # stub
from dexsim.tasks.piano.piano_env_cfg import PianoEnvCfg, NUM_KEYS, NUM_FINGERS
from dexsim.tasks.piano.piano_env import PianoEnv

# ---------- 1. config sizing ----------
cfg = PianoEnvCfg()
ids = cfg.obs_key_indices()
assert ids == list(range(19, 27)) + list(range(63, 71)), ids
assert cfg.n_obs_keys() == 16
assert cfg.observation_space == 368, cfg.observation_space
assert cfg.critic_extra_dim() == 11
assert cfg.state_space == 379, cfg.state_space
print(f"default: K=16 obs={cfg.observation_space} critic={cfg.state_space}")

cfg.fold_to_reach = False; cfg.refresh_spaces()
assert cfg.n_obs_keys() == 88 and cfg.observation_space == 1304 and cfg.state_space == 1315
print(f"no_fold: K=88 obs={cfg.observation_space} critic={cfg.state_space}")
cfg.fold_to_reach = True; cfg.goal_lookahead = 4; cfg.refresh_spaces()
assert cfg.observation_space == 368 - 6 * 16
print(f"lookahead=4: obs={cfg.observation_space}")
cfg.goal_lookahead = 10; cfg.critic_obs_sounding = True; cfg.refresh_spaces()
assert cfg.state_space == 368 + 27
cfg.critic_obs_sounding = False; cfg.critic_obs = False; cfg.refresh_spaces()
assert cfg.state_space == 0
cfg.critic_obs = True; cfg.refresh_spaces()
print("cfg sizing OK")

# ---------- 2. real _get_observations on fake sim data ----------
E, DOF, L = 7, 25, cfg.goal_lookahead
env = object.__new__(PianoEnv)
env.cfg = cfg; env.num_envs = E; env.device = "cpu"; env.per_arm_dof = DOF
env.scene = isaaclab.Stub(env_origins=torch.randn(E, 3))
def robot(): 
    r = isaaclab.Stub(); r.data = isaaclab.Stub(joint_pos=torch.randn(E, DOF), joint_vel=torch.randn(E, DOF)); return r
env.left_robot, env.right_robot = robot(), robot()
env.piano = isaaclab.Stub(); env.piano.data = isaaclab.Stub(joint_pos=torch.randn(E, NUM_KEYS), joint_vel=torch.randn(E, NUM_KEYS))
env.key_sounding = torch.rand(E, NUM_KEYS) > 0.5
env.obs_key_ids = torch.tensor(cfg.obs_key_indices())
# song goals: 2 songs, Tmax+L steps, 88 keys
env.goal_padded = (torch.rand(2, 40 + L, NUM_KEYS) > 0.8).float()
env.song_id = torch.randint(0, 2, (E,)); env.song_step = torch.randint(0, 40, (E,))
tips = torch.randn(E, NUM_FINGERS, 3); press = torch.randn(E, NUM_FINGERS, 3)
env._fingertips_world = lambda: tips
env._key_top_world = lambda: torch.zeros(E, NUM_KEYS, 3)
env._finger_targets_world = lambda kt: (None, press, None)
tipf = torch.rand(E, NUM_FINGERS); col = torch.rand(E) > 0.5
env._tip_forces = lambda: tipf
env._hands_collided = lambda t: col

out = env._get_observations()
pol, cri = out["policy"], out["critic"]
assert pol.shape == (E, cfg.observation_space), pol.shape
assert cri.shape == (E, cfg.state_space), cri.shape
assert torch.equal(cri[:, :pol.shape[1]], pol)
kid = env.obs_key_ids; K = 16
o = 0
def chunk(n):
    global o; c = pol[:, o:o + n]; o += n; return c
assert torch.equal(chunk(DOF), env.left_robot.data.joint_pos)
assert torch.equal(chunk(DOF), env.left_robot.data.joint_vel)
assert torch.equal(chunk(DOF), env.right_robot.data.joint_pos)
assert torch.equal(chunk(DOF), env.right_robot.data.joint_vel)
assert torch.allclose(chunk(30), (tips - env.scene.env_origins.unsqueeze(1)).reshape(E, -1))
assert torch.equal(chunk(K), env.piano.data.joint_pos[:, kid])
assert torch.equal(chunk(K), env.piano.data.joint_vel[:, kid])
assert torch.equal(chunk(K).bool(), env.key_sounding[:, kid])
exp_goal = env._goal_lookahead()[:, :, kid].reshape(E, -1)
assert exp_goal.shape == (E, L * K) and torch.equal(chunk(L * K), exp_goal)
assert torch.allclose(chunk(30), (press - env.scene.env_origins.unsqueeze(1)).reshape(E, -1))
assert o == pol.shape[1], (o, pol.shape)
ex = cri[:, pol.shape[1]:]
assert torch.equal(ex[:, :10], tipf) and torch.equal(ex[:, 10].bool(), col)
print(f"obs assembly OK: policy {tuple(pol.shape)} critic {tuple(cri.shape)}, all chunks verified")

# goal SDF path + 88-key fallback still assemble
cfg.obs_goal_sdf = True; cfg.fold_to_reach = False; cfg.refresh_spaces()
env.obs_key_ids = torch.tensor(cfg.obs_key_indices())
out = env._get_observations()
assert out["policy"].shape == (E, cfg.observation_space) == (E, 1304 + 88)
assert out["critic"].shape == (E, cfg.state_space)
print(f"sdf + no_fold OK: policy {tuple(out['policy'].shape)}")
print("ALL PASSED")
