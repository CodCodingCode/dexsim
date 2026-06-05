"""Off-policy SAC training on the SLIDER piano env (skrl).

PPO collapses on this task (peaks ~0.17 then degrades) -- the reason RoboPianist used
off-policy DroQ/SAC. skrl ships SAC + an Isaac Lab vec-env wrapper, so we can run
off-policy here: sample-efficient (replay buffer), no on-policy collapse.

  python scripts/train/train_slider_sac.py --headless --num_envs 64 \
      --songs_npz data/multisong/repertoire40.npz --max_songs 24 --timesteps 300000
"""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--timesteps", type=int, default=300000)
parser.add_argument("--songs_npz", default=None)
parser.add_argument("--max_songs", type=int, default=0)
parser.add_argument("--midi", default=None)
parser.add_argument("--tag", default="sac")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--mem_size", type=int, default=12000, help="replay capacity per env")
parser.add_argument("--goal_lookahead", type=int, default=5)
parser.add_argument("--gradient_steps", type=int, default=1)
parser.add_argument("--false_press_weight", type=float, default=1.0)
parser.add_argument("--key_press_weight", type=float, default=4.0)
parser.add_argument("--target_entropy", type=float, default=None)
parser.add_argument("--slider_stiffness", type=float, default=0.0)
parser.add_argument("--slider_residual", type=float, default=0.05)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
import torch.nn as nn
import gymnasium as gym

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.sac import SAC, SAC_CFG
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.preprocessors.torch import RunningStandardScaler
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import dexsim.tasks  # noqa: F401
from dexsim.tasks.piano import PianoEnvCfg

cfg = PianoEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.use_slider = True
cfg.slider_stiffness = args.slider_stiffness
cfg.slider_residual = args.slider_residual
# trim the obs (goal_lookahead*88 dominates the 1220-d obs) -> ~2x faster off-policy
# updates -> ~2x more env-steps per wall-hour. The slider positions the hand, so the
# policy needs less future lookahead; drop the analytic goal-SDF (88) too.
cfg.goal_lookahead = args.goal_lookahead
cfg.obs_goal_sdf = False
cfg.__post_init__()
if args.midi:
    cfg.midi_path = args.midi
if args.songs_npz:
    cfg.songs_npz = args.songs_npz
    cfg.max_songs = args.max_songs
cfg.__post_init__()

# F1-ALIGNED reward: the slider already POSITIONS the hand, so the positioning shaping
# (fingering / arm-base reward) is "free" -> the policy harvests it WITHOUT pressing
# (high reward, low F1). Zero the positioning shaping; reward only correct key presses
# (key_press) minus wrong ones (false_press), plus onset timing. Now reward == F1.
cfg.fingering_weight = 0.0
cfg.arm_base_weight = 0.0
cfg.key_press_weight = args.key_press_weight
cfg.false_press_weight = args.false_press_weight
cfg.onset_weight = 2.0
cfg.energy_weight = 0.0

env = gym.make("Dexsim-Piano-Bimanual-v0", cfg=cfg, render_mode=None)
env = SkrlVecEnvWrapper(env)
device = env.device
obs_space, act_space = env.observation_space, env.action_space
print(f"[sac] obs={obs_space} act={act_space} num_envs={env.num_envs}")


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space=observation_space,
                       action_space=action_space, device=device)
        # clip_actions=False: the env's action_space is unbounded; PianoEnv already
        # clamps actions to [-1,1] in _pre_physics_step, so skrl needn't (and can't).
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True,
                               min_log_std=-5.0, max_log_std=2.0)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role=""):
        obs = inputs.get("observations")        # skrl 2.1.0 key
        return torch.tanh(self.net(obs)), {"log_std": self.log_std_parameter}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space=observation_space,
                       action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 1))

    def compute(self, inputs, role=""):
        obs = inputs.get("observations")
        act = inputs.get("taken_actions")
        return self.net(torch.cat([obs, act], dim=-1)), {}


models = {
    "policy": Policy(obs_space, act_space, device),
    "critic_1": Critic(obs_space, act_space, device),
    "critic_2": Critic(obs_space, act_space, device),
    "target_critic_1": Critic(obs_space, act_space, device),
    "target_critic_2": Critic(obs_space, act_space, device),
}

memory = RandomMemory(memory_size=args.mem_size, num_envs=env.num_envs, device=device)

sac_cfg = SAC_CFG()
sac_cfg.gradient_steps = args.gradient_steps
sac_cfg.batch_size = 4096
sac_cfg.discount_factor = 0.99
sac_cfg.polyak = 0.005
sac_cfg.learning_rate = args.lr
sac_cfg.random_timesteps = 1000
sac_cfg.learning_starts = 1000
sac_cfg.grad_norm_clip = 1.0
sac_cfg.learn_entropy = True
if args.target_entropy is not None:
    sac_cfg.target_entropy = args.target_entropy
sac_cfg.observation_preprocessor = RunningStandardScaler
sac_cfg.observation_preprocessor_kwargs = {"size": obs_space, "device": device}
sac_cfg.experiment.directory = "logs/skrl_sac"
sac_cfg.experiment.experiment_name = args.tag
sac_cfg.experiment.write_interval = 200
sac_cfg.experiment.checkpoint_interval = 5000

agent = SAC(models=models, memory=memory, cfg=sac_cfg,
            observation_space=obs_space, action_space=act_space, device=device)

trainer = SequentialTrainer(cfg={"timesteps": args.timesteps, "headless": True},
                            env=env, agents=agent)
print(f"[sac] training {args.timesteps} timesteps ...")
trainer.train()
print("[sac] done")
env.close()
app.close()
