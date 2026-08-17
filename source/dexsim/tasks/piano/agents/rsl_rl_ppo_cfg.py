"""rsl_rl PPO config for the bimanual piano task."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PianoPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 5000
    save_interval = 10
    experiment_name = "piano_bimanual"
    empirical_normalization = True
    # --- logging: Weights & Biases (key in ~/.netrc via `wandb login`) ---
    logger = "wandb"
    wandb_project = "dexsim-piano"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,   # was 1.0 -> 60-DoF residual flailed, slamming the
        #                       arms (NaN physics -> NaN reward -> std>=0 crash)
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,      # exploration (0 -> stuck mashing keys). Safe now
        #                          that the hand-only action + log-std fix the crash.
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,    # was 5e-4 -> gentler for stability
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,       # was 1.0 -> tighter grad clip
    )

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        # Isaac Lab >= 2.3 (rsl-rl-lib 3.x) additions, version-gated so this one
        # cfg drives BOTH stacks (.venv = 4.5/2.1, .venv-isaac51 = 5.1/2.3.2):
        #  * obs_groups is a new REQUIRED mapping of algorithm obs sets -> env
        #    obs groups; our env emits a single "policy" group used by both nets.
        #  * per-network obs normalization replaces empirical_normalization.
        #  * noise_std_type is now a first-class field (train_piano.py's kwargs
        #    hack becomes redundant on the new stack but stays harmless).
        if "obs_groups" in getattr(self, "__dataclass_fields__", {}):
            self.obs_groups = {"policy": ["policy"], "critic": ["policy"]}
            self.policy.actor_obs_normalization = True
            self.policy.critic_obs_normalization = True
            self.policy.noise_std_type = "log"
