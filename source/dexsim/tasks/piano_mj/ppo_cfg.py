"""rsl_rl (>=5.x) PPO config for the MuJoCo piano task.

Hyper-parameters mirror the Isaac ``PianoPPORunnerCfg``
(source/dexsim/tasks/piano/agents/rsl_rl_ppo_cfg.py): same network sizes,
same PPO constants, same init noise. Expressed in the rsl_rl 5.x dict format
(obs_groups / actor / critic / algorithm with class_name resolution).
"""

from __future__ import annotations

import copy

_BASE: dict = {
    "num_steps_per_env": 32,
    "save_interval": 50,
    "logger": "tensorboard",
    "wandb_project": "dexsim-piano-mj",
    "run_name": "",
    "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
    "actor": {
        "class_name": "MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": True,     # == Isaac empirical_normalization
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 0.5,           # 1.0 flailed the 60-DoF residual (Isaac note)
        },
    },
    "critic": {
        "class_name": "MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": True,
    },
    "algorithm": {
        "class_name": "PPO",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.006,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 3.0e-4,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 0.5,
        "rnd_cfg": None,
        "symmetry_cfg": None,
    },
}


def piano_ppo_cfg(**overrides) -> dict:
    """Deep copy of the base cfg with optional flat overrides.

    Recognized override keys: any top-level key, plus the shortcuts
    ``learning_rate``, ``entropy_coef``, ``desired_kl`` (-> algorithm),
    ``init_std`` (-> actor distribution), ``obs_normalization`` (-> both nets).
    ``None`` values are ignored.
    """
    cfg = copy.deepcopy(_BASE)
    for k, v in overrides.items():
        if v is None:
            continue
        if k in ("learning_rate", "entropy_coef", "desired_kl"):
            cfg["algorithm"][k] = v
        elif k == "init_std":
            cfg["actor"]["distribution_cfg"]["init_std"] = v
        elif k == "obs_normalization":
            cfg["actor"]["obs_normalization"] = v
            cfg["critic"]["obs_normalization"] = v
        else:
            cfg[k] = v
    return cfg
