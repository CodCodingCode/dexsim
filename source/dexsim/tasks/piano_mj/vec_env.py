"""CPU-vectorized PianoMjEnv + the rsl_rl (>=5.x) VecEnv wrapper.

One compiled ``MjModel`` and one :class:`SongBank` are shared read-only across
``num_envs`` :class:`PianoMjEnv` instances (each owns its ``MjData``); steps
fan out over a thread pool (``mj_step`` releases the GIL). Multi-song bundles
assign songs round-robin, matching the Isaac env.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from dexsim.mjcf import compile_scene
from .piano_mj_env import PianoMjEnv
from .song_bank import SongBank


class PianoMjVecEnv:
    """Synchronous numpy vector env with auto-reset."""

    def __init__(self, cfg, num_envs: int, threads: int | None = None):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.bank = SongBank(cfg)
        self.model = compile_scene(cfg)
        self.envs = [
            PianoMjEnv(cfg, model=self.model, bank=self.bank,
                       song_id=i % self.bank.num_songs)
            for i in range(self.num_envs)
        ]
        self.num_actions = cfg.action_space
        self.num_obs = cfg.observation_space
        n_workers = threads if threads is not None else min(self.num_envs, 16)
        self._pool = (ThreadPoolExecutor(max_workers=n_workers)
                      if n_workers > 1 else None)
        self._last_logs: dict[str, float] = {}

    def reset(self) -> np.ndarray:
        return np.stack([e.reset() for e in self.envs]).astype(np.float32)

    def _step_one(self, env: PianoMjEnv, action: np.ndarray):
        obs, reward, terminated, truncated, logs = env.step(action)
        done = terminated or truncated
        if done:
            obs = env.reset()
        return obs, reward, done, truncated, logs

    def step(self, actions: np.ndarray):
        """actions (E, A) -> (obs (E,O), rew (E,), done (E,), time_out (E,), logs)."""
        acts = np.asarray(actions, dtype=np.float64)
        if self._pool is not None:
            results = list(self._pool.map(self._step_one, self.envs, acts))
        else:
            results = [self._step_one(e, a) for e, a in zip(self.envs, acts)]
        obs = np.stack([r[0] for r in results]).astype(np.float32)
        rew = np.array([r[1] for r in results], dtype=np.float32)
        done = np.array([r[2] for r in results], dtype=bool)
        timeout = np.array([r[3] for r in results], dtype=bool)
        # mean logs over envs; play/* accuracy metrics only over envs that had
        # a goal this step (mirrors the Isaac env's has_goal masking)
        keys = results[0][4].keys()
        logs = {}
        hg = np.array([r[4].get("play/has_goal", 1.0) for r in results])
        for k in keys:
            v = np.array([r[4][k] for r in results], dtype=np.float64)
            if k.startswith("play/") and k != "play/has_goal" and hg.sum() > 0:
                logs[k] = float(v[hg > 0].mean())
            else:
                logs[k] = float(v.mean())
        logs.pop("play/has_goal", None)
        self._last_logs = logs
        return obs, rew, done, timeout, logs

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False)


def make_rsl_rl_env(venv: PianoMjVecEnv, device: str = "cpu"):
    """Wrap a PianoMjVecEnv for rsl_rl >= 5.x (TensorDict obs groups)."""
    import torch
    from tensordict import TensorDict
    from rsl_rl.env import VecEnv

    class RslRlPianoMjVecEnv(VecEnv):
        def __init__(self):
            self.venv = venv
            self.num_envs = venv.num_envs
            self.num_actions = venv.num_actions
            self.max_episode_length = venv.envs[0].max_episode_length
            self.episode_length_buf = torch.zeros(
                self.num_envs, dtype=torch.long, device=device)
            self.device = device
            self.cfg = venv.cfg.to_dict()
            self._obs = venv.reset()

        def _obs_td(self):
            obs = torch.as_tensor(self._obs, dtype=torch.float32, device=self.device)
            return TensorDict({"policy": obs}, batch_size=[self.num_envs],
                              device=self.device)

        def get_observations(self):
            return self._obs_td()

        def reset(self):
            self._obs = self.venv.reset()
            self.episode_length_buf.zero_()
            return self._obs_td()

        def step(self, actions):
            obs, rew, done, timeout, logs = self.venv.step(
                actions.detach().cpu().numpy())
            self._obs = obs
            self.episode_length_buf += 1
            done_t = torch.as_tensor(done, device=self.device)
            self.episode_length_buf[done_t] = 0
            extras = {
                "time_outs": torch.as_tensor(timeout, device=self.device),
                "log": logs,
            }
            return (self._obs_td(),
                    torch.as_tensor(rew, dtype=torch.float32, device=self.device),
                    done_t,
                    extras)

    return RslRlPianoMjVecEnv()
