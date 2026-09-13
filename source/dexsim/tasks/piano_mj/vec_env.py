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
from .piano_mj_env import PianoMjEnv, measure_finger_offsets
from .song_bank import SongBank


class PianoMjVecEnv:
    """Synchronous numpy vector env with auto-reset."""

    def __init__(self, cfg, num_envs: int, threads: int | None = None):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.model = compile_scene(cfg)
        self.bank = SongBank(cfg, finger_offsets=measure_finger_offsets(self.model))
        self.envs = [
            PianoMjEnv(cfg, model=self.model, bank=self.bank,
                       song_id=i % self.bank.num_songs, env_index=i)
            for i in range(self.num_envs)
        ]
        self.num_actions = cfg.action_space
        self.num_obs = cfg.observation_space
        self.max_episode_length = self.envs[0].max_episode_length
        self.num_priv_obs = cfg.critic_extra_dim()   # critic-only extras (0 = symmetric)
        self._priv = np.zeros((self.num_envs, self.num_priv_obs), dtype=np.float32)
        n_workers = threads if threads is not None else min(self.num_envs, 16)
        self._pool = (ThreadPoolExecutor(max_workers=n_workers)
                      if n_workers > 1 else None)
        self._last_logs: dict[str, float] = {}

    def reset(self) -> np.ndarray:
        obs = np.stack([e.reset() for e in self.envs]).astype(np.float32)
        if self.num_priv_obs:
            self._priv = np.stack([e.critic_extras() for e in self.envs])
        return obs

    def critic_priv(self) -> np.ndarray:
        """(E, num_priv_obs) privileged critic features for the CURRENT obs
        (refreshed by reset()/step())."""
        return self._priv

    def _step_one(self, env: PianoMjEnv, action: np.ndarray):
        obs, reward, terminated, truncated, logs = env.step(action)
        done = terminated or truncated
        if done:
            obs = env.reset()
        priv = env.critic_extras() if self.num_priv_obs else None
        return obs, reward, done, truncated, logs, priv

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
        if self.num_priv_obs:
            self._priv = np.stack([r[5] for r in results]).astype(np.float32)
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


def _reduce_logs(per_env: list[dict]) -> dict:
    """Mean logs over envs; play/* accuracy metrics only over envs that had a
    goal this step (mirrors the Isaac env's has_goal masking)."""
    # union of keys: some diagnostics (finger/online_*) exist only on steps
    # where an env had goal keys, so average each key over the envs that
    # reported it
    keys = list(dict.fromkeys(k for r in per_env for k in r))
    hg = np.array([r.get("play/has_goal", 1.0) for r in per_env])
    logs = {}
    for k in keys:
        have = np.array([k in r for r in per_env])
        v = np.array([r[k] for r in per_env if k in r], dtype=np.float64)
        if k.startswith("play/") and k != "play/has_goal" and hg[have].sum() > 0:
            logs[k] = float(v[hg[have] > 0].mean())
        else:
            logs[k] = float(v.mean())
    logs.pop("play/has_goal", None)
    return logs


def _worker_main(conn, cfg, env_ids: list[int]):
    """One process owning a shard of PianoMjEnvs (its own MjModel + SongBank).
    Protocol: ("reset",) -> (obs, priv); ("step", acts) -> (obs, rew, done,
    timeout, per_env_logs, priv); ("close",) -> exit."""
    try:
        model = compile_scene(cfg)
        bank = SongBank(cfg, finger_offsets=measure_finger_offsets(model))
        envs = [PianoMjEnv(cfg, model=model, bank=bank, song_id=i % bank.num_songs,
                           env_index=i)
                for i in env_ids]
        n_priv = cfg.critic_extra_dim()
        while True:
            msg = conn.recv()
            if msg[0] == "info":
                conn.send({"max_episode_length": envs[0].max_episode_length})
            elif msg[0] == "reset":
                obs = np.stack([e.reset() for e in envs]).astype(np.float32)
                priv = (np.stack([e.critic_extras() for e in envs]).astype(np.float32)
                        if n_priv else None)
                conn.send((obs, priv))
            elif msg[0] == "step":
                acts = msg[1]
                obs, rew, done, tout, logs, priv = [], [], [], [], [], []
                for e, a in zip(envs, acts):
                    o, r, term, trunc, lg = e.step(a)
                    d = term or trunc
                    if d:
                        o = e.reset()
                    obs.append(o); rew.append(r); done.append(d); tout.append(trunc)
                    logs.append(lg)
                    if n_priv:
                        priv.append(e.critic_extras())
                conn.send((np.stack(obs).astype(np.float32),
                           np.array(rew, dtype=np.float32),
                           np.array(done, dtype=bool),
                           np.array(tout, dtype=bool),
                           logs,
                           np.stack(priv).astype(np.float32) if n_priv else None))
            elif msg[0] == "close":
                break
    except Exception as e:  # surface worker crashes to the parent
        import traceback
        conn.send(("__error__", traceback.format_exc()))
    finally:
        conn.close()


class PianoMjSubprocVecEnv:
    """PianoMjVecEnv API, but the envs are sharded across ``workers`` OS
    processes so the per-step Python work (reward, obs, contact bookkeeping)
    runs truly in parallel instead of serializing on the GIL. Same song
    round-robin as the threaded env (global env index % num_songs)."""

    def __init__(self, cfg, num_envs: int, workers: int):
        import multiprocessing as mp
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.workers = max(1, min(int(workers), self.num_envs))
        # vendor Menagerie / validate the scene ONCE before forking
        compile_scene(cfg)
        self.num_actions = cfg.action_space
        self.num_obs = cfg.observation_space
        self.num_priv_obs = cfg.critic_extra_dim()
        self.max_episode_length = int(round(cfg.episode_length_s / cfg.control_dt))
        ctx = mp.get_context("fork" if hasattr(__import__("os"), "fork") else "spawn")
        self._shards = np.array_split(np.arange(self.num_envs), self.workers)
        self._conns, self._procs = [], []
        for ids in self._shards:
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker_main, args=(child, cfg, [int(i) for i in ids]),
                            daemon=True)
            p.start(); child.close()
            self._conns.append(parent); self._procs.append(p)
        # episode length may derive from the song (episode_length_s == 0), which
        # only the workers' SongBank knows
        self._conns[0].send(("info",))
        info = self._conns[0].recv()
        if isinstance(info, tuple) and info and info[0] == "__error__":
            raise RuntimeError(info[1])
        self.max_episode_length = int(info["max_episode_length"])
        self._priv = np.zeros((self.num_envs, self.num_priv_obs), dtype=np.float32)
        self._last_logs: dict[str, float] = {}

    def _recv(self, conn):
        r = conn.recv()
        if isinstance(r, tuple) and len(r) == 2 and isinstance(r[0], str) and r[0] == "__error__":
            raise RuntimeError(f"PianoMj worker crashed:\n{r[1]}")
        return r

    def reset(self) -> np.ndarray:
        for c in self._conns:
            c.send(("reset",))
        rs = [self._recv(c) for c in self._conns]
        if self.num_priv_obs:
            self._priv = np.concatenate([r[1] for r in rs])
        return np.concatenate([r[0] for r in rs])

    def critic_priv(self) -> np.ndarray:
        return self._priv

    def step(self, actions: np.ndarray):
        acts = np.asarray(actions, dtype=np.float64)
        for c, ids in zip(self._conns, self._shards):
            c.send(("step", acts[ids]))
        rs = [self._recv(c) for c in self._conns]
        obs = np.concatenate([r[0] for r in rs])
        rew = np.concatenate([r[1] for r in rs])
        done = np.concatenate([r[2] for r in rs])
        timeout = np.concatenate([r[3] for r in rs])
        per_env = [lg for r in rs for lg in r[4]]
        if self.num_priv_obs:
            self._priv = np.concatenate([r[5] for r in rs])
        logs = _reduce_logs(per_env)
        self._last_logs = logs
        return obs, rew, done, timeout, logs

    def close(self):
        for c in self._conns:
            try:
                c.send(("close",))
            except (BrokenPipeError, OSError):
                pass
        for p in self._procs:
            p.join(timeout=5)


def make_rsl_rl_env(venv, device: str = "cpu"):
    """Wrap a PianoMjVecEnv for rsl_rl >= 5.x (TensorDict obs groups)."""
    import torch
    from tensordict import TensorDict
    from rsl_rl.env import VecEnv

    class RslRlPianoMjVecEnv(VecEnv):
        def __init__(self):
            self.venv = venv
            self.num_envs = venv.num_envs
            self.num_actions = venv.num_actions
            self.max_episode_length = venv.max_episode_length
            self.episode_length_buf = torch.zeros(
                self.num_envs, dtype=torch.long, device=device)
            self.device = device
            self.cfg = venv.cfg.to_dict()
            self._obs = venv.reset()

        def _obs_td(self):
            obs = torch.as_tensor(self._obs, dtype=torch.float32, device=self.device)
            groups = {"policy": obs}
            if venv.num_priv_obs:
                # asymmetric critic: rsl_rl concatenates ["policy", "critic_priv"]
                # for the critic (see ppo_cfg / train_piano_mj.build_train_cfg)
                groups["critic_priv"] = torch.as_tensor(
                    venv.critic_priv(), dtype=torch.float32, device=self.device)
            return TensorDict(groups, batch_size=[self.num_envs], device=self.device)

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
