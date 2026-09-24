"""Vectorized KeyboardMjEnv (threaded or multi-process) with the same API as
the piano vec envs, so ``dexsim.tasks.piano_mj.vec_env.make_rsl_rl_env``
wraps it unchanged."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .keyboard_mj_env import KeyboardMjEnv, compile_keyboard_model


def _reduce_logs(per_env: list[dict]) -> dict:
    keys = list(dict.fromkeys(k for r in per_env for k in r))
    out = {}
    for k in keys:
        v = np.array([r[k] for r in per_env if k in r], dtype=np.float64)
        out[k] = float(v.mean())
    return out


class KeyboardMjVecEnv:
    """Synchronous vector env with auto-reset (one shared MjModel)."""

    def __init__(self, cfg, num_envs: int, threads: int | None = None):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.model = compile_keyboard_model(cfg)
        self.envs = [KeyboardMjEnv(cfg, model=self.model, env_index=i)
                     for i in range(self.num_envs)]
        self.num_actions = cfg.action_space
        self.num_obs = cfg.observation_space
        self.num_priv_obs = 0
        self.max_episode_length = max(e.max_episode_length for e in self.envs)
        n = threads if threads is not None else min(self.num_envs, 16)
        self._pool = ThreadPoolExecutor(max_workers=n) if n > 1 else None
        self._last_logs = {}

    def reset(self):
        return np.stack([e.reset() for e in self.envs]).astype(np.float32)

    def critic_priv(self):
        return np.zeros((self.num_envs, 0), dtype=np.float32)

    @staticmethod
    def _step_one(env, action):
        obs, rew, term, trunc, logs = env.step(action)
        done = term or trunc
        if done:
            obs = env.reset()
        return obs, rew, done, trunc, logs

    def step(self, actions):
        acts = np.asarray(actions, dtype=np.float64)
        if self._pool is not None:
            rs = list(self._pool.map(self._step_one, self.envs, acts))
        else:
            rs = [self._step_one(e, a) for e, a in zip(self.envs, acts)]
        obs = np.stack([r[0] for r in rs]).astype(np.float32)
        rew = np.array([r[1] for r in rs], dtype=np.float32)
        done = np.array([r[2] for r in rs], dtype=bool)
        tout = np.array([r[3] for r in rs], dtype=bool)
        logs = _reduce_logs([r[4] for r in rs])
        self._last_logs = logs
        return obs, rew, done, tout, logs

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False)


def _worker_main(conn, cfg, env_ids):
    try:
        model = compile_keyboard_model(cfg)
        envs = [KeyboardMjEnv(cfg, model=model, env_index=i) for i in env_ids]
        while True:
            msg = conn.recv()
            if msg[0] == "info":
                conn.send({"max_episode_length": max(e.max_episode_length for e in envs)})
            elif msg[0] == "reset":
                conn.send(np.stack([e.reset() for e in envs]).astype(np.float32))
            elif msg[0] == "step":
                obs, rew, done, tout, logs = [], [], [], [], []
                for e, a in zip(envs, msg[1]):
                    o, r, term, trunc, lg = e.step(a)
                    d = term or trunc
                    if d:
                        o = e.reset()
                    obs.append(o); rew.append(r); done.append(d); tout.append(trunc); logs.append(lg)
                conn.send((np.stack(obs).astype(np.float32), np.array(rew, dtype=np.float32),
                           np.array(done, dtype=bool), np.array(tout, dtype=bool), logs))
            elif msg[0] == "close":
                break
    except Exception:
        import traceback
        conn.send(("__error__", traceback.format_exc()))
    finally:
        conn.close()


class KeyboardMjSubprocVecEnv:
    """Envs sharded across ``workers`` processes (true parallelism)."""

    def __init__(self, cfg, num_envs: int, workers: int):
        import multiprocessing as mp
        import os
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.workers = max(1, min(int(workers), self.num_envs))
        compile_keyboard_model(cfg)             # validate + build legend PNGs once
        self.num_actions = cfg.action_space
        self.num_obs = cfg.observation_space
        self.num_priv_obs = 0
        ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
        self._shards = np.array_split(np.arange(self.num_envs), self.workers)
        self._conns, self._procs = [], []
        for ids in self._shards:
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker_main, args=(child, cfg, [int(i) for i in ids]),
                            daemon=True)
            p.start(); child.close()
            self._conns.append(parent); self._procs.append(p)
        self._conns[0].send(("info",))
        info = self._recv(self._conns[0])
        self.max_episode_length = int(info["max_episode_length"])
        self._last_logs = {}

    def _recv(self, conn):
        r = conn.recv()
        if isinstance(r, tuple) and len(r) == 2 and isinstance(r[0], str) and r[0] == "__error__":
            raise RuntimeError(f"KeyboardMj worker crashed:\n{r[1]}")
        return r

    def reset(self):
        for c in self._conns:
            c.send(("reset",))
        return np.concatenate([self._recv(c) for c in self._conns])

    def critic_priv(self):
        return np.zeros((self.num_envs, 0), dtype=np.float32)

    def step(self, actions):
        acts = np.asarray(actions, dtype=np.float64)
        for c, ids in zip(self._conns, self._shards):
            c.send(("step", acts[ids]))
        rs = [self._recv(c) for c in self._conns]
        obs = np.concatenate([r[0] for r in rs])
        rew = np.concatenate([r[1] for r in rs])
        done = np.concatenate([r[2] for r in rs])
        tout = np.concatenate([r[3] for r in rs])
        logs = _reduce_logs([lg for r in rs for lg in r[4]])
        self._last_logs = logs
        return obs, rew, done, tout, logs

    def close(self):
        for c in self._conns:
            try:
                c.send(("close",))
            except (BrokenPipeError, OSError):
                pass
        for p in self._procs:
            p.join(timeout=5)
