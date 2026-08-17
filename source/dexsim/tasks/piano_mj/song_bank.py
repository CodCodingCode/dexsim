"""Song goals + fingering, precomputed once and shared by every env instance.

MuJoCo port of the song machinery in ``PianoEnv.__init__`` / ``_gather_songs``:
loads one MIDI (or a multi-song .npz bundle), folds it into the hands'
reachable key windows, plans a fingering, and stacks everything into padded
numpy tensors indexed ``[song_id, song_step]``. Pure numpy -- no sim.
"""

from __future__ import annotations

import numpy as np

from dexsim.piano import load_song, plan_fingering, geometry, NUM_FINGERS, NUM_KEYS
from dexsim.piano.midi import fold_into_reach


def compute_swap_hands(cfg) -> bool:
    """GEOMETRY GUARDRAIL flag for plan_fingering (== PianoEnv._compute_swap_hands):
    True when the low-pitch keys are physically nearer the RIGHT robot (e.g.
    the 180°-flipped piano of the locked layout), so each hand is assigned the
    keys in its own vicinity instead of reaching across the body."""
    loc = geometry.key_local_top_positions()                 # (88,3) piano-local
    q = np.asarray(cfg.piano_rot, dtype=float)               # wxyz
    p = np.asarray(cfg.piano_pos, dtype=float)
    u = q[1:]

    def world_y(v):
        t = 2.0 * np.cross(u, v)
        return float((p + v + q[0] * t + np.cross(u, t))[1])

    wy = np.array([world_y(v) for v in loc])
    split = NUM_KEYS // 2
    low_y = float(wy[:split].mean())
    lb = float(cfg.left_base_pos[1])
    rb = float(cfg.right_base_pos[1])
    swap = abs(low_y - rb) < abs(low_y - lb)
    print(f"[PianoMjEnv] hand-side guardrail: low-pitch keys mean worldY={low_y:+.2f}, "
          f"left_robot Y={lb:+.2f}, right_robot Y={rb:+.2f} -> swap_hands={swap}")
    return bool(swap)


class SongBank:
    """Padded per-song tensors, shared (read-only) across all envs."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.swap_hands = compute_swap_hands(cfg)
        L = int(cfg.goal_lookahead)
        songs = self._gather_songs(cfg)
        self.song_names = [s[0] for s in songs]
        self.num_songs = len(songs)
        lens = [s[1].shape[0] for s in songs]
        Tmax = max(lens)
        self.song_lens = np.asarray(lens, dtype=np.int64)
        self.song_len = int(Tmax)

        goals, onsets, fkeys, factives = [], [], [], []
        finger_home = None
        for name, act, ons in songs:
            T = act.shape[0]
            pad = np.zeros((Tmax + L - T, NUM_KEYS), dtype=np.float32)
            goals.append(np.concatenate([act.astype(np.float32), pad], 0))
            onsets.append(np.concatenate([ons.astype(np.float32), pad.copy()], 0))
            plan = plan_fingering(act, swap_hands=self.swap_hands)
            pfk, pfa = plan.finger_key.copy(), plan.finger_active.copy()
            if getattr(cfg, "remap_thumb_to_middle", False):
                for th, mid in ((0, 2), (5, 7)):
                    mv = pfa[:, th] & ~pfa[:, mid]
                    pfk[mv, mid] = pfk[mv, th]
                    pfa[mv, mid] = True
                    pfa[mv, th] = False
            fk = np.concatenate([pfk, np.repeat(pfk[-1:], Tmax + L - T, 0)], 0)
            fa = np.concatenate(
                [pfa, np.zeros((Tmax + L - T, NUM_FINGERS), dtype=bool)], 0)
            fkeys.append(fk)
            factives.append(fa)
            if finger_home is None:
                finger_home = plan.home_key.copy()
        self.goal = np.stack(goals, 0)             # (N, Tmax+L, 88) f32
        self.onset = np.stack(onsets, 0)           # (N, Tmax+L, 88) f32
        self.finger_key = np.stack(fkeys, 0)       # (N, Tmax+L, 10) i64
        self.finger_active = np.stack(factives, 0)  # (N, Tmax+L, 10) bool
        self.finger_home = finger_home             # (10,) i64

        # time-dilated onset window for the onset-timing metric (+/-W steps)
        W = int(getattr(cfg, "onset_tol_steps", 3))
        ow = self.onset.copy()
        for s in range(1, W + 1):
            fut = np.zeros_like(self.onset)
            fut[:, :-s] = self.onset[:, s:]
            pst = np.zeros_like(self.onset)
            pst[:, s:] = self.onset[:, :-s]
            ow = np.maximum(ow, np.maximum(fut, pst))
        self.onset_win = ow

        names = ", ".join(self.song_names[:4]) + ("..." if self.num_songs > 4 else "")
        print(f"[PianoMjEnv] {self.num_songs} song(s) [{names}]: longest "
              f"{self.song_len} steps @ {1 / cfg.control_dt:.0f}Hz")

    @staticmethod
    def _gather_songs(cfg):
        """[(name, key_activation (T,88) bool, onsets (T,88) bool)] -- one from
        cfg.midi_path, or N from cfg.songs_npz (rising-edge onsets)."""

        def _fold(act, ons):
            if cfg.fold_to_reach:
                act, ons = fold_into_reach(
                    act, ons, left_window=tuple(cfg.left_key_window),
                    right_window=tuple(cfg.right_key_window))
            return act, ons

        npz = getattr(cfg, "songs_npz", None)
        if npz:
            d = np.load(npz, allow_pickle=True)
            G, lens, names = d["goals"], d["lens"], d["names"]
            off = int(getattr(cfg, "song_offset", 0))
            cap = int(getattr(cfg, "max_songs", 0)) or (G.shape[0] - off)
            out = []
            for i in range(off, min(off + cap, G.shape[0])):
                T = int(lens[i])
                act = G[i, :T].astype(bool)
                ons = np.zeros_like(act)
                ons[0] = act[0]
                ons[1:] = act[1:] & ~act[:-1]
                act, ons = _fold(act, ons)
                out.append((str(names[i]), act, ons))
            print(f"[PianoMjEnv] MULTI-SONG: {len(out)} songs from {npz} "
                  f"(fold_to_reach={cfg.fold_to_reach})")
            return out

        song = load_song(cfg.midi_path, control_dt=cfg.control_dt)
        act, ons = _fold(song.key_activation, song.onsets)
        if cfg.fold_to_reach:
            print(f"[PianoMjEnv] folded '{song.name}' into reach: "
                  f"{int(act.any(0).sum())} distinct keys")
        return [(song.name, act, ons)]
