"""Imitation data: games where agent seats record (observation, teacher label, final score).

The label is always the JS KI planner's choice. The action actually played is chosen by an
"actor": the teacher itself with some random moves mixed in (round 1), or a trained network
(DAgger rounds), so the data also covers states the teacher alone would never reach.

    python -m hochciv_ki.data --games 4000 --out data/bc0 --eps 0.15
    python -m hochciv_ki.data --games 4000 --out data/bc1 --actor ckpt:checkpoints/bc/latest.pt
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np

from .env import Agent, play_game
from .model import grid_to_u8
from .pool import BridgePool

SEAT_MIX = [   # (weight, players, map choices)
    (3, ["agent", "agent"], ["plaettchen", "plaettchen", "original", "gross", "customgen"]),
    (2, ["agent", "ki:schwer"], ["plaettchen", "original"]),
    (1, ["agent", "ki:mittel"], ["plaettchen"]),
    (1, ["agent", "agent", "agent"], ["plaettchen", "gross", "customgen"]),
    (2, ["agent", "agent", "agent", "agent"], ["plaettchen", "original", "gross", "customgen"]),
    (1, ["agent", "agent", "ki:schwer", "ki:schwer"], ["plaettchen"]),
    (1, ["agent", "ki:schwer", "ki:mittel", "bot:koenig"], ["plaettchen", "original"]),
]


def seat_spec(s):
    if s == "agent":
        return {"kind": "agent"}
    kind, _, lvl = s.partition(":")
    return {"kind": "ki", "kiLevel": lvl} if kind == "ki" else {"kind": "bot", "diff": lvl}


def sample_setup(rng: random.Random):
    w = [m[0] for m in SEAT_MIX]
    _, seats, maps = rng.choices(SEAT_MIX, weights=w)[0]
    seats = seats[:]
    rng.shuffle(seats)
    return [seat_spec(s) for s in seats], rng.choice(maps)


class ExploringActor(Agent):
    """Plays `base` (or the teacher) but takes a uniformly random candidate with prob. eps."""

    def __init__(self, eps, seed, base: Agent | None = None):
        self.eps, self.rng, self.base = eps, random.Random(seed), base

    def act(self, obs):
        if self.rng.random() < self.eps:
            return self.rng.randrange(len(obs.tags)), None
        if self.base is not None:
            return self.base.act(obs)
        return obs.teacher, None


def gen_game(bridge, dims, job, make_actor):
    rng = random.Random(job["seed"])
    players, map_key = sample_setup(rng)
    actor = make_actor(job["seed"])
    agents = {i: actor for i, p in enumerate(players) if p["kind"] == "agent"}
    r = play_game(bridge, dims, players, agents, seed=job["seed"], map_key=map_key, record=True,
                  startPlayer=rng.randrange(len(players)))
    rows = []
    for pi, traj in r.trajectories.items():
        z = r.score(pi) if r.over else 1.0 / len(players)
        for obs, _a, _extra in traj:
            rows.append((grid_to_u8(obs.grid), obs.glob.copy(), obs.cand.copy(), obs.pos.copy(), obs.teacher, z, len(players)))
    return rows, r


def save_shard(path: Path, rows):
    shapes = np.array([r[0].shape[1:] for r in rows], np.int16)
    grids = np.concatenate([r[0].reshape(-1) for r in rows])
    globs = np.stack([r[1] for r in rows])
    lens = np.array([r[2].shape[0] for r in rows], np.int32)
    cands = np.concatenate([r[2] for r in rows]).astype(np.float16)
    poss = np.concatenate([r[3] for r in rows]).astype(np.int8)
    labels = np.array([r[4] for r in rows], np.int32)
    z = np.array([r[5] for r in rows], np.float32)
    nplayers = np.array([r[6] for r in rows], np.int8)
    np.savez_compressed(path, grid=grids, gshape=shapes, glob=globs.astype(np.float16), lens=lens, cand=cands, pos=poss,
                        label=labels, z=z, nplayers=nplayers)


def generate(pool: BridgePool, out: Path, games: int, make_actor, base_seed: int, shard_games=250):
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    done, total_rows = 0, 0
    for s0 in range(0, games, shard_games):
        jobs = [{"seed": base_seed + s0 + k} for k in range(min(shard_games, games - s0))]
        results = pool.map(lambda b, j: gen_game(b, pool.dims, j, make_actor), jobs)
        rows = [row for rr, _ in results for row in rr]
        save_shard(out / f"shard_{base_seed + s0:08d}.npz", rows)
        done += len(jobs)
        total_rows += len(rows)
        dt = time.perf_counter() - t0
        print(f"{done}/{games} games, {total_rows} samples, {dt:.0f}s ({done / dt:.1f} games/s)", flush=True)
    return total_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--out", default="data/bc0")
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--actor", default="teacher", help="teacher or ckpt:<path>")
    ap.add_argument("--base-seed", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    kw = {"workers": args.workers} if args.workers else {}
    with BridgePool(**kw) as pool:
        base = None
        if args.actor.startswith("ckpt:"):
            from .model import load_agent
            base = load_agent(args.actor[5:], greedy=False)
        print(f"engine {pool.version}, {pool.workers} workers, actor {args.actor}, eps {args.eps}")
        generate(pool, Path(args.out), args.games, lambda seed: ExploringActor(args.eps, seed, base), args.base_seed)


if __name__ == "__main__":
    main()
