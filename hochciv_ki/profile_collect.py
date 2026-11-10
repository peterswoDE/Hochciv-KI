"""Where does the time of a PPO iteration go? Plays training games exactly like train_ppo's
collect (same mix, maps and options, the network on the GPU) with timing in the Node workers
(HOCHCIV_PROFILE=1) and on the Python side (bridge calls incl. the pipe, inference wait).

    python -m hochciv_ki.profile_collect --init checkpoints/v2.pt --games 56 --workers 14

Engine and feature times are inclusive: kiCandidates and kiClone also run inside kiTurn
(the KI seats) and evaluate (the candidate evaluation of the agent's observation).
"""
from __future__ import annotations

import argparse
import os
import random
import threading
import time
from collections import defaultdict

from .bridge import Bridge
from .infer import InferenceServer, ServerAgent
from .model import load_net
from .train_ppo import MAPS, MAPS_V2, MIXES, collect, game_options


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="checkpoints/v2.pt")
    ap.add_argument("--mix", default="v2", choices=list(MIXES))
    ap.add_argument("--games", type=int, default=56)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--infer-wait", type=float, default=0.002, help="InferenceServer batch wait (s)")
    args = ap.parse_args()

    os.environ["HOCHCIV_PROFILE"] = "1"       # the workers inherit it
    from .pool import BridgePool

    # Python side: time of every bridge call by command, and the wait for the network
    calls, lock = defaultdict(lambda: [0.0, 0]), threading.Lock()
    orig_call = Bridge.call

    def call(self, cmd, **kw):
        t0 = time.perf_counter()
        try:
            return orig_call(self, cmd, **kw)
        finally:
            with lock:
                c = calls["bridge." + cmd]
                c[0] += time.perf_counter() - t0
                c[1] += 1

    Bridge.call = call
    orig_act = ServerAgent.act

    def act(self, obs):
        t0 = time.perf_counter()
        try:
            return orig_act(self, obs)
        finally:
            with lock:
                c = calls["inference (wait + sample)"]
                c[0] += time.perf_counter() - t0
                c[1] += 1

    ServerAgent.act = act

    net, _ = load_net(args.init, "cuda")
    learner, snap = (InferenceServer(net.eval(), "cuda", wait=args.infer_wait),
                     InferenceServer(load_net(args.init, "cuda")[0].eval(), "cuda", wait=args.infer_wait))
    rng = random.Random(args.seed)
    mix = MIXES[args.mix]
    jobs = []
    for g in range(args.games):
        seats = rng.choices([m[1] for m in mix], weights=[m[0] for m in mix])[0]
        maps = (MAPS_V2 if args.mix == "v2" else MAPS)[str(len(seats))]
        jobs.append({"seed": 60_000_000 + g, "seats": seats, "map": rng.choice(maps),
                     "opts": game_options(rng, args.mix)})
    with BridgePool(workers=args.workers) as pool:
        def one(b, j):
            t = time.perf_counter()
            res = collect(b, pool.dims, j, learner, snap)
            return res, time.perf_counter() - t, len(j["seats"])

        t0 = time.perf_counter()
        out = pool.map(one, jobs)
        wall = time.perf_counter() - t0
        results = [r for r, _, _ in out]
        js = defaultdict(lambda: [0.0, 0])
        for b in pool._all:
            for k, v in orig_call(b, "profile")["profile"].items():
                js[k][0] += v["ms"] / 1000
                js[k][1] += v["n"]
    learner.close()
    snap.close()
    for name, srv in (("learner", learner), ("snapshot", snap)):
        st = srv.stats
        if st["batches"]:
            print(f"inference {name}: {st['batches']} batches, {st['items'] / st['batches']:.1f} per batch, "
                  f"{1000 * st['fwd_s'] / st['batches']:.1f} ms per batch (collate + forward), "
                  f"{100 * st['fwd_s'] / wall:.0f} % of wall time busy")

    samples = sum(len(rec) for recs, _ in results for rec, _ in recs)
    busy = wall * args.workers
    print(f"{args.games} games, {samples} learner decisions, {wall:.0f} s wall, {args.workers} workers "
          f"({busy:.0f} worker-seconds), {samples / wall:.0f} decisions/s")
    game_s = [d for _, d, _ in out]
    print(f"games: {sum(game_s):.0f} worker-seconds busy = {100 * sum(game_s) / busy:.0f} % of the pool, "
          f"longest game {max(game_s):.0f} s, median {sorted(game_s)[len(game_s) // 2]:.1f} s")
    by_n = defaultdict(list)
    for _, d, n in out:
        by_n[n].append(d)
    print("  seconds per game by players: " + ", ".join(
        f"{n}p {sum(v) / len(v):.1f} (max {max(v):.0f})" for n, v in sorted(by_n.items())))
    print("\nshare of worker time (inclusive, so the rows overlap):")
    for name, (s, n) in sorted({**js, **calls}.items(), key=lambda kv: -kv[1][0]):
        print(f"  {name:28s} {s:8.1f} s  {100 * s / busy:5.1f} %  {n:8d} calls  {1000 * s / max(1, n):8.2f} ms/call")


if __name__ == "__main__":
    main()
