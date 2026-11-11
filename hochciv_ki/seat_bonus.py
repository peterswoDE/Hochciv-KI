"""Seat advantage and the start bonus: how often each place in the turn order wins, with the
engine's start bonus (Startausgleich) or a different one.

Every seat is played by the same player (KI schwer, or one network in every seat), so a
difference between the places comes from the turn order alone. The bonus is the engine's
rule with other numbers (bridge/worker.js, engineStartBonus): coins in the first turn, by
place in the turn order; with two empires only the second gets `duel`.

    python -m hochciv_ki.seat_bonus --players 2 3 4 --steps 0 2 4 6 --duels 0 4 8 12 --games 600
    python -m hochciv_ki.seat_bonus --seat ckpt:checkpoints/v5.pt ...

--steps s means coins 0, s, 2s, 3s for places 1-4 (the engine today: s = 2, duel 4).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter

from .env import play_game
from .eval import agent_factory, is_net, opp_seat
from .pool import BridgePool


def run(pool, n, bonus, games, seat, map_key, base_seed):
    agent = agent_factory(seat)() if is_net(seat) else None
    players = [opp_seat(seat)] * n

    def one(bridge, s):
        agents = {i: agent for i in range(n)} if agent else {}
        r = play_game(bridge, pool.dims, players, agents, seed=base_seed + s, map_key=map_key,
                      startBonus=bonus)
        start = next(pi for pi, slot in r.slot_of.items() if slot == 0)
        return [(pi - start) % n for pi in r.winners] if r.over else []

    wins, unfinished = Counter(), 0
    for w in pool.map(one, range(games)):
        if not w:
            unfinished += 1
        for k in w:
            wins[k] += 1 / len(w)
    done = games - unfinished
    share = [wins[k] / max(1, done) for k in range(n)]
    ci = [1.96 * math.sqrt(max(p * (1 - p), 1e-9) / max(1, done)) for p in share]
    return {"players": n, "bonus": bonus, "games": games, "unfinished": unfinished,
            "share": [round(x, 3) for x in share], "ci95": [round(x, 3) for x in ci],
            "spread": round(max(share) - min(share), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 2, 4, 6],
                    help="3-4 empires: coins 0, s, 2s, 3s by place (engine: 2)")
    ap.add_argument("--duels", type=int, nargs="+", default=[0, 4, 8, 12],
                    help="2 empires: coins for the second (engine: 4)")
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--seat", default="ki:schwer", help="who plays every seat")
    ap.add_argument("--map", default="plaettchen")
    ap.add_argument("--base-seed", type=int, default=600_000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    out = []
    with BridgePool() as pool:
        print(f"engine {pool.version}, every seat {args.seat}, map {args.map}, {args.games} games each", flush=True)
        for n in args.players:
            variants = [{"duel": d, "coins": [0]} for d in args.duels] if n == 2 else \
                [{"coins": [k * s for k in range(n)], "duel": 0} for s in args.steps]
            for b in variants:
                t0 = time.perf_counter()
                r = run(pool, n, b, args.games, args.seat, args.map, args.base_seed)
                out.append(r)
                label = f"2nd +{b['duel']}" if n == 2 else "coins " + "/".join(map(str, b["coins"]))
                cells = "  ".join(f"{p:.3f}±{c:.3f}" for p, c in zip(r["share"], r["ci95"]))
                print(f"{n}p  {label:18s} {cells}   spread {r['spread']:.3f}  ({r['unfinished']} open, "
                      f"{time.perf_counter() - t0:.0f}s)", flush=True)
                if args.json:
                    with open(args.json, "w", encoding="utf-8") as f:
                        json.dump({"seat": args.seat, "map": args.map, "results": out}, f, indent=1)


if __name__ == "__main__":
    main()
