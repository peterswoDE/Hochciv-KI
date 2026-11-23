"""Strength of every nation and ability for a network (plan item 5: choosing the nation).

The bot joins online games with civ "random", ability "random". In lobbies with free choice
(civMode fixed) and in the draft it could pick instead; this measures what to pick. Every arm
(nation:ability) plays the same seeds and seats against the same opponents, so the arms differ
only in the nation (common random numbers); the opponents draw their nations from the seed as
usual (with the experimental ones and wonders on, so every nation is in the pool).

    python -m hochciv_ki.civ_eval --agent ckpt:checkpoints/v5x.pt --seeds 40 --json runs/civ_v5x.json
    python -m hochciv_ki.civ_eval --report runs/civ_v5x.json

Modes (--modes): duel = 2p tiles vs KI schwer, four = 4p tiles vs KI schwer, rush = 2p tiles vs
hc:rusher. rel = score / fair share. The table sorts by the mean rel over the modes.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict

from .env import play_game
from .eval import agent_factory, opp_seat
from .hcbot import hc_agent, is_hc
from .pool import BridgePool

MODES = {
    "duel": (2, "plaettchen", "ki:schwer"),
    "four": (4, "plaettchen", "ki:schwer"),
    "rush": (2, "plaettchen", "hc:rusher"),
}
OPTS = {"expCivs": True, "wonders": True}


def arms_of(info, only=None):
    out = []
    for civ, abils in info["abilities"].items():
        if only and civ not in only:
            continue
        out += [f"{civ}:{a}" for a in abils]
    return out


def run(pool, agent_spec, arms, modes, seeds, base_seed, progress=True):
    agent = agent_factory(agent_spec)()
    jobs = []
    for mode in modes:
        n, mp, opp = MODES[mode]
        for s in range(seeds):
            seat = s % n
            for arm in arms:
                jobs.append({"mode": mode, "n": n, "map": mp, "opp": opp, "seed": base_seed + s, "seat": seat, "arm": arm})

    def one(bridge, job):
        civ, ab = job["arm"].split(":")
        players, agents = [None] * job["n"], {job["seat"]: agent}
        for i in range(job["n"]):
            if i == job["seat"]:
                players[i] = {"kind": "agent", "civ": civ, "ability": ab}
            else:
                players[i] = opp_seat(job["opp"])
                if is_hc(job["opp"]):
                    agents[i] = hc_agent(job["opp"])
        r = play_game(bridge, pool.dims, players, agents, seed=job["seed"], map_key=job["map"], **OPTS)
        pi = next(p for p, slot in r.slot_of.items() if slot == job["seat"])
        return job, r.score(pi), r.over

    t0, done, out = time.perf_counter(), 0, []
    futs = [pool.submit(one, j) for j in jobs]
    for f in futs:
        out.append(f.result())
        done += 1
        if progress and done % 200 == 0:
            dt = time.perf_counter() - t0
            print(f"{done}/{len(jobs)} games, {dt:.0f}s, eta {dt / done * (len(jobs) - done) / 60:.0f} min", flush=True)
    res = defaultdict(lambda: defaultdict(list))
    unfinished = 0
    for job, sc, over in out:
        res[job["arm"]][job["mode"]].append(sc)
        unfinished += not over
    return {"agent": agent_spec, "seeds": seeds, "base_seed": base_seed, "modes": modes, "opts": OPTS,
            "unfinished": unfinished, "seconds": time.perf_counter() - t0,
            "scores": {a: dict(m) for a, m in res.items()}}


def table(data):
    rows = []
    for arm, by_mode in data["scores"].items():
        row = {"arm": arm}
        rels = []
        for mode, sc in by_mode.items():
            n = MODES[mode][0]
            m = sum(sc) / len(sc)
            sd = math.sqrt(sum((x - m) ** 2 for x in sc) / max(1, len(sc) - 1))
            row[mode] = (m * n, 1.96 * sd / math.sqrt(len(sc)) * n)
            rels.append(m * n)
        row["mean"] = sum(rels) / len(rels)
        rows.append(row)
    rows.sort(key=lambda r: -r["mean"])
    modes = data["modes"]
    print(f"{'arm':28s} {'mean':>6s} " + " ".join(f"{m:>14s}" for m in modes))
    for r in rows:
        print(f"{r['arm']:28s} {r['mean']:6.2f} " + " ".join(f"{r[m][0]:6.2f} ± {r[m][1]:5.2f}" for m in modes))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="ckpt:checkpoints/v5x.pt")
    ap.add_argument("--modes", nargs="+", default=["duel", "four", "rush"], choices=list(MODES))
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--base-seed", type=int, default=700_000)
    ap.add_argument("--civs", nargs="*", help="only these nations")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--json")
    ap.add_argument("--report", help="print the table of a saved run")
    args = ap.parse_args()
    if args.report:
        table(json.load(open(args.report, encoding="utf-8")))
        return
    pool = BridgePool(args.workers) if args.workers else BridgePool()
    arms = arms_of(pool.info, args.civs)
    print(f"engine {pool.version}: {len(arms)} arms, modes {args.modes}, {args.seeds} seeds each", flush=True)
    data = run(pool, args.agent, arms, args.modes, args.seeds, args.base_seed)
    if args.json:
        json.dump(data, open(args.json, "w", encoding="utf-8"))
    table(data)


if __name__ == "__main__":
    main()
