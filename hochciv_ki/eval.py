"""Paired evaluation against the JS KI and bots.

Every seed is played once per seat rotation: the agent takes each seat in turn, the opponents
the others, so every empire, map position and turn position is played by the agent once.
That cancels out the start player advantage (as `tools_ki.js stufen` does).

    python -m hochciv_ki.eval --agent teacher --opp ki:schwer --players 2 --seeds 50
    python -m hochciv_ki.eval --agent ckpt:checkpoints/bc/latest.pt --players 4

Opponents can be networks too, and a comma-separated list fills the opponent seats in turn
(head-to-head: does a new model beat the old one, not only the KI?):

    python -m hochciv_ki.eval --agent ckpt:checkpoints/ppo3/best.pt --opp ckpt:checkpoints/v2.pt
    python -m hochciv_ki.eval --agent ckpt:a.pt --opp ckpt:b.pt,ki:schwer --players 4
"""
from __future__ import annotations

import argparse
import math
import time
from collections import Counter

from .env import KiTurnAgent, TeacherAgent, play_game
from .pool import BridgePool


def opp_seat(spec: str):
    """'ki:schwer', 'ki:mittel', 'ki:leicht', 'bot:prinz', 'bot:koenig', ...; a network
    ('ckpt:…', 'ckpts:…') is an agent seat."""
    if is_net(spec):
        return {"kind": "agent"}
    kind, _, lvl = spec.partition(":")
    if kind == "ki":
        return {"kind": "ki", "kiLevel": lvl or "schwer"}
    if kind == "bot":
        return {"kind": "bot", "diff": lvl or "prinz"}
    raise ValueError(spec)


def mode_name(n_players, map_key, opts):
    o = opts or {}
    tags = [f"{n_players}p", map_key]
    if o.get("wonders"):
        tags.append("wonders")
    if o.get("events"):
        tags.append("events " + o.get("eventMode", "hard"))
    return ", ".join(tags)


def is_net(spec: str):
    return spec.startswith("ckpt:") or spec.startswith("ckpts:")


def jobs_for(n_players, seeds, base_seed, map_key):
    for s in range(seeds):
        for seat in range(n_players):
            yield {"seed": base_seed + s, "seat": seat, "n": n_players, "map": map_key}


def run_eval(pool: BridgePool, make_agent, opp: str, n_players=2, seeds=50, base_seed=10_000,
             map_key="plaettchen", progress=True, opts=None):
    """opts: game options for the bridge (events, eventMode, wonders)."""
    agent = make_agent()
    specs = opp.split(",")
    nets = {sp: agent_factory(sp)() for sp in set(specs) if is_net(sp)}
    jobs = list(jobs_for(n_players, seeds, base_seed, map_key))

    def one(bridge, job):
        players, agents = [None] * job["n"], {job["seat"]: agent}
        others = [i for i in range(job["n"]) if i != job["seat"]]
        for k, i in enumerate(others):
            sp = specs[k % len(specs)]
            players[i] = opp_seat(sp)
            if sp in nets:
                agents[i] = nets[sp]
        players[job["seat"]] = {"kind": "agent"}
        r = play_game(bridge, pool.dims, players, agents, seed=job["seed"], map_key=job["map"],
                      **(opts or {}))
        return job, r

    t0 = time.perf_counter()
    results = pool.map(one, jobs)
    dt = time.perf_counter() - t0
    scores, hows, rounds, unfinished = [], Counter(), [], 0
    for job, r in results:
        pi = next(p for p, slot in r.slot_of.items() if slot == job["seat"])
        scores.append(r.score(pi))
        if not r.over:
            unfinished += 1
        hows[(r.how or "open").split(" (")[0] + (" won" if pi in r.winners else " lost")] += 1
        rounds.append(r.round)
    n = len(scores)
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / max(1, n - 1)
    ci = 1.96 * math.sqrt(var / n)
    fair = 1.0 / n_players
    rounds.sort()
    out = {
        "games": n, "score": mean, "ci95": ci, "fair": fair, "unfinished": unfinished,
        "median_round": rounds[n // 2], "how": dict(hows), "seconds": dt,
    }
    if progress:
        print(f"{n} games ({mode_name(n_players, map_key, opts)}) vs {opp}: score {mean:.3f} ± {ci:.3f} "
              f"(fair share {fair:.3f}), median end round {out['median_round']}, {dt:.0f}s")
        for k, v in hows.most_common():
            print(f"    {v:4d}  {k}")
    return out


def agent_factory(spec: str, device="cuda"):
    if spec == "teacher":
        return TeacherAgent
    if spec == "kiturn":
        return KiTurnAgent
    if spec.startswith("ckpt:") or spec.startswith("ckpts:"):
        # ckpt: greedy, ckpts: samples from the policy
        from .model import load_agent
        greedy, path = spec.startswith("ckpt:"), spec.split(":", 1)[1]
        return lambda: load_agent(path, device=device, greedy=greedy)
    raise ValueError(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="teacher")
    ap.add_argument("--opp", default="ki:schwer")
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--map", default="plaettchen")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--base-seed", type=int, default=10_000)
    ap.add_argument("--wonders", action="store_true")
    ap.add_argument("--events", choices=["hard", "easy"], default=None)
    args = ap.parse_args()
    opts = {"wonders": args.wonders}
    if args.events:
        opts.update(events=True, eventMode=args.events)
    kw = {"workers": args.workers} if args.workers else {}
    with BridgePool(**kw) as pool:
        print(f"engine {pool.version}, {pool.workers} workers")
        run_eval(pool, agent_factory(args.agent), args.opp, args.players, args.seeds, args.base_seed, args.map,
                 opts=opts)


if __name__ == "__main__":
    main()
