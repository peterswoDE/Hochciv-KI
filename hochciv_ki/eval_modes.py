"""Evaluate an agent against KI schwer over a suite of game modes (v2: wonders, events, marathon,
5–8 empires, next to the v1 modes). Score = share of wins, fair share 1/n.

    python -m hochciv_ki.eval_modes --agent ckpt:checkpoints/v1w.pt --suite v2 --scale 1
"""
from __future__ import annotations

import argparse
import json

from .eval import agent_factory, mode_name, run_eval
from .pool import BridgePool

HE = {"events": True, "eventMode": "hard"}
EE = {"events": True, "eventMode": "easy"}
WO = {"wonders": True}

# (players, map, options, seeds at scale 1)
SUITES = {
    "v1": [
        (2, "plaettchen", {}, 60),
        (4, "plaettchen", {}, 15),
    ],
    "v2": [
        (2, "plaettchen", WO, 60),
        (2, "plaettchen", HE, 60),
        (2, "original", {**WO, **EE}, 40),
        (3, "customgen", {**WO, **HE}, 20),
        (4, "plaettchen", {**WO, **HE}, 15),
        (6, "plaettchen", {}, 10),
        (8, "customgen", WO, 6),
        (4, "marathon", {}, 10),
        (3, "customgen-marathon", WO, 10),
        # the v1 modes, against regressions
        (2, "plaettchen", {}, 60),
        (4, "customgen", {}, 15),
    ],
}


def run_suite(pool, make_agent, suite, scale=1.0, base_seed=90_000, opp="ki:schwer", progress=True):
    out = []
    for k, (n, m, opts, seeds) in enumerate(SUITES[suite]):
        r = run_eval(pool, make_agent, opp, n, max(1, round(seeds * scale)), base_seed + 1000 * k, m,
                     progress=progress, opts=opts)
        out.append({"mode": mode_name(n, m, opts), "score": round(r["score"], 4), "ci95": round(r["ci95"], 4),
                    "fair": round(r["fair"], 4), "games": r["games"], "rel": round(r["score"] / r["fair"], 3)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="teacher")
    ap.add_argument("--suite", default="v2", choices=list(SUITES))
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies the seeds of every mode")
    ap.add_argument("--base-seed", type=int, default=90_000)
    ap.add_argument("--opp", default="ki:schwer")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--json", default=None, help="write the results here")
    args = ap.parse_args()
    kw = {"workers": args.workers} if args.workers else {}
    with BridgePool(**kw) as pool:
        print(f"engine {pool.version}, {pool.workers} workers, agent {args.agent}", flush=True)
        res = run_suite(pool, agent_factory(args.agent), args.suite, args.scale, args.base_seed, args.opp)
    print()
    for r in res:
        print(f"{r['mode']:42s} {r['score']:.3f} ± {r['ci95']:.3f}  fair {r['fair']:.3f}  ({r['games']} games)")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"agent": args.agent, "suite": args.suite, "results": res}, f, indent=1)


if __name__ == "__main__":
    main()
