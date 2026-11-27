"""Evaluate a bot against KI schwer (or other opponents) over many seeds and every seat.

    python evaluate.py rusher                      # all modes, 12 seeds each
    python evaluate.py rusher --modes tile2 orig4 --seeds 40 --jobs 8
    python evaluate.py rusher --seed0 1000 -v      # verbose single-process run

Modes: tile2 tile3 tile4 (tile map with placement), orig4 (original map, 4 players),
custom2 custom3 custom4 (generated custom maps); with the experimental nations of v111+ in the
draw (not in the default set): xtile2, xtile3 and xcustom4 (the last two with wonders). The bot sits in every seat in turn, so seat
order and civilisation even out. A fair share is 1/n; the goal is far above it.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "template"), str(ROOT)]

from hcbots.bridge import Bridge          # noqa: E402
from hcbots.runner import play_game       # noqa: E402

MODES = {"tile2": ("plaettchen", 2), "tile3": ("plaettchen", 3), "tile4": ("plaettchen", 4),
         "orig4": ("original", 4), "custom2": ("customgen", 2), "custom3": ("customgen", 3),
         "custom4": ("customgen", 4),
         # the experimental nations (Rome, Venice, Aztecs, Egypt, Germany, Maya, China, USA)
         "xtile2": ("plaettchen", 2, {"expCivs": True}),
         "xtile3": ("plaettchen", 3, {"expCivs": True, "wonders": True}),
         "xcustom4": ("customgen", 4, {"expCivs": True, "wonders": True})}
BASE_MODES = [m for m in MODES if not m.startswith("x")]


def load_bot(name: str, opts: dict | None = None):
    mod = importlib.import_module(f"bots.{name}.bot")
    cls = next(v for v in vars(mod).values() if isinstance(v, type) and getattr(v, "name", None) == name)
    return cls(**(opts or {}))


def parse_opts(items) -> dict:
    """key=value pairs for the bot's constructor; values are read as Python literals."""
    import ast
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def run_jobs(args):
    bot_name, jobs, opp_level, verbose, opts = args
    out = []
    with Bridge() as br:
        for mode, seed, seat in jobs:
            map_, n, *rest = MODES[mode]
            seats = [{"kind": "ki", "kiLevel": opp_level} for _ in range(n)]
            seats[seat] = {"kind": "py"}
            r = play_game(br, {seat: load_bot(bot_name, opts)}, seats, seed, map_, verbose=verbose,
                          **(rest[0] if rest else {}))
            r.update(mode=mode, seat=seat, win=seat in r["winners"])
            out.append(r)
            if verbose:
                print(mode, seed, seat, "WIN" if r["win"] else "loss", r["how"], "round", r["rounds"],
                      flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bot")
    ap.add_argument("--modes", nargs="*", default=BASE_MODES)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--level", default="schwer")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--losses", action="store_true", help="list every lost game")
    ap.add_argument("--opt", nargs="*", default=[], help="bot options key=value (constructor kwargs)")
    a = ap.parse_args()

    jobs = []
    for mode in a.modes:
        n = MODES[mode][1]
        for s in range(a.seed0, a.seed0 + a.seeds):
            jobs.append((mode, s, s % n))          # rotating seat
    if a.verbose or a.jobs <= 1:
        results = run_jobs((a.bot, jobs, a.level, a.verbose, parse_opts(a.opt)))
    else:
        chunks = [jobs[i::a.jobs] for i in range(a.jobs)]
        with ProcessPoolExecutor(a.jobs) as ex:
            results = [r for part in ex.map(run_jobs, [(a.bot, c, a.level, False, parse_opts(a.opt)) for c in chunks]) for r in part]

    by = defaultdict(list)
    for r in results:
        by[r["mode"]].append(r)
    print(f"\n{a.bot} {' '.join(a.opt)} vs ki {a.level}")
    print(f"{'mode':8} {'games':>5} {'wins':>5} {'rate':>6} {'fair':>5} {'rounds':>6}  how (wins) / how (losses)")
    total = wins = 0
    for mode in a.modes:
        rs = by[mode]
        w = sum(r["win"] for r in rs)
        total += len(rs)
        wins += w
        hw = Counter(r["how"] for r in rs if r["win"])
        hl = Counter(r["how"] or "timeout" for r in rs if not r["win"])
        avg = sum(r["rounds"] for r in rs) / max(len(rs), 1)
        err = sum(r["errors"] for r in rs)
        print(f"{mode:8} {len(rs):5} {w:5} {w / max(len(rs), 1):6.0%} {1 / MODES[mode][1]:5.0%} {avg:6.1f}  "
              f"{dict(hw)} / {dict(hl)}" + (f"  errors={err}" if err else ""))
    print(f"{'all':8} {total:5} {wins:5} {wins / max(total, 1):6.0%}")
    if a.losses:
        for r in sorted(results, key=lambda r: (r["mode"], r["seed"])):
            if not r["win"]:
                print(f"  {r['mode']} seed {r['seed']} seat {r['seat']} round {r['rounds']}: {r['how']}")


if __name__ == "__main__":
    main()
