"""Calibrate the shaping potential (DESIGN_SHAPING.md): play KI schwer against itself over the
v2 modes, record every seat's victory progress at the start of each round and the winners, fit
β of the soft win share softmax(β · progress²) and show how well it predicts the winner by
game phase. Φ is only worth using if it clearly beats the uniform guess well before the end.

    python -m hochciv_ki.calibrate_phi --games 1000 --out runs/calib_phi.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict

import numpy as np

from .pool import BridgePool
from .train_ppo import MAPS_V2, game_options

SIZES = [2, 2, 2, 3, 3, 4, 4, 4, 6, 8]       # player counts, drawn per game
BETAS = [0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24]
PHASES = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]


def play(bridge, job):
    seats = [{"kind": "agent", "kiLevel": "schwer"}] * job["n"]
    st = bridge.reset(seats, seed=job["seed"], map=job["map"], startPlayer=job["start"], **job["opts"])
    gid, rounds, last, turns = st["gid"], [], None, 0
    try:
        while not st["over"] and turns < 400:
            if st["round"] != last:
                last = st["round"]
                rounds.append((st["round"], [x and (None if x.get("dead") else x["prog"])
                                             for x in bridge.call("progress", gid=gid)["progress"]]))
            st = bridge.ki_turn(gid)
            turns += 1
    finally:
        bridge.close_game(gid)
    return {**job, "over": st["over"], "winners": st.get("winners", []), "how": (st.get("how") or "").split(" (")[0],
            "last_round": st["round"], "rounds": rounds}


def shares(prog, beta):
    """Soft win share per seat (0 for seats that can't win)."""
    alive = np.array([p is not None for p in prog])
    x = np.where(alive, beta * np.array([p or 0.0 for p in prog]) ** 2, -np.inf)
    e = np.exp(x - x[alive].max())
    return e / e.sum()


def records(games, by_round=False):
    """(phase, n, prog, target) per game round; target = share of the win per seat.
    by_round: the round number instead of the phase."""
    out = []
    for g in games:
        if not g["over"] or not g["winners"]:
            continue
        for rnd, prog in g["rounds"]:
            if sum(p is not None for p in prog) < 2:
                continue
            tgt = np.zeros(len(prog))
            tgt[g["winners"]] = 1.0 / len(g["winners"])
            out.append((rnd if by_round else (rnd - 1) / max(1, g["last_round"]), g["n"], prog, tgt))
    return out


def log_loss(recs, beta):
    if not recs:
        return float("nan")
    tot = 0.0
    for _, _, prog, tgt in recs:
        w = shares(prog, beta) if beta is not None else uniform(prog)
        tot -= float((tgt * np.log(np.maximum(w, 1e-9))).sum())
    return tot / len(recs)


def uniform(prog):
    alive = np.array([p is not None for p in prog], float)
    return alive / alive.sum()


def accuracy(recs, beta):
    """How often the seat with the highest share is a winner."""
    if not recs:
        return float("nan")
    return sum(tgt[int(np.argmax(shares(prog, beta)))] > 0 for _, _, prog, tgt in recs) / len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=70_000_000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="runs/calib_phi.json")
    ap.add_argument("--load", default=None, help="analyse saved games instead of playing")
    args = ap.parse_args()

    if args.load:
        games = json.load(open(args.load))
    else:
        rng = random.Random(args.seed)
        jobs = []
        for g in range(args.games):
            n = rng.choice(SIZES)
            jobs.append({"seed": args.seed + g, "n": n, "map": rng.choice(MAPS_V2[str(n)]),
                         "start": rng.randrange(n), "opts": game_options(rng, "v2")})
        kw = {"workers": args.workers} if args.workers else {}
        t0 = time.perf_counter()
        with BridgePool(**kw) as pool:
            print(f"engine {pool.version}, {pool.workers} workers, {len(jobs)} games", flush=True)
            games = pool.map(play, jobs)
        print(f"played in {time.perf_counter() - t0:.0f} s", flush=True)
        with open(args.out, "w") as f:
            json.dump(games, f)

    print("\nhow the games ended:")
    for k, v in Counter(g["how"] or "unfinished" for g in games).most_common():
        print(f"  {v:5d}  {k}")
    rounds = [g["last_round"] for g in games if g["over"]]
    print(f"rounds: median {np.median(rounds):.0f}, 90% {np.percentile(rounds, 90):.0f}")

    recs = records(games)
    fit = {b: log_loss(recs, b) for b in BETAS}
    best = min(fit, key=fit.get)
    print(f"\n{len(recs)} game rounds. log loss by beta (uniform guess {log_loss(recs, None):.4f}):")
    print("  " + "  ".join(f"{b}: {v:.4f}" for b, v in fit.items()))
    print(f"best beta = {best}")

    print("\nby game phase (share of the game's rounds) - log loss uniform / phi, top seat wins:")
    by = defaultdict(list)
    for r in recs:
        for lo, hi in PHASES:
            if lo <= r[0] < hi:
                by[(lo, hi)].append(r)
    for (lo, hi), rs in sorted(by.items()):
        print(f"  {lo:.2f}-{min(hi, 1):.2f}  {len(rs):6d} rounds  {log_loss(rs, None):.3f} / {log_loss(rs, best):.3f}"
              f"  top seat wins {accuracy(rs, best):.2f}")
    print("\nby players - log loss uniform / phi in the first half of the game:")
    for n in sorted({r[1] for r in recs}):
        rs = [r for r in recs if r[1] == n and r[0] < 0.5]
        print(f"  {n}p  {len(rs):6d} rounds  {log_loss(rs, None):.3f} / {log_loss(rs, best):.3f}"
              f"  top seat wins {accuracy(rs, best):.2f} (chance {1 / n:.2f})")

    print("\nbest beta by round (train_ppo uses min(8, slope * (round - 1))):")
    rr = records(games, by_round=True)
    for rnd in range(1, 11):
        rs = [r for r in rr if (r[0] == rnd if rnd < 10 else r[0] >= 10)]
        if rs:
            f = {b: log_loss(rs, b) for b in [0.5] + BETAS[1:6] + [8]}
            b = min(f, key=f.get)
            print(f"  round {rnd}{'+' if rnd == 10 else ' '} {len(rs):5d}  uniform {log_loss(rs, None):.3f}"
                  f"  best beta {b} -> {f[b]:.3f}")


if __name__ == "__main__":
    main()
