"""The scripted rusher (bridge/rush.js): it plays its seats without errors, follows its plan
(slavery sacrifices, armies by round 3) and beats KI schwer more often than not – it is meant
to be the opponent KI schwer never is.

    python -m tests.test_rush
"""
from __future__ import annotations

import sys

from hochciv_ki.bridge import Bridge

GAMES = 16


def play(b, seed, rush_first):
    seats = [{"kind": "rush"}, {"kind": "ki", "kiLevel": "schwer"}]
    if not rush_first:
        seats.reverse()
    r = b.reset(seats, seed=seed, map="plaettchen")
    assert r["over"], r                      # no agent seat: the worker plays the whole game
    S = b.state(r["gid"])
    b.close_game(r["gid"])
    rush = next(i for i, p in enumerate(S["players"]) if p["slot"] == (0 if rush_first else 1))
    return S, rush, rush in (r.get("winners") or [])


def main():
    wins = sacrificed = early_army = 0
    with Bridge() as b:
        for k in range(GAMES):
            S, rush, won = play(b, 7000 + k, k % 2 == 0)
            wins += won
            name = S["players"][rush]["civ"]
            log = [e for e in S["log"] if e.get("c") == "act"]
            sacrificed += any("geopfert" in e["m"] for e in log)
            # an army built by the rusher in rounds 1–3 (civ names are shared in mirror games,
            # so any army counts there)
            early_army += any("Armee gebaut" in e["m"] and e["r"] <= 3 for e in log)
            print(f"seed {7000 + k}: rusher ({name}) {'won' if won else 'lost'} in round {S['round']}", flush=True)
    print(f"rusher won {wins}/{GAMES}, sacrificed in {sacrificed}, army by round 3 in {early_army}")
    ok = wins > GAMES / 2 and sacrificed >= GAMES // 4 and early_army >= GAMES // 2
    print("OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
