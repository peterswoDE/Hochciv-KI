"""Online-Zugschleife ohne Server testen: ein zweiter Engine-Worker spielt den Server (eigene
Würfel, Zustand ohne S.seed wie stateForPlayer), der Bot spielt über online.client.Brain.play
gegen KI schwer – wie online, Aktion für Aktion.

    python tests/online_sim.py --mode tile2 --seeds 10
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "template"), str(ROOT)]

from evaluate import MODES                 # noqa: E402
from hcbots.bridge import Bridge           # noqa: E402
from online.client import Brain            # noqa: E402


def public(state):
    out = dict(state)
    out.pop("seed", None)
    return out


def play(brain, server, mode, seed, seat):
    map_, n, *rest = MODES[mode]
    opts = rest[0] if rest else {}         # x-modes: experimental nations (v111+)
    seats = [{"kind": "ki", "kiLevel": "schwer"} for _ in range(n)]
    seats[seat] = {"kind": "py"}
    placements = {}
    if map_ == "plaettchen":
        tp = server.call("tile_plan", players=seats, seed=seed, map=map_, **opts)
        for s in tp["seats"]:
            o, cell = brain.place(tp["plan"], s["idx"], {"civ": s["civ"], "ability": s["ability"]})
            placements[str(s["idx"])] = {"o": o, "cell": cell}
    gid = server.call("reset", players=seats, seed=seed, map=map_, placements=placements, **opts)["gid"]
    slot_of = [p["slot"] for p in server.call("state", gid=gid)["state"]["players"]]
    me = slot_of.index(seat)
    sent = [0]

    def latest():
        return public(server.call("state", gid=gid)["state"])

    def send(typ, params):
        err = server.call("act", gid=gid, type=typ, params=params)["err"]
        if err:
            raise RuntimeError(err)
        sent[0] += 1

    try:
        for _ in range(130):
            st = server.call("status", gid=gid)
            if st["over"] or st["round"] > 120:
                break
            assert st["cur"] == me, st
            if not brain.play(latest(), me, f"sim{seed}", send, latest):
                return None
        st = server.call("status", gid=gid)
        return {"win": me in st.get("winners", []), "how": st.get("how"), "round": st["round"], "sent": sent[0]}
    finally:
        server.close_game(gid)
        brain.forget(f"sim{seed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="tile2")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=1001)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    brain = Brain("rusher")
    wins = 0
    with Bridge() as server:
        for i in range(args.seeds):
            seed = args.seed0 + i
            r = play(brain, server, args.mode, seed, i % MODES[args.mode][1])
            wins += bool(r and r["win"])
            print(args.mode, seed, r, flush=True)
    print(f"{args.mode}: {wins}/{args.seeds}")
    brain.close()


if __name__ == "__main__":
    main()
