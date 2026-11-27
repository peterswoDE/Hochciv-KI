"""Offline games: Python bots against the JS KI (default level "schwer") on the local engine."""
from __future__ import annotations

import time
import traceback

from .bridge import Bridge
from .turn import Turn

MAX_ROUNDS = 120


def free_blocked_armies(t: Turn) -> None:
    """endTurn is refused while an own army stands in a city and could leave: move it out,
    to the reachable tile nearest to where it stands (the bot should have done this)."""
    for a in t.armies():
        if not t.city_at(a["r"], a["c"]):
            continue
        reach = t.q("armyReach", "$S", {"$army": a["id"]}) or []
        if reach:
            k, _ = min(reach, key=lambda kv: kv[1])
            r, c = map(int, k.split(","))
            t.act("moveArmy", armyId=a["id"], r=r, c=c)


def end_turn(t: Turn) -> str | None:
    err = t.act("endTurn")
    if err:
        free_blocked_armies(t)
        err = t.act("endTurn")
    return err


def play_game(bridge: Bridge, bots: dict, seats: list[dict], seed: int, map_: str = "plaettchen",
              verbose: bool = False, on_end=None, **opts) -> dict:
    """bots: {seat index in `seats`: bot}. seats: [{kind: 'py'|'ki'|'bot', civ?, kiLevel?}].
    on_end(bridge, gid): called when the game is over, before it is closed (debugging: logs).
    Returns {winners (seat indices), how, rounds, seconds, errors}."""
    t0 = time.time()
    placements = {}
    if map_ in ("plaettchen", "plaettchen-hex"):
        # same options as reset: with expCivs the nations drawn depend on them
        tp = bridge.call("tile_plan", players=seats, seed=seed, map=map_, **opts)
        for s in tp["seats"]:
            bot = bots[s["idx"]]
            choice = bot.place(s, tp["plan"]) if hasattr(bot, "place") else None
            if choice:
                placements[str(s["idx"])] = choice
    res = bridge.call("reset", players=seats, seed=seed, map=map_, placements=placements, **opts)
    gid = res["gid"]
    st = bridge.call("state", gid=gid)
    slot_of = [p["slot"] for p in st["state"]["players"]]     # state index -> seat index
    errors = refused = 0
    try:
        while not res["over"] and res["round"] <= MAX_ROUNDS:
            t = Turn(bridge, gid)
            bot = bots[slot_of[t.pi]]
            try:
                bot.turn(t)
            except Exception:
                errors += 1
                if verbose or errors <= 3:
                    traceback.print_exc()
            refused += len(t.errors)
            if verbose and t.errors:
                for e in t.errors:
                    print("  refused:", e)
            if t.over:
                res = bridge.call("status", gid=gid)
                break
            end_turn(t)
            res = bridge.call("status", gid=gid)
        if on_end:
            on_end(bridge, gid)
        winners = [slot_of[w] for w in res.get("winners", [])] if res["over"] else []
        return {"winners": winners, "how": res.get("how"), "rounds": res["round"],
                "seconds": round(time.time() - t0, 1), "errors": errors, "refused": refused, "seed": seed, "map": map_,
                "n": len(seats)}
    finally:
        bridge.close_game(gid)
