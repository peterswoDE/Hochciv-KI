"""End-to-end test of a Rundenspiel on staging: the password account in online-staging.env
hosts one (the client's "host" mode), the bot (HOCHCIV_TOKEN) joins it while its client is
not running. Once it is the bot's turn the bot's client is started and must play that turn
on launch; then both play to the end. Staging only.

    python -m tests.staging_async [--map plaettchen] [--agent teacher] [--timeout 1200]
"""
from __future__ import annotations

import argparse
import logging
import time

from hochciv_ki.online.client import Account, Brain, Manager, read_env

log = logging.getLogger("test")


def wait(what, fn, timeout, every=2):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = fn()
        if r:
            return r
        time.sleep(every)
    raise SystemExit(f"timed out: {what}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="plaettchen")
    ap.add_argument("--agent", default="teacher")
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    env = read_env("online-staging.env")
    url = env["HOCHCIV_URL"]
    assert "staging" in url, "staging only"
    assert env.get("HOCHCIV_TOKEN"), "needs HOCHCIV_TOKEN (a staging bot account)"

    brain = Brain("teacher")
    host = Manager(Account(url, env["HOCHCIV_USER"], env["HOCHCIV_PASSWORD"]), brain, settings={
        "mode": "host", "host_live": 0, "host_async": 1,
        "async": {"mapKey": args.map, "playerCount": 2, "kiCount": 0, "civMode": "random", "isPublic": True}})
    host.start()
    hacc = host.acc

    def hosted():
        mine = hacc.get("/api/async").json().get("mine", [])
        return next((s for s in mine if s.get("isHost") and s.get("freeSeats")), None)
    game = wait("hosted Rundenspiel", hosted, 120)
    gid = game["id"]
    host.update_settings({"mode": "select"})        # no second game after this one
    log.info("Rundenspiel %s (code %s) hosted", gid, game["joinCode"])

    # the bot joins with its client off
    bacc = Account(url, token=env["HOCHCIV_TOKEN"])
    bacc.login()
    r = bacc.post(f"/api/async/{gid}/join", json={"civ": "random", "ability": "random"})
    assert r.ok and not r.json().get("error"), r.text

    def bot_game():
        return next((s for s in bacc.get("/api/async").json().get("mine", []) if s["id"] == gid), None)
    # placement is simultaneous: wait until the host has placed and the bot is the one due
    def bots_turn():
        g = bot_game() or {}
        return g.get("yourTurn") and g.get("waitingFor") == "HochCiv-KI" and g
    g = wait("bot's turn", bots_turn, 300)
    log.info("bot's turn while its client is off: %s, round %s", g.get("status"), g.get("round"))

    t0 = time.time()
    bot = Manager(bacc, Brain(args.agent), settings={"mode": "select"})
    bot.start()
    g = wait("bot plays on launch", lambda: (lambda x: x if not x.get("yourTurn") or x.get("status") != g.get("status") or x.get("round") != g.get("round") else None)(bot_game() or {}), 120, every=1)
    log.info("bot played %.0f s after launch -> %s, round %s", time.time() - t0, g.get("status"), g.get("round"))

    def finished():
        g = bot_game()
        return g is None or g.get("status") == "finished"
    try:
        wait("game over", finished, args.timeout, every=10)
    except SystemExit:
        g = bot_game() or {}
        log.error("not finished: status %s, round %s, waiting for %s", g.get("status"), g.get("round"), g.get("waitingFor"))
        hacc.post(f"/api/async/{gid}/cancel")
        raise
    log.info("Rundenspiel %s finished", gid)


if __name__ == "__main__":
    main()
