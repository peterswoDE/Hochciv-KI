"""End-to-end test of a live game on staging: a guest hosts a public lobby that allows
player-bots (the client's hosting path: starts when the lobby is full), the bot
(HOCHCIV_TOKEN in online-staging.env, else the password account there) picks it from the
public list, joins and plays; the host plays its own seat with the same engine. Staging only.

    python -m tests.staging_live [--map random] [--agent teacher]
    python -m tests.staging_live --map marathon --players 4 --ki 2 --wonders --events hard
"""
from __future__ import annotations

import argparse
import logging
import time

from hochciv_ki.online.client import Account, Brain, Manager, read_env

log = logging.getLogger("test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="random")
    ap.add_argument("--agent", default="teacher")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--players", type=int, default=2, help="empires")
    ap.add_argument("--ki", type=int, default=0, help="KI seats")
    ap.add_argument("--wonders", action="store_true")
    ap.add_argument("--events", choices=["off", "easy", "hard"], default="off")
    ap.add_argument("--drop-at", type=int, default=0, help="drop the bot's connection in this round (reconnect test)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    env = read_env("online-staging.env")
    url = env["HOCHCIV_URL"]
    assert "staging" in url, "staging only"

    host = Manager(Account(url, name="KI-Testhost"), Brain("teacher"),
                   settings={"greeting": "", "host": {"setups": [
                       {"id": "test", "kind": "live", "mapKey": args.map, "playerCount": args.players, "kiCount": args.ki,
                        "turnTime": 0, "wonders": args.wonders, "events": args.events}]}})
    host.start()
    token = env.get("HOCHCIV_TOKEN")
    bot = Manager(Account(url, env.get("HOCHCIV_USER"), env.get("HOCHCIV_PASSWORD"), token=token,
                          name=None if token else "HochCiv-KI [Bot]"), Brain(args.agent))
    bot.acc.login()                  # no Rundenspiele/matchmaking connections in this test

    hg = host.start_live(host_cfg=dict(host.settings["host"]["setups"][0]))
    deadline = time.time() + 30
    while not (hg.g and hg.g.get("sessionId")) and time.time() < deadline and not hg.done.is_set():
        time.sleep(0.5)
    assert hg.g, f"lobby not opened: {hg.note}"
    log.info("lobby %s open", hg.join_code)

    lobby = next((s for s in bot.game_lists()["lobbies"] if s["joinCode"] == hg.join_code), None)
    assert lobby, "lobby not in the public list"
    assert not lobby["why"], f"lobby does not fit: {lobby['why']}"
    bot.join("live", lobby["id"])
    bg = bot.live[-1]

    deadline = time.time() + args.timeout
    while time.time() < deadline and not (hg.done.is_set() and bg.done.is_set()):
        time.sleep(1)
        S = (bg.g or {}).get("state") or {}
        if args.drop_at and S.get("round", 0) >= args.drop_at:
            args.drop_at = 0
            # the next poll goes to an unknown engine.io session: the server answers 400 and
            # socket.io reconnects by itself, as after a timed-out transport
            log.info("dropping the bot's connection")
            e = bg.sio.eio
            e.base_url = e.base_url.replace(f"sid={e.sid}", "sid=gone")
    log.info("host: %s (%s), bot: %s (%s)", hg.phase, hg.note, bg.phase, bg.note)
    raise SystemExit(0 if hg.phase == "over" and bg.phase == "over" else 1)


if __name__ == "__main__":
    main()
