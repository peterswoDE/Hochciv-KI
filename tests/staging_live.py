"""End-to-end test of a live game on staging: a guest hosts a public lobby that allows
player-bots (the client's hosting path: starts when the lobby is full), the bot
(HOCHCIV_TOKEN in online-staging.env, else the password account there) picks it from the
public list, joins and plays; the host plays its own seat with the same engine. Staging only.

    python -m tests.staging_live [--map random] [--agent teacher]
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
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    env = read_env("online-staging.env")
    url = env["HOCHCIV_URL"]
    assert "staging" in url, "staging only"

    host = Manager(Account(url, name="KI-Testhost"), Brain("teacher"),
                   settings={"greeting": "", "live": {"mapKey": args.map, "playerCount": 2, "kiCount": 0, "turnTime": 0}})
    host.start()
    token = env.get("HOCHCIV_TOKEN")
    bot = Manager(Account(url, env.get("HOCHCIV_USER"), env.get("HOCHCIV_PASSWORD"), token=token,
                          name=None if token else "HochCiv-KI [Bot]"), Brain(args.agent))
    bot.acc.login()                  # no Rundenspiele/matchmaking connections in this test

    hg = host.start_live(host_cfg=dict(host.settings["live"]))
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
    log.info("host: %s (%s), bot: %s (%s)", hg.phase, hg.note, bg.phase, bg.note)
    raise SystemExit(0 if hg.phase == "over" and bg.phase == "over" else 1)


if __name__ == "__main__":
    main()
