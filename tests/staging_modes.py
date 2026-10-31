"""End-to-end test of the settings switches on staging, through the managers' own loops: the
bot (HOCHCIV_TOKEN) hosts two ranked live lobbies at once from one host setup, the password
account has "join games" on and must find and join one by itself; both play to the end and
the bot's MMR changes. Staging only.

    python -m tests.staging_modes [--agent teacher] [--timeout 600]
"""
from __future__ import annotations

import argparse
import logging
import threading
import time

from hochciv_ki.online.client import Account, Brain, Manager, read_env

log = logging.getLogger("test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="teacher")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    env = read_env("online-staging.env")
    url = env["HOCHCIV_URL"]
    assert "staging" in url and env.get("HOCHCIV_TOKEN"), "staging with a bot token only"

    bot = Manager(Account(url, token=env["HOCHCIV_TOKEN"]), Brain(args.agent), settings={"host": {"on": True, "setups": [
        {"id": "t1", "kind": "live", "count": 2, "mapKey": "random", "playerCount": 2, "kiCount": 0, "turnTime": 0,
         "ranked": True}]}})
    human = Manager(Account(url, env["HOCHCIV_USER"], env["HOCHCIV_PASSWORD"]), Brain("teacher"), settings={
        "greeting": "", "join": {"on": True, "live": True, "max_live": 1, "async": False, "maps": ["tiles"], "max_empires": 2}})
    bot.acc.login()
    mmr0 = bot.acc.get("/api/auth/me").json().get("mmr")

    threading.Thread(target=bot.run, daemon=True).start()
    deadline = time.time() + 120
    while sum(1 for g in bot.active() if g.kind == "host" and g.phase == "lobby") < 2 and time.time() < deadline:
        time.sleep(1)
    hosted = [g for g in bot.active() if g.kind == "host" and g.phase == "lobby"]
    assert len(hosted) == 2, f"two hosted lobbies expected, got {[g.join_code for g in hosted]}"
    assert all(g.g["cfg"].get("ranked") for g in hosted), "lobbies are not ranked"
    log.info("bot hosts two ranked lobbies: %s", ", ".join(g.join_code for g in hosted))
    threading.Thread(target=human.run, daemon=True).start()

    def played():
        jg = next((g for g in human.live if g.kind == "join"), None)
        hg = jg and jg.g and next((g for g in bot.live if g.kind == "host" and g.g and g.g["sessionId"] == jg.g["sessionId"]), None)
        return hg, jg
    deadline = time.time() + args.timeout
    hg = jg = None
    while time.time() < deadline:
        hg, jg = played()
        if hg and jg and hg.done.is_set() and jg.done.is_set():
            break
        time.sleep(1)
    time.sleep(3)                    # the server rates the game after game:over
    mmr1 = bot.acc.get("/api/auth/me").json().get("mmr")
    log.info("bot hosted: %s (%s), human joined: %s (%s), bot MMR %s -> %s", hg and hg.phase, hg and hg.note,
             jg and jg.phase, jg and jg.note, mmr0, mmr1)
    raise SystemExit(0 if hg and jg and hg.phase == jg.phase == "over" and mmr1 != mmr0 else 1)


if __name__ == "__main__":
    main()
