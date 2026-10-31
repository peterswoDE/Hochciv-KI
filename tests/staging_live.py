"""End-to-end test of a live game on staging: a guest host opens a public lobby that allows
player-bots, the bot (HOCHCIV_TOKEN in online-staging.env, else the password account there)
finds it, joins and plays; the host plays its own seat with the same engine. Staging only.

    python -m tests.staging_live [--map plaettchen] [--agent teacher]
"""
from __future__ import annotations

import argparse
import logging
import queue
import secrets
import threading
import time

from hochciv_ki.online.client import Brain, Client, read_env

log = logging.getLogger("host")


class Host(Client):
    """A guest that hosts a lobby and plays its seat (no login)."""

    def login(self):
        pass

    def open_lobby(self, map_key, players):
        cid = secrets.token_hex(16)
        r = self.http.post(f"{self.url}/api/sessions", timeout=20, json={
            "config": {"mapKey": map_key, "playerCount": players, "kiCount": 0, "isPublic": True,
                       "playerBots": True},
            "host": {"name": "KI-Testhost", "clientId": cid, "civ": "random", "ability": "random"}})
        r.raise_for_status()
        b = r.json()
        self.enter_live(b["sessionId"], b["password"], cid)
        return b

    def enter_live(self, session_id, password, client_id):
        ack = self.call("session:connect", {"sessionId": session_id, "password": password, "clientId": client_id})
        self.live_game = {"sessionId": session_id, "password": password, "clientId": client_id,
                          "players": ack.get("players"), "cfg": ack.get("gameConfig") or {}, "lobbyIndex": ack.get("playerIndex")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="plaettchen")
    ap.add_argument("--agent", default="teacher")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    env = read_env("online-staging.env")
    url = env["HOCHCIV_URL"]
    assert "staging" in url, "staging only"

    host = Host(url, None, None, Brain("teacher"), rundenspiele=False)
    host.connect()
    lobby = host.open_lobby(args.map, 2)
    log.info("lobby %s open", lobby["joinCode"])

    token = env.get("HOCHCIV_TOKEN")
    bot = Client(url, env.get("HOCHCIV_USER"), env.get("HOCHCIV_PASSWORD"), Brain(args.agent), rundenspiele=False,
                 name=None if token else "HochCiv-KI [Bot]", token=token)
    result = {}
    t = threading.Thread(target=lambda: result.setdefault("bot", bot.run(once=True)), daemon=True)
    t.start()

    # wait for the bot, then start
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            ev, data = host.events.get(timeout=2)
        except queue.Empty:
            continue
        if ev == "player:joined" and len(data.get("players", [])) >= 2:
            break
        host.handle(ev, data)
    else:
        raise SystemExit("bot did not join")
    log.info("bot joined, starting")
    host.call("game:start", {})

    over = None
    deadline = time.time() + args.timeout
    while time.time() < deadline and t.is_alive():
        try:
            ev, data = host.events.get(timeout=2)
        except queue.Empty:
            continue
        if ev == "game:over":
            over = data
            break
        host.handle(ev, data)
    t.join(timeout=30)
    log.info("result: %s", (over or {}).get("how") if over else "no game:over")
    raise SystemExit(0 if over else 1)


if __name__ == "__main__":
    main()
