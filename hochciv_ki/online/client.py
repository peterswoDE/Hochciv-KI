"""Online client: plays live lobbies and Rundenspiele on a HochCiv multiplayer server.

    python -m hochciv_ki.online.client --config online.env --agent ckpt:checkpoints/ppo/best.pt
    python -m hochciv_ki.online.client --config online.env --agent teacher --once

online.env (git-ignored, KEY=VALUE):
    HOCHCIV_URL=https://staging.hochciv.de
    HOCHCIV_USER=...
    HOCHCIV_PASSWORD=...

What it does, in a loop:
  * Rundenspiele: joins open public games that fit (2–4 empires, no events/wonders/marathon,
    current engine version) and plays every game where it is its turn.
  * Live lobbies: joins an open public lobby that fits, says in the chat that it is an AI,
    places its start tile and plays the game to the end. One live game at a time.
  * Matchmaking: only with --matchmaking (off until the server has a bot account type).

Every move is computed by the local bridge worker on the server's state: the candidate the
agent picks is recorded as engine calls and sent as `action { type, params }`.
"""
from __future__ import annotations

import argparse
import logging
import queue
import secrets
import threading
import time
from pathlib import Path

import requests
import socketio

from ..bridge import Bridge
from ..env import decode_obs
from ..eval import agent_factory

log = logging.getLogger("hochciv-ki")

GREETING = {
    "de": "Hallo! Ich bin HochCiv-KI, ein neuronales Netz (Bot). Viel Spaß!",
}
ALLOWED_MAPS = {"plaettchen", "random", "0", "original", "gross", "1", "eigene"}
MAX_PLAYERS = 4


def read_env(path):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def config_ok(cfg: dict, n_players=None) -> str | None:
    """Why a game configuration is outside what the AI plays (None = fine)."""
    if cfg.get("events"):
        return "events"
    if cfg.get("wonders"):
        return "wonders"
    if cfg.get("marathon") or cfg.get("mapKey") == "marathon":
        return "marathon"
    if (cfg.get("customMap") or {}).get("marathon"):
        return "marathon map"
    n = n_players or cfg.get("playerCount") or cfg.get("humanCount")
    if n and n > MAX_PLAYERS:
        return f"{n} empires"
    return None


class Brain:
    """The local engine copy plus the agent. One turn = repeated (load, observe, act)."""

    def __init__(self, agent_spec, engine_dir=None):
        self.bridge = Bridge(engine_dir)
        info = self.bridge.info()
        self.version, self.dims = info["version"], info["dims"]
        self.agent = agent_factory(agent_spec)()

    def next_actions(self, state, pi):
        """Server actions for the next decision of seat `pi` (ends with endTurn when the turn
        is over)."""
        self.bridge.call("load", gid=1, state=state, pi=pi)
        obs = decode_obs(self.bridge.call("observe", gid=1), self.dims)
        a, _ = self.agent.act(obs)
        res = self.bridge.call("step_online", gid=1, i=a)
        return obs.tags[a], [tuple(x) for x in res["actions"]]

    def place(self, plan, seat_idx, players, setup, seed):
        r = self.bridge.call("place", plan=plan, seatIdx=seat_idx, players=players, setup=setup, seed=seed)
        return r["o"], r["cell"]

    def roll_setup(self, players, seed, cfg):
        return self.bridge.call("rollSetup", cfg={"players": players, "seed": seed, "wonders": bool(cfg.get("wonders")),
                                                  "events": bool(cfg.get("events")), "eventMode": cfg.get("eventMode", "hard")})["setup"]


class Client:
    def __init__(self, url, user, password, brain: Brain, live=True, rundenspiele=True, matchmaking=False,
                 name=None):
        self.url = url.rstrip("/")
        self.user, self.password = user, password
        self.brain = brain
        self.live, self.async_on, self.matchmaking = live, rundenspiele, matchmaking
        self.name = name or user
        self.http = requests.Session()
        self.events: queue.Queue = queue.Queue()
        self.sio = None
        self.live_game = None        # {sessionId, password, clientId, yourIndex, players, cfg}
        self.stash = []              # events put aside while waiting for an update
        self.async_open = None       # id of the opened Rundenspiel
        self.last_async_poll = 0.0
        self.async_every = 60
        self.server_version = None

    # ---------------------------------------------------------------- connection
    def login(self):
        r = self.http.post(f"{self.url}/api/auth/login", json={"username": self.user, "password": self.password}, timeout=20)
        r.raise_for_status()
        body = r.json()
        if not body.get("ok", True) or body.get("mfa") or body.get("needs"):
            raise RuntimeError(f"login needs more than a password: {body}")
        me = self.http.get(f"{self.url}/api/auth/me", timeout=20).json()
        log.info("logged in as %s", (me.get("user") or me).get("username", self.user))

    def connect(self):
        cookie = "; ".join(f"{c.name}={c.value}" for c in self.http.cookies)
        self.sio = socketio.Client(reconnection=True, logger=False, engineio_logger=False)
        for ev in ("game:start", "state:update", "placement:start", "game:over", "session:kicked",
                   "async:update", "match:found", "match:ready", "match:cancelled", "turn:timeout",
                   "lobby:player:updated", "lobby:config:update", "session:rematch", "player:joined"):
            self.sio.on(ev, (lambda e: lambda data=None: self.events.put((e, data)))(ev))
        self.sio.on("disconnect", lambda *a: self.events.put(("disconnect", None)))
        self.sio.on("connect", lambda: self.events.put(("connect", None)))
        self.sio.connect(self.url, headers={"Cookie": cookie, "Origin": self.url}, transports=["polling", "websocket"],
                         wait_timeout=20)

    def call(self, event, data, timeout=30):
        res = self.sio.call(event, data, timeout=timeout)
        if isinstance(res, dict) and res.get("error"):
            raise RuntimeError(f"{event}: {res['error']}")
        return res

    # ---------------------------------------------------------------- main loop
    def run(self, once=False):
        self.login()
        self.connect()
        if self.matchmaking:
            self.call("match:join", {})
        last_lobby_poll = 0.0
        while True:
            try:
                ev, data = self.stash.pop(0) if self.stash else self.events.get(timeout=2)
                self.handle(ev, data)
            except queue.Empty:
                pass
            now = time.time()
            if self.live and not self.live_game and not self.matchmaking and now - last_lobby_poll > 15:
                last_lobby_poll = now
                self.guarded(self.try_join_lobby)
            if self.async_on and now - self.last_async_poll > self.async_every:
                self.last_async_poll = now
                self.guarded(self.poll_rundenspiele)
            if once and self.live_game is None and self._played_live:
                return

    _played_live = False

    def handle(self, ev, data):
        if ev in ("game:start", "state:update") and self.live_game:
            if ev == "game:start":
                self.live_game["yourIndex"] = data["yourIndex"]
            self.live_game["state"] = data["state"]
            self.drain()                      # act on the newest state only
            self.guarded(self.live_turn)
        elif ev == "placement:start" and self.live_game:
            self.guarded(self.live_place, data)
        elif ev == "lobby:player:updated" and self.live_game:
            self.live_game["players"] = data if isinstance(data, list) else data.get("players", data)
        elif ev == "lobby:config:update" and self.live_game:
            self.live_game["cfg"] = data
            why = config_ok(data)
            if why:
                self.leave_live(f"lobby changed to {why}")
        elif ev in ("game:over", "session:kicked"):
            if self.live_game:
                log.info("live game ended: %s", (data or {}).get("how") or (data or {}).get("reason") or ev)
                self.leave_live(None)
                self._played_live = True
        elif ev == "async:update" and data:
            self._async_view = data
            self.drain()
            self.guarded(self.async_turn, self._async_view)
        elif ev == "match:found":
            self.call("match:accept", {"matchId": data["matchId"]})
        elif ev == "match:ready":
            self.enter_live(data["sessionId"], data["password"], data["clientId"])
        elif ev == "connect" and self.live_game:
            # reconnect: bind the seat again
            self.call("session:connect", {"sessionId": self.live_game["sessionId"], "password": self.live_game["password"],
                                          "clientId": self.live_game["clientId"]})
        elif ev == "turn:timeout":
            log.warning("turn timed out")

    def guarded(self, fn, *args):
        try:
            fn(*args)
        except Exception:  # noqa: BLE001 - one bad turn must not stop the client
            log.exception("%s failed", fn.__name__)

    # ---------------------------------------------------------------- live lobbies
    def try_join_lobby(self):
        try:
            lst = self.http.get(f"{self.url}/api/public-sessions", timeout=20).json()
        except Exception as e:  # noqa: BLE001
            log.warning("public-sessions: %s", e)
            return
        for s in lst:
            if s.get("status") != "lobby" or s.get("playersCount", 0) >= s.get("maxPlayers", 0):
                continue
            if str(s.get("mapKey")) not in ALLOWED_MAPS:
                continue
            client_id = secrets.token_hex(16)
            try:
                r = self.http.post(f"{self.url}/api/sessions/join", timeout=20, json={
                    "joinCode": s["joinCode"], "password": s["password"],
                    "player": {"name": self.name, "clientId": client_id, "civ": "random", "ability": "random"}})
                if r.status_code != 200:
                    continue
                self.enter_live(r.json()["sessionId"], s["password"], client_id)
                return
            except Exception as e:  # noqa: BLE001
                log.warning("join %s: %s", s.get("joinCode"), e)

    def enter_live(self, session_id, password, client_id):
        ack = self.call("session:connect", {"sessionId": session_id, "password": password, "clientId": client_id})
        self.live_game = {"sessionId": session_id, "password": password, "clientId": client_id,
                          "players": ack.get("players"), "cfg": ack.get("gameConfig") or {}, "lobbyIndex": ack.get("playerIndex")}
        why = config_ok(self.live_game["cfg"])
        if why:
            self.leave_live(f"lobby is {why}")
            return
        log.info("joined lobby %s (seat %s)", session_id, ack.get("playerIndex"))
        try:
            self.call("chat:send", {"kind": "text", "text": GREETING["de"]})
        except Exception as e:  # noqa: BLE001
            log.info("greeting not sent: %s", e)

    def leave_live(self, why):
        if why:
            log.info("leaving lobby: %s", why)
        self.live_game = None
        # leaving = dropping the socket (the server removes a disconnected lobby player)
        try:
            self.sio.disconnect()
        except Exception:  # noqa: BLE001
            pass
        self.connect()

    def live_place(self, data):
        g = self.live_game
        me = g["lobbyIndex"]
        if me not in [s["idx"] for s in data.get("queue", [])] and me not in data.get("queue", []):
            return
        players = [{"civ": p.get("civ"), "kind": p.get("kind", "human"), "ability": p.get("ability", "basis")}
                   for p in (g.get("players") or [])]
        cfg = data.get("cfg") or g["cfg"]
        setup = self.brain.roll_setup(players, data["seed"], cfg)
        o, cell = self.brain.place(data["plan"], me, players, setup, data["seed"])
        self.call("placement:action", {"o": o, "cell": cell})
        log.info("placed start tile o=%s cell=%s", o, cell)

    def live_turn(self):
        g = self.live_game
        S, me = g["state"], g.get("yourIndex", -1)
        if S.get("over") or me is None or me < 0 or S["cur"] != me:
            return
        def send(typ, params):
            # live: the server acks, then broadcasts exactly one state:update (none on errors)
            self.call("action", {"type": typ, "params": params})
            self.wait_event("state:update")
        self.play_turn(S, me, send, lambda: self.live_game and self.live_game["state"])

    # ---------------------------------------------------------------- shared turn logic
    def play_turn(self, state, me, send, latest):
        """Decide and send actions until the turn is ended. `send(type, params)` raises on a
        rejected action; `latest()` returns the newest server state."""
        rejected, sent, tags = 0, 0, []
        for _ in range(80):
            tag, actions = self.brain.next_actions(state, me)
            tags.append(tag)
            ended = False
            for typ, params in actions:
                try:
                    send(typ, params)
                    sent += 1
                except RuntimeError as e:
                    rejected += 1
                    log.warning("rejected %s %s (%s): %s", typ, params, tag, e)
                    break
                ended = typ == "endTurn"
            if ended:
                log.info("%s: round %s, %d actions: %s", self.name, state.get("round"), sent, " ".join(tags))
                return
            if rejected >= 5:
                log.warning("too many rejected actions, ending the turn")
                send("endTurn", {})
                return
            state = latest() or state
            if state.get("over") or state["cur"] != me:
                return
        send("endTurn", {})

    def take_update(self, ev, data):
        """Apply a state update without starting a turn. True if it was one."""
        if ev == "state:update" and self.live_game:
            self.live_game["state"] = data["state"]
            return True
        if ev == "async:update" and data and self.async_open == data.get("id"):
            self._async_view = data
            return True
        return False

    def drain(self):
        """Take all queued state updates (the newest wins); other events wait in the stash."""
        while True:
            try:
                ev, data = self.events.get_nowait()
            except queue.Empty:
                return
            if not self.take_update(ev, data):
                self.stash.append((ev, data))

    def wait_event(self, kind, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                ev, data = self.events.get(timeout=max(0.01, deadline - time.time()))
            except queue.Empty:
                break
            if ev == kind and self.take_update(ev, data):
                return
            if not self.take_update(ev, data):
                self.stash.append((ev, data))
        log.warning("no %s within %ss", kind, timeout)

    # ---------------------------------------------------------------- Rundenspiele
    def poll_rundenspiele(self):
        try:
            lst = self.http.get(f"{self.url}/api/async", timeout=20).json()
        except Exception as e:  # noqa: BLE001
            log.warning("async list: %s", e)
            return
        for s in lst.get("open", []):
            if s.get("engine") and s["engine"] != self.brain.version and s["engine"] != "current":
                continue
            if config_ok(s.get("config") or {}, len(s.get("seats") or [])) or not s.get("freeSeats"):
                continue
            r = self.http.post(f"{self.url}/api/async/{s['id']}/join", json={"civ": "random", "ability": "random"}, timeout=20)
            if r.ok:
                log.info("joined Rundenspiel %s", s["id"])
        for s in lst.get("mine", []):
            if s.get("yourTurn") or s.get("status") == "placement":
                self.open_async(s["id"])

    def open_async(self, gid):
        res = self.call("async:open", {"id": gid})
        self.async_open = gid
        self.async_turn(res["view"])

    def async_turn(self, view):
        gid = view.get("id")
        if not gid:
            return
        if view.get("status") == "placement" and view.get("placement") and not view["placement"].get("placed"):
            pl = view["placement"]
            o, cell = self.brain.place(pl["plan"], view["mySlot"], pl["players"], pl.get("setup"), 1)
            self.call("async:place", {"id": gid, "o": o, "cell": cell})
            log.info("Rundenspiel %s: placed start tile", gid)
            return
        S, me = view.get("state"), view.get("myIndex", -1)
        if view.get("status") != "playing" or not S or S.get("over") or me < 0 or S["cur"] != me or view.get("hold"):
            return
        self._async_view = view
        def send(typ, params):
            # Rundenspiele broadcast the new view before the ack: it is already queued
            self.call("async:action", {"id": gid, "type": typ, "params": params})
            self.drain()
        self.play_turn(S, me, send, lambda: (self._async_view or {}).get("state"))
        log.info("Rundenspiel %s: turn played (round %s)", gid, S.get("round"))

    _async_view = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="online.env")
    ap.add_argument("--agent", default="teacher")
    ap.add_argument("--no-live", action="store_true")
    ap.add_argument("--no-rundenspiele", action="store_true")
    ap.add_argument("--matchmaking", action="store_true",
                    help="enter ranked matchmaking (keep off until the bot account type exists)")
    ap.add_argument("--once", action="store_true", help="stop after one live game")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    env = read_env(args.config)
    brain = Brain(args.agent)
    log.info("engine %s, agent %s", brain.version, args.agent)
    Client(env["HOCHCIV_URL"], env["HOCHCIV_USER"], env["HOCHCIV_PASSWORD"], brain,
           live=not args.no_live, rundenspiele=not args.no_rundenspiele, matchmaking=args.matchmaking,
           name=env.get("HOCHCIV_NAME")).run(once=args.once)


if __name__ == "__main__":
    threading.current_thread().name = "main"
    main()
