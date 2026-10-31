"""Online client: plays live games and Rundenspiele on a HochCiv multiplayer server, steered
from a small local control panel (panel.py).

    python -m hochciv_ki.online.client --config online.env --agent ckpt:checkpoints/v1.pt
    -> control panel on http://127.0.0.1:8765

online.env (git-ignored, KEY=VALUE):
    HOCHCIV_URL=https://hochciv.de
    HOCHCIV_TOKEN=hcbot_...        # bot account (account page, tab "Meine Bots")
or, for a regular account (only where bot accounts don't exist yet):
    HOCHCIV_USER=...
    HOCHCIV_PASSWORD=...

A bot account plays as a player-bot (multiplayer-server docs/reference/bot-accounts.md): it
only joins lobbies and Rundenspiele whose host allows player-bots and that have room for one,
and plays under its account name.

What it does is set in the panel (saved in runs/online-settings.json); the switches work
together:
  * join  – join public lobbies (up to N live games at once) and open Rundenspiele (up to M)
            that match the join filters (maps, empires, KI seats, ranked, turn timer, bot hosts)
  * host  – a list of setups (live lobby or Rundenspiel, map, empires, KI, ...), each kept
            open/running so many times
  * match – ranked matchmaking (bot accounts only: humans opt in to play bots), one game at
            a time, back in the queue after every game
  * games picked in the panel's game list are joined whatever the filters say
The bot always plays every game it is in. It never plays events, wonders, marathon, more
than 4 empires, or Rundenspiele of an older engine (config_ok).

Every move is computed by the local bridge worker on the server's state: the candidate the
agent picks is recorded as engine calls and sent as `action { type, params }`. Each live game
has its own socket (the server binds a socket to one game); Rundenspiele share one socket and
matchmaking has one. All of them share one Brain.
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import os
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

GREETING = "Hallo! Ich bin HochCiv-KI, ein neuronales Netz (Bot). Viel Spaß!"
MAX_PLAYERS = 4
# map keys of live lobbies and Rundenspiele -> the map groups in the settings
MAP_GROUPS = {"0": "original", "1": "original", "original": "original", "gross": "gross",
              "random": "tiles", "plaettchen": "tiles", "eigene": "custom"}
HOST_MAPS = {"live": ("random", "0", "gross"), "async": ("plaettchen", "0", "gross")}
TURN_TIMES = (0, 60, 120, 180, 300, 600)
KI_LEVELS = ("leicht", "mittel", "schwer")
MAX_SETUPS = 12

SETUP = {"kind": "live", "count": 1, "mapKey": "random", "playerCount": 2, "kiCount": 0, "difficulty": "schwer",
         "turnTime": 180, "randomCivs": False, "civMode": "random", "isPublic": True, "playerBotLimit": None,
         "ranked": False}
DEFAULTS = {
    "version": 2,
    "join": {
        "on": False,
        "live": True, "max_live": 1,         # live games joined at once
        "async": True, "max_async": 5,       # Rundenspiele joined at once
        "maps": ["tiles", "original", "gross", "custom"],
        "min_empires": 2, "max_empires": MAX_PLAYERS,
        "ki": True,                          # games with KI seats
        "ranked": "any",                     # any | ranked | unranked
        "timer": False,                      # live: only lobbies with a turn timer
        "bot_hosts": True,                   # games hosted by other bots
    },
    "host": {
        "on": False,
        "setups": [dict(SETUP, id="live1"), dict(SETUP, id="async1", kind="async", mapKey="plaettchen", count=0)],
        "start_after": 5,                    # live: start after so many minutes ...
        "min_players": 2,                    # ... with at least so many players
    },
    "match": False,                          # ranked matchmaking (bot accounts)
    "greeting": GREETING,
}


def read_env(path):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def fingerprint(state) -> int:
    """Changes with every applied action (resources, positions, logs, ...)."""
    return hash(json.dumps(state, sort_keys=True))


def config_ok(cfg: dict, n_players=None, max_empires=MAX_PLAYERS) -> str | None:
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
    if n and n > max_empires:
        return f"{n} empires"
    return None


def join_filter(j: dict, cfg: dict, empires=None, ki=None, host_is_bot=False, live=True) -> str | None:
    """Why a game doesn't match the join filters of the settings (None = it does). Only for
    joining on its own: a game picked in the panel just has to be playable (config_ok)."""
    if MAP_GROUPS.get(str(cfg.get("mapKey"))) not in j["maps"]:
        return f"map {cfg.get('mapKey')}"
    n = empires or cfg.get("playerCount")
    if n and not j["min_empires"] <= n <= j["max_empires"]:
        return f"{n} empires"
    if ki is None:
        ki = cfg.get("kiCount") or 0
    if ki and not j["ki"]:
        return "KI seats"
    if j["ranked"] != "any" and bool(cfg.get("ranked")) != (j["ranked"] == "ranked"):
        return "ranked" if cfg.get("ranked") else "unranked"
    if live and j["timer"] and not cfg.get("turnTime"):
        return "no turn timer"
    if host_is_bot and not j["bot_hosts"]:
        return "hosted by a bot"
    return None


def _num(v, lo, hi, default):
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


def clean_setup(raw: dict, old: dict | None = None) -> dict:
    s = dict(old or SETUP)
    s["id"] = str(raw.get("id") or s.get("id") or secrets.token_hex(3))[:12]
    if raw.get("kind") in HOST_MAPS:
        s["kind"] = raw["kind"]
    if raw.get("mapKey") in HOST_MAPS[s["kind"]]:
        s["mapKey"] = raw["mapKey"]
    if s["mapKey"] not in HOST_MAPS[s["kind"]]:
        s["mapKey"] = HOST_MAPS[s["kind"]][0]
    s["count"] = _num(raw.get("count", s["count"]), 0, 30 if s["kind"] == "async" else 8, s["count"])
    s["playerCount"] = _num(raw.get("playerCount", s["playerCount"]), 2, MAX_PLAYERS, s["playerCount"])
    s["kiCount"] = min(_num(raw.get("kiCount", s["kiCount"]), 0, MAX_PLAYERS, 0), s["playerCount"] - 2)  # two humans
    if raw.get("difficulty") in KI_LEVELS:
        s["difficulty"] = raw["difficulty"]
    if _num(raw.get("turnTime"), 0, 600, -1) in TURN_TIMES:
        s["turnTime"] = int(raw["turnTime"])
    if raw.get("civMode") in ("random", "fixed"):
        s["civMode"] = raw["civMode"]
    for k in ("randomCivs", "isPublic", "ranked"):
        if k in raw:
            s[k] = bool(raw[k])
    if s["ranked"]:
        s["difficulty"] = KI_LEVELS[-1]      # ranked: the KI plays on the hardest level (as the server does)
    if "playerBotLimit" in raw:
        v = raw["playerBotLimit"]
        s["playerBotLimit"] = None if v in (None, "", 0, "0") else _num(v, 1, 8, None)
    return s


def clean_settings(raw: dict, old: dict | None = None) -> dict:
    """Merge `raw` over `old` (or the defaults) and keep every value in range. `raw` may be
    partial; a given `host.setups` replaces the whole list."""
    s = json.loads(json.dumps(old if old and old.get("version") == DEFAULTS["version"] else DEFAULTS))
    raw = raw or {}
    j, rj = s["join"], raw.get("join") or {}
    for k in ("on", "live", "async", "ki", "timer", "bot_hosts"):
        if k in rj:
            j[k] = bool(rj[k])
    j["max_live"] = _num(rj.get("max_live", j["max_live"]), 0, 8, j["max_live"])
    j["max_async"] = _num(rj.get("max_async", j["max_async"]), 0, 30, j["max_async"])
    j["min_empires"] = _num(rj.get("min_empires", j["min_empires"]), 2, MAX_PLAYERS, j["min_empires"])
    j["max_empires"] = _num(rj.get("max_empires", j["max_empires"]), j["min_empires"], MAX_PLAYERS, j["max_empires"])
    if isinstance(rj.get("maps"), list):
        j["maps"] = [m for m in DEFAULTS["join"]["maps"] if m in rj["maps"]]
    if rj.get("ranked") in ("any", "ranked", "unranked"):
        j["ranked"] = rj["ranked"]
    h, rh = s["host"], raw.get("host") or {}
    if "on" in rh:
        h["on"] = bool(rh["on"])
    h["start_after"] = _num(rh.get("start_after", h["start_after"]), 0, 120, h["start_after"])
    h["min_players"] = _num(rh.get("min_players", h["min_players"]), 2, MAX_PLAYERS, h["min_players"])
    if isinstance(rh.get("setups"), list):
        olds = {x["id"]: x for x in h["setups"]}
        h["setups"] = [clean_setup(x, olds.get(x.get("id"))) for x in rh["setups"][:MAX_SETUPS] if isinstance(x, dict)]
    if "match" in raw:
        s["match"] = bool(raw["match"])
    if isinstance(raw.get("greeting"), str):
        s["greeting"] = raw["greeting"].strip()[:300]
    return s


def setup_key(cfg: dict):
    """What tells hosted Rundenspiele of one setup apart from those of another."""
    n = cfg.get("playerCount")
    humans = cfg.get("humanCount") if cfg.get("humanCount") is not None else n - (cfg.get("kiCount") or 0)
    return str(cfg.get("mapKey")), n, humans, bool(cfg.get("ranked"))


class Tagged(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"[{self.extra['tag']}] {msg}", kwargs


class Brain:
    """The local engine copy plus the agent. One turn = repeated (load, observe, act). Shared by
    all games: one decision at a time."""

    def __init__(self, agent_spec, engine_dir=None):
        self.bridge = Bridge(engine_dir)
        info = self.bridge.info()
        self.version, self.dims = info["version"], info["dims"]
        self.agent = agent_factory(agent_spec)()
        self.spec = agent_spec
        self.lock = threading.Lock()

    def next_actions(self, state, pi):
        """Server actions for the next decision of seat `pi` (ends with endTurn when the turn
        is over)."""
        with self.lock:
            self.bridge.call("load", gid=1, state=state, pi=pi)
            obs = decode_obs(self.bridge.call("observe", gid=1), self.dims)
            a, _ = self.agent.act(obs)
            res = self.bridge.call("step_online", gid=1, i=a)
        return obs.tags[a], [tuple(x) for x in res["actions"]]

    def place(self, plan, seat_idx, players, setup, seed):
        with self.lock:
            r = self.bridge.call("place", plan=plan, seatIdx=seat_idx, players=players, setup=setup, seed=seed)
        return r["o"], r["cell"]

    def roll_setup(self, players, seed, cfg):
        with self.lock:
            return self.bridge.call("rollSetup", cfg={"players": players, "seed": seed, "wonders": bool(cfg.get("wonders")),
                                                      "events": bool(cfg.get("events")), "eventMode": cfg.get("eventMode", "hard")})["setup"]


class Account:
    """The login (bot token, password, or none = guest) and its REST session."""

    def __init__(self, url, user=None, password=None, token=None, name=None):
        self.url = url.rstrip("/")
        self.user, self.password, self.token = user, password, token
        self.name = name
        self.is_bot = False          # logged in as a bot account (set by login)
        self.http = requests.Session()
        self.lock = threading.Lock()  # the cookie jar is shared by all threads

    def get(self, path, **kw):
        with self.lock:
            return self.http.get(self.url + path, timeout=20, **kw)

    def post(self, path, **kw):
        with self.lock:
            return self.http.post(self.url + path, timeout=20, **kw)

    def cookie(self):
        with self.lock:
            return "; ".join(f"{c.name}={c.value}" for c in self.http.cookies)

    def login(self):
        if self.token:
            # no Origin header: the server's origin check lets non-browser bot logins through
            r = self.post("/api/auth/bot", headers={"Authorization": f"Bearer {self.token}"})
            if r.status_code in (401, 403):
                raise RuntimeError(f"bot login refused: {r.json().get('error', r.status_code)}")
        elif self.user:
            r = self.post("/api/auth/login", json={"username": self.user, "password": self.password})
        else:
            self.name = self.name or "Gast"
            log.info("playing as guest %s", self.name)
            return
        r.raise_for_status()
        body = r.json()
        if not body.get("ok", True) or body.get("mfa") or body.get("needs"):
            raise RuntimeError(f"login needs more than a password: {body}")
        me = self.get("/api/auth/me").json()
        me = me.get("user") or me
        self.user = me.get("username", self.user)
        self.is_bot = me.get("accountType") == "bot"
        if self.is_bot:
            self.name = self.user        # a player-bot plays under its account name
        self.name = self.name or self.user
        log.info("logged in as %s (%s)", self.user, "bot account" if self.is_bot else "regular account")

    def check_login(self) -> bool:
        """Sessions expire, and a renewed token or a banned owner ends them: log in again (the
        bot login fails loudly when the token no longer works). True if it logged in again."""
        if not (self.token or self.user) or self.get("/api/auth/me").status_code != 401:
            return False
        log.warning("session ended, logging in again")
        self.login()
        return True


class Conn:
    """One socket.io connection with an ordered event queue, handled in its own thread."""

    EVENTS = ("game:start", "state:update", "placement:start", "game:over", "session:kicked",
              "async:update", "match:found", "match:ready", "match:cancelled", "turn:timeout",
              "lobby:player:updated", "lobby:config:update", "session:rematch", "player:joined", "player:left")

    def __init__(self, mgr, tag):
        self.mgr, self.acc, self.brain = mgr, mgr.acc, mgr.brain
        self.log = Tagged(log, {"tag": tag})
        self.events: queue.Queue = queue.Queue()
        self.stash = []              # events put aside while waiting for an update
        self.sio = None

    def connect(self, tries=4):
        """Open the socket. The first request may take a while on a slow network (engine.io's
        default timeout is 5 s): longer timeout, a few tries with a pause in between."""
        for attempt in range(1, tries + 1):
            try:
                return self._connect()
            except socketio.exceptions.ConnectionError as e:
                if attempt == tries:
                    raise
                self.log.warning("connect failed (%s), try %d of %d in %d s", e, attempt + 1, tries, 5 * attempt)
                time.sleep(5 * attempt)

    def _connect(self):
        self.sio = socketio.Client(reconnection=True, logger=False, engineio_logger=False, request_timeout=20)
        # engine.io handles every message in its own thread, so an ack could overtake the
        # broadcast of the previous action and the next decision would see a stale state.
        # Our handlers only queue events, so handling them in order in the read thread is safe.
        trigger = self.sio.eio._trigger_event

        def in_order(event, *args, **kw):
            if event == "message":
                kw["run_async"] = False
            return trigger(event, *args, **kw)
        self.sio.eio._trigger_event = in_order
        for ev in self.EVENTS:
            self.sio.on(ev, (lambda e: lambda data=None: self.events.put((e, data)))(ev))
        self.sio.on("disconnect", lambda *a: self.events.put(("disconnect", None)))
        self.sio.on("connect", lambda: self.events.put(("connect", None)))
        self.sio.connect(self.acc.url, headers={"Cookie": self.acc.cookie(), "Origin": self.acc.url},
                         transports=["polling", "websocket"], wait_timeout=20)

    def close(self):
        try:
            if self.sio:
                self.sio.disconnect()
        except Exception:  # noqa: BLE001
            pass

    def reconnect(self):
        self.close()
        self.connect()

    def call(self, event, data, timeout=30):
        res = self.sio.call(event, data, timeout=timeout)
        if isinstance(res, dict) and res.get("error"):
            raise RuntimeError(f"{event}: {res['error']}")
        return res

    def command(self, name, data=None):
        """Run `name` in this connection's thread (from the panel or the manager)."""
        self.events.put(("cmd:" + name, data))

    def next_event(self, timeout=2):
        return self.stash.pop(0) if self.stash else self.events.get(timeout=timeout)

    def guarded(self, fn, *args):
        try:
            return fn(*args)
        except Exception:  # noqa: BLE001 - one bad turn must not stop the client
            self.log.exception("%s failed", fn.__name__)

    def take_update(self, ev, data):
        """Apply a state update without starting a turn. True if it was one."""
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

    @staticmethod
    def newer(new, old):
        """A state is never replaced by an older one (logSeq only grows)."""
        if not old or not new:
            return bool(new)
        return (new.get("logSeq") or 0) >= (old.get("logSeq") or 0)

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
                    cur = latest() or state
                    if os.environ.get("HOCHCIV_KI_DUMP"):
                        Path(os.environ["HOCHCIV_KI_DUMP"]).write_text(json.dumps(
                            {"state": cur, "decided_on": state, "action": [typ, params], "me": me, "error": str(e)}))
                    city = next((c for c in cur.get("cities", []) if c.get("id") == params.get("cityId")), None)
                    self.log.warning("rejected %s %s (%s): %s [logSeq %s, cur %s, me %s, round %s, cities %s, city %s]",
                                     typ, params, tag, e, cur.get("logSeq"), cur.get("cur"), me, cur.get("round"),
                                     len(cur.get("cities", [])), city and {k: city.get(k) for k in ("grown", "freeUsed", "born", "pop", "owner")})
                    break
                ended = typ == "endTurn"
            if ended:
                self.log.info("round %s, %d actions: %s", state.get("round"), sent, " ".join(tags))
                return
            if rejected >= 5:
                self.log.warning("too many rejected actions, ending the turn")
                send("endTurn", {})
                return
            state = latest() or state
            if state.get("over") or state["cur"] != me:
                return
        send("endTurn", {})


class LiveGame(Conn):
    """One live game: joined from the public list, hosted by the bot, or from matchmaking."""

    DONE = ("over", "left", "failed")

    def __init__(self, mgr, lobby=None, host_cfg=None, match=None, picked=False):
        """host_cfg: a setup from the settings (its id ties the game to the setup)."""
        self.id = secrets.token_hex(3)
        super().__init__(mgr, f"live {self.id}")
        self.kind = "host" if host_cfg else "match" if match else "join"
        self.lobby, self.host_cfg, self.match, self.picked = lobby, host_cfg, match, picked
        self.g = None                # {sessionId, password, clientId, yourIndex, players, cfg, state}
        self.setup_id = (host_cfg or {}).get("id")
        self.join_code = (lobby or {}).get("joinCode") or (match or {}).get("joinCode")
        self.phase = "connecting"    # lobby, placement, playing, over, left, failed
        self.note = ""
        self.since = time.time()
        self.last_start_try = self.last_ping = 0.0
        self.connected = False       # the first connect event is the initial connection
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True, name=f"live-{self.id}")

    def start(self):
        self.thread.start()
        return self

    def info(self):
        g = self.g or {}
        cfg, S = g.get("cfg") or {}, g.get("state") or {}
        return {"id": self.id, "kind": self.kind, "picked": self.picked, "setup": self.setup_id,
                "phase": self.phase, "note": self.note,
                "joinCode": self.join_code, "sessionId": g.get("sessionId"), "password": g.get("password"),
                "mapKey": cfg.get("mapKey"), "empires": cfg.get("playerCount"), "kiCount": cfg.get("kiCount"),
                "ranked": bool(cfg.get("ranked")),
                "players": [p.get("name") for p in (g.get("players") or [])], "round": S.get("round"),
                "myTurn": bool(S) and not S.get("over") and S.get("cur") == g.get("yourIndex"),
                "since": self.since}

    def leave(self, why=""):
        """Leave the game (in a running game the server lets the KI take over the seat)."""
        self.command("leave", why)

    # ---------------------------------------------------------------- thread
    def run(self):
        try:
            self.connect()
            if self.host_cfg:
                self.open_lobby()
            elif self.match:
                self.enter(self.match["sessionId"], self.match["password"], self.match["clientId"])
            else:
                self.join()
            while self.phase not in self.DONE:
                try:
                    ev, data = self.next_event(2)
                    self.handle(ev, data)
                except queue.Empty:
                    pass
                if self.kind == "host" and self.phase == "lobby":
                    self.guarded(self.host_tick)
        except Exception as e:  # noqa: BLE001
            self.log.exception("live game failed")
            self.phase, self.note = "failed", str(e)
        finally:
            self.close()
            self.done.set()
            self.mgr.live_ended(self)

    def join(self):
        s, client_id = self.lobby, secrets.token_hex(16)
        r = self.acc.post("/api/sessions/join", json={
            "joinCode": s["joinCode"], "password": s["password"],
            "player": {"name": self.acc.name, "clientId": client_id, "civ": "random", "ability": "random"}})
        if r.status_code != 200:
            raise RuntimeError(f"join refused: {r.json().get('error', r.status_code)}")
        self.enter(r.json()["sessionId"], s["password"], client_id)

    def open_lobby(self):
        client_id = secrets.token_hex(16)
        keys = ("mapKey", "playerCount", "kiCount", "difficulty", "turnTime", "randomCivs", "isPublic", "playerBotLimit", "ranked")
        cfg = dict({k: self.host_cfg[k] for k in keys if k in self.host_cfg}, events=False, eventMode="hard",
                   wonders=False, noKi=False, playerBots=True, seed=secrets.randbelow(2 ** 31))
        cfg["ranked"] = bool(cfg.get("ranked"))
        r = self.acc.post("/api/sessions", json={
            "config": cfg, "host": {"name": self.acc.name, "clientId": client_id, "civ": "random", "ability": "random"}})
        if r.status_code not in (200, 201):
            raise RuntimeError(f"lobby not created: {r.json().get('error', r.status_code)}")
        b = r.json()
        self.join_code = b.get("joinCode")
        self.enter(b["sessionId"], b["password"], client_id)
        self.log.info("hosting lobby %s (%s, %s empires, %s KI%s)", self.join_code, cfg.get("mapKey"),
                      cfg.get("playerCount"), cfg.get("kiCount"), ", ranked" if cfg["ranked"] else "")

    def enter(self, session_id, password, client_id):
        ack = self.call("session:connect", {"sessionId": session_id, "password": password, "clientId": client_id})
        self.g = {"sessionId": session_id, "password": password, "clientId": client_id,
                  "players": ack.get("players"), "cfg": ack.get("gameConfig") or {}, "lobbyIndex": ack.get("playerIndex")}
        self.phase = "lobby" if ack.get("status", "lobby") == "lobby" else "playing"
        if self.kind == "join":
            # the public list doesn't show the whole configuration: check it now
            players, hi = ack.get("players") or [], ack.get("hostIndex", 0)
            host_is_bot = 0 <= hi < len(players) and bool(players[hi].get("botAccount"))
            cfg = self.g["cfg"]
            why = config_ok(cfg) or (None if self.picked else join_filter(self.mgr.settings["join"], cfg, host_is_bot=host_is_bot))
            if why:
                self.mgr.refuse(session_id)
                self.drop("left", f"lobby: {why}")
                return
        self.log.info("in lobby %s (seat %s)", session_id, ack.get("playerIndex"))
        if self.mgr.settings["greeting"]:
            try:
                self.call("chat:send", {"kind": "text", "text": self.mgr.settings["greeting"]})
            except Exception as e:  # noqa: BLE001
                self.log.info("greeting not sent: %s", e)

    def drop(self, phase, why):
        if why:
            self.log.info("leaving: %s", why)
        self.phase, self.note = phase, why or self.note
        # leaving = dropping the socket (the server removes a disconnected lobby player)
        self.close()

    def host_tick(self):
        """A hosted lobby starts when it is full, or after `start_after` minutes with at least
        `min_players` (the free seats become KI). Pings keep the lobby from timing out."""
        st, now = self.mgr.settings["host"], time.time()
        cfg = self.g["cfg"]
        humans = len(self.g.get("players") or [])
        need = (cfg.get("playerCount") or 2) - (cfg.get("kiCount") or 0)
        waited = now - self.since >= st["start_after"] * 60
        if (humans >= need or (waited and humans >= min(st["min_players"], need))) and humans >= 2 \
                and now - self.last_start_try > 20:
            self.last_start_try = now
            try:
                self.call("game:start", {})
                self.log.info("started with %d players", humans)
            except RuntimeError as e:
                self.note = str(e)
                self.log.warning("start: %s", e)
        if now - self.last_ping > 240:
            self.last_ping = now
            self.call("lobby:player:update", {})

    def handle(self, ev, data):
        if ev == "cmd:leave":
            self.drop("left", data or "left by the panel")
        elif ev in ("game:start", "state:update") and self.g:
            if ev == "game:start":
                self.g["yourIndex"] = data["yourIndex"]
                self.g["state"] = data["state"]
                self.phase = "playing"
            else:
                self.take_update(ev, data)
            self.drain()                      # act on the newest state only
            self.guarded(self.live_turn)
        elif ev == "placement:start" and self.g:
            self.phase = "placement"
            self.guarded(self.live_place, data)
        elif ev in ("player:joined", "player:left") and self.g and data:
            self.g["players"] = data.get("players", self.g.get("players"))
        elif ev == "lobby:player:updated" and self.g:
            self.g["players"] = data if isinstance(data, list) else data.get("players", data)
        elif ev == "lobby:config:update" and self.g:
            self.g["cfg"] = data
            why = config_ok(data)
            if why and self.kind != "host":
                self.mgr.refuse(self.g["sessionId"])
                self.drop("left", f"lobby changed to {why}")
        elif ev in ("game:over", "session:kicked"):
            why = (data or {}).get("how") or (data or {}).get("reason") or ev
            self.log.info("game ended: %s", why)
            self.drop("over" if ev == "game:over" else "left", why)
        elif ev == "connect" and not self.connected:
            self.connected = True
        elif ev == "connect" and self.g:
            # reconnect: bind the seat again
            self.call("session:connect", {"sessionId": self.g["sessionId"], "password": self.g["password"],
                                          "clientId": self.g["clientId"]})
        elif ev == "turn:timeout":
            self.log.warning("turn timed out")

    def live_place(self, data):
        g = self.g
        me = g["lobbyIndex"]
        if me not in [s["idx"] for s in data.get("queue", [])] and me not in data.get("queue", []):
            return
        players = [{"civ": p.get("civ"), "kind": p.get("kind", "human"), "ability": p.get("ability", "basis")}
                   for p in (g.get("players") or [])]
        cfg = data.get("cfg") or g["cfg"]
        setup = self.brain.roll_setup(players, data["seed"], cfg)
        o, cell = self.brain.place(data["plan"], me, players, setup, data["seed"])
        self.call("placement:action", {"o": o, "cell": cell})
        self.log.info("placed start tile o=%s cell=%s", o, cell)

    def live_turn(self):
        g = self.g
        S, me = g["state"], g.get("yourIndex", -1)
        if S.get("over") or me is None or me < 0 or S["cur"] != me:
            return

        def send(typ, params):
            # live: the server acks, then broadcasts the new state (nothing on errors). The
            # server also rebroadcasts the unchanged state every 15 s (sync loop), so wait for
            # an update that differs from the state before the action.
            before = fingerprint(self.g["state"])
            self.call("action", {"type": typ, "params": params})
            self.wait_event("state:update", changed_from=before)
        self.play_turn(S, me, send, lambda: self.g and self.g["state"])

    def take_update(self, ev, data):
        if ev == "state:update" and self.g:
            if self.newer(data["state"], self.g.get("state")):
                self.g["state"] = data["state"]
            return True
        return False

    def wait_event(self, kind, timeout=10, changed_from=None):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                ev, data = self.events.get(timeout=max(0.01, deadline - time.time()))
            except queue.Empty:
                break
            if ev == kind and self.take_update(ev, data):
                if changed_from is None or fingerprint(self.g["state"]) != changed_from:
                    return
                continue
            if not self.take_update(ev, data):
                self.stash.append((ev, data))
        self.log.warning("no %s within %ss", kind, timeout)


class Rundenspiele(Conn):
    """All Rundenspiele on one socket: plays every turn that is due, and joins or hosts new
    games as the mode says."""

    def __init__(self, mgr):
        super().__init__(mgr, "rundenspiele")
        self.mine, self.open = [], []    # from the last GET /api/async
        self.async_open = None           # id of the opened Rundenspiel
        self.view = None                 # its newest view
        self.refused = {}                # id -> why the server refused us
        self.last_poll = 0.0
        self.host_retry = 0.0            # no new hosting attempt before this time
        self.thread = threading.Thread(target=self.run, daemon=True, name="rundenspiele")

    def run(self):
        self.connect()
        while True:
            try:
                ev, data = self.next_event(2)
                self.handle(ev, data)
            except queue.Empty:
                pass
            if time.time() - self.last_poll > 60:
                self.guarded(self.poll)

    def handle(self, ev, data):
        if ev == "cmd:poll":
            self.guarded(self.poll)
        elif ev == "cmd:join":
            self.guarded(self.join, data, True)
            self.guarded(self.poll)
        elif ev == "cmd:leave":
            r = self.acc.post(f"/api/async/{data}/leave")
            self.log.info("left Rundenspiel %s: %s", data, r.status_code)
            self.guarded(self.poll)
        elif ev == "cmd:reconnect":
            self.guarded(self.reconnect)
        elif ev == "async:update" and data:
            self.accept(data)
            self.drain()
            self.guarded(self.turn, self.view)

    def fit(self, s):
        """(why, hard) for the open Rundenspiel `s`: hard = it can't be joined or played at
        all, else only the join filters are against it. why None = fine."""
        if s.get("engine") and s["engine"] not in (self.brain.version, "current"):
            return f"engine {s['engine']}", True
        cfg = s.get("config") or {}
        seats = s.get("seats") or []
        why = config_ok(cfg, len(seats)) or ("full" if not s.get("freeSeats") else None) \
            or ("no player-bots" if self.acc.is_bot and not cfg.get("playerBots") else None)
        if why:
            return why, True
        if s.get("id") in self.refused:
            return self.refused[s["id"]], False
        host = seats[cfg["startPlayer"]] if isinstance(cfg.get("startPlayer"), int) and cfg["startPlayer"] < len(seats) else {}
        n = cfg.get("playerCount") or len(seats)
        return join_filter(self.mgr.settings["join"], cfg, empires=n, ki=n - (cfg.get("humanCount") or n),
                           host_is_bot=bool(host.get("botAccount")), live=False), False

    def poll(self):
        self.last_poll = time.time()
        lst = self.acc.get("/api/async").json()
        self.mine, self.open = lst.get("mine", []), lst.get("open", [])
        st = self.mgr.settings
        j = st["join"]
        if j["on"] and j["async"]:
            joined = sum(1 for s in self.mine if not s.get("isHost"))
            for s in self.open:
                if joined >= j["max_async"]:
                    break
                if not self.fit(s)[0] and self.join(s["id"]):
                    joined += 1
        if st["host"]["on"] and time.time() >= self.host_retry:
            created = False
            for setup, missing in self.missing_hosted():
                for _ in range(missing):
                    if not self.host(setup):
                        break
                    created = True
            if created:
                self.mine = self.acc.get("/api/async").json().get("mine", self.mine)
        due = [s for s in self.mine if s.get("yourTurn")]
        if due:
            self.log.info("due: %s", ", ".join(f"{s.get('joinCode')} ({s.get('status')}, round {s.get('round')})" for s in due))
        for s in due:
            self.guarded(self.open_game, s["id"])

    def join(self, gid, picked=False):
        r = self.acc.post(f"/api/async/{gid}/join", json={"civ": "random", "ability": "random"})
        if r.ok and not r.json().get("error"):
            self.log.info("joined Rundenspiel %s%s", gid, " (picked)" if picked else "")
            return True
        err = (r.json() if r.headers.get("content-type", "").startswith("application/json") else {}).get("error") or r.status_code
        if r.status_code < 500:
            self.refused[gid] = f"refused: {err}"    # e.g. no room for another player-bot
        self.log.info("Rundenspiel %s refused us: %s", gid, err)
        return False

    def missing_hosted(self):
        """[(setup, how many more to host)] for the Rundenspiel setups. Hosted games belong to
        the first setup with their map, empires and humans that still has room."""
        setups = [x for x in self.mgr.settings["host"]["setups"] if x["kind"] == "async" and x["count"] > 0]
        have = {x["id"]: 0 for x in setups}
        for g in self.mine:
            if not g.get("isHost"):
                continue
            key = setup_key(g.get("config") or {})
            for x in setups:
                if setup_key(x) == key and have[x["id"]] < x["count"]:
                    have[x["id"]] += 1
                    break
        return [(x, x["count"] - have[x["id"]]) for x in setups if have[x["id"]] < x["count"]]

    def host(self, setup):
        keys = ("mapKey", "playerCount", "kiCount", "difficulty", "civMode", "isPublic", "playerBotLimit", "ranked")
        cfg = dict({k: setup.get(k) for k in keys}, events=False, eventMode="hard", wonders=False, playerBots=True)
        cfg["ranked"] = bool(cfg["ranked"])
        r = self.acc.post("/api/async", json={"config": cfg, "civ": "random", "ability": "random"})
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if not r.ok or body.get("error"):
            self.host_retry = time.time() + 600
            self.mgr.note = f"Rundenspiel not created: {body.get('error') or r.status_code}"
            self.log.warning(self.mgr.note)
            return False
        self.log.info("hosting Rundenspiel %s (code %s)", body.get("id"), body.get("joinCode"))
        return True

    def open_game(self, gid):
        res = self.call("async:open", {"id": gid})
        self.async_open = gid
        self.accept(res["view"])
        self.turn(self.view)

    def accept(self, view):
        """Keep the newest view of the opened game. Views without a state (setup, placement)
        are always newer than none; a state never goes back (logSeq)."""
        old = self.view if (self.view or {}).get("id") == view.get("id") else None
        if not old or not view.get("state") or self.newer(view["state"], old.get("state")):
            self.view = view

    def take_update(self, ev, data):
        if ev == "async:update" and data and self.async_open == data.get("id"):
            self.accept(data)
            return True
        return False

    def turn(self, view):
        gid = (view or {}).get("id")
        if not gid:
            return
        if view.get("status") == "placement" and view.get("placement") and not view["placement"].get("placed"):
            pl = view["placement"]
            o, cell = self.brain.place(pl["plan"], view["mySlot"], pl["players"], pl.get("setup"), 1)
            self.call("async:place", {"id": gid, "o": o, "cell": cell})
            self.log.info("%s: placed start tile", gid)
            return
        S, me = view.get("state"), view.get("myIndex", -1)
        if view.get("status") != "playing" or not S or S.get("over") or me < 0 or S["cur"] != me or view.get("hold"):
            return
        self.view = view

        def send(typ, params):
            # Rundenspiele broadcast the new view before the ack: it is already queued
            self.call("async:action", {"id": gid, "type": typ, "params": params})
            self.drain()
        self.play_turn(S, me, send, lambda: (self.view or {}).get("state"))
        self.log.info("%s: turn played (round %s)", gid, S.get("round"))


class Matchmaker(Conn):
    """Ranked matchmaking: in the queue while the mode is "match" and no match game runs."""

    def __init__(self, mgr):
        super().__init__(mgr, "matchmaking")
        self.queued = False
        self.waiting = None
        self.thread = threading.Thread(target=self.run, daemon=True, name="matchmaking")

    def run(self):
        self.connect()
        while True:
            try:
                ev, data = self.next_event(2)
                self.handle(ev, data)
            except queue.Empty:
                pass
            want = self.mgr.settings["match"] and not any(g.kind == "match" for g in self.mgr.active())
            if want and not self.queued:
                self.guarded(self.join_queue)
            elif not want and self.queued:
                self.guarded(self.leave_queue)

    def join_queue(self):
        res = self.call("match:join", {})
        self.queued = True
        self.log.info("%s", "match pending" if res.get("pending") else f"in the queue ({res.get('queued')} waiting)")

    def leave_queue(self):
        self.queued = False
        self.call("match:leave", {})
        self.log.info("left the queue")

    def handle(self, ev, data):
        if ev == "match:found":
            self.log.info("match found: %s", ", ".join(p.get("name", "?") for p in data.get("players", [])))
            self.call("match:accept", {"matchId": data["matchId"]})
        elif ev == "match:ready":
            self.mgr.start_live(match=data)
            self.guarded(self.leave_queue)
        elif ev == "match:cancelled":
            self.log.info("match cancelled: %s", (data or {}).get("reason"))
            if not (data or {}).get("requeued"):
                self.queued = False
        elif ev == "connect":
            self.queued = False          # the queue entry belongs to the old socket
        elif ev == "cmd:reconnect":
            self.guarded(self.reconnect)


class LogBuffer(logging.Handler):
    def __init__(self, n=300):
        super().__init__()
        self.lines = collections.deque(maxlen=n)
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))

    def emit(self, record):
        self.lines.append(self.format(record))


class Manager:
    """Settings, modes and all connections. tick() runs in the main thread."""

    def __init__(self, acc: Account, brain: Brain, settings_path=None, settings=None):
        self.acc, self.brain = acc, brain
        self.path = Path(settings_path) if settings_path else None
        saved = {}
        if self.path and self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.settings = clean_settings(settings or saved)
        self.lock = threading.RLock()
        self.live: list[LiveGame] = []
        self.refused = {}                # session id -> until when we don't join it again
        self.note = ""
        self.last_lobby_poll = self.last_host_try = self.last_auth_check = 0.0
        self.rs = self.mm = None
        self.logs = LogBuffer()
        log.addHandler(self.logs)

    def start(self):
        """Log in and open the Rundenspiele and matchmaking connections."""
        self.acc.login()
        if not self.acc.is_bot:
            self.settings["match"] = False
        self.last_auth_check = time.time()
        if self.acc.token or self.acc.user:
            self.rs = Rundenspiele(self)
            self.rs.thread.start()
        if self.acc.is_bot:
            self.mm = Matchmaker(self)
            self.mm.thread.start()

    def run(self):
        self.start()
        while True:
            self.guarded_tick()
            time.sleep(3)

    def guarded_tick(self):
        try:
            self.tick()
        except Exception:  # noqa: BLE001
            log.exception("tick failed")

    # ---------------------------------------------------------------- live games
    def active(self):
        with self.lock:
            return [g for g in self.live if not g.done.is_set()]

    def start_live(self, **kw):
        g = LiveGame(self, **kw)
        with self.lock:
            self.live.append(g)
            ended = [x for x in self.live if x.done.is_set()]
            for x in ended[:-20]:        # keep the last 20 finished games for the panel
                self.live.remove(x)
        return g.start()

    def live_ended(self, g):
        if g.kind == "host" and g.phase == "failed":
            self.note = g.note
            self.last_host_try = time.time() + 60       # don't hammer the server

    def refuse(self, session_id, minutes=30):
        self.refused[session_id] = time.time() + minutes * 60

    def lobby_fit(self, s):
        """(why, hard) for the public lobby `s`: hard = it can't be joined, else only the join
        filters are against it. why None = fine. The list shows only the map and the human
        seats; the rest of the configuration is checked after joining (LiveGame.enter)."""
        j = self.settings["join"]
        if s.get("status") != "lobby":
            return "running", True
        if any(g.g and g.g.get("sessionId") == s.get("id") for g in self.active()):
            return "joined", True
        if s.get("playersCount", 0) >= s.get("maxPlayers", 0):
            return "full", True
        if str(s.get("mapKey")) not in MAP_GROUPS:
            return f"map {s.get('mapKey')}", True
        if self.acc.is_bot and not s.get("playerBots"):
            return "no player-bots", True
        if self.acc.is_bot and s.get("playerBotsFree") == 0:
            return "no room for bots", True
        if s.get("maxPlayers", 0) > MAX_PLAYERS:
            return f"{s['maxPlayers']} seats", True
        if self.refused.get(s.get("id"), 0) > time.time():
            return "left before", False
        if MAP_GROUPS[str(s.get("mapKey"))] not in j["maps"]:
            return f"map {s.get('mapKey')}", False
        if s.get("maxPlayers", 0) > j["max_empires"]:
            return f"{s['maxPlayers']} seats", False
        return None, False

    def public_lobbies(self):
        return self.acc.get("/api/public-sessions").json()

    def tick(self):
        now, st = time.time(), self.settings
        if now - self.last_auth_check > 600:
            self.last_auth_check = now
            if self.acc.check_login():
                for c in (self.rs, self.mm):
                    if c:
                        c.command("reconnect")
        active = self.active()
        j, h = st["join"], st["host"]
        joined = [g for g in active if g.kind == "join"]
        if j["on"] and j["live"] and len(joined) < j["max_live"] and now - self.last_lobby_poll > 15:
            self.last_lobby_poll = now
            for s in self.public_lobbies():
                if not self.lobby_fit(s)[0]:
                    self.start_live(lobby=s)
                    break
        # hosted live games: per setup, so many open or running; lobbies nobody needs any
        # more (hosting off, setup removed, fewer wanted) are closed
        setups = {x["id"]: x for x in h["setups"] if x["kind"] == "live"}
        for sid in {g.setup_id for g in active if g.kind == "host"} | set(setups):
            mine = [g for g in active if g.kind == "host" and g.setup_id == sid]
            want = setups[sid]["count"] if h["on"] and sid in setups else 0
            waiting = [g for g in mine if g.phase in ("connecting", "lobby")]
            for g in waiting[max(0, want - (len(mine) - len(waiting))):]:
                g.leave("no longer hosted")
            if len(mine) < want and now - self.last_host_try > 20:
                self.last_host_try = now
                self.start_live(host_cfg=dict(setups[sid]))

    # ---------------------------------------------------------------- panel API
    def status(self):
        rs = self.rs
        return {
            "account": {"name": self.acc.name, "isBot": self.acc.is_bot, "url": self.acc.url},
            "engine": self.brain.version, "agent": self.brain.spec,
            "settings": self.settings, "note": self.note,
            "live": [g.info() for g in reversed(self.live)],
            "rundenspiele": [{"id": s.get("id"), "joinCode": s.get("joinCode"), "status": s.get("status"),
                              "empires": (s.get("config") or {}).get("playerCount"),
                              "ranked": bool((s.get("config") or {}).get("ranked")),
                              "isHost": s.get("isHost"), "round": s.get("round"), "yourTurn": s.get("yourTurn"),
                              "waitingFor": s.get("waitingFor"), "mapKey": (s.get("config") or {}).get("mapKey"),
                              "players": [x.get("name") or ("KI" if x.get("kind") == "ki" else "–") for x in s.get("seats") or []]}
                             for s in (rs.mine if rs else [])],
            "queued": bool(self.mm and self.mm.queued),
            "log": list(self.logs.lines)[-80:],
        }

    def game_lists(self):
        lobbies = []
        for s in self.public_lobbies():
            if s.get("status") != "lobby":
                continue
            why, hard = self.lobby_fit(s)
            lobbies.append({"id": s.get("id"), "joinCode": s.get("joinCode"), "host": s.get("hostName"),
                            "mapKey": s.get("mapKey"), "players": s.get("playersCount"), "seats": s.get("maxPlayers"),
                            "playerBots": s.get("playerBots"), "why": why, "hard": hard})
        games = []
        if self.rs:
            for s in self.acc.get("/api/async").json().get("open", []):
                cfg = s.get("config") or {}
                why, hard = self.rs.fit(s)
                games.append({"id": s.get("id"), "joinCode": s.get("joinCode"), "mapKey": cfg.get("mapKey"),
                              "status": s.get("status"), "empires": cfg.get("playerCount"),
                              "free": len(s.get("freeSeats") or []), "round": s.get("round"),
                              "host": next((x.get("name") for x in s.get("seats") or [] if x.get("userId")), None),
                              "playerBots": cfg.get("playerBots"), "why": why, "hard": hard})
        return {"lobbies": lobbies, "rundenspiele": games}

    def join(self, kind, gid):
        if kind == "live":
            s = next((x for x in self.public_lobbies() if x.get("id") == gid), None)
            if not s:
                raise ValueError("lobby not found")
            why, hard = self.lobby_fit(s)
            if hard:
                raise ValueError(why)
            self.refused.pop(gid, None)
            self.start_live(lobby=s, picked=True)
        elif kind == "async" and self.rs:
            self.rs.refused.pop(gid, None)
            self.rs.command("join", gid)
        else:
            raise ValueError("unknown game kind")

    def leave(self, kind, gid):
        if kind == "live":
            g = next((x for x in self.active() if x.id == gid), None)
            if not g:
                raise ValueError("game not found")
            g.leave("left by the panel")
        elif kind == "async" and self.rs:
            self.rs.command("leave", gid)
        else:
            raise ValueError("unknown game kind")

    def update_settings(self, raw):
        new = clean_settings(raw, self.settings)
        if new["match"] and not self.acc.is_bot:
            raise ValueError("matchmaking is for bot accounts only")
        self.settings = new
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(new, indent=2), encoding="utf-8")
        log.info("settings saved (join %s, host %s, matchmaking %s)", *("on" if x else "off" for x in
                 (new["join"]["on"], new["host"]["on"], new["match"])))
        if self.rs:
            self.rs.command("poll")
        return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="online.env")
    ap.add_argument("--agent", default="teacher")
    ap.add_argument("--settings", default="runs/online-settings.json", help="where the panel settings are saved")
    ap.add_argument("--port", type=int, default=8765, help="control panel on http://127.0.0.1:PORT")
    ap.add_argument("--no-panel", action="store_true", help="run with the saved settings, no control panel")
    ap.add_argument("--open", action="store_true", help="open the control panel in the browser")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    env = read_env(args.config)
    brain = Brain(args.agent)
    log.info("engine %s, agent %s", brain.version, args.agent)
    acc = Account(env["HOCHCIV_URL"], env.get("HOCHCIV_USER"), env.get("HOCHCIV_PASSWORD"),
                  token=env.get("HOCHCIV_TOKEN"), name=env.get("HOCHCIV_NAME"))
    mgr = Manager(acc, brain, args.settings)
    if not args.no_panel:
        from .panel import serve
        serve(mgr, args.port, open_browser=args.open)
    mgr.run()


if __name__ == "__main__":
    threading.current_thread().name = "main"
    main()
