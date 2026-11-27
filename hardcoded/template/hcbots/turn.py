"""The interface between a bot and a game: one Turn object per turn of a py seat.

A bot reads the state (`t.S`), asks the engine rule questions (`t.q`, `t.call`) and acts with
`t.act(type, **params)` – actions in the server's format (docs/bots/writing-a-bot.md in the
multiplayer repo). Every action runs on the local engine copy at once, so the state is always
current, and is recorded in `t.actions`; online, the client sends exactly these to the server.
The bot does not end the turn – the runner or the online client does.
"""
from __future__ import annotations

import copy

from .bridge import Bridge
from .hexgrid import Grid


class Turn:
    def __init__(self, bridge: Bridge, gid: int, state: dict | None = None):
        self.b = bridge
        self.gid = gid
        self.S = state if state is not None else bridge.call("state", gid=gid)["state"]
        self.pi = self.S["cur"]
        self.grid = Grid(self.S["map"]["rows"])
        self.actions: list[tuple[str, dict]] = []
        self.errors: list[tuple[str, dict, str]] = []

    # -------------------------------------------------------------- state access
    @property
    def me(self) -> dict:
        return self.S["players"][self.pi]

    @property
    def round(self) -> int:
        return self.S["round"]

    def const(self, name: str):
        """A constant of the engine (e.g. SLAVERY_COINS), cached per bridge."""
        cache = self.b.__dict__.setdefault("_consts", {})
        if name not in cache:
            cache[name] = self.b.call("consts", names=[name])["consts"][name]
        return cache[name]

    def has(self, tech: str, pi: int | None = None) -> bool:
        p = self.S["players"][self.pi if pi is None else pi]
        return bool(p["techs"].get(tech))

    def cities(self, pi: int | None = None) -> list[dict]:
        pi = self.pi if pi is None else pi
        return [c for c in self.S["cities"] if c["owner"] == pi]

    def armies(self, pi: int | None = None) -> list[dict]:
        pi = self.pi if pi is None else pi
        return [a for a in self.S["armies"] if a["owner"] == pi]

    def capital(self, pi: int | None = None) -> dict | None:
        cs = self.cities(pi)
        return next((c for c in cs if c.get("cap")), cs[0] if cs else None)

    def city(self, cid: int) -> dict | None:
        return next((c for c in self.S["cities"] if c["id"] == cid), None)

    def army(self, aid: int) -> dict | None:
        return next((a for a in self.S["armies"] if a["id"] == aid), None)

    def city_at(self, r: int, c: int) -> dict | None:
        return next((x for x in self.S["cities"] if x["r"] == r and x["c"] == c), None)

    def army_at(self, r: int, c: int) -> dict | None:
        return next((x for x in self.S["armies"] if x["r"] == r and x["c"] == c), None)

    def enemies(self) -> list[int]:
        """Living empires other than me (not the barbarians)."""
        return [i for i, p in enumerate(self.S["players"])
                if i != self.pi and not p.get("dead") and p.get("kind") != "barbar"]

    # -------------------------------------------------------------- rule questions
    def call(self, calls: list[tuple]) -> list:
        """Several engine calls in one round trip: [(fn, *args)] -> values. Arguments may be
        "$S", {"$city": id}, {"$army": id}, {"$player": pi}. Errors come back as None."""
        res = self.b.call("call", gid=self.gid, calls=[{"fn": c[0], "args": list(c[1:])} for c in calls])
        return [r.get("v") if "error" not in r else None for r in res["results"]]

    def q(self, fn: str, *args):
        """One engine call; raises on error."""
        r = self.b.call("call", gid=self.gid, calls=[{"fn": fn, "args": list(args)}])["results"][0]
        if "error" in r:
            raise RuntimeError(f"{fn}: {r['error']}")
        return r.get("v")

    # -------------------------------------------------------------- actions
    def act(self, type_: str, **params) -> str | None:
        """Apply a server action locally; returns the engine's error text or None."""
        res = self.b.call("act", gid=self.gid, type=type_, params=params, state=True)
        self.S = res["state"]
        err = res.get("err")
        if err:
            self.errors.append((type_, params, err))
        else:
            self.actions.append((type_, params))
        return err

    # -------------------------------------------------------------- trying things out
    def fork(self) -> "Turn":
        """A copy of this turn on a sandboxed copy of the game: act freely, then .close() it.
        Use it to try a plan before running it on the real game."""
        gid = self.b.call("fork", gid=self.gid)["gid"]
        return Turn(self.b, gid, state=copy.deepcopy(self.S))

    def trials(self, trials: list[tuple[list, list]]) -> list:
        """Many throwaway tries in one round trip. Each trial is (acts, calls): acts =
        [(type, params)] run on a fresh copy of the game, then calls = [(fn, *args)] asked on
        that copy. Returns per trial None if an action failed, else the list of call values
        (None for a failed call). The copies roll their own dice."""
        req = [{"acts": [{"type": a, "params": p} for a, p in acts],
                "calls": [{"fn": c[0], "args": list(c[1:])} for c in calls]} for acts, calls in trials]
        res = self.b.call("try", gid=self.gid, trials=req)["results"]
        return [None if r.get("err") else [x.get("v") if "error" not in x else None for x in r["results"]]
                for r in res]

    def close(self) -> None:
        self.b.close_game(self.gid)

    @property
    def over(self) -> bool:
        return bool(self.S.get("over"))
