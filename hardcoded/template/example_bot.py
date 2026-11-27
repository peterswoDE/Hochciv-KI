"""The smallest possible bot – a starting point to copy into bots/<name>/bot.py.

A bot is a class with two methods:

    place(seat, plan) -> {"o": int, "cell": int} | None
        Start tile on tile maps. seat["choices"][o][cell] is the first-turn income
        {sci, food, coins} of a capital there, or None where it may not stand.
        None lets the worker place it like a bot (random).

    turn(t: hcbots.turn.Turn) -> None
        Play one turn: read t.S, ask the engine with t.q / t.call, act with t.act(type, **params).
        Do not end the turn – the runner (offline) or the online client does that. An army left
        in a city is moved out automatically before endTurn.
"""
from __future__ import annotations


class ExampleBot:
    name = "example"

    def place(self, seat, plan):
        best = None
        for o, row in enumerate(seat["choices"]):
            for cell, y in enumerate(row):
                if y and (best is None or sum(y.values()) > best[0]):
                    best = (sum(y.values()), o, cell)
        return {"o": best[1], "cell": best[2]} if best else None

    def turn(self, t):
        # research the cheapest tech we can pay for with science
        open_ = t.q("researchable", "$S", t.pi)
        costs = t.call([("techCost", "$S", t.pi, x) for x in open_])
        for tech, cost in sorted(zip(open_, costs), key=lambda tc: tc[1]):
            if cost <= t.me["res"]["sci"]:
                t.act("research", tech=tech["k"])
        # everything else into power (resources do not carry over to the next turn)
        price = t.q("powerPrice", "$S", t.pi)
        n = t.q("available", "$S", t.pi, "coins") // price
        if n > 0:
            t.act("buyPower", n=n)
