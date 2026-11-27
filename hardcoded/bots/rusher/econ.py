"""Economy planner: greedy best action per resource, every candidate tried on a copy.

Each step lists the candidate actions (research, free picks, growth, founding, roads), runs
every one on a throwaway copy of the game (Turn.trials, one round trip for all of them) and
values the copy with `value()` below. The action with the best gain per resource spent runs
on the real game; then the next step starts from the new state. Stops when nothing gains.

The value is a hand-written estimate of what a state is worth to the economy: weighted
income over a horizon, population, cities, techs (with a bonus for effects income doesn't
show) and progress through the ages. All of it is computed here in Python; the engine only
answers what the income, population etc. of a copy are.
"""
from __future__ import annotations

import copy
import math
import heapq

from hcbots.hexgrid import hex_distance, key, neighbors, within

W = {"sci": 1.0, "food": 1.05, "coins": 0.8}      # value per unit of income per round
PRICE = {"sci": 1.0, "food": 1.4, "coins": 1.0}   # price of a resource spent now
SCI_RESERVE = 3      # science spent on anything but research counts this much more
GAMMA = 0.86
END_ROUND = 11
W_POP, W_CITY, W_TECH, W_AGE = 1.2, 3.0, 0.8, 2.2
AGE_VALUE = [0, 1, 2.3, 3.8, 5.6]
GROW_STEP = 1.5      # value per city and round of one more growth step (keramik, dampfmaschine)
THREAT_W = 300       # weight of an enemy's economy victory coming close
# Effects income doesn't show (military for the rush, growth and trade helpers)
MIL_BONUS = {"sklaverei": 40, "belagerung": 60, "eisenverarbeitung": 6, "demokratie": 6,
             "taktik": 2, "nationalismus": 14, "gewehre": 6, "rittertum": 5,
             "militaergericht": 7, "dynamit": 25, "stadtmauern": 10, "stahl": 6,
             "maschinengewehr": 10, "theologie": 3, "philosophie": 4, "navigation": 5}


def enemy_growth(t, e: int) -> int:
    """Most population enemy e can add in its next turn: its resources then are its income
    (they reset each turn); every city grows up to its paid steps (cost 1 food + 1 coin per
    population), new cities cost 1/3/6/... + distance food; the cheapest steps first, and no
    more growth than the food income carries (unless gentechnik/massenmedien)."""
    p = t.S["players"][e]
    inc, base, rates, lim = t.call([("income", "$S", e), ("baseIncome", "$S", e), ("rates", "$S", e),
                                    ("growLimits", "$S", e)])
    if not inc:
        return 2 + len(t.cities(e))
    food, coins = inc["food"], inc["coins"]
    if p["techs"].get("alchemie"):
        coins += inc["sci"]
    c2f = rates.get("coinsToFood", 2) if rates else 2
    cities = t.cities(e)
    prices = t.call([("growPrice", "$S", e, {"$city": c["id"]}) for c in cities])
    units = []
    for c, pr in zip(cities, prices):
        if not pr:
            continue
        n = max(1, c["pop"])
        for k in range((lim or {}).get("paid", 1)):
            sc = (c["pop"] + k) / n
            units.append((pr["food"] * sc, pr["coins"] * sc, "g"))
        for k in range((lim or {}).get("free", 0)):
            units.append((0, 0, "f"))
    m = len(cities)
    for k in range(4):
        units.append((max(1, (m + k) * (m + k + 1) // 2 + 3), 0, "c"))
    units.sort(key=lambda u: u[0] * c2f + u[1])
    feed = p["techs"].get("gentechnik") or p["techs"].get("massenmedien")
    cap = 99 if feed else max(0, (base or inc)["food"])
    grow = 0
    for f, c, kind in units:
        if kind == "g" and cap <= 0:
            continue
        if c > coins:
            continue
        rest = coins - c
        need = max(0, f - food)
        if need * c2f > rest:
            continue
        food -= min(food, f)
        coins = rest - need * c2f
        grow += 1
        if kind == "g":
            cap -= 1
    return grow


def horizon(rnd: int) -> float:
    r = max(1, min(8, END_ROUND - rnd + 1))
    return sum(GAMMA ** i for i in range(r))


class EconPlanner:
    def __init__(self, t, pool: list, slavery_pays: bool, threat_w: float = THREAT_W,
                 grow_step: float = GROW_STEP, coin_price: float = PRICE["coins"], def_w: float = 0.0,
                 alch_bonus: float = 0.0, chunk: bool = True, hold=(), mil_scale: float = 1.0, mil_from: int = 1,
                 grow_est: bool = True, grow_margin: int = 1, threat_exp: float = 10,
                 late_bonus: bool = True, rails: bool = True,
                 w_city: float = W_CITY, chunk_k: float = 2.5, chunk_r: float = 0.45,
                 method_first: bool = True, sing_path: bool = True,
                 race_w: float = 4.0, race_turns: float = 4.0,
                 eta_h: bool = False, race_mode: str = "hold", lookahead: bool = True,
                 lookahead_k: int = 1, slav_scale: float | None = None):
        self.lookahead = lookahead
        self.lookahead_k = lookahead_k
        # slavery was tuned at 7 coins per population (v108); v110 pays 4
        self.slav_scale = slav_scale if slav_scale is not None else (t.const("SLAVERY_COINS") or 7) / 7
        self.alch_bonus = alch_bonus
        self.chunk = chunk
        self.threat_exp = threat_exp
        self.late_bonus, self.rails = late_bonus, rails
        self._pop = 0
        self.w_city, self.chunk_k, self.chunk_r = w_city, chunk_k, chunk_r
        self.method_first = method_first
        self.sing_path = sing_path
        self.race_w, self.race_turns, self.race_mode = race_w, race_turns, race_mode
        self.hold = set(hold)      # techs left to the strike (researched in the turn they hit)
        # military bonus only from round mil_from on (scaled): the rush techs crowd out the economy
        self.mil_scale = mil_scale if t.round >= mil_from else 0.0
        self.t = t
        self.def_w = def_w
        cap = t.capital()
        self.cap_id = cap["id"] if cap else None
        self.threat_w, self.grow_step = threat_w, grow_step
        self.price = dict(PRICE, coins=coin_price)
        self.pool = {x["k"]: x for x in pool}
        self.H = horizon(t.round)
        self.slavery_pays = slavery_pays
        # enemies' population at the end of the round (with what they can still grow) and
        # their economy-victory threshold
        self.threats = []
        for e in t.enemies():
            vo = t.q("victoryOption", "$S", {"$player": e})
            pop = sum(c["pop"] for c in t.cities(e))
            grow = (enemy_growth(t, e) + grow_margin if grow_est else 2 + len(t.cities(e))) if self.moves_after(e) else 0
            if vo and vo.get("frac"):
                self.threats.append((pop + grow, grow, vo["frac"]))
        self.race = False
        if (self.race_w or eta_h) and len(self.pool) and t.round >= 3:
            etas = [self.sing_eta(pi) for pi in [t.pi] + t.enemies()]
            self.race = bool(self.race_w) and t.round >= 4 and min(etas) <= self.race_turns
            if eta_h:
                # like the KI: the game probably ends one round after the fastest singularity,
                # so income counts for fewer rounds and points (pop, techs) for more
                r = max(1, min(8, END_ROUND - t.round + 1, math.ceil(min(etas)) + 1))
                self.H = sum(GAMMA ** i for i in range(r))

    def sing_need(self, p: dict) -> float:
        """Science still needed for the singularity: the cheapest tech of each missing age in
        every field (available ones first), then the singularity itself."""
        meth = bool(p["techs"].get("wiss_methode"))
        disc = (2, 3, 4, 5, 10)
        if p["techs"].get("singularitaet"):
            return 0
        need = max(0, 120 - (disc[4] if meth else 0))
        for f in range(4):
            a = max([x["age"] for k, x in self.pool.items() if x.get("f") == f and p["techs"].get(k)], default=-1)
            for age in range(a + 1, 4):
                lst = [x for x in self.pool.values() if x.get("f") == f and x.get("age") == age]
                if not lst:
                    continue
                av = [x for x in lst if p.get("avail", {}).get(x["k"]) and not p["techs"].get(x["k"])]
                need += min(max(0, x["c"] - (disc[age] if meth else 0)) for x in (av or lst))
        return need

    def sing_eta(self, pi: int) -> float:
        t = self.t
        p = t.S["players"][pi]
        need = self.sing_need(p)
        inc = t.q("income", "$S", pi)
        per = max(1, inc["sci"] + max(0, inc["coins"]) / 2)
        have = t.q("available", "$S", pi, "sci")
        return max(0, need - have) / per

    def moves_after(self, e: int) -> bool:
        """Does enemy e still move this round after me?"""
        t = self.t
        n, first = len(t.S["players"]), t.S.get("startIdx", 0) or 0
        order = [(first + i) % n for i in range(n)]
        return order.index(e) > order.index(t.pi)

    # ------------------------------------------------------------------ value
    def calls(self):
        pi = self.t.pi
        return [("baseIncome", "$S", pi), ("popOf", "$S", pi), ("worldPop", "$S"),
                ("victoryOption", "$S", {"$player": pi}), ("$player", pi),
                ("defenseValue", "$S", {"$city": self.cap_id}) if self.cap_id is not None else ("popOf", "$S", pi),
                ("available", "$S", pi, "sci")]

    def tech_bonus(self, k: str, ncities: int, inc: dict) -> float:
        H = self.H
        if k == "singularitaet":
            return 5000
        v = MIL_BONUS.get(k, 0) * self.mil_scale * (self.slav_scale if k == "sklaverei" else 1)
        # one more paid growth step per city and round: growth is the bottleneck
        if k == "keramik":
            v += self.grow_step * H * min(ncities, 8)
        elif k == "rad":
            v += 1.2 * H * max(0, ncities - 1) + 2
        elif k == "dampfmaschine":
            v += self.grow_step * H * min(ncities, 8) + 0.35 * H * ncities
        elif k == "kartografie":
            v += 1.5 * max(0, 7 - ncities) + 1
        elif k == "alchemie":
            v += 0.12 * H * max(0, inc["sci"]) + 2 + self.alch_bonus
        elif k == "gilden":
            v += 0.12 * H * max(0, inc["coins"]) + 2
        elif k == "wiss_methode":
            v += 18 + 4 * min(4, H)
        elif self.late_bonus:
            if k == "eisenbahn":
                # +4 instead of +1 on all three yields per city on a continuous rail line
                v += 3.0 * H * max(0, ncities - 1) + 2
            elif k == "computertechnik":
                v += 0.2 * H * max(0, inc["coins"]) + 3
            elif k == "verbundwerkstoffe":
                v += ncities * H * max(1, H) / 3
            elif k in ("massenmedien", "gentechnik"):
                v += 0.3 * H * min(self._pop, 20) / 2 + 3
        t = self.pool.get(k)
        if t and t["age"] >= 3 and self.slavery_pays:
            v -= 15                         # the modern age ends slavery
        return v

    def value(self, res) -> tuple[float, dict]:
        inc, pop, world, vo, p, dcap, sci_av = res
        self._pop = pop
        techs = [k for k, v in (p or {}).get("techs", {}).items() if v]
        ncities = self._ncities
        v = self.H * sum(W[r] * inc[r] for r in W)
        v += W_POP * pop + self.w_city * ncities + W_TECH * len(techs)
        v += sum(self.tech_bonus(k, ncities, inc) for k in techs)
        if self.def_w and self.cap_id is not None:
            v += self.def_w * (dcap or 0)
        best = [-1, -1, -1, -1]
        for k in techs:
            x = self.pool.get(k)
            if x and 0 <= x["f"] < 4:
                best[x["f"]] = max(best[x["f"]], x["age"])
        late = min(1.8, 0.6 + self.t.round / 10)
        v += W_AGE * late * sum(AGE_VALUE[a + 1] for a in best) + (25 if min(best) >= 3 else 0)
        if world and vo and vo.get("frac"):
            q = min(1.2, pop / world / vo["frac"])
            v += 40 * q ** 6
        for epop, grow, frac in self.threats:
            q = min(1.3, epop / max(1, world + grow) / frac)
            v -= self.threat_w * q ** self.threat_exp
        if self.race and p and sci_av is not None:
            # singularity race: science kept towards what the singularity still costs
            if self.race_mode == "progress":
                # resources do not carry over: reward what is left of the way, not what is kept
                v -= self.race_w * self.sing_need(p)
            else:
                v += self.race_w * min(0, sci_av - self.sing_need(p))
        return v, p["res"]

    # ------------------------------------------------------------------ candidates
    def candidates(self):
        t = self.t
        out = []
        sci = t.q("available", "$S", t.pi, "sci")
        open_ = t.q("researchable", "$S", t.pi) or []
        costs = t.call([("techCost", "$S", t.pi, x) for x in open_])
        for x, c in zip(open_, costs):
            if c is not None and c <= sci and x["k"] not in self.hold:
                out.append(("tech:" + x["k"], [("research", {"tech": x["k"]})]))
        if self.sing_path:
            path = self.singularity_path()
            if path:
                out.append(("sing:path", [("research", {"tech": k}) for k in path]))
        for kind, fn in (("freeTech", "freeTechOptions"), ("backPick", "backPickOptions"),
                         ("freePick", "freePickOptions")):
            for x in t.q(fn, "$S", t.pi) or []:
                if x["k"] in self.hold:
                    continue
                out.append(("pick:" + x["k"], [(kind, {"tech": x["k"]})]))
        for city in t.cities():
            cid = city["id"]
            out.append(("gfree:%d" % cid, [("growCity", {"cityId": cid, "mode": "free"})]))
            out.append(("grow:%d" % cid, [("growCity", {"cityId": cid, "mode": "paid"})]))
        for r, c in self.sites()[:6]:
            out.append(("found:%d,%d" % (r, c), [("foundCity", {"r": r, "c": c})]))
        if t.has("rad"):
            for tag, tiles in self.road_plans()[:4]:
                out.append((tag, [("buildRoad", {"r": r, "c": c, "target": 1}) for r, c in tiles]))
        if self.rails and t.has("eisenbahn"):
            for tag, tiles in self.road_plans(2)[:4]:
                out.append(("rail" + tag[4:], [("buildRoad", {"r": r, "c": c, "target": 2}) for r, c in tiles]))
        return out

    def singularity_path(self, max_len: int = 10):
        """Research chain that reaches the singularity in this turn, built greedily on a fork:
        the scientific method first, then the cheapest tech that raises a field still short of
        the modern age, then the singularity. None if it is out of reach."""
        t = self.t
        if t.has("singularitaet"):
            return None
        best = [-1] * 4
        for k, on in t.me["techs"].items():
            x = self.pool.get(k)
            if on and x and 0 <= x["f"] < 4:
                best[x["f"]] = max(best[x["f"]], x["age"])
        if min(best) < 2 or sum(3 - a for a in best) > 3:
            return None
        f = t.fork()
        path = []
        try:
            for _ in range(max_len):
                sci = f.q("available", "$S", f.pi, "sci")
                open_ = f.q("researchable", "$S", f.pi) or []
                costs = f.call([("techCost", "$S", f.pi, x) for x in open_])
                opts = [(c, x["k"]) for x, c in zip(open_, costs) if c is not None and c <= sci]
                done = [o for o in opts if o[1] == "singularitaet"]
                if done:
                    path.append("singularitaet")
                    return path
                meth = [o for o in opts if o[1] == "wiss_methode"]
                step = [o for o in opts if o[1] in self.pool and 0 <= self.pool[o[1]]["f"] < 4
                        and self.pool[o[1]]["age"] >= 3 and best[self.pool[o[1]]["f"]] < 3]
                pick = min(meth or step, default=None)
                if pick is None or f.act("research", tech=pick[1]):
                    return None
                path.append(pick[1])
                x = self.pool.get(pick[1])
                if x and 0 <= x["f"] < 4:
                    best[x["f"]] = max(best[x["f"]], x["age"])
            return None
        finally:
            f.close()

    def sites(self):
        t = self.t
        food = t.q("available", "$S", t.pi, "food")
        mine = t.cities()
        if food <= 0 or not mine:
            return []
        cand = set()
        for c in mine:
            for (r, cc) in within(c["r"], c["c"], 6):
                if t.grid.is_land(r, cc):
                    cand.add((r, cc))
        cand = [x for x in cand if all(hex_distance(x[0], x[1], o["r"], o["c"]) >= 3 for o in t.S["cities"])
                and not t.army_at(*x)]
        if not cand:
            return []
        calls = []
        for r, c in cand:
            calls += [("foundSiteError", "$S", t.pi, r, c), ("foundCost", "$S", t.pi, r, c),
                      ("settleGain", "$S", t.pi, r, c)]
        res = t.call(calls)
        scored = []
        for i, (r, c) in enumerate(cand):
            err, cost, g = res[3 * i: 3 * i + 3]
            if err or cost is None or cost > food or not g:
                continue
            v = self.H * (W["sci"] * g["sci"] + W["food"] * max(g["food"], -1) + W["coins"] * g["coins"])
            scored.append((v - 1.2 * cost, r, c))
        scored.sort(reverse=True)
        return [(r, c) for _, r, c in scored]

    def road_plans(self, level: int = 1):
        """For every own city off the capital's road net: the cheapest tiles to connect it
        (Dijkstra over own or neutral land, a tile with road or an own city costs nothing)."""
        t = self.t
        cap = t.capital()
        if not cap:
            return []
        roads = t.S.get("roads") or {}
        foreign = set()
        for i in range(len(t.S["players"])):
            if i != t.pi:
                foreign.update(t.q("controlledTiles", "$S", i) or [])
        own_city = {(c["r"], c["c"]) for c in t.cities()}
        other_city = {(c["r"], c["c"]) for c in t.S["cities"]} - own_city

        def free(rc):
            return rc in own_city or roads.get(key(*rc), 0) >= level

        def ok(rc):
            return (t.grid.is_tile(*rc) and t.grid.terrain(*rc) not in ("M",)
                    and rc not in other_city and (key(*rc) not in foreign or rc in own_city))
        # the capital's net
        net, stack = {(cap["r"], cap["c"])}, [(cap["r"], cap["c"])]
        while stack:
            rc = stack.pop()
            for n in neighbors(*rc):
                if n not in net and ok(n) and free(n):
                    net.add(n)
                    stack.append(n)
        plans = []
        for city in t.cities():
            goal = (city["r"], city["c"])
            if city.get("cap") or goal in net:
                continue
            dist = {s: 0 for s in net}
            prev = {}
            q = [(0, s) for s in net]
            heapq.heapify(q)
            found = None
            while q:
                d, rc = heapq.heappop(q)
                if d > dist.get(rc, 99) or d > 8:
                    continue
                if rc == goal:
                    found = d
                    break
                for n in neighbors(*rc):
                    if not ok(n) or not t.grid.is_land(*n) and n not in own_city:
                        continue
                    nd = d + (0 if free(n) else 1)
                    if nd < dist.get(n, 99):
                        dist[n] = nd
                        prev[n] = rc
                        heapq.heappush(q, (nd, n))
            if not found:
                continue
            tiles, rc = [], goal
            while rc in prev:
                rc = prev[rc]
                if not free(rc):
                    tiles.append(rc)
            if tiles:
                plans.append((found, "road:%d" % city["id"], tiles))
        plans.sort()
        return [(tag, tiles) for _, tag, tiles in plans]

    # ------------------------------------------------------------------ plan
    def rollout(self, acts, max_steps: int = 20) -> float:
        """Value at the end of the turn after acts and then the greedy plan, on a copy."""
        f = self.t.fork()
        try:
            for a, p in acts:
                if f.act(a, **p):
                    return float("-inf")
            sub = copy.copy(self)
            sub.t, sub.lookahead = f, False
            sub.run(max_steps)
            res = f.call(sub.calls())
            sub._ncities = len(f.cities())
            return sub.value(res)[0]
        finally:
            f.close()

    def run(self, max_steps: int = 40, keep_coins: int = 0) -> list:
        t = self.t
        done = []
        failed = set()
        for _ in range(max_steps):
            if t.over:
                break
            self._ncities = len(t.cities())
            cands = [c for c in self.candidates() if c[0] not in failed]
            if not cands:
                break
            calls = self.calls()
            trials = [([], calls)] + [(acts, calls) for _, acts in cands]
            res = t.trials(trials)
            if res[0] is None:
                break
            base, before = self.value(res[0])
            best = None
            top = None
            method = None
            scored = []
            for (tag, acts), r in zip(cands, res[1:]):
                if r is None:
                    continue
                self._ncities = len(t.cities()) + (1 if tag.startswith("found:") else 0)
                v, after = self.value(r)
                self._ncities = len(t.cities())
                dv = v - base
                if dv <= 0.05:
                    continue
                if after["coins"] < keep_coins and after["coins"] < before["coins"]:
                    continue
                research = tag.startswith(("tech:", "pick:"))
                spent = sum(self.price[k] * (before[k] - after[k]) * (1 if k != "sci" or research else SCI_RESERVE)
                            for k in self.price)
                ratio = dv / (max(0, spent) + 2)
                scored.append((ratio, dv, tag, acts))
                if tag.endswith(":wiss_methode") and (method is None or ratio > method[0]):
                    method = (ratio, dv, tag, acts)
                if best is None or ratio > best[0]:
                    best = (ratio, dv, tag, acts)
                if top is None or dv > top[1]:
                    top = (ratio, dv, tag, acts)
            if best is None:
                break
            # big chunks: an action worth much more in total wins if its ratio is not much
            # worse (else small things eat the food a new city needs)
            if self.chunk and top is not best and top[1] > self.chunk_k * best[1] and top[0] > self.chunk_r * best[0]:
                best = top
            # the scientific method makes every tech paid after it cheaper: take it first
            # whenever this step pays a tech and the method is worth something on its own
            if self.method_first and method and best[2].startswith("tech:") and best is not method:
                best = method
            # cheap now vs. big now: play both plans to the end of the turn on a copy and keep
            # the better end (a growth step can eat the food a new city needs)
            if self.lookahead and self.lookahead_k > 1:
                # several plans: the best few by ratio and by size, each played to the end
                pool, seen = [best], {best[2]}
                for x in sorted(scored, key=lambda x: -x[0])[:self.lookahead_k] + sorted(scored, key=lambda x: -x[1])[:2]:
                    if x[2] not in seen:
                        seen.add(x[2])
                        pool.append(x)
                if len(pool) > 1:
                    best = max(pool, key=lambda x: self.rollout(x[3]))
            elif self.lookahead and top is not best and top[2] != best[2]:
                if self.rollout(top[3]) > self.rollout(best[3]):
                    best = top
            _, dv, tag, acts = best
            err = None
            for a, p in acts:
                err = t.act(a, **p)
                if err:
                    break
            if err:
                failed.add(tag)
            else:
                done.append((tag, round(dv, 1)))
        return done


class ClaimPlanner(EconPlanner):
    """An enemy has claimed a victory this round (singularity): the round ends with a points
    comparison (population + techs) among everyone who claimed by then. Value: having a
    claim myself (singularity, or the economy threshold at the end of the round) and beating
    the best claimant's points; every step tried on a copy like the economy planner."""

    def __init__(self, t, pool: list, target: int, **kw):
        kw.setdefault("hold", ())
        super().__init__(t, pool, False, **kw)
        self.target = target
        self.mine = any(c["pi"] == t.pi for c in (t.S.get("claims") or []))

    def value(self, res):
        inc, pop, world, vo, p, dcap = res[:6]
        techs = sum(1 for v in (p or {}).get("techs", {}).values() if v)
        claim = self.mine or bool(p["techs"].get("singularitaet"))
        if not claim and vo and world and len(self.t.S["cities"]) > 1 and vo.get("den"):
            claim = pop * vo["den"] > world * vo["num"] if vo.get("strict") else pop * vo["den"] >= world * vo["num"]
        s = pop + techs
        # progress towards the singularity: a modern tech in each field
        modern = {self.pool[k]["f"] for k, on in p["techs"].items() if on and k in self.pool and self.pool[k]["age"] >= 3}
        v = 8 * len(modern) + 1000 * claim + 40 * min(1, max(0, s - self.target + 1)) + 40 * min(1, max(0, s - self.target)) + s
        return v, p["res"]
