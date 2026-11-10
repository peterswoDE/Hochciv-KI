"""Search at decision time, stage 1 (DESIGN_SEARCH.md): a one-step look-ahead with the value head.

At every decision the policy's top k candidates are played on copies of the game
(bridge `fork`): an action inside the turn is followed by the next decision point of the same
player, "end" by the other seats' turns (played by KI schwer) up to the player's next decision
point. Each successor is scored with the value head (a finished game with its score), "end" is
averaged over a few seeds. "end" is only searched when the policy ranks it in its top k or
gives it at least `p_end` (its copies cost the other seats' turns, and ending early is rarely
right: unspent resources expire). The pick maximises value + c · log p.

A time budget per turn keeps it inside online turn timers: once it is used up, the rest of the
turn is the plain policy.

    python -m hochciv_ki.eval_modes --agent search:checkpoints/v2.pt:4 --suite v2 --scale 0.5
"""
from __future__ import annotations

import random
import time

import torch

from .env import Agent, Obs, decode_obs


class SearchAgent(Agent):
    needs_game = True

    def __init__(self, net_agent, k=4, c=0.02, end_seeds=2, p_end=0.1, budget=None, seed=0):
        self.net, self.k, self.c, self.end_seeds, self.budget = net_agent, k, c, end_seeds, budget
        self.p_end = p_end
        self.rng = random.Random(seed)
        self._spent = {}                             # (bridge, gid, player, round) -> seconds

    def act(self, obs: Obs, game=None):
        logits, v = self.net.evaluate(obs)
        logp = torch.log_softmax(logits.float(), -1)
        greedy = int(logits.argmax())
        if game is None or len(obs.tags) == 1:
            return greedy, {"v": v}
        turn = (id(game[0]), game[1], obs.pi, obs.round)
        if len(self._spent) > 10_000:
            self._spent.clear()
        if self.budget is not None and self._spent.get(turn, 0.0) >= self.budget:
            return greedy, {"v": v, "search": "budget"}
        t0 = time.perf_counter()
        end = len(obs.tags) - 1
        top = [int(i) for i in torch.topk(logp, min(self.k, end + 1)).indices]
        if end not in top and float(logp[end].exp()) >= self.p_end:
            top.append(end)
        if len(top) == 1:
            return top[0], {"v": v}
        scores = {}
        for a in top:
            vals = [self._value_after(game, obs.pi, a) for _ in range(self.end_seeds if a == end else 1)]
            scores[a] = sum(vals) / len(vals) + self.c * float(logp[a])
        self._spent[turn] = self._spent.get(turn, 0.0) + time.perf_counter() - t0
        best = max(scores, key=scores.get)
        return best, {"v": v, "search": {obs.tags[a]: round(s, 4) for a, s in scores.items()}}

    def _value_after(self, game, pi, a):
        """Play action a on a copy of the game; the value of the player's next decision point."""
        bridge, gid = game
        f = bridge.call("fork", gid=gid, seed=self.rng.randrange(2**31))["gid"]
        try:
            st = bridge.call("step", gid=f, i=a)
            if st.get("err"):
                return -1.0                          # failed on the copy: never pick it
            if st["over"]:
                w = st.get("winners", [])
                return 1.0 / len(w) if pi in w else 0.0
            if st["cur"] != pi:                      # cannot happen: the other seats are the KI
                return 0.0
            return self.net.evaluate(decode_obs(bridge.call("observe", gid=f), self.dims))[1]
        finally:
            bridge.call("close", gid=f)

    @property
    def dims(self):
        return self.net.net.dims
