"""Search stage 3 (DESIGN_SEARCH.md): the network learns from the search.

During a training game a share of the learner's decisions is searched like search.py does:
the top k candidates are played on copies of the game and scored with the value head. That
gives an improved policy, the target

    target(a) ∝ p(a) · exp((Q(a) − v) / τ),   Q(a) = v for the candidates not searched

so an unsearched candidate keeps its prior and one noisy value cannot force a label. The move
played is still sampled from the plain policy: the game stays on-policy for PPO, and only the
decisions that were searched carry a target. After the PPO update, `search_phase` trains the
policy on those targets.

    python -m hochciv_ki.expert --init checkpoints/ppo6/best.pt --games 28
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch

from .env import decode_obs
from .infer import InferenceServer, ServerAgent, softmax
from .model import collate, grid_to_u8


def improved_policy(p, v, q, tau):
    """p: the policy over all candidates, v: the value now, q: {candidate: Q} for those searched."""
    adv = np.zeros(len(p), np.float64)
    for a, x in q.items():
        adv[a] = x - v
    w = np.log(np.maximum(p, 1e-12)) + adv / tau
    return softmax(w).astype(np.float32)


class ExpertAgent(ServerAgent):
    """Samples from the policy like ServerAgent; searched decisions also record a target. The
    rules of what is searched are those of SearchAgent, the values come from the InferenceServer."""

    needs_game = True

    def __init__(self, server, frac=0.25, k=3, tau=0.03, end_seeds=2, p_end=0.1, seed=None):
        super().__init__(server, seed=seed)
        self.frac, self.k, self.tau, self.end_seeds, self.p_end = frac, k, tau, end_seeds, p_end
        self.pyrng = random.Random(seed)

    def act(self, obs, game=None):
        a, extra = super().act(obs)
        if game is not None and len(obs.tags) > 1 and self.pyrng.random() < self.frac:
            logits, v = self.server.request(extra["sample"]).result()
            p = softmax(logits)
            extra["target"] = improved_policy(p, v, self._search(game, obs, p), self.tau)
            extra["p"] = p.astype(np.float32)
        return a, extra

    def _search(self, game, obs, p):
        end = len(p) - 1
        top = [int(i) for i in np.argsort(-p)[:self.k]]
        if end not in top and p[end] >= self.p_end:
            top.append(end)
        q = {}
        for a in top:
            vals = [self._value_after(game, obs.pi, a) for _ in range(self.end_seeds if a == end else 1)]
            q[a] = sum(vals) / len(vals)
        return q

    def _value_after(self, game, pi, a):
        """As SearchAgent._value_after. A move that fails on the copy counts as the sure loss."""
        bridge, gid = game
        f = bridge.call("fork", gid=gid, seed=self.pyrng.randrange(2**31))["gid"]
        try:
            st = bridge.call("step", gid=f, i=a)
            if st.get("err"):
                return 0.0
            if st["over"]:
                w = st.get("winners", [])
                return 1.0 / len(w) if pi in w else 0.0
            if st["cur"] != pi:
                return 0.0
            obs = decode_obs(bridge.call("observe", gid=f), self.server.net.dims)
            sample = (grid_to_u8(obs.grid, self.scale), obs.glob, obs.cand, obs.pos)
            return self.server.request(sample).result()[1]
        finally:
            bridge.call("close", gid=f)


def pad(arrays, n):
    out = np.zeros((len(arrays), n), np.float32)
    for i, x in enumerate(arrays):
        out[i, :len(x)] = x
    return out


def search_phase(net, opt, items, device, mb=256, epochs=1, max_kl=0.05):
    """Train the policy on the search targets. items: list of (sample, target, p at collection).
    Stops when the policy has moved `max_kl` from the one that played. Returns statistics:
    gap = KL(target ‖ policy) before and after (what is left to learn), kl = how far it moved."""
    if not items:
        return {"search_n": 0}
    net.train()
    idx = np.arange(len(items))
    gaps, kl, steps, stop = [], 0.0, 0, False
    for _ in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, len(idx), mb):
            b = idx[s:s + mb]
            grid, glob, cand, pos, mask = collate([items[i][0] for i in b], device)
            N = mask.shape[1]
            T = torch.from_numpy(pad([items[i][1] for i in b], N)).to(device)
            P = torch.from_numpy(pad([items[i][2] for i in b], N)).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = net(grid, glob, cand, pos, mask)
            logp = torch.log_softmax(logits.float(), -1).masked_fill(~mask, 0)
            loss = -(T * logp).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                gaps.append((T * (torch.log(T.clamp_min(1e-12)) - logp)).sum(-1).mean().item())
                kl = (P * (torch.log(P.clamp_min(1e-12)) - logp)).sum(-1).mean().item()
            steps += 1
            if kl > max_kl:
                stop = True
                break
        if stop:
            break
    return {"search_n": len(items), "search_steps": steps, "search_gap0": round(gaps[0], 4),
            "search_gap1": round(gaps[-1], 4), "search_kl": round(kl, 4)}


def main():
    """How much does the search cost and how much does it change the policy? Plays learner
    seats against KI schwer, with and without search, on the same seeds."""
    from .env import play_game
    from .model import load_net
    from .pool import BridgePool
    from .train_ppo import MAPS_V2, game_options

    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="checkpoints/ppo6/best.pt")
    ap.add_argument("--games", type=int, default=28)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.03)
    args = ap.parse_args()

    net, _ = load_net(args.init, "cuda")
    srv = InferenceServer(net.eval(), "cuda")
    rng = random.Random(1)
    jobs = []
    for g in range(args.games):
        n = rng.choice([2, 2, 2, 3, 4, 4, 6])
        jobs.append({"seed": 80_000_000 + g, "n": n, "map": rng.choice(MAPS_V2[str(n)]), "opts": game_options(rng, "v2")})

    with BridgePool(workers=args.workers) as pool:
        def play(b, j, frac):
            players = [{"kind": "agent"}] + [{"kind": "ki", "kiLevel": "schwer"}] * (j["n"] - 1)
            agent = ExpertAgent(srv, frac=frac, k=args.k, tau=args.tau, seed=j["seed"])
            return play_game(b, pool.dims, players, {0: agent}, seed=j["seed"], map_key=j["map"], record=True,
                             **j["opts"])

        times = {}
        for frac in (0.0, args.frac):
            t0 = time.perf_counter()
            res = pool.map(lambda b, j, f=frac: play(b, j, f), jobs)
            times[frac] = time.perf_counter() - t0
    srv.close()

    ex = [e for r in res for t in r.trajectories.values() for _, _, e in t]
    tg = [e for e in ex if "target" in e]
    gap = [float((e["target"] * (np.log(e["target"] + 1e-12) - np.log(e["p"] + 1e-12))).sum()) for e in tg]
    moved = sum(int(e["target"].argmax()) != int(e["p"].argmax()) for e in tg)
    print(f"{args.games} games: plain {times[0.0]:.0f} s, frac {args.frac} k {args.k}: {times[args.frac]:.0f} s "
          f"({times[args.frac] / times[0.0]:.2f}x)")
    print(f"{len(ex)} decisions, {len(tg)} searched; KL(target || policy) mean {np.mean(gap):.4f}, "
          f"median {np.median(gap):.4f}; best move changed in {100 * moved / max(1, len(tg)):.1f} %")


if __name__ == "__main__":
    main()
