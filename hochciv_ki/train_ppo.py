"""PPO self-play against a league: the current network, earlier snapshots, the JS KI and bots.

Every iteration plays a batch of games in parallel (one inference server for the learner, one
for a snapshot), then runs a few PPO epochs on the decisions of the learner's seats. The
reward is the final score (1 for a win, shared on ties, 0 otherwise). A small imitation term
(cross entropy towards the JS KI planner's choice) keeps early training stable; its weight
decays over the run. Checkpoints and the league are saved, so a run can be stopped and
resumed (--resume).

    python -m hochciv_ki.train_ppo --init checkpoints/bc/best.pt --out checkpoints/ppo --mix 1v1
    python -m hochciv_ki.train_ppo --out checkpoints/ppo --resume
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import queue
import random
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .env import play_game
from .eval import run_eval
from .expert import ExpertAgent, search_phase
from .infer import InferenceServer, ServerAgent
from .model import collate, load_net, save_net
from .pool import BridgePool

MAPS = {"2": ["plaettchen", "plaettchen", "plaettchen", "original", "customgen"],
        "3": ["plaettchen", "plaettchen", "customgen"],
        "4": ["plaettchen", "plaettchen", "original", "customgen"]}
# v2: also marathon maps, and 5–8 empires (tile and custom maps allow up to 8); hex tiles since v100
MAPS_V2 = {"2": MAPS["2"] + ["customgen-marathon", "plaettchen-hex"],
           "3": MAPS["3"] + ["customgen-marathon", "plaettchen-hex"],
           "4": MAPS["4"] + ["marathon", "marathon", "customgen-marathon", "plaettchen-hex"],
           **{str(n): ["plaettchen", "plaettchen", "customgen", "customgen", "customgen-marathon", "plaettchen-hex"]
              for n in range(5, 9)}}

# (weight, seats) – L: learner (records), S: snapshot, ki:*, bot:*
MIXES = {
    "1v1": [
        (4, ["L", "ki:schwer"]),
        (3, ["L", "L"]),
        (2, ["L", "S"]),
        (1, ["L", "ki:mittel"]),
    ],
    "all": [
        (3, ["L", "ki:schwer"]),
        (2, ["L", "L"]),
        (1, ["L", "S"]),
        (1, ["L", "ki:schwer", "ki:schwer"]),
        (1, ["L", "L", "S"]),
        (2, ["L", "ki:schwer", "ki:schwer", "ki:schwer"]),
        (1, ["L", "L", "ki:schwer", "ki:schwer"]),
        (1, ["L", "S", "ki:schwer", "bot:koenig"]),
    ],
}
KI = "ki:schwer"
# v2: the v1 mix plus 5–8 empires; game options (wonders, events) are drawn per game
MIXES["v2"] = MIXES["all"] + [
    (1, ["L"] + [KI] * 4),
    (1, ["L", "L"] + [KI] * 4),
    (1, ["L"] + [KI] * 6),
    (1, ["L", "S"] + [KI] * 6),
    (1, ["L", "L", "S", KI, KI, "bot:koenig", KI]),
]


# v3: the v2 mix with more 1v1 and more marathon, where ppo12 lost ground against v4 (v106)
MIXES["v3"] = MIXES["v2"] + [
    (3, ["L", KI]),
    (1, ["L", "S"]),
    (2, ["L", KI, KI, KI]),
]
MAPS_V3 = {**MAPS_V2,
           "2": MAPS_V2["2"] + ["original", "customgen-marathon"],
           "3": MAPS_V2["3"] + ["customgen-marathon"],
           "4": MAPS_V2["4"] + ["marathon", "marathon", "customgen-marathon"]}


def game_options(rng, mix):
    """v2, v3: wonders in 40 % of the games, events in half of them (hard or easy)."""
    if mix not in ("v2", "v3"):
        return {}
    o = {"wonders": rng.random() < 0.4}
    ev = rng.random()
    if ev < 0.5:
        o.update(events=True, eventMode="hard" if ev < 0.25 else "easy")
    return o


def seat(s):
    if s in ("L", "S"):
        return {"kind": "agent"}
    kind, _, lvl = s.partition(":")
    return {"kind": "ki", "kiLevel": lvl} if kind == "ki" else {"kind": "bot", "diff": lvl}


BETA_MAX = 8.0


def phi(prog, me, rnd, n0, slope=1.0):
    """Potential for reward shaping (DESIGN_SHAPING.md): the soft share of the win
    softmax(beta * progress²) of seat `me` over the seats that can still win, minus the fair
    share 1/n0 at the start of the game. beta = min(8, slope * (round − 1)) – fitted on KI
    games (calibrate_phi.py): progress says nothing in round 1 and more every round."""
    beta = min(BETA_MAX, slope * (rnd - 1))
    if not prog or prog[me] < 0:
        return -1.0 / n0
    x = beta * np.array([p for p in prog if p >= 0], np.float64) ** 2
    m = x.max()
    return float(np.exp(beta * prog[me] ** 2 - m) / np.exp(x - m).sum()) - 1.0 / n0


def gae(traj, z, lam, shape=0.0, slope=1.0):
    """traj: list of (sample, a, logp, v, teacher, prog, pi, round, …). Reward z at the end, gamma 1.
    shape > 0 adds c * (Φ(s') − Φ(s)) to the advantages, Φ(end) = 0; the value targets stay
    on the unshaped reward (the value head is the chance to win)."""
    n = len(traj)
    v = np.array([x[3] for x in traj], np.float32)
    nxt = np.append(v[1:], 0.0)
    r = np.zeros(n, np.float32)
    r[-1] = z
    base = r + nxt - v
    if shape:
        prog0 = traj[0][5]
        n0 = sum(1 for x in prog0 if x >= 0) if prog0 else 2
        ph = np.array([phi(x[5], x[6], x[7], n0, slope) for x in traj], np.float32)
        delta = base + shape * (np.append(ph[1:], 0.0) - ph)
    else:
        delta = base

    def disc(d):
        out, acc = np.zeros(n, np.float32), 0.0
        for t in range(n - 1, -1, -1):
            acc = d[t] + lam * acc
            out[t] = acc
        return out

    adv = disc(delta)
    ret = (adv if delta is base else disc(base)) + v
    return adv, ret


class PPO:
    def __init__(self, net, lr, clip=0.2, vcoef=0.5, ent=0.01, target_kl=0.03, ref=None, anchor=0.0):
        self.net, self.clip, self.vcoef, self.ent, self.target_kl = net, clip, vcoef, ent, target_kl
        # anchor: KL(ref || net) over the candidates, ref = the starting network (frozen), so
        # that a run improves on it instead of drifting away (ppo10, ppo12)
        self.ref, self.anchor = ref, anchor
        self.opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    def update(self, batch, epochs, mb, device, imit):
        """batch: list of (sample, a, logp_old, adv, ret, teacher)."""
        # the network has no batch-dependent layers (ChannelNorm, no dropout): train and eval
        # mode compute the same, so the probability ratio starts at 1
        self.net.train()
        adv = np.array([b[3] for b in batch], np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        stats = {"pol": 0.0, "val": 0.0, "ent": 0.0, "imit": 0.0, "kl": 0.0, "clipfrac": 0.0, "anchor": 0.0, "n": 0}
        extra = {"kl0": None, "epochs_done": 0}
        idx = np.arange(len(batch))
        stop = False
        for ep in range(epochs):
            ep_kl, ep_n = 0.0, 0
            np.random.shuffle(idx)
            for s in range(0, len(idx), mb):
                # early stop: the policy moved far enough from the one that collected the data
                if ep_n and ep_kl / ep_n > 1.5 * self.target_kl:
                    stop = True
                    break
                b = idx[s:s + mb]
                grid, glob, cand, pos, mask = collate([batch[i][0] for i in b], device)
                a = torch.tensor([batch[i][1] for i in b], device=device)
                lp_old = torch.tensor([batch[i][2] for i in b], device=device)
                A = torch.tensor(adv[b], device=device)
                R = torch.tensor([batch[i][4] for i in b], device=device, dtype=torch.float32)
                T = torch.tensor([batch[i][5] for i in b], device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, v = self.net(grid, glob, cand, pos, mask)
                logp_all = torch.log_softmax(logits.float(), -1)
                lp = logp_all.gather(1, a[:, None]).squeeze(1)
                ratio = torch.exp(lp - lp_old)
                pl = -torch.min(ratio * A, ratio.clamp(1 - self.clip, 1 + self.clip) * A).mean()
                vl = F.mse_loss(torch.sigmoid(v.float()), R)
                p_all = logp_all.exp()
                ent = -(p_all * logp_all.masked_fill(~mask, 0)).sum(-1).mean()
                il = F.nll_loss(logp_all.masked_fill(~mask, -1e9), T)
                loss = pl + self.vcoef * vl - self.ent * ent + imit * il
                if self.anchor:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        ref_logits, _ = self.ref(grid, glob, cand, pos, mask)
                    ref_lp = torch.log_softmax(ref_logits.float(), -1)
                    ak = (ref_lp.exp() * (ref_lp - logp_all)).masked_fill(~mask, 0).sum(-1).mean()
                    loss = loss + self.anchor * ak
                    stats["anchor"] += ak.item() * len(b)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
                n = len(b)
                kl = (lp_old - lp).mean().item()
                if extra["kl0"] is None:
                    extra["kl0"] = round(kl, 5)     # before any step: should be ~0
                ep_kl += kl * n
                ep_n += n
                stats["n"] += n
                stats["pol"] += pl.item() * n
                stats["val"] += vl.item() * n
                stats["ent"] += ent.item() * n
                stats["imit"] += il.item() * n
                stats["kl"] += kl * n
                stats["clipfrac"] += ((ratio - 1).abs() > self.clip).float().mean().item() * n
            extra["epochs_done"] = round(ep + ep_n / len(idx), 2)
            if stop:
                break
        n = max(1, stats.pop("n"))
        return {**{k: v / n for k, v in stats.items()}, **extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="checkpoints/bc/best.pt")
    ap.add_argument("--out", default="checkpoints/ppo")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--mix", default="1v1", choices=list(MIXES))
    ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--games", type=int, default=168, help="games per iteration")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--mb", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--imit", type=float, default=0.3, help="initial imitation weight")
    ap.add_argument("--imit-half", type=float, default=100, help="iterations until it halves")
    ap.add_argument("--shape", type=float, default=0.0, help="weight of the shaped reward (0: off)")
    ap.add_argument("--shape-slope", type=float, default=1.0, help="sharpness of the soft win share per round")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-seeds", type=int, default=60)
    ap.add_argument("--snap-every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--continuous", action="store_true", help="keep every worker playing (class Collector)")
    ap.add_argument("--search-frac", type=float, default=0.0, help="share of the learner's decisions searched (expert.py)")
    ap.add_argument("--search-k", type=int, default=3)
    ap.add_argument("--search-tau", type=float, default=0.03)
    ap.add_argument("--search-epochs", type=int, default=1)
    ap.add_argument("--search-kl", type=float, default=0.05, help="the search phase stops at this KL from the played policy")
    ap.add_argument("--anchor-kl", type=float, default=0.0, help="weight of KL(start network || network) in the loss")
    ap.add_argument("--anchor-frac", type=float, default=0.0,
                    help="share of the snapshot seats played by the start network instead of a league snapshot")
    args = ap.parse_args()

    device = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    if args.resume and (out / "latest.pt").exists():
        net, ck = load_net(out / "latest.pt", device)
        st = json.loads(state_path.read_text())
    else:
        net, ck = load_net(args.init, device)
        st = {"iter": 0, "league": [], "best": None, "history": []}
    snap_net, _ = load_net(args.init if not st["league"] else st["league"][-1], device)
    learner_srv = InferenceServer(copy.deepcopy(net).eval(), device)
    snap_srv = InferenceServer(snap_net, device)
    ref = None
    if args.anchor_kl:
        ref = load_net(args.init, device)[0].eval()
        for q in ref.parameters():
            q.requires_grad_(False)
    ppo = PPO(net, args.lr, target_kl=args.target_kl, ref=ref, anchor=args.anchor_kl)
    if args.resume and "opt" in ck:
        ppo.opt.load_state_dict(ck["opt"])
        for g in ppo.opt.param_groups:      # the command line wins over the saved schedule
            g["lr"] = args.lr
    kw = {"workers": args.workers} if args.workers else {}
    mix = MIXES[args.mix]
    rng = random.Random(12345 + st["iter"])

    job_no = itertools.count(st["iter"] * 1000)
    search = {"frac": args.search_frac, "k": args.search_k, "tau": args.search_tau} if args.search_frac else None

    def make_job():
        seats = rng.choices([m[1] for m in mix], weights=[m[0] for m in mix])[0]
        maps = {"v2": MAPS_V2, "v3": MAPS_V3}.get(args.mix, MAPS)[str(len(seats))]
        return {"seed": 50_000_000 + next(job_no), "seats": seats, "map": rng.choice(maps),
                "opts": game_options(rng, args.mix)}

    with BridgePool(**kw) as pool:
        print(f"engine {pool.version}, {pool.workers} workers, mix {args.mix}, start at iter {st['iter']}", flush=True)

        def play(b, j):
            return collect(b, pool.dims, j, learner_srv, snap_srv, search)

        collector = Collector(pool, make_job, play) if args.continuous else None
        if collector:
            collector.resume()
        while st["iter"] < args.iters:
            it = st["iter"] + 1
            t0 = time.perf_counter()
            if collector:
                results = collector.take(args.games)
            else:
                jobs = [make_job() for _ in range(args.games)]
                results = pool.map(play, jobs)
            batch, wins, games_vs_ki, steps, shaped, searched = [], 0.0, 0, 0, [], []
            for recs, info in results:
                for rec, z in recs:
                    adv, ret = gae(rec, z, args.lam, args.shape, args.shape_slope)
                    if args.shape:
                        shaped.append(float((adv - gae(rec, z, args.lam)[0]).std()))
                    for k, (sample, a, logp, v, teacher, *rest) in enumerate(rec):
                        batch.append((sample, a, logp, adv[k], ret[k], teacher))
                        if rest[-2] is not None:
                            searched.append((sample, rest[-2], rest[-1]))
                if info["vs_ki"]:
                    games_vs_ki += 1
                    wins += info["score"]
                steps += info["steps"]
            t1 = time.perf_counter()
            imit = args.imit * 0.5 ** (it / args.imit_half)
            stats = ppo.update(batch, args.epochs, args.mb, device, imit)
            if search:
                stats.update(search_phase(net, ppo.opt, searched, device, args.mb, args.search_epochs, args.search_kl))
            learner_srv.load_state(net.state_dict())
            t2 = time.perf_counter()
            st["iter"] = it
            rec = {"iter": it, "samples": len(batch), "train_score_vs_ki": wins / max(1, games_vs_ki),
                   "collect_s": round(t1 - t0), "update_s": round(t2 - t1), "imit_w": imit,
                   **({"shape_std": round(float(np.mean(shaped)), 4)} if shaped else {}),
                   **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()}}
            print(json.dumps(rec), flush=True)
            if it % args.snap_every == 0:
                p = out / f"snap_{it:05d}.pt"
                save_net(net, p, iter=it)
                st["league"] = (st["league"] + [str(p)])[-8:]
                pick = args.init if rng.random() < args.anchor_frac else rng.choice(st["league"])
                snap_srv.load_state(load_net(pick, device)[0].state_dict())
            if it % args.eval_every == 0:
                if collector:
                    collector.pause()
                ev = evaluate(pool, net, device, args.eval_seeds, args.mix)
                if collector:
                    collector.resume()
                rec["eval"] = ev
                print(json.dumps({"iter": it, "eval": ev}), flush=True)
                if st["best"] is None or best_key(ev) > best_key(st["best"]):
                    st["best"] = dict(ev, iter=it)
                    save_net(net, out / "best.pt", iter=it, eval=ev)
            st["history"].append(rec)
            save_net(net, out / "latest.pt", iter=it, opt=ppo.opt.state_dict())
            state_path.write_text(json.dumps(st, indent=1))
        if collector:
            collector.pause()
    learner_srv.close()
    snap_srv.close()


class Collector:
    """Continuous collection: every worker always plays a game, and an iteration takes the next
    `n` finished games. Long games carry over into a later iteration instead of keeping the
    other workers idle, and games go on during the network update. Every sample keeps the
    log-probability and value of the policy that played it, so a game that spans an update is
    still valid PPO data, one update stale. pause() lets the running games finish (their
    results stay queued), so the pool is free for an evaluation."""

    def __init__(self, pool, make_job, play):
        self.pool, self.make_job, self.play = pool, make_job, play
        self.done: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.running = 0
        self.feeding = False

    def _feed(self):
        with self.lock:
            if not self.feeding:
                return
            self.running += 1
            job = self.make_job()
        self.pool.submit(self.play, job).add_done_callback(self._finished)

    def _finished(self, fut):
        with self.lock:
            self.running -= 1
        self.done.put(fut)
        self._feed()

    def resume(self):
        with self.lock:
            self.feeding = True
            missing = self.pool.workers - self.running
        for _ in range(missing):
            self._feed()

    def pause(self):
        with self.lock:
            self.feeding = False
        while True:
            with self.lock:
                if self.running == 0:
                    return
            time.sleep(0.2)

    def take(self, n):
        return [self.done.get().result() for _ in range(n)]


def collect(bridge, dims, job, learner, snap, search=None):
    rng = random.Random(job["seed"])
    seats = job["seats"][:]
    rng.shuffle(seats)
    players = [seat(s) for s in seats]
    agents, learner_slots = {}, set()
    for i, s in enumerate(seats):
        if s == "L":
            seed = job["seed"] * 31 + i
            agents[i] = ExpertAgent(learner, seed=seed, **search) if search else ServerAgent(learner, seed=seed)
            learner_slots.add(i)
        elif s == "S":
            agents[i] = ServerAgent(snap, seed=job["seed"] * 31 + i)
    r = play_game(bridge, dims, players, agents, seed=job["seed"], map_key=job["map"], record=True,
                  startPlayer=rng.randrange(len(players)), **job.get("opts", {}))
    recs = []
    learner_pis = [pi for pi, traj in r.trajectories.items() if traj and traj[0][2] is not None
                   and r.slot_of.get(pi) in learner_slots]
    for pi in learner_pis:
        z = r.score(pi) if r.over else 1.0 / len(players)
        recs.append(([(e["sample"], a, e["logp"], e["v"], obs.teacher, obs.prog, obs.pi, obs.round,
                       e.get("target"), e.get("p"))
                      for obs, a, e in r.trajectories[pi]], z))
    vs_ki = all(s == "L" or s.startswith("ki") for s in seats) and seats.count("L") == 1
    score = r.score(learner_pis[0]) if (vs_ki and learner_pis) else 0.0
    return recs, {"vs_ki": vs_ki, "score": score, "steps": r.steps}


def best_key(ev):
    """v1 runs: the 1v1 score. v2 runs: the mean score relative to the fair share over the suite."""
    if "suite" in ev:
        return sum(r["rel"] for r in ev["suite"]) / len(ev["suite"])
    return ev["2p"]["score"]


def evaluate(pool, net, device, seeds, mix="all"):
    from .model import NetAgent
    agent = NetAgent(copy.deepcopy(net).eval(), device, greedy=True)
    if mix in ("v2", "v3"):
        from .eval_modes import run_suite
        # a quarter of the full suite, on its own seeds (the full suite is for final numbers)
        return {"suite": run_suite(pool, lambda: agent, "v2", scale=seeds / 240, base_seed=30_000, progress=False)}
    res = {}
    res["2p"] = run_eval(pool, lambda: agent, "ki:schwer", 2, seeds, base_seed=10_000, progress=False)
    res["4p"] = run_eval(pool, lambda: agent, "ki:schwer", 4, max(5, seeds // 4), base_seed=20_000, progress=False)
    return {k: {"score": round(v["score"], 4), "ci95": round(v["ci95"], 4), "games": v["games"]} for k, v in res.items()}


if __name__ == "__main__":
    main()
