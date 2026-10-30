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
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .env import play_game
from .eval import run_eval
from .infer import InferenceServer, ServerAgent
from .model import collate, load_net, save_net
from .pool import BridgePool

MAPS = {"2": ["plaettchen", "plaettchen", "plaettchen", "original", "gross", "customgen"],
        "3": ["plaettchen", "plaettchen", "gross", "customgen"],
        "4": ["plaettchen", "plaettchen", "original", "gross", "customgen"]}

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


def seat(s):
    if s in ("L", "S"):
        return {"kind": "agent"}
    kind, _, lvl = s.partition(":")
    return {"kind": "ki", "kiLevel": lvl} if kind == "ki" else {"kind": "bot", "diff": lvl}


def gae(traj, z, lam):
    """traj: list of (sample, a, logp, v, teacher). Reward z at the end, gamma 1."""
    n = len(traj)
    adv = np.zeros(n, np.float32)
    nxt_v, acc = 0.0, 0.0
    for t in range(n - 1, -1, -1):
        r = z if t == n - 1 else 0.0
        v = traj[t][3]
        delta = r + nxt_v - v
        acc = delta + lam * acc
        adv[t] = acc
        nxt_v = v
    ret = adv + np.array([x[3] for x in traj], np.float32)
    return adv, ret


class PPO:
    def __init__(self, net, lr, clip=0.2, vcoef=0.5, ent=0.01, target_kl=0.03):
        self.net, self.clip, self.vcoef, self.ent, self.target_kl = net, clip, vcoef, ent, target_kl
        self.opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    def update(self, batch, epochs, mb, device, imit):
        """batch: list of (sample, a, logp_old, adv, ret, teacher)."""
        # the network has no batch-dependent layers (ChannelNorm, no dropout): train and eval
        # mode compute the same, so the probability ratio starts at 1
        self.net.train()
        adv = np.array([b[3] for b in batch], np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        stats = {"pol": 0.0, "val": 0.0, "ent": 0.0, "imit": 0.0, "kl": 0.0, "clipfrac": 0.0, "n": 0}
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
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-seeds", type=int, default=60)
    ap.add_argument("--snap-every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=None)
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
    ppo = PPO(net, args.lr, target_kl=args.target_kl)
    if args.resume and "opt" in ck:
        ppo.opt.load_state_dict(ck["opt"])
    kw = {"workers": args.workers} if args.workers else {}
    mix = MIXES[args.mix]
    rng = random.Random(12345 + st["iter"])

    with BridgePool(**kw) as pool:
        print(f"engine {pool.version}, {pool.workers} workers, mix {args.mix}, start at iter {st['iter']}", flush=True)
        while st["iter"] < args.iters:
            it = st["iter"] + 1
            t0 = time.perf_counter()
            jobs = []
            for g in range(args.games):
                seats = rng.choices([m[1] for m in mix], weights=[m[0] for m in mix])[0]
                jobs.append({"seed": 50_000_000 + it * 1000 + g, "seats": seats,
                             "map": rng.choice(MAPS[str(len(seats))])})
            results = pool.map(lambda b, j: collect(b, pool.dims, j, learner_srv, snap_srv), jobs)
            batch, wins, games_vs_ki, steps = [], 0.0, 0, 0
            for recs, info in results:
                for rec, z in recs:
                    adv, ret = gae(rec, z, args.lam)
                    for k, (sample, a, logp, v, teacher) in enumerate(rec):
                        batch.append((sample, a, logp, adv[k], ret[k], teacher))
                if info["vs_ki"]:
                    games_vs_ki += 1
                    wins += info["score"]
                steps += info["steps"]
            t1 = time.perf_counter()
            imit = args.imit * 0.5 ** (it / args.imit_half)
            stats = ppo.update(batch, args.epochs, args.mb, device, imit)
            learner_srv.load_state(net.state_dict())
            t2 = time.perf_counter()
            st["iter"] = it
            rec = {"iter": it, "samples": len(batch), "train_score_vs_ki": wins / max(1, games_vs_ki),
                   "collect_s": round(t1 - t0), "update_s": round(t2 - t1), "imit_w": imit,
                   **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()}}
            print(json.dumps(rec), flush=True)
            if it % args.snap_every == 0:
                p = out / f"snap_{it:05d}.pt"
                save_net(net, p, iter=it)
                st["league"] = (st["league"] + [str(p)])[-8:]
                snap_srv.load_state(load_net(rng.choice(st["league"]), device)[0].state_dict())
            if it % args.eval_every == 0:
                ev = evaluate(pool, net, device, args.eval_seeds)
                rec["eval"] = ev
                print(json.dumps({"iter": it, "eval": ev}), flush=True)
                if st["best"] is None or ev["2p"]["score"] > st["best"]["2p"]["score"]:
                    st["best"] = dict(ev, iter=it)
                    save_net(net, out / "best.pt", iter=it, eval=ev)
            st["history"].append(rec)
            save_net(net, out / "latest.pt", iter=it, opt=ppo.opt.state_dict())
            state_path.write_text(json.dumps(st, indent=1))
    learner_srv.close()
    snap_srv.close()


def collect(bridge, dims, job, learner, snap):
    rng = random.Random(job["seed"])
    seats = job["seats"][:]
    rng.shuffle(seats)
    players = [seat(s) for s in seats]
    agents, learner_slots = {}, set()
    for i, s in enumerate(seats):
        if s == "L":
            agents[i] = ServerAgent(learner, seed=job["seed"] * 31 + i)
            learner_slots.add(i)
        elif s == "S":
            agents[i] = ServerAgent(snap, seed=job["seed"] * 31 + i)
    r = play_game(bridge, dims, players, agents, seed=job["seed"], map_key=job["map"], record=True,
                  startPlayer=rng.randrange(len(players)))
    recs = []
    learner_pis = [pi for pi, traj in r.trajectories.items() if traj and traj[0][2] is not None
                   and r.slot_of.get(pi) in learner_slots]
    for pi in learner_pis:
        z = r.score(pi) if r.over else 1.0 / len(players)
        recs.append(([(e["sample"], a, e["logp"], e["v"], obs.teacher) for obs, a, e in r.trajectories[pi]], z))
    vs_ki = all(s == "L" or s.startswith("ki") for s in seats) and seats.count("L") == 1
    score = r.score(learner_pis[0]) if (vs_ki and learner_pis) else 0.0
    return recs, {"vs_ki": vs_ki, "score": score, "steps": r.steps}


def evaluate(pool, net, device, seeds):
    from .model import NetAgent
    agent = NetAgent(copy.deepcopy(net).eval(), device, greedy=True)
    res = {}
    res["2p"] = run_eval(pool, lambda: agent, "ki:schwer", 2, seeds, base_seed=10_000, progress=False)
    res["4p"] = run_eval(pool, lambda: agent, "ki:schwer", 4, max(5, seeds // 4), base_seed=20_000, progress=False)
    return {k: {"score": round(v["score"], 4), "ci95": round(v["ci95"], 4), "games": v["games"]} for k, v in res.items()}


if __name__ == "__main__":
    main()
