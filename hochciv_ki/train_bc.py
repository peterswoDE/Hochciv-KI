"""Behaviour cloning: train the network to pick the JS KI planner's candidate.

    python -m hochciv_ki.train_bc --data data/bc0 data/bc1 --epochs 8 --out checkpoints/bc
    python -m hochciv_ki.train_bc --data data/bc1 --init checkpoints/bc/latest.pt ...
"""
from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import Net, collate, load_net, save_net


class Dataset:
    """All shards of the given directories in memory (grids as uint8, cropped to the map)."""

    def __init__(self, dirs):
        self.grid, self.glob, self.cand, self.pos, self.label, self.z = [], [], [], [], [], []
        for d in dirs:
            for f in sorted(Path(d).glob("shard_*.npz")):
                self._load(f)
        self.n = len(self.label)

    def _load(self, f):
        z = np.load(f)
        lens, shapes = z["lens"], z["gshape"]
        C = 32
        # candidates and seat vectors stay float16 in memory (converted per batch in collate)
        grid, cand, pos = z["grid"], z["cand"], z["pos"]
        go = co = 0
        for i in range(len(lens)):
            h, w = int(shapes[i][0]), int(shapes[i][1])
            self.grid.append(grid[go:go + C * h * w].reshape(C, h, w))
            go += C * h * w
            n = int(lens[i])
            self.cand.append(cand[co:co + n])
            self.pos.append(pos[co:co + n])
            co += n
        self.glob.extend(z["glob"])
        self.label.extend(z["label"].tolist())
        self.z.extend(z["z"].tolist())

    def batches(self, idx, batch, device, shuffle=True, prefetch=4):
        """Yields collated batches; collation runs in a background thread."""
        idx = np.array(idx)
        if shuffle:
            np.random.shuffle(idx)
        q = queue.Queue(maxsize=prefetch)

        def work():
            for s in range(0, len(idx), batch):
                b = idx[s:s + batch]
                t = collate([(self.grid[i], self.glob[i], self.cand[i], self.pos[i]) for i in b], device)
                q.put((t, torch.tensor([self.label[i] for i in b], device=device),
                       torch.tensor([self.z[i] for i in b], device=device, dtype=torch.float32)))
            q.put(None)

        threading.Thread(target=work, daemon=True).start()
        while (item := q.get()) is not None:
            yield item


def run_epoch(net, data, idx, opt, device, batch, train=True, vweight=0.5):
    net.train(train)
    tot = acc = vl = pl = 0.0
    with torch.set_grad_enabled(train):
        for (grid, glob, cand, pos, mask), label, z in data.batches(idx, batch, device, shuffle=train):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, v = net(grid, glob, cand, pos, mask)
            logits = logits.float()
            lp = F.cross_entropy(logits, label)
            lv = F.binary_cross_entropy_with_logits(v.float(), z)
            loss = lp + vweight * lv
            if not torch.isfinite(loss):
                print("skipping a batch with a non-finite loss", flush=True)
                continue
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            n = len(label)
            tot += n
            acc += (logits.argmax(1) == label).sum().item()
            pl += lp.item() * n
            vl += lv.item() * n
    return {"acc": acc / tot, "policy": pl / tot, "value": vl / tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default="checkpoints/bc")
    ap.add_argument("--init", default=None, help="continue from a checkpoint")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val", type=float, default=0.05)
    args = ap.parse_args()
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    t0 = time.perf_counter()
    data = Dataset(args.data)
    print(f"{data.n} samples loaded in {time.perf_counter() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(0)
    perm = rng.permutation(data.n)
    nval = int(data.n * args.val)
    val_idx, train_idx = perm[:nval], perm[nval:]

    if args.init:
        net, _ = load_net(args.init, device)
    else:
        from .bridge import Bridge
        with Bridge() as b:
            dims = b.info()["dims"]
        net = Net(dims).to(device)
    print(f"{sum(p.numel() for p in net.parameters()) / 1e6:.1f}M parameters", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(train_idx) // args.batch + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    class Stepper:  # step the schedule with the optimizer
        def __init__(s, o): s.o = o
        def zero_grad(s, **k): s.o.zero_grad(**k)
        def step(s): s.o.step(); sched.step()

    best = -1
    for ep in range(args.epochs):
        t = time.perf_counter()
        tr = run_epoch(net, data, train_idx, Stepper(opt), device, args.batch, train=True)
        va = run_epoch(net, data, val_idx, None, device, args.batch, train=False)
        print(f"epoch {ep + 1}/{args.epochs} {time.perf_counter() - t:.0f}s  "
              f"train acc {tr['acc']:.3f} pol {tr['policy']:.3f} val {tr['value']:.3f} | "
              f"val acc {va['acc']:.3f} pol {va['policy']:.3f} val {va['value']:.3f}", flush=True)
        save_net(net, out / "latest.pt", epoch=ep + 1, val=va)
        if va["acc"] > best:
            best = va["acc"]
            save_net(net, out / "best.pt", epoch=ep + 1, val=va)


if __name__ == "__main__":
    main()
