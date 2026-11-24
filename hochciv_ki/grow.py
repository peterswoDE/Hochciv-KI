"""A deeper map tower for a trained network, without changing how it plays (plan step 5).

New residual blocks go after the old ones with the last norm's scale and shift at zero: the
block adds nothing (relu(x + 0) * m = x, x is already relu'd and masked), so the grown network
computes exactly what the old one did, and PPO then trains the new blocks from there.

    python -m hochciv_ki.grow checkpoints/v6.pt checkpoints/v6deep.pt --blocks 10
"""
from __future__ import annotations

import argparse

import torch

from .model import Net, load_net, save_net


def grow(net: Net, blocks: int) -> Net:
    cfg = dict(net.cfg)
    if blocks < cfg["blocks"]:
        raise ValueError(f"{blocks} blocks: the network has {cfg['blocks']} already")
    old = cfg["blocks"]
    cfg["blocks"] = blocks
    big = Net(net.dims, **cfg).to(next(net.parameters()).device)
    missing, unexpected = big.load_state_dict(net.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("tower.") for k in missing), (missing, unexpected)
    with torch.no_grad():
        for blk in big.tower[old:]:
            blk.b2.ln.weight.zero_()
            blk.b2.ln.bias.zero_()
    big.eval()
    return big


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--blocks", type=int, default=10)
    args = ap.parse_args()
    net, _ = load_net(args.src, device="cpu")
    big = grow(net, args.blocks)
    save_net(big, args.dst, grown_from=args.src)
    n = lambda m: sum(p.numel() for p in m.parameters())
    print(f"{args.src}: {net.cfg['blocks']} blocks, {n(net):,} weights -> {args.dst}: {args.blocks} blocks, {n(big):,} weights")


if __name__ == "__main__":
    main()
