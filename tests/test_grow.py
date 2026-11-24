"""grow.py: the deeper network computes exactly what the old one did.

    python tests/test_grow.py
"""
import sys

import torch

from hochciv_ki.grow import grow
from hochciv_ki.model import Net

DIMS = {"GRID_C": 12, "GLOB_N": 20, "CAND_F": 16, "names": None}


def main():
    torch.manual_seed(0)
    net = Net(DIMS, ch=16, blocks=2, d=32, layers=1, heads=2).eval()
    big = grow(net, 5)
    assert big.cfg["blocks"] == 5 and len(big.tower) == 5
    B, N, H, W = 3, 7, 6, 8
    g = torch.randint(0, 4, (B, 12, H, W), dtype=torch.uint8)
    glob, cand = torch.randn(B, 20), torch.randn(B, N, 16)
    pos = torch.stack([torch.randint(-1, H, (B, N)), torch.randint(-1, W, (B, N))], -1)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[0, 5:] = False
    with torch.no_grad():
        a, b = net(g, glob, cand, pos, mask), big(g, glob, cand, pos, mask)
    ok = torch.isfinite(a[0])
    assert torch.allclose(a[0][ok], b[0][ok], atol=1e-6) and torch.allclose(a[1], b[1], atol=1e-6)
    # the new blocks learn: their zeroed norm gets a gradient
    big.train()
    lo, v = big(g, glob, cand, pos, mask)
    (lo[ok].sum() + v.sum()).backward()
    assert big.tower[4].b2.ln.weight.grad.abs().sum() > 0
    print("ok    grow keeps the function, new blocks get gradients")
    return 0


if __name__ == "__main__":
    sys.exit(main())
