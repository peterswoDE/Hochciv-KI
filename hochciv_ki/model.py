"""The policy/value network and the agent that plays with it.

State: a residual CNN over the map planes plus an MLP over the per-seat vectors.
Policy: one row per candidate (its features plus the map features at its target cell),
mixed with the state and with each other by a small transformer; one logit per candidate.
Value: the observer's expected win share (sigmoid).
"""
from __future__ import annotations

import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import Agent, Obs

# Integer scale of every map plane (features.js stores value / scale); used to keep the
# planes as uint8 on disk and in memory.
GRID_SCALE = np.array([1] * 8 + [3] * 3 + [1] * 4 + [5] * 4 + [1] + [1] * 4 + [4] + [2] + [1] * 4 + [2] + [1],
                      dtype=np.float32)


def grid_to_u8(grid: np.ndarray) -> np.ndarray:
    """Quantize to uint8 and crop to the map (rows/columns with any terrain)."""
    g = np.clip(np.rint(grid * GRID_SCALE[:, None, None]), 0, 255).astype(np.uint8)
    terrain = g[:8].any(0)
    h = int(np.nonzero(terrain.any(1))[0].max()) + 1
    w = int(np.nonzero(terrain.any(0))[0].max()) + 1
    return np.ascontiguousarray(g[:, :h, :w])


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class Net(nn.Module):
    def __init__(self, dims, ch=96, blocks=6, d=256, layers=2, heads=4):
        super().__init__()
        self.dims = dict(dims)
        self.cfg = dict(ch=ch, blocks=blocks, d=d, layers=layers, heads=heads)
        C, G, CF = dims["GRID_C"], dims["GLOB_N"], dims["CAND_F"]
        self.register_buffer("gscale", torch.tensor(GRID_SCALE[:C]))
        self.stem = nn.Sequential(nn.Conv2d(C, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU())
        self.tower = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.glob = nn.Sequential(nn.Linear(G, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.state = nn.Sequential(nn.Linear(2 * ch + d, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.cand = nn.Sequential(nn.Linear(CF + ch, d), nn.ReLU(), nn.Linear(d, d))
        self.mix = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True), layers)
        self.logit = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.value = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, grid_u8, glob, cand, pos, mask):
        """grid_u8 [B,C,H,W] uint8, glob [B,G], cand [B,N,F], pos [B,N,2] long, mask [B,N] bool
        (True = real candidate). Returns logits [B,N] (-inf on padding) and value logits [B]."""
        B, C, H, W = grid_u8.shape
        g = grid_u8.float() / self.gscale.view(1, C, 1, 1)
        valid = g[:, C - 1:C]                                   # last plane: inside the map
        x = self.tower(self.stem(g))
        n_valid = valid.sum((2, 3)).clamp(min=1)
        mean = (x * valid).sum((2, 3)) / n_valid
        mx = (x - 1e4 * (1 - valid)).amax((2, 3))
        s = self.state(torch.cat([mean, mx, self.glob(glob)], 1))
        # map features at every candidate's target cell
        r, c = pos[..., 0], pos[..., 1]
        has = (r >= 0) & (c >= 0)
        idx = (r.clamp(min=0) * W + c.clamp(min=0)).unsqueeze(1).expand(-1, x.shape[1], -1)
        at = torch.gather(x.flatten(2), 2, idx).transpose(1, 2) * has.unsqueeze(-1)
        h = self.cand(torch.cat([cand, at], -1)) + s.unsqueeze(1)
        h = self.mix(h, src_key_padding_mask=~mask)
        logits = self.logit(torch.cat([h, s.unsqueeze(1).expand_as(h)], -1)).squeeze(-1)
        logits = logits.masked_fill(~mask, float("-inf"))
        return logits, self.value(s).squeeze(-1)


def collate(samples, device):
    """samples: list of (grid_u8 [C,h,w], glob [G], cand [n,F], pos [n,2]) -> batch tensors.
    Grids are padded to the largest map in the batch (padding is outside the map)."""
    B = len(samples)
    N = max(s[2].shape[0] for s in samples)
    F_ = samples[0][2].shape[1]
    C = samples[0][0].shape[0]
    H = max(s[0].shape[1] for s in samples)
    W = max(s[0].shape[2] for s in samples)
    grid = np.zeros((B, C, H, W), np.uint8)
    for i, s in enumerate(samples):
        grid[i, :, :s[0].shape[1], :s[0].shape[2]] = s[0]
    grid = torch.from_numpy(grid)
    glob = torch.from_numpy(np.stack([s[1] for s in samples]))
    cand = np.zeros((B, N, F_), np.float32)
    pos = np.full((B, N, 2), -1, np.int64)
    mask = np.zeros((B, N), bool)
    for i, s in enumerate(samples):
        n = s[2].shape[0]
        cand[i, :n] = s[2]
        pos[i, :n] = s[3]
        mask[i, :n] = True
    to = lambda t: t.to(device, non_blocking=True)
    return to(grid), to(glob), to(torch.from_numpy(cand)), to(torch.from_numpy(pos)), to(torch.from_numpy(mask))


def save_net(net: Net, path, **extra):
    torch.save({"dims": net.dims, "cfg": net.cfg, "state": net.state_dict(), **extra}, path)


def load_net(path, device="cuda") -> tuple[Net, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = Net(ck["dims"], **ck["cfg"]).to(device)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck


class NetAgent(Agent):
    """Plays with a network. Thread-safe (a lock around the forward pass); `greedy` picks the
    best candidate, otherwise it samples with `temperature`."""

    def __init__(self, net: Net, device="cuda", greedy=True, temperature=1.0):
        self.net, self.device, self.greedy, self.temperature = net, device, greedy, temperature
        self._lock = threading.Lock()

    @torch.no_grad()
    def act(self, obs: Obs):
        sample = (grid_to_u8(obs.grid), obs.glob, obs.cand, obs.pos)
        with self._lock:
            logits, v = self.net(*collate([sample], self.device))
        logits = logits[0]
        if self.greedy:
            a = int(logits.argmax())
        else:
            a = int(torch.distributions.Categorical(logits=logits / self.temperature).sample())
        return a, {"v": float(torch.sigmoid(v[0]))}


def load_agent(path, device="cuda", greedy=True):
    net, _ = load_net(path, device)
    return NetAgent(net, device, greedy=greedy)
