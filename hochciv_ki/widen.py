"""Carry a trained network over to a new observation layout (the bridge's current dims).

Inputs are matched by name (dims["names"], features.js): the weights of an input that exists
in both layouts are copied, new inputs start at zero weight, so the widened network plays
exactly like the old one wherever the new inputs are zero. Only the three input layers depend
on the layout (the map stem, the global MLP, the candidate MLP); everything else is copied.

Checkpoints from before the names existed (layout v1: 4 seats, no wonder/event inputs) get
their names reconstructed from the v1 rules.

    python -m hochciv_ki.widen checkpoints/v1.pt checkpoints/v1w.pt
"""
from __future__ import annotations

import argparse

import torch

from .bridge import Bridge
from .model import Net, load_net, save_net

V1_SEAT_BASE = ['present', 'dead', 'cur', 'turnPos', 'resSci', 'resFood', 'resCoins', 'incSci',
                'incFood', 'incCoins', 'power', 'pop', 'econProgress', 'cities', 'armies', 'techs',
                'capitalsTaken', 'freeArmies', 'backPicks', 'wonders', 'powerOf', 'claim']
V1_GAME = ['round', 'players', 'duel', 'marathon', 'wonders', 'events', 'claims', 'worldPop']
V1_NUM = ['powerN', 'strikeN', 'defendN', 'dv', 'spent', 'ratio', 'gain', 'dResSci', 'dResFood',
          'dResCoins', 'dIncSci', 'dIncFood', 'dIncCoins', 'dPop', 'dCities', 'dArmies', 'dPower',
          'dTechs', 'dEPop', 'dECities', 'dEArmies', 'win', 'lose', 'dvBase']


def v1_names(old_dims, new_dims):
    """Input names of layout v1, using the nation, ability and tech keys of the new layout."""
    seat0 = [n.split(":", 1)[1] for n in new_dims["names"]["glob"] if n.startswith("seat0:")]
    keys = lambda pre: [x for x in seat0 if x.startswith(pre)]
    seat = V1_SEAT_BASE + keys("civ:") + keys("abil:") + keys("tech:") + keys("avail:")
    ns = 4
    grid = ([f"terrain{j}" for j in range(8)] + [f"yield{j}" for j in range(3)] + [f"ctrl{j}" for j in range(ns)]
            + [f"city{j}" for j in range(ns)] + ["cap"] + [f"army{j}" for j in range(ns)] + ["mp", "road"]
            + [f"bought{j}" for j in range(ns)] + ["siege", "valid"])
    glob = V1_GAME + [f"seat{k}:{x}" for k in range(ns) for x in seat]
    cand = ["type:" + k for k in old_dims["CAND_TYPES"]] + ["tech:" + k for k in old_dims["TECH_KEYS"]] + V1_NUM
    assert len(grid) == old_dims["GRID_C"] and len(glob) == old_dims["GLOB_N"] and len(cand) == old_dims["CAND_F"], \
        "checkpoint is not layout v1"
    return {"grid": grid, "glob": glob, "cand": cand}


_DIMS = None


def current_dims():
    """The bridge's observation layout (asked once per process)."""
    global _DIMS
    if _DIMS is None:
        with Bridge() as b:
            _DIMS = b.info()["dims"]
    return _DIMS


def copy_columns(new_w, old_w, old_names, new_names, extra=0):
    """new_w[:, j] = old_w[:, i] where the names match; the last `extra` columns 1:1."""
    idx = {n: i for i, n in enumerate(old_names)}
    new_w.zero_()
    hit = 0
    for j, n in enumerate(new_names):
        if n in idx:
            new_w[:, j] = old_w[:, idx[n]]
            hit += 1
    if extra:
        new_w[:, -extra:] = old_w[:, -extra:]
    return hit


def widen(old: Net, new_dims) -> tuple[Net, dict]:
    old_names = old.dims.get("names") or v1_names(old.dims, new_dims)
    new_names = new_dims["names"]
    net = Net(new_dims, **old.cfg).to(next(old.parameters()).device)
    sd_old = old.state_dict()
    sd = net.state_dict()
    report = {}
    with torch.no_grad():
        for k, v in sd.items():
            if k == "gscale":
                continue
            if k == "stem.0.weight":
                report["grid"] = copy_columns(v, sd_old[k], old_names["grid"], new_names["grid"])
            elif k == "glob.0.weight":
                report["glob"] = copy_columns(v, sd_old[k], old_names["glob"], new_names["glob"])
            elif k == "cand.0.weight":
                report["cand"] = copy_columns(v, sd_old[k], old_names["cand"], new_names["cand"], extra=old.cfg["ch"])
            else:
                v.copy_(sd_old[k])
    for part in ("grid", "glob", "cand"):
        report[part] = f"{report[part]} of {len(new_names[part])} inputs from {len(old_names[part])}"
    return net, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args()
    new_dims = current_dims()
    old, ck = load_net(args.src, "cuda", adapt=False)
    net, report = widen(old, new_dims)
    for k, v in report.items():
        print(f"{k}: {v}")
    extra = {k: v for k, v in ck.items() if k not in ("dims", "cfg", "state", "opt")}
    save_net(net, args.dst, widened_from=str(args.src), **extra)


if __name__ == "__main__":
    main()
