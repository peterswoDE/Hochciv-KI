"""Search stage 3: the improved policy, recording during a real game, the training phase.

    python -m tests.test_expert
"""
from __future__ import annotations

import copy
import sys
import traceback

import numpy as np
import torch

from hochciv_ki.bridge import Bridge
from hochciv_ki.env import play_game
from hochciv_ki.expert import ExpertAgent, improved_policy, search_phase
from hochciv_ki.infer import InferenceServer, softmax
from hochciv_ki.model import load_net

CKPT = "checkpoints/v3.pt"


def test_improved_policy():
    p = np.array([0.5, 0.3, 0.2])
    # nothing searched, or searched at exactly v: the policy stays
    assert np.allclose(improved_policy(p, 0.5, {}, 0.03), p, atol=1e-6)
    assert np.allclose(improved_policy(p, 0.5, {0: 0.5, 1: 0.5}, 0.03), p, atol=1e-6)
    t = improved_policy(p, 0.5, {1: 0.56, 0: 0.47}, 0.03)
    assert abs(t.sum() - 1) < 1e-5
    assert t[1] > t[0] and t[1] > p[1] and t[0] < p[0], t
    # the candidate not searched keeps its share against nothing but the searched ones
    assert t[2] / p[2] > t[0] / p[0], t
    # a bigger tau trusts the search less
    assert improved_policy(p, 0.5, {1: 0.56}, 0.3)[1] < improved_policy(p, 0.5, {1: 0.56}, 0.03)[1]


def play(frac, k=3):
    net, _ = load_net(CKPT, "cuda")
    srv = InferenceServer(net.eval(), "cuda")
    try:
        with Bridge() as b:
            dims = b.info()["dims"]
            agent = ExpertAgent(srv, frac=frac, k=k, seed=5)
            r = play_game(b, dims, [{"kind": "agent"}, {"kind": "ki", "kiLevel": "schwer"}], {0: agent}, seed=9,
                          map_key="original", record=True)
            # the forks are gone: only the games the test itself opens exist
            st = b.reset([{"kind": "agent"}, {"kind": "ki"}], seed=1, map="original")
            assert st["gid"] is not None
        return r
    finally:
        srv.close()


def test_searched_decisions_carry_targets():
    r = play(0.5)
    assert r.steps > 0
    ex = [e for t in r.trajectories.values() for _, _, e in t]
    tg = [e for e in ex if "target" in e]
    assert tg and len(tg) < len(ex), (len(tg), len(ex))
    for e in tg:
        assert abs(float(e["target"].sum()) - 1) < 1e-4
        assert e["target"].shape == e["p"].shape == (e["sample"][2].shape[0],)
    assert all("logp" in e and "v" in e and "sample" in e for e in ex)


def test_without_search_nothing_is_recorded():
    ex = [e for t in play(0.0).trajectories.values() for _, _, e in t]
    assert ex and not any("target" in e for e in ex)


def test_the_game_is_not_changed_by_searching():
    """Forks must leave the game alone: same seed, same moves, with or without search."""
    a, b = play(0.0), play(1.0)
    moves = lambda r: [m for t in r.trajectories.values() for _, m, _ in t][:8]
    assert moves(a) == moves(b), (moves(a), moves(b))


def test_search_phase_moves_the_policy_to_the_targets():
    net, _ = load_net(CKPT, "cuda")
    with Bridge() as b:
        dims = b.info()["dims"]
        srv = InferenceServer(copy.deepcopy(net).eval(), "cuda")
        agent = ExpertAgent(srv, frac=1.0, k=3, seed=5)
        items = []
        for seed in (21, 22):
            r = play_game(b, dims, [{"kind": "agent"}, {"kind": "ki", "kiLevel": "schwer"}], {0: agent}, seed=seed,
                          map_key="original", record=True)
            items += [(e["sample"], e["target"], e["p"]) for t in r.trajectories.values() for _, _, e in t
                      if "target" in e]
        srv.close()
    assert items
    # a target far from the policy, so the phase has something to learn
    items = [(s, softmax(np.log(t + 1e-12) * 0.0 + (np.arange(len(t)) == 0) * 5.0).astype(np.float32), p)
             for s, t, p in items]
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    res = search_phase(net, opt, items, "cuda", mb=32, epochs=4, max_kl=10.0)
    assert res["search_steps"] > 0 and res["search_gap1"] < res["search_gap0"], res
    res = search_phase(net, opt, items, "cuda", mb=4, epochs=4, max_kl=0.0)
    assert res["search_steps"] == 1, res                       # the brake works
    assert search_phase(net, opt, [], "cuda")["search_n"] == 0


def main():
    fails = 0
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        try:
            t()
            print("ok   ", t.__name__)
        except Exception:  # noqa: BLE001
            fails += 1
            traceback.print_exc()
            print("FAIL ", t.__name__)
    print(f"{len(tests) - fails} of {len(tests)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
