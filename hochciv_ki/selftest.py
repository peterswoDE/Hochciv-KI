"""M0 selftest: play complete games with a random agent against the JS KI and bots.

    python -m hochciv_ki.selftest [--games 4]
"""
from __future__ import annotations

import argparse
import random
import time

from .bridge import Bridge

SETUPS = [
    ("1v1 tile map vs KI", "plaettchen", [{"kind": "agent"}, {"kind": "ki", "kiLevel": "leicht"}]),
    ("1v1 original map vs bot", "original", [{"kind": "agent"}, {"kind": "bot"}]),
    ("4p tile map", "plaettchen", [{"kind": "agent"}, {"kind": "ki", "kiLevel": "leicht"}, {"kind": "bot"}, {"kind": "bot"}]),
    ("3p gross map", "gross", [{"kind": "agent"}, {"kind": "bot"}, {"kind": "bot"}]),
]


def play_random(bridge: Bridge, map_key, players, seed, max_steps=5000):
    rng = random.Random(seed)
    st = bridge.reset(players, seed=seed, map=map_key)
    gid, steps, errors = st["gid"], 0, 0
    while not st["over"] and steps < max_steps:
        cands = bridge.candidates(gid)
        # Random policy that ends the turn now and then so games move on
        i = len(cands) - 1 if rng.random() < 0.25 else rng.randrange(len(cands))
        st = bridge.step(gid, i)
        if st.get("err"):
            errors += 1
        steps += 1
    bridge.close_game(gid)
    return st, steps, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=len(SETUPS))
    args = ap.parse_args()
    try:
        import torch
        cuda = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA"
        print(f"torch {torch.__version__}, {cuda}")
    except ImportError:
        print("torch not installed")
    with Bridge() as b:
        info = b.info()
        print(f"engine {info['version']} from {info['engineDir']}")
        ok = True
        for g in range(args.games):
            name, map_key, players = SETUPS[g % len(SETUPS)]
            t0 = time.perf_counter()
            st, steps, errors = play_random(b, map_key, players, seed=1000 + g)
            dt = time.perf_counter() - t0
            agent_won = any(w in st["agents"] for w in st.get("winners", []))
            print(f"{name:28s} seed {1000 + g}: {'over' if st['over'] else 'NOT OVER'} in round {st['round']}, "
                  f"{steps} steps ({errors} rejected), agent {'won' if agent_won else 'lost'}, "
                  f"{st.get('how', '')[:50]!r}, {dt:.1f}s")
            ok &= st["over"]
        print("OK" if ok else "FAILED")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
