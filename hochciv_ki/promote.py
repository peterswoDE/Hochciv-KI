"""Promotion gate: may a candidate network replace the production one?

Training evals are small (20-60 games per mode, ±0.12) and best.pt is the best of several, so
it is picked at a lucky high point (v4, ppo10, ppo12). The gate looks only at evaluations on
seeds that training never saw, all made with eval_modes --json:

  h2h        candidate against production (--opp ckpt:<production>), suite v2
  cand_ki    candidate against KI schwer          } same suite, same --base-seed, so the
  prod_ki    production against KI schwer         } modes are paired game for game

Rules (all must hold):
  1. head to head: the pooled excess over the fair share, summed over all games, is above
     zero with one-sided 95 % confidence (z > 1.645)
  2. against KI schwer the candidate's mean relative score (score / fair share) is at least
     the production one's minus 0.05
  3. no mode collapses: in no mode against KI schwer is the candidate more than 0.5 (relative)
     below production while also below the fair share (ppo12: marathon 0.000 against 0.438)

    python -m hochciv_ki.promote --h2h runs/eval_a_vs_v4.json --cand-ki runs/eval_a.json --prod-ki runs/eval_v4.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys


def results(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["results"] if isinstance(d, dict) else d


def pooled(res):
    """(excess wins over the fair share, its standard error, games) over all modes."""
    ex = var = n = 0.0
    for r in res:
        g = r["games"]
        ex += (r["score"] - r["fair"]) * g
        var += (r["ci95"] / 1.96 * g) ** 2
        n += g
    return ex, math.sqrt(var), int(n)


def mean_rel(res):
    return sum(r["rel"] for r in res) / len(res)


def gate(h2h, cand_ki, prod_ki):
    """(passed, lines of the report)."""
    out, ok = [], True
    ex, se, n = pooled(h2h)
    z = ex / se if se else (math.inf if ex > 0 else 0.0)
    good = z > 1.645
    ok &= good
    out.append(f"{'ok  ' if good else 'FAIL'} head to head: {ex:+.1f} wins over the fair share in {n} games, "
               f"z = {z:.2f} (needs > 1.645), mean relative {mean_rel(h2h):.3f}")
    c, p = mean_rel(cand_ki), mean_rel(prod_ki)
    good = c >= p - 0.05
    ok &= good
    out.append(f"{'ok  ' if good else 'FAIL'} against KI schwer: mean relative {c:.3f}, production {p:.3f}")
    prod = {r["mode"]: r for r in prod_ki}
    for r in cand_ki:
        q = prod.get(r["mode"])
        if not q:
            continue
        bad = r["rel"] < q["rel"] - 0.5 and r["score"] < r["fair"]
        ok &= not bad
        out.append(f"{'FAIL' if bad else '    '}   {r['mode']:42s} {r['score']:.3f} (production {q['score']:.3f}, "
                   f"fair {r['fair']:.3f})")
    return ok, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h2h", required=True, help="candidate against production")
    ap.add_argument("--cand-ki", required=True, help="candidate against KI schwer")
    ap.add_argument("--prod-ki", required=True, help="production against KI schwer, same seeds")
    ap.add_argument("--json", default=None, help="write the verdict here")
    args = ap.parse_args()
    ok, lines = gate(results(args.h2h), results(args.cand_ki), results(args.prod_ki))
    print("\n".join(lines))
    print("PROMOTE" if ok else "KEEP PRODUCTION")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"promote": ok, "report": lines}, f, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
