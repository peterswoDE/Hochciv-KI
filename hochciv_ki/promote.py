"""Promotion gate: may a candidate network replace the production one?

Training evals are small (20-60 games per mode, ±0.12) and best.pt is the best of several, so
it is picked at a lucky high point (v4, ppo10, ppo12). The gate looks only at evaluations on
seeds that training never saw, all made with eval_modes --json:

  h2h        candidate against production (--opp ckpt:<production>), suite v2
  cand_ki    candidate against KI schwer          } same suite, same --base-seed, so the
  prod_ki    production against KI schwer         } modes are paired game for game
  cand_rush  candidate against the scripted rusher  } suite rush, optional (since ppo16)
  prod_rush  production against the scripted rusher }

Rules (all must hold):
  1. head to head: the pooled excess over the fair share, summed over all games, is above
     zero with one-sided 95 % confidence (z > 1.645)
  2. against KI schwer the candidate's mean relative score (score / fair share) is at least
     the production one's minus 0.05
  3. no mode collapses: in no mode against KI schwer is the candidate more than 0.5 (relative)
     below production while also below the fair share (ppo12: marathon 0.000 against 0.438)
  4. with the rush evaluations: against the rusher the candidate's mean relative score is at
     least the production one's minus 0.05 (the rush is how people beat v4 online)

  5. with --marathon (since ppo31): in no marathon mode against KI schwer is the candidate
     significantly below production (score difference below -1.645 standard errors of the
     difference). ppo28-30 passed rules 1-4 while 4p marathon fell from 0.525 to 0.275; the
     online client then plays marathon games with --marathon-agent (the production network).
     The verdict file also says "core" (rules 1-4) and "marathon" (rule 5) separately.

With --core (since ppo18) rules 1 and 2 count only the core modes (2 to 4 empires, no
marathon); rule 3 still looks at every mode, so 5-8 empires and marathon may not collapse.

    python -m hochciv_ki.promote --h2h runs/eval_a_vs_v4.json --cand-ki runs/eval_a.json --prod-ki runs/eval_v4.json         --cand-rush runs/eval_a_rush.json --prod-rush runs/eval_v4_rush.json
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


def marathon_check(cand_ki, prod_ki):
    """(ok, report lines): rule 5."""
    out, ok = [], True
    prod = {r["mode"]: r for r in prod_ki}
    for r in cand_ki:
        q = prod.get(r["mode"])
        if not q or "marathon" not in r["mode"]:
            continue
        se = math.hypot(r["ci95"], q["ci95"]) / 1.96
        d = r["score"] - q["score"]
        good = d >= -1.645 * se
        ok &= good
        out.append(f"{'ok  ' if good else 'FAIL'} marathon {r['mode']}: {r['score']:.3f}, production {q['score']:.3f} "
                   f"(difference {d:+.3f}, may not be below {-1.645 * se:.3f})")
    return ok, out


def gate(h2h, cand_ki, prod_ki, cand_rush=None, prod_rush=None, core=False):
    """(passed, lines of the report)."""
    out, ok = [], True
    if core:
        from .eval_modes import core_mode
        h2h = [r for r in h2h if core_mode(r["mode"])]
        out.append("core modes only (2 to 4 empires, no marathon) for head to head and KI schwer")
    ex, se, n = pooled(h2h)
    z = ex / se if se else (math.inf if ex > 0 else 0.0)
    good = z > 1.645
    ok &= good
    out.append(f"{'ok  ' if good else 'FAIL'} head to head: {ex:+.1f} wins over the fair share in {n} games, "
               f"z = {z:.2f} (needs > 1.645), mean relative {mean_rel(h2h):.3f}")
    focus = (lambda res: [r for r in res if core_mode(r["mode"])]) if core else (lambda res: res)
    c, p = mean_rel(focus(cand_ki)), mean_rel(focus(prod_ki))
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
    if cand_rush and prod_rush:
        c, p = mean_rel(cand_rush), mean_rel(prod_rush)
        good = c >= p - 0.05
        ok &= good
        out.append(f"{'ok  ' if good else 'FAIL'} against the rusher: mean relative {c:.3f}, production {p:.3f}")
        prod = {r["mode"]: r for r in prod_rush}
        for r in cand_rush:
            q = prod.get(r["mode"])
            if q:
                out.append(f"       {r['mode']:42s} {r['score']:.3f} (production {q['score']:.3f}, "
                           f"fair {r['fair']:.3f})")
    return ok, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h2h", required=True, help="candidate against production")
    ap.add_argument("--cand-ki", required=True, help="candidate against KI schwer")
    ap.add_argument("--prod-ki", required=True, help="production against KI schwer, same seeds")
    ap.add_argument("--cand-rush", default=None, help="candidate against the rusher (suite rush)")
    ap.add_argument("--prod-rush", default=None, help="production against the rusher, same seeds")
    ap.add_argument("--core", action="store_true",
                    help="head to head and KI schwer on the core modes only (2-4 empires, no marathon)")
    ap.add_argument("--marathon", action="store_true",
                    help="rule 5: no marathon mode significantly below production")
    ap.add_argument("--json", default=None, help="write the verdict here")
    args = ap.parse_args()
    if bool(args.cand_rush) != bool(args.prod_rush):
        ap.error("--cand-rush and --prod-rush go together")
    rush = (results(args.cand_rush), results(args.prod_rush)) if args.cand_rush else (None, None)
    core_ok, lines = gate(results(args.h2h), results(args.cand_ki), results(args.prod_ki), *rush, core=args.core)
    mara_ok = True
    if args.marathon:
        mara_ok, more = marathon_check(results(args.cand_ki), results(args.prod_ki))
        lines += more
    ok = core_ok and mara_ok
    print("\n".join(lines))
    print("PROMOTE" if ok else ("PROMOTE WITHOUT MARATHON (client --marathon-agent)" if core_ok else "KEEP PRODUCTION"))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"promote": ok, "core": core_ok, "marathon": mara_ok, "report": lines}, f, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
