"""Nation choice for the online client: which nation and ability to take where the game lets
the bot choose (live lobbies and Rundenspiele with civMode fixed, and the nation draft).

The scores come from civ_eval (civ_pref.json: mean rel of every nation:ability over the modes
duel, four and rush). A nation or ability the table doesn't know (a new one of a later
engine) gets the median score: it is taken, but after the known strong ones, and never
crashes the client.
"""
from __future__ import annotations

import json
from pathlib import Path

PREF_FILE = Path(__file__).with_name("civ_pref.json")
RANDOM = ("random", "zufall", None, "")


def load_scores(path=PREF_FILE) -> dict:
    try:
        return dict(json.loads(Path(path).read_text(encoding="utf-8")).get("scores") or {})
    except (OSError, ValueError):
        return {}


SCORES = load_scores()


def _median(scores):
    v = sorted(scores.values())
    return v[len(v) // 2] if v else 0.0


def is_marathon(cfg: dict) -> bool:
    cfg = cfg or {}
    custom = cfg.get("mapKey") == "eigene"
    return cfg.get("mapKey") == "marathon" or bool(custom and (cfg.get("customMap") or {}).get("marathon"))


def civ_mode(cfg: dict) -> str:
    """The server's CivDraft.modesOf: civMode, also for configs with only randomCivs."""
    cfg = cfg or {}
    m = cfg.get("civMode")
    if m in ("fixed", "random", "draft", "randomDraft"):
        return m
    return "random" if cfg.get("randomCivs") else "fixed"


def best_ability(civ: str, scores=None):
    """(ability, score) of the best known ability of `civ`; ("random", median) for an unknown nation."""
    scores = SCORES if scores is None else scores
    known = [(s, arm.split(":", 1)[1]) for arm, s in scores.items() if arm.split(":", 1)[0] == civ]
    if not known:
        return "random", _median(scores)
    s, ab = max(known)
    return ab, s


def ranked_civs(scores=None, marathon=False):
    """[(civ, ability)] best first, one entry per nation. Marathon drafts the abilities: 'basis'."""
    scores = SCORES if scores is None else scores
    civs = {arm.split(":", 1)[0] for arm in scores}
    out = []
    for c in civs:
        ab, s = best_ability(c, scores)
        if marathon:
            s = scores.get(f"{c}:basis", s)
            ab = "basis"
        out.append((s, c, ab))
    return [(c, ab) for s, c, ab in sorted(out, key=lambda x: (-x[0], x[1]))]


def entry_score(e: dict, noability: bool, scores=None) -> tuple[float, str | None]:
    """Score and ability to send for a pool entry of the nation draft."""
    scores = SCORES if scores is None else scores
    civ, ab = e.get("civ"), e.get("ability")
    if ab:                                     # randomDraft: the ability comes with the entry
        return scores.get(f"{civ}:{ab}", best_ability(civ, scores)[1]), None
    if noability:                              # Marathon: the ability is drafted later
        return scores.get(f"{civ}:basis", best_ability(civ, scores)[1]), None
    ab, s = best_ability(civ, scores)
    return s, (None if ab in RANDOM else ab)


def draft_pick(draft: dict, scores=None):
    """(entry id, ability or None) for the nation draft payload, or None if nothing is left."""
    pool = [e for e in (draft or {}).get("pool") or [] if (e.get("left") or 0) > 0]
    if not pool:
        return None
    noab = bool(draft.get("noAbility")) or draft.get("mode") == "randomDraft"
    best = max(pool, key=lambda e: (entry_score(e, noab, scores)[0], str(e.get("id"))))
    return best.get("id"), entry_score(best, noab, scores)[1]
