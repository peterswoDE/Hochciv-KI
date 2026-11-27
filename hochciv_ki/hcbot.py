"""Hard-coded Python bots (hardcoded/, formerly the HochCiv-HardcodedBots repo) as opponents.

A seat of kind 'ext' is a human seat to the engine that our worker does not play. When it is
on turn, the state goes to a HardcodedBots worker (its `load`: a sandbox where nothing moves
by itself), the bot plays its turn there and ends it, and the state comes back
(external_done), where our worker plays the KI seats up to the next agent or external seat.

The code is found at hardcoded/ of this repo or HOCHCIV_HCBOTS_DIR. Its worker loads the
engine from HOCHCIV_ENGINE_DIR like ours, so both run the same version.
"""
from __future__ import annotations

import importlib
import os
import sys
import threading
from pathlib import Path

HC_DIR = Path(os.environ.get("HOCHCIV_HCBOTS_DIR")
              or Path(__file__).resolve().parents[1] / "hardcoded")
_local = threading.local()
_imported = False
_lock = threading.Lock()


def _import():
    global _imported
    with _lock:
        if not _imported:
            if not (HC_DIR / "template" / "hcbots").is_dir():
                raise FileNotFoundError(f"hard-coded bots not found at {HC_DIR}")
            sys.path[:0] = [str(HC_DIR / "template"), str(HC_DIR)]
            _imported = True
    from hcbots.bridge import Bridge
    from hcbots.runner import end_turn
    from hcbots.turn import Turn
    return Bridge, Turn, end_turn


def hc_bridge():
    """The HardcodedBots worker of this thread (one per thread, like the bridge pool)."""
    b = getattr(_local, "bridge", None)
    if b is None:
        Bridge, _, _ = _import()
        b = _local.bridge = Bridge(os.environ.get("HOCHCIV_ENGINE_DIR") or None)
    return b


def load_bot(name: str, opts: dict | None = None):
    _import()
    mod = importlib.import_module(f"bots.{name}.bot")
    cls = next(v for v in vars(mod).values() if isinstance(v, type) and getattr(v, "name", None) == name)
    return cls(**(opts or {}))


class HcAgent:
    """One hard-coded bot for one game (the bots keep per-game caches): make a new one per game."""
    external = True

    def __init__(self, name: str = "rusher", **opts):
        self.name = name
        self.bot = load_bot(name, opts)
        self.errors = 0

    def play_turn(self, bridge, gid) -> dict:
        """Play the external seat on turn in our game `gid`; returns the new status."""
        _, Turn, end_turn = _import()
        hc = hc_bridge()
        S = bridge.call("external_state", gid=gid)["state"]
        hg = hc.call("load", state=S, pi=S["cur"])["gid"]
        try:
            t = Turn(hc, hg)
            try:
                self.bot.turn(t)
            except Exception:
                self.errors += 1
            if not t.over:
                if end_turn(t):
                    # refused even after freeing armies: our worker ends it as it stands
                    self.errors += 1
                    return bridge.call("external_done", gid=gid, state=hc.call("state", gid=hg)["state"],
                                       force=True)
            S2 = hc.call("state", gid=hg)["state"]
        finally:
            hc.close_game(hg)
        return bridge.call("external_done", gid=gid, state=S2)


def is_hc(spec: str) -> bool:
    return spec.startswith("hc:")


def hc_agent(spec: str) -> HcAgent:
    """'hc:rusher' -> a fresh HcAgent."""
    return HcAgent(spec.partition(":")[2] or "rusher")
