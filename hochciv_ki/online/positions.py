"""The bot's own positions in online games, to learn from the games it lost (plan step 3).

At the first decision of every round the client appends the server's state (one JSON line:
engine, round, pi, state) to runs/online_positions/open/<game>.jsonl. When the game ends, a
lost game's file moves to lost/, every other one (won, shared win, left) is deleted.
train_ppo --loss-dir runs/online_positions/lost starts some of its games from these positions
(bridge start_from: the bot's seat, everybody else KI schwer, a new seed).

Files of games whose end the client never saw (it was off, the game was cancelled) are
dropped after PRUNE_DAYS.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path

log = logging.getLogger("hochciv_ki.online")
PRUNE_DAYS = 30


def safe(key) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(key))[:80]


class Positions:
    def __init__(self, root):
        self.root = Path(root)
        self.open_dir, self.lost_dir = self.root / "open", self.root / "lost"
        self.open_dir.mkdir(parents=True, exist_ok=True)
        self.lost_dir.mkdir(parents=True, exist_ok=True)
        self.last_round = {}         # key -> round saved last
        self.lock = threading.Lock()

    def _last_round(self, f: Path):
        try:
            with f.open("rb") as fh:
                lines = fh.read().splitlines()
            return json.loads(lines[-1]).get("round") if lines else None
        except (OSError, ValueError, IndexError):
            return None

    def record(self, key, state, pi, engine):
        """Save the position if it is the first one of its round in game `key`."""
        try:
            rnd = state.get("round")
            if state.get("over") or rnd is None:
                return
            f = self.open_dir / f"{safe(key)}.jsonl"
            with self.lock:
                if key not in self.last_round and f.exists():
                    self.last_round[key] = self._last_round(f)
                if self.last_round.get(key) == rnd:
                    return
                self.last_round[key] = rnd
                with f.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"engine": engine, "round": rnd, "pi": pi, "key": str(key),
                                         "saved": int(time.time()), "state": state}, separators=(",", ":")) + "\n")
        except Exception:  # noqa: BLE001 - never stop a game for this
            log.exception("position of %s not saved", key)

    def finish(self, key, won):
        """Game `key` ended: lost (won False) -> lost/, otherwise the positions are deleted."""
        f = self.open_dir / f"{safe(key)}.jsonl"
        with self.lock:
            self.last_round.pop(key, None)
            try:
                if not f.exists():
                    return
                if won is False:
                    f.replace(self.lost_dir / f.name)
                    log.info("lost game %s: positions kept for training", key)
                else:
                    f.unlink()
            except OSError:
                log.exception("positions of %s", key)

    def prune(self, days=PRUNE_DAYS):
        cut = time.time() - days * 86400
        for f in self.open_dir.glob("*.jsonl"):
            try:
                if f.stat().st_mtime < cut:
                    f.unlink()
            except OSError:
                pass


def load_lost(folder, engine=None):
    """[(file, line no)] of every saved position in `folder` (engine `engine` only)."""
    out = []
    for f in sorted(Path(folder).glob("*.jsonl")):
        try:
            with f.open(encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    head = line[:200]
                    m = re.search(r'"engine":"([^"]*)"', head)
                    if engine and m and m.group(1) != engine:
                        continue
                    out.append((str(f), i))
        except OSError:
            continue
    return out


def read_position(ref):
    f, i = ref
    with open(f, encoding="utf-8") as fh:
        for k, line in enumerate(fh):
            if k == i:
                return json.loads(line)
    raise IndexError(ref)
