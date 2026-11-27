"""Hex grid helpers, a port of js/hex.js (pointy top, odd rows shifted right: "odd-r")."""
from __future__ import annotations

from collections import deque

DIRS_EVEN = [(0, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)]
DIRS_ODD = [(0, 1), (1, 1), (1, 0), (0, -1), (-1, 0), (-1, 1)]

# Terrain yields [sci, food, coins] and passability, as TERRAIN in js/data.js
TERRAIN = {
    "G": {"yield": (0, 1, 0), "land": True},
    "W": {"yield": (0, 0, 1), "land": True},
    "B": {"yield": (1, 0, 1), "land": True},
    "F": {"yield": (0, 1, 1), "land": True},
    "M": {"yield": (0, 0, 1), "land": False},
    "I": {"yield": (1, 1, 1), "land": True},
    "V": {"yield": (0, 0, 0), "land": True, "block": True},
    "X": {"yield": (0, 0, 0), "land": False, "off": True},
}


def key(r: int, c: int) -> str:
    return f"{r},{c}"


def unkey(k: str) -> tuple[int, int]:
    r, c = k.split(",")
    return int(r), int(c)


def neighbors(r: int, c: int) -> list[tuple[int, int]]:
    d = DIRS_ODD if r & 1 else DIRS_EVEN
    return [(r + dr, c + dc) for dr, dc in d]


def _x(r: int, c: int) -> int:
    return c - (r - (r & 1)) // 2


def hex_distance(r1: int, c1: int, r2: int, c2: int) -> int:
    x1, x2 = _x(r1, c1), _x(r2, c2)
    return max(abs(x1 - x2), abs((-x1 - r1) - (-x2 - r2)), abs(r1 - r2))


def within(r: int, c: int, n: int) -> list[tuple[int, int]]:
    out = []
    for dr in range(-n, n + 1):
        for dc in range(-n - 1, n + 2):
            rr, cc = r + dr, c + dc
            d = hex_distance(r, c, rr, cc)
            if 0 < d <= n:
                out.append((rr, cc))
    return out


class Grid:
    """The map of a state: terrain lookups and distance fields."""

    def __init__(self, rows: list[str]):
        self.rows = rows
        self.h = len(rows)

    def terrain(self, r: int, c: int) -> str | None:
        if r < 0 or r >= self.h:
            return None
        row = self.rows[r]
        return row[c] if 0 <= c < len(row) else None

    def is_land(self, r: int, c: int) -> bool:
        t = self.terrain(r, c)
        return bool(t) and TERRAIN.get(t, {}).get("land", False) and not TERRAIN[t].get("block")

    def is_tile(self, r: int, c: int) -> bool:
        t = self.terrain(r, c)
        return bool(t) and not TERRAIN.get(t, {}).get("off")

    def cells(self):
        for r, row in enumerate(self.rows):
            for c in range(len(row)):
                yield r, c

    def distance_field(self, sources, passable=None, limit: int = 200) -> dict[tuple[int, int], int]:
        """Steps from the nearest source over passable cells (default: land). The sources
        themselves need not be passable (a city target)."""
        passable = passable or self.is_land
        dist = {}
        q = deque()
        for s in sources:
            dist[tuple(s)] = 0
            q.append(tuple(s))
        while q:
            r, c = q.popleft()
            d = dist[(r, c)]
            if d >= limit:
                continue
            for nr, nc in neighbors(r, c):
                if (nr, nc) in dist or not passable(nr, nc):
                    continue
                dist[(nr, nc)] = d + 1
                q.append((nr, nc))
        return dist
