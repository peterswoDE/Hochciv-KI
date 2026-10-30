"""A pool of bridge workers, one Node process per thread (the threads wait on pipes, so the
GIL is not the bottleneck; the engine runs in parallel in the Node processes)."""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

from .bridge import Bridge

DEFAULT_WORKERS = max(1, (os.cpu_count() or 4) - 2)


class BridgePool:
    def __init__(self, workers: int = DEFAULT_WORKERS, engine_dir=None):
        self.workers = workers
        self._local = threading.local()
        self._all: list[Bridge] = []
        self._lock = threading.Lock()
        self._engine_dir = engine_dir
        self._ex = ThreadPoolExecutor(max_workers=workers)
        with Bridge(engine_dir) as b:
            info = b.info()
        self.version = info["version"]
        self.dims = info["dims"]

    def bridge(self) -> Bridge:
        b = getattr(self._local, "bridge", None)
        if b is None:
            b = self._local.bridge = Bridge(self._engine_dir)
            with self._lock:
                self._all.append(b)
        return b

    def map(self, fn, items):
        """fn(bridge, item) for every item, in parallel; results in order."""
        return list(self._ex.map(lambda it: fn(self.bridge(), it), items))

    def submit(self, fn, item):
        return self._ex.submit(lambda: fn(self.bridge(), item))

    def close(self):
        self._ex.shutdown(wait=True)
        for b in self._all:
            b.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
