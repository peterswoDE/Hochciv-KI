"""Python client for bridge/worker.js (one Node process per Bridge)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

WORKER = Path(__file__).resolve().parent.parent / "bridge" / "worker.js"


class BridgeError(RuntimeError):
    pass


class Bridge:
    """A Node worker running the HochCiv engine. Requests are synchronous."""

    def __init__(self, engine_dir: str | os.PathLike | None = None, node: str | None = None):
        env = dict(os.environ)
        if engine_dir:
            env["HOCHCIV_ENGINE_DIR"] = str(engine_dir)
        node = node or shutil.which("node") or "node"
        self._proc = subprocess.Popen(
            [node, str(WORKER)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            env=env, text=True, encoding="utf-8", bufsize=1,
        )
        self._next_id = 0

    def call(self, cmd: str, **args):
        self._next_id += 1
        req = dict(args, id=self._next_id, cmd=cmd)
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise BridgeError(f"worker exited (code {self._proc.poll()})")
        res = json.loads(line)
        if not res.get("ok"):
            raise BridgeError(res.get("error", "unknown error"))
        return res

    # Convenience wrappers -------------------------------------------------

    def info(self):
        return self.call("info")

    def reset(self, players, seed=None, map="plaettchen", **kw):
        return self.call("reset", players=players, seed=seed, map=map, **kw)

    def candidates(self, gid):
        return self.call("candidates", gid=gid)["cands"]

    def step(self, gid, i):
        return self.call("step", gid=gid, i=i)

    def ki_turn(self, gid):
        return self.call("ki_turn", gid=gid)

    def state(self, gid):
        return self.call("state", gid=gid)["state"]

    def close_game(self, gid):
        return self.call("close", gid=gid)

    def close(self):
        if self._proc.poll() is None:
            self._proc.stdin.close()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
