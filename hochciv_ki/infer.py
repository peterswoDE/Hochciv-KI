"""Batched inference: many game threads ask one GPU thread for policy outputs.

The engine runs in the Node workers, so the threads mostly wait; a request is answered with
the logits and value of a whole batch of observations collected within a millisecond or two.
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import Future

import numpy as np
import torch

from .env import Agent, Obs
from .model import Net, collate, grid_scale, grid_to_u8


class InferenceServer:
    def __init__(self, net: Net, device="cuda", max_batch=64, wait=0.002):
        self.net, self.device, self.max_batch, self.wait = net, device, max_batch, wait
        self.q: queue.Queue = queue.Queue()
        self.lock = threading.Lock()          # held while the weights are swapped
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def request(self, sample) -> Future:
        f = Future()
        self.q.put((sample, f))
        return f

    def load_state(self, state_dict):
        with self.lock:
            self.net.load_state_dict(state_dict)
            self.net.eval()

    def _run(self):
        while not self._stop:
            try:
                first = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            items = [first]
            try:
                while len(items) < self.max_batch:
                    items.append(self.q.get(timeout=self.wait))
            except queue.Empty:
                pass
            try:
                with self.lock, torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, v = self.net(*collate([s for s, _ in items], self.device))
                logits = logits.float().cpu().numpy()
                v = torch.sigmoid(v.float()).cpu().numpy()
                for i, (s, f) in enumerate(items):
                    f.set_result((logits[i, :s[2].shape[0]], float(v[i])))
            except Exception as e:  # noqa: BLE001 - hand the error to every waiting thread
                for _, f in items:
                    if not f.done():
                        f.set_exception(e)

    def close(self):
        self._stop = True
        self._t.join(timeout=2)


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class ServerAgent(Agent):
    """An agent backed by an InferenceServer. Samples (or picks greedily) and records the
    log-probability and value of its choice for PPO."""

    def __init__(self, server: InferenceServer, greedy=False, seed=None):
        self.server, self.greedy = server, greedy
        self.scale = grid_scale(server.net.dims)
        self.rng = np.random.default_rng(seed)

    def act(self, obs: Obs):
        sample = (grid_to_u8(obs.grid, self.scale), obs.glob, obs.cand, obs.pos)
        logits, v = self.server.request(sample).result()
        p = softmax(logits)
        a = int(p.argmax()) if self.greedy else int(self.rng.choice(len(p), p=p))
        return a, {"logp": float(np.log(p[a] + 1e-12)), "v": v, "sample": sample}
