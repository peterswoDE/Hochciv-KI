"""Game loop on top of the bridge: observations as numpy arrays and agents that pick actions."""
from __future__ import annotations

import base64
from dataclasses import dataclass, field

import numpy as np

from .bridge import Bridge


@dataclass
class Obs:
    pi: int
    round: int
    tags: list[str]            # candidate tags, the last is "end"
    teacher: int               # index the JS KI planner would take
    glob: np.ndarray           # [GLOB_N]
    cand: np.ndarray           # [n, CAND_F]
    pos: np.ndarray            # [n, 2] int32
    grid: np.ndarray | None    # [C, H, W]


def _f32(s, shape=None):
    a = np.frombuffer(base64.b64decode(s), dtype=np.float32)
    return a.reshape(shape) if shape else a


def decode_obs(res, dims) -> Obs:
    n = len(res["tags"])
    grid = None
    if "grid" in res:
        grid = _f32(res["grid"], (dims["GRID_C"], *res["gshape"]))
    pos = np.frombuffer(base64.b64decode(res["pos"]), dtype=np.int32).reshape(n, 2)
    return Obs(res["pi"], res["round"], res["tags"], res["teacher"], _f32(res["glob"]),
               _f32(res["cand"], (n, dims["CAND_F"])), pos, grid)


@dataclass
class GameResult:
    over: bool
    round: int
    winners: list[int]
    agents: list[int]
    how: str = ""
    steps: int = 0
    trajectories: dict = field(default_factory=dict)   # pi -> list of (Obs, action, extra)
    slot_of: dict = field(default_factory=dict)        # pi (turn order) -> seat in `players`

    def score(self, pi):
        """1 for a win, 1/k for a shared win, 0 otherwise."""
        if pi in self.winners:
            return 1.0 / len(self.winners)
        return 0.0


class Agent:
    """Picks a candidate index from an observation. `extra` is stored with the trajectory."""

    def act(self, obs: Obs):
        raise NotImplementedError

    def act_batch(self, obs_list):
        return [self.act(o) for o in obs_list]


class TeacherAgent(Agent):
    """Follows the JS KI planner one step at a time (the imitation target)."""

    def act(self, obs):
        return obs.teacher, None


class KiTurnAgent(Agent):
    """Lets the JS KI play the agent's whole turn with kiTurn (the reference strength)."""
    full_turn = True


def play_game(bridge: Bridge, dims, players, agents: dict, seed, map_key="plaettchen",
              record=False, max_steps=4000, **kw) -> GameResult:
    """Play one game. `agents` maps a seat index in `players` (lobby order) to an Agent; the
    bridge reports agent seats in turn order, so they are matched by slot."""
    st = bridge.reset(players, seed=seed, map=map_key, **kw)
    gid = st["gid"]
    state = bridge.state(gid)
    slot_of = {i: p["slot"] for i, p in enumerate(state["players"])}
    by_pi = {pi: agents[slot_of[pi]] for pi in st["agents"]}
    traj = {pi: [] for pi in st["agents"]}
    steps = 0
    try:
        while not st["over"] and steps < max_steps:
            pi = st["cur"]
            if getattr(by_pi[pi], "full_turn", False):
                st = bridge.ki_turn(gid)
                steps += 1
                continue
            obs = decode_obs(bridge.call("observe", gid=gid), dims)
            a, extra = by_pi[pi].act(obs)
            if record:
                traj[pi].append((obs, a, extra))
            st = bridge.step(gid, a)
            steps += 1
    finally:
        bridge.close_game(gid)
    return GameResult(st["over"], st["round"], st.get("winners", []), st["agents"], st.get("how", ""),
                      steps, traj, slot_of)
