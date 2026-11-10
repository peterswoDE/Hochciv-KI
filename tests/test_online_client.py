"""The online client without a server: several Rundenspiele on one socket, turn context per
game, refused actions and the safe end of a turn.

    python -m tests.test_online_client
"""
from __future__ import annotations

import logging
import sys
import traceback
from types import SimpleNamespace

from hochciv_ki.bridge import Bridge
from hochciv_ki.env import TeacherAgent, decode_obs
from hochciv_ki.online.client import Brain, Conn, LiveGame, Rundenspiele, engine_dir_of

ME = 0


def view(gid, log_seq=5, cur=ME, status="playing", **state):
    return {"id": gid, "status": status, "myIndex": ME,
            "state": {"game": gid, "logSeq": log_seq, "cur": cur, "round": 1, "cities": [], **state}}


class FakeBrain:
    """One research per turn, then the end. Remembers every state it decided on."""

    def __init__(self):
        self.seen = []

    def on(self, engine):
        return self

    def next_actions(self, state, pi, key):
        self.seen.append((key, state["game"], state["logSeq"]))
        if state.get("researched"):
            return "end", [("endTurn", {})]
        return "tech:a", [("research", {"tech": "a"})]

    def refuse(self, key, tag):
        pass

    def end_actions(self, state, pi, key):
        return [("endTurn", {})]


class FakeServer:
    """Rundenspiele as sockets/async.js has them: a socket is in the room of the game it
    opened last only, and a broadcast reaches the room before the ack."""

    def __init__(self, *views):
        self.views = {v["id"]: v for v in views}
        self.room = None
        self.actions = []
        self.opened = []
        self.after_action = None

    def update(self, client, gid):
        client.events.put(("async:update", self.views[gid]))

    def call(self, client, event, data):
        gid = data["id"]
        if event == "async:open":
            self.room = gid
            self.opened.append(gid)
            return {"ok": True, "view": self.views[gid]}
        self.actions.append((gid, data["type"]))
        old = self.views[gid]["state"]
        if data["type"] == "endTurn":
            self.views[gid] = view(gid, old["logSeq"] + 1, cur=ME + 1)
        else:
            self.views[gid] = view(gid, old["logSeq"] + 1, researched=True)
        if self.room == gid:
            self.update(client, gid)
        if self.after_action:
            hook, self.after_action = self.after_action, None
            hook()
        return {"ok": True}


class FakeRundenspiele(Rundenspiele):
    def __init__(self, server, brain):
        super().__init__(SimpleNamespace(acc=None, brain=brain, settings={}))
        self.server = server

    def call(self, event, data, timeout=30):
        return self.server.call(self, event, data)

    def pump(self):
        while self.stash or not self.events.empty():
            self.handle(*self.next_event(0))


def test_update_of_another_game_during_a_turn():
    brain, server = FakeBrain(), FakeServer(view("A"), view("B", 9))
    rs = FakeRundenspiele(server, brain)
    # still on its way from before A was opened: the socket is no longer in B's room
    server.after_action = lambda: server.update(rs, "B")
    rs.open_game("A")
    assert brain.seen == [("A", "A", 5), ("A", "A", 6)], brain.seen
    assert server.actions == [("A", "research"), ("A", "endTurn")], server.actions
    rs.pump()
    # B is opened again, so that its own updates reach the next decision
    assert server.opened == ["A", "B"], server.opened
    assert brain.seen[2:] == [("B", "B", 9), ("B", "B", 10)], brain.seen
    assert server.actions[2:] == [("B", "research"), ("B", "endTurn")], server.actions
    assert rs.views["A"]["state"]["game"] == "A" and rs.views["B"]["state"]["game"] == "B"


def test_stale_update_after_the_turn_changes_nothing():
    brain, server = FakeBrain(), FakeServer(view("A"))
    rs = FakeRundenspiele(server, brain)
    stale = view("A")
    rs.open_game("A")
    actions = list(server.actions)
    rs.events.put(("async:update", stale))
    rs.pump()
    assert server.actions == actions, server.actions


def test_older_and_stateless_views_never_replace_newer_ones():
    rs = FakeRundenspiele(FakeServer(), FakeBrain())
    rs.accept(view("A", 7))
    rs.accept(view("A", 6))
    assert rs.views["A"]["state"]["logSeq"] == 7
    rs.accept({"id": "A", "status": "placement"})
    assert rs.views["A"]["state"]["logSeq"] == 7
    rs.accept(view("A", 7, marker=1))            # an action that logs nothing keeps logSeq
    assert rs.views["A"]["state"]["marker"] == 1
    rs.accept({"id": "B", "status": "setup"})
    rs.accept({"id": "B", "status": "placement"})
    assert rs.views["B"]["status"] == "placement"
    rs.accept(view("B", 1))
    assert rs.views["B"]["state"]


def test_reconnect_and_poll_drop_views():
    rs = FakeRundenspiele(FakeServer(), FakeBrain())
    rs.poll = lambda: None
    rs.accept(view("A"))
    rs.opened = "A"
    rs.handle("connect", None)
    assert rs.views == {} and rs.opened is None

    rs = FakeRundenspiele(FakeServer(), FakeBrain())
    rs.accept(view("A"))
    rs.accept(view("B"))
    rs.mgr.settings = {"join": {"on": False, "async": False}, "host": {"on": False}}
    rs.acc = SimpleNamespace(get=lambda path: SimpleNamespace(json=lambda: {"mine": [{"id": "A"}], "open": []}))
    rs.poll()
    assert set(rs.views) == {"A"}, rs.views


def position():
    """(server, its gid, state, seat) of a real game with an army in a city on the agent's turn."""
    agent, server = TeacherAgent(), Bridge()
    with Bridge() as client:
        dims = client.info()["dims"]
        for seed in (506, *range(500, 530)):       # 506: right in round 1 (engine v98)
            st = server.reset([{"kind": "agent"}, {"kind": "ki", "kiLevel": "schwer"}], seed=seed, map="original")
            gid, me = st["gid"], st["agents"][0]
            for _ in range(200):
                if st["over"] or st["round"] > 15:
                    break
                S = server.state(gid)
                cells = {(c["r"], c["c"]) for c in S["cities"]}
                if S["cur"] == me and any(a["owner"] == me and (a["r"], a["c"]) in cells for a in S["armies"]) \
                        and server.call("server_action", gid=gid, type="endTurn", params={})["err"]:
                    return server, gid, S, me
                client.call("load", gid=1, state=S, pi=me)
                a, _ = agent.act(decode_obs(client.call("observe", gid=1), dims))
                for typ, params in client.call("step_online", gid=1, i=a)["actions"]:
                    st = server.call("server_action", gid=gid, type=typ, params=params)
                    if st["over"]:
                        break
            server.close_game(gid)
    raise AssertionError("no position with an army in a city")


def conn(brain):
    return Conn(SimpleNamespace(acc=None, brain=brain), "test")


def test_turn_context_per_game():
    with Bridge() as b:
        st = b.reset([{"kind": "agent"}, {"kind": "ki"}], seed=7, map="original")
        S, me = b.state(st["gid"]), st["agents"][0]
        b.call("load", gid=1, state=S, pi=me, key="A")
        tag = b.call("candidates", gid=1)["cands"][0]
        assert b.call("refuse", gid=1, key="A", tag=tag)["refused"]
        assert tag not in b.call("candidates", gid=1)["cands"]
        b.call("load", gid=1, state=S, pi=me, key="A")           # the same game: kept
        assert tag not in b.call("candidates", gid=1)["cands"]
        # a refusal of A while another game is loaded must not reach that game
        b.call("load", gid=1, state=S, pi=me, key="B")           # same seat and round: fresh
        assert tag in b.call("candidates", gid=1)["cands"]
        assert not b.call("refuse", gid=1, key="A", tag=tag)["refused"]
        assert tag in b.call("candidates", gid=1)["cands"]
        # without a key (test_online_offline) the turn goes on as before
        b.call("load", gid=2, state=S, pi=me)
        b.call("candidates", gid=2)
        assert b.call("refuse", gid=2, tag=tag)["refused"]
        b.call("load", gid=2, state=S, pi=me)
        assert tag not in b.call("candidates", gid=2)["cands"]


def test_refused_tag_is_not_decided_again():
    brain = Brain("teacher", device="cpu")
    try:
        with Bridge() as server:
            st = server.reset([{"kind": "agent"}, {"kind": "ki"}], seed=7, map="original")
            gid, me = st["gid"], st["agents"][0]
            sent, refused = [], None

            def send(typ, params):
                nonlocal refused
                sent.append((typ, params))
                if refused is None and typ != "endTurn":
                    refused = (typ, params)
                if (typ, params) == refused:
                    raise RuntimeError("refused")
                err = server.call("server_action", gid=gid, type=typ, params=params)["err"]
                assert not err, err
            assert conn(brain).play_turn(server.state(gid), me, "A", send, lambda: server.state(gid))
            assert refused and sent.count(refused) == 1, sent
            assert sent[-1][0] == "endTurn", sent
    finally:
        brain.bridge.close()


def test_forced_end_moves_armies_first():
    brain = Brain("teacher", device="cpu")
    server, gid, S, me = position()
    try:
        sent = []

        def send(typ, params):
            sent.append(typ)
            err = server.call("server_action", gid=gid, type=typ, params=params)["err"]
            if err:
                raise RuntimeError(err)
        assert conn(brain).force_end(S, me, "A", send)
        assert sent[-1] == "endTurn" and "moveArmy" in sent, sent
        assert sent.index("moveArmy") < sent.index("endTurn"), sent
    finally:
        brain.bridge.close()
        server.close()


def test_too_many_rejected_actions_end_the_turn_safely():
    brain = Brain("teacher", device="cpu")
    server, gid, S, me = position()
    try:
        sent = []

        def send(typ, params):
            sent.append(typ)
            if typ not in ("moveArmy", "endTurn"):
                raise RuntimeError("refused")
            err = server.call("server_action", gid=gid, type=typ, params=params)["err"]
            if err:
                raise RuntimeError(err)
        assert conn(brain).play_turn(S, me, "A", send, lambda: server.state(gid))
        assert sent[-1] == "endTurn", sent
        after = server.state(gid)
        assert after["cur"] != me or after["round"] != S["round"], sent
    finally:
        brain.bridge.close()
        server.close()


def test_refused_end_does_not_raise():
    brain = Brain("teacher", device="cpu")
    server, gid, S, me = position()
    try:
        def send(typ, params):
            if typ == "endTurn":
                raise RuntimeError("Eine Armee steht noch in einer Stadt")
            err = server.call("server_action", gid=gid, type=typ, params=params)["err"]
            if err:
                raise RuntimeError(err)
        assert conn(brain).play_turn(S, me, "A", send, lambda: server.state(gid)) is False
    finally:
        brain.bridge.close()
        server.close()


def test_live_game_tries_a_refused_end_again():
    brain = FakeBrain()
    mgr = SimpleNamespace(acc=None, brain=brain, settings={})
    g = LiveGame(mgr, lobby={})
    g.g = {"sessionId": "S1", "yourIndex": ME, "state": dict(view("S1")["state"], researched=True)}

    def call(event, data, timeout=30):
        raise RuntimeError("action: Eine Armee steht noch in einer Stadt")
    g.call = call
    g.live_turn()
    assert g.retry_at, "no retry planned"
    g.g["state"]["cur"] = ME + 1                 # e.g. the turn timer ended the turn
    g.live_turn()
    assert not g.retry_at


def test_old_engines_play_their_own_rules():
    brain = Brain("teacher", device="cpu")
    try:
        for version in ("v100", "v98"):
            old = brain.on(version)
            assert old is brain.on(version) and old.lock is brain.lock
            assert old.bridge.info()["version"] == version
            with Bridge(engine_dir_of(version)) as server:
                st = server.reset([{"kind": "agent"}, {"kind": "ki"}], seed=7, map="original")
                gid, me = st["gid"], st["agents"][0]
                sent = []

                def send(typ, params):
                    sent.append(typ)
                    err = server.call("server_action", gid=gid, type=typ, params=params)["err"]
                    assert not err, (version, typ, err)
                assert conn(brain).play_turn(server.state(gid), me, "A", send, lambda: server.state(gid), old)
                assert sent[-1] == "endTurn", sent
    finally:
        brain.close()


def test_rundenspiel_is_played_on_its_engine():
    brain, server = FakeBrain(), FakeServer(dict(view("A"), engine="v98"))
    asked = []
    brain.on = lambda engine: asked.append(engine) or brain
    FakeRundenspiele(server, brain).open_game("A")
    assert asked[0] == "v98", asked


def test_unknown_engine_is_refused():
    brain = Brain("teacher", device="cpu")
    try:
        try:
            brain.on("v1")
        except RuntimeError as e:
            assert "v1" in str(e)
        else:
            raise AssertionError("played a game without its engine")
    finally:
        brain.close()


def test_claimed_tile_is_sent_to_the_server():
    """Kolonialismus (v103+): a claim the agent picks goes to the server as claimTile."""
    with Bridge() as server, Bridge() as client:
        if not server.call("info")["version"] >= "v103":
            return
        st = server.reset([{"kind": "agent"}, {"kind": "ki"}], seed=7, map="plaettchen")
        me = st["agents"][0]
        S = server.state(st["gid"])
        S["players"][me]["techs"]["kolonialismus"] = True
        server.call("load", gid=50, state=S, pi=me)
        client.call("load", gid=1, state=S, pi=me, key="A")
        tags = client.call("candidates", gid=1)["cands"]
        i = next(i for i, t in enumerate(tags) if t.startswith("col:"))
        actions = client.call("step_online", gid=1, i=i)["actions"]
        assert [a[0] for a in actions] == ["claimTile"], actions
        err = server.call("server_action", gid=50, type="claimTile", params=actions[0][1])["err"]
        assert not err, err


def test_the_server_engine_is_followed():
    """A deploy of a new engine: new games are played on it, a version without a local copy
    leaves the current one."""
    from hochciv_ki.online import client as cl
    brain = Brain("teacher", device="cpu")
    mgr = SimpleNamespace(acc=SimpleNamespace(url="x"), brain=brain, last_engine_check=0)
    real = cl.server_version
    try:
        cl.server_version = lambda url: "v98"
        cl.Manager.check_engine(mgr)
        assert brain.current == "v98" and brain.on(None).version == "v98"
        assert conn(brain).brain.on(None) is brain.on("v98")
        cl.server_version = lambda url: "v1"
        cl.Manager.check_engine(mgr)
        assert brain.current == "v98"
        cl.server_version = lambda url: None
        cl.Manager.check_engine(mgr)
        assert brain.current == "v98"
        cl.server_version = lambda url: brain.version
        cl.Manager.check_engine(mgr)
        assert brain.on(None) is brain
    finally:
        cl.server_version = real
        brain.close()


def test_logs_name_the_game():
    class Lines(logging.Handler):
        def __init__(self):
            super().__init__()
            self.lines = []

        def emit(self, record):
            self.lines.append(record.getMessage())
    lines = Lines()
    logger = logging.getLogger("hochciv-ki")
    logger.addHandler(lines)
    try:
        brain, server = FakeBrain(), FakeServer(view("A"))
        rs = FakeRundenspiele(server, brain)
        rs.call = lambda event, data, timeout=30: (_ for _ in ()).throw(RuntimeError("refused"))
        rs.accept(view("A"))
        rs.opened = "A"
        rs.turn("A")
        assert any(x.startswith("[rundenspiele] A: rejected ") for x in lines.lines), lines.lines
        assert any("A: too many rejected" in x for x in lines.lines), lines.lines
        rs.turn = lambda gid: 1 / 0
        rs.handle("async:update", view("A"))
        assert any(x == "[rundenspiele] <lambda> failed (A)" for x in lines.lines), lines.lines
    finally:
        logger.removeHandler(lines)


def main():
    logging.disable(logging.CRITICAL)
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        logging.disable(logging.NOTSET if name == "test_logs_name_the_game" else logging.CRITICAL)
        try:
            fn()
            print(f"ok    {name}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed} of {len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
