"""The online path without a server: one worker plays "server" (full rules, other seats), a
second one is the bot client, which only gets states and answers with server actions.

    python -m tests.test_online_offline [games]
"""
from __future__ import annotations

import random
import sys

from hochciv_ki.bridge import Bridge
from hochciv_ki.env import TeacherAgent, decode_obs


def run(games=6, agent=None):
    agent = agent or TeacherAgent()
    bad = 0
    with Bridge() as server, Bridge() as client:
        dims = client.info()["dims"]
        for g in range(games):
            n = [2, 2, 3, 4][g % 4]
            players = [{"kind": "agent"}] + [{"kind": "ki", "kiLevel": "schwer"}] * (n - 1)
            random.Random(g).shuffle(players)
            st = server.reset(players, seed=500 + g, map=["plaettchen", "original", "gross"][g % 3])
            gid, me = st["gid"], st["agents"][0]
            sent = rejected = 0
            while not st["over"]:
                state = server.state(gid)
                client.call("load", gid=1, state=state, pi=me)
                obs = decode_obs(client.call("observe", gid=1), dims)
                a, _ = agent.act(obs)
                res = client.call("step_online", gid=1, i=a)
                for typ, params in res["actions"]:
                    r = server.call("server_action", gid=gid, type=typ, params=params)
                    sent += 1
                    if r["err"]:
                        rejected += 1
                        print(f"  rejected {typ} {params}: {r['err']}")
                    st = r
                    if st["over"]:
                        break
            won = me in st.get("winners", [])
            print(f"game {g}: {n}p, round {st['round']}, {sent} actions sent, {rejected} rejected, "
                  f"{'won' if won else 'lost'}: {st.get('how', '')[:50]}")
            bad += rejected
            server.close_game(gid)
    return bad


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    agent = None
    if len(sys.argv) > 2:          # a checkpoint: the network plays instead of the teacher
        from hochciv_ki.model import load_agent
        agent = load_agent(sys.argv[2])
    sys.exit(1 if run(n, agent) else 0)
