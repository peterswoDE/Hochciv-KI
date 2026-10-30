# Hochciv-KI

Neural-network AI for HochCiv (Python + PyTorch). See [PLAN.md](PLAN.md).

## Setup (Windows, cmd.exe)

```
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
```

Node.js must be on the PATH. The rules engine is loaded read-only from
`..\multiplayer-server\HochCiv\js`; set `HOCHCIV_ENGINE_DIR` to use another copy.

## Selftest

```
.venv\Scripts\python -m hochciv_ki.selftest
```

Plays complete games with a random agent against the JS KI and bots on tile, fixed and big maps.

## Layout

| Path | What |
|---|---|
| `bridge/worker.js` | Node worker: loads the engine, runs games, JSON lines over stdin/stdout |
| `hochciv_ki/bridge.py` | Python client for the worker |
| `bridge/features.js` | Observation features, candidate evaluation on copies, teacher label (one kiPlan step) |
| `hochciv_ki/env.py` | Observations as numpy arrays, agents, `play_game` |
| `hochciv_ki/pool.py` | Pool of bridge workers (one Node process per thread) |
| `hochciv_ki/eval.py` | Paired evaluation against the JS KI and bots |
| `hochciv_ki/selftest.py` | M0 selftest |

## Evaluation

```
.venv\Scripts\python -m hochciv_ki.eval --agent teacher --opp ki:schwer --players 2 --seeds 100
```

Agents: `kiturn` (the JS KI's full turn), `teacher` (the JS KI planner one step at a time, the
imitation target), `ckpt:<path>` (a trained network). Every seed is played once per seat.

Baselines (engine v98, tile map, against KI schwer; score = share of wins, fair share 1/n):

| Agent | 2 players | 4 players |
|---|---|---|
| kiturn | 0.500 ± 0.069 (200 games) | |
| teacher | 0.525 ± 0.069 (200 games) | 0.292 ± 0.082 (120 games) |
