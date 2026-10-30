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
| `hochciv_ki/model.py` | The network (residual CNN over the map, transformer over candidates) |
| `hochciv_ki/data.py` | Imitation data (labels from the JS KI planner) |
| `hochciv_ki/train_bc.py` | Behaviour cloning |
| `hochciv_ki/infer.py` | Batched GPU inference for many game threads |
| `hochciv_ki/train_ppo.py` | PPO self-play against a league (snapshots, JS KI, bots) |
| `hochciv_ki/online/client.py` | Online client: live lobbies, Rundenspiele, matchmaking (off) |
| `hochciv_ki/selftest.py` | M0 selftest |
| `tests/test_online_offline.py` | Online path against a local "server" worker |
| `tests/staging_live.py` | Live game on staging: guest host + bot |

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
| BC network (730k samples, 6 epochs) | 0.540 ± 0.069 (200 games) | 0.283 ± 0.081 (120 games) |

## Training

```
rem 1. imitation data from the JS KI planner, then behaviour cloning
.venv\Scripts\python -m hochciv_ki.data --games 5000 --out data\bc0 --eps 0.15
.venv\Scripts\python -m hochciv_ki.train_bc --data data\bc0 --epochs 8 --out checkpoints\bc

rem 2. DAgger round: the network plays, the JS KI labels
.venv\Scripts\python -m hochciv_ki.data --games 3000 --out data\bc1 --actor ckpt:checkpoints\bc\best.pt --eps 0.05
.venv\Scripts\python -m hochciv_ki.train_bc --data data\bc0 data\bc1 --init checkpoints\bc\best.pt --out checkpoints\bc1

rem 3. PPO self-play (resumable with --resume)
.venv\Scripts\python -m hochciv_ki.train_ppo --init checkpoints\bc1\best.pt --out checkpoints\ppo --mix 1v1
.venv\Scripts\python -m hochciv_ki.train_ppo --out checkpoints\ppo --resume --mix all
```

`data\` and `checkpoints\` are not in git.

## Playing online

Create `online.env` (not in git):

```
HOCHCIV_URL=https://staging.hochciv.de
HOCHCIV_USER=<bot account>
HOCHCIV_PASSWORD=<password>
HOCHCIV_NAME=HochCiv-KI [Bot]
```

```
.venv\Scripts\python -m hochciv_ki.online.client --config online.env --agent ckpt:checkpoints\ppo\best.pt
```

It joins fitting public lobbies (2–4 empires, no events, wonders or marathon), says in the chat
that it is an AI, and plays; it also joins and plays open public Rundenspiele. Matchmaking
(`--matchmaking`) stays off until the server has a bot account type. The account must log in
with a password only (no second factor).
