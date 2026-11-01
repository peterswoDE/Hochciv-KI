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
| `hochciv_ki/eval_modes.py` | Evaluation over a suite of game modes (wonders, events, marathon, 5–8 empires) |
| `hochciv_ki/widen.py` | Carries a network over to a new observation layout (inputs matched by name) |
| `hochciv_ki/model.py` | The network (residual CNN over the map, transformer over candidates) |
| `hochciv_ki/data.py` | Imitation data (labels from the JS KI planner) |
| `hochciv_ki/train_bc.py` | Behaviour cloning |
| `hochciv_ki/infer.py` | Batched GPU inference for many game threads |
| `hochciv_ki/train_ppo.py` | PPO self-play against a league (snapshots, JS KI, bots) |
| `hochciv_ki/online/client.py` | Online client: live lobbies, Rundenspiele, matchmaking, modes |
| `hochciv_ki/online/panel.py`, `panel.html` | Local control panel for the online client |
| `hochciv_ki/selftest.py` | M0 selftest |
| `tests/test_online_offline.py` | Online path against a local "server" worker |
| `tests/staging_live.py`, `staging_async.py`, `staging_modes.py` | Live game, Rundenspiel and join/host switches on staging |
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

M4 (1v1): after 87 PPO iterations (about 14 000 games), on 600 fresh games (seeds 70000–70299)
against KI schwer: **PPO 0.559 ± 0.040**, BC on the same games 0.523 ± 0.040. Sampling from the
policy instead of the greedy choice plays about the same (0.575 vs 0.573 on 200 games).

v1 model (`checkpoints/v1.pt`, PPO iteration ~118 with all v1 modes since iteration 92), fresh
seeds 80000+, against KI schwer:

| Mode | Score | Fair share |
|---|---|---|
| 1v1, tile map (300 games) | **0.610 ± 0.055** | 0.500 |
| 1v1, original map (200) | 0.583 ± 0.068 | 0.500 |
| 1v1, random custom map (200) | 0.562 ± 0.069 | 0.500 |
| 3 players, big map (180) | **0.472 ± 0.073** | 0.333 |
| 4 players, tile map (240) | 0.310 ± 0.059 | 0.250 |
| 4 players, random custom map (160) | 0.338 ± 0.073 | 0.250 |

## v2: every game mode

Observation layout v2 (`bridge/features.js`) covers every mode the server offers: 8 seat
slots (5–8 empires), the barbarians on their own map planes, wonders (pool, built, owned,
cost, wonder candidates by name), events (mode, this round's event, the next one with the
oracle, nuclear ban) and the two drafted marathon abilities. Every input has a name;
`widen.py` copies the weights of a network to a new layout by name and starts new inputs at
zero weight, and `load_net` does that by itself for older checkpoints (v1.pt still works).
The widened v1 plays exactly like v1 (max logit difference 5e-4 over 640 decisions).

Bridge game options: `events`, `eventMode` (`hard`/`easy`), `wonders`; maps `marathon` (24 × 36,
4 empires, abilities drafted like the JS KI) and `customgen-marathon` (random custom map with the
marathon rules).

```
.venv\Scripts\python -m hochciv_ki.eval_modes --agent ckpt:checkpoints\v1w.pt --suite v2
.venv\Scripts\python -m hochciv_ki.train_ppo --init checkpoints\v1w.pt --out checkpoints\ppo2 --mix v2 --lr 3e-5 --target-kl 0.05 --imit 0.1 --imit-half 150
```

Baseline before v2 training: the widened v1 model against KI schwer (seeds 90000+):

| Mode | v1w | Fair share |
|---|---|---|
| 2p, tile map, wonders (120 games) | 0.575 ± 0.089 | 0.500 |
| 2p, tile map, events hard (120) | 0.492 ± 0.090 | 0.500 |
| 2p, original map, wonders, events easy (80) | 0.613 ± 0.107 | 0.500 |
| 3p, custom map, wonders, events hard (60) | 0.483 ± 0.128 | 0.333 |
| 4p, tile map, wonders, events hard (60) | 0.350 ± 0.122 | 0.250 |
| 6p, tile map (60) | 0.200 ± 0.102 | 0.167 |
| 8p, custom map, wonders (48) | 0.135 ± 0.096 | 0.125 |
| 4p, marathon (40) | 0.250 ± 0.136 | 0.250 |
| 3p, custom marathon map, wonders (30) | 0.500 ± 0.182 | 0.333 |

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

Create a bot account on hochciv.de (account page, tab *Meine Bots*; needs the role
`botcreator`) and put its token in `online.env` (not in git):

```
HOCHCIV_URL=https://hochciv.de
HOCHCIV_TOKEN=hcbot_...
```

(Without a bot account: `HOCHCIV_USER`, `HOCHCIV_PASSWORD` and optionally `HOCHCIV_NAME` for a
regular account that logs in with a password only.)

```
.venv\Scripts\python -m hochciv_ki.online.client --config online.env --agent ckpt:checkpoints\v1.pt --open
```

It logs in with the token (`POST /api/auth/bot`) and opens a control panel on
<http://127.0.0.1:8765/> (`--port`, `--open` opens it in the browser, `--no-panel` runs with
the saved settings). The panel shows the running games, the public lobbies and open
Rundenspiele with a *Join* button, the log, and the settings in a column on the right. Every
change is saved at once (`runs/online-settings.json`). The switches work together:

| Setting | What the bot does |
|---|---|
| Join games | Joins public lobbies (live games: on/off and how many at once) and open Rundenspiele (on/off, how many at once) that match the join filters: maps, empires min–max, KI seats, ranked/unranked, live only with a turn timer, games hosted by other bots |
| Host games | A list of setups: live lobby or Rundenspiel, map, empires, KI seats and level, turn timer, nations, other player-bots, public, ranked (🔥: MMR counts, KI on Schwer); each setup is kept open or running × times. A live lobby starts when it is full, or after X minutes with at least Y players; free seats become KI |
| Matchmaking | Ranked queue (bot accounts only; it only meets players who opted in to bots), one game at a time |
| Chat greeting | Sent on entering a live lobby |

The game list's *Join* works for every game the bot can play, whatever the join filters say;
the reasons a game doesn't match are shown next to it (red ones block joining: full, no
player-bots, events, wonders, marathon, more than 4 empires, an older engine). The bot always
plays every game it is in. The public lobby list shows only the map and the human seats: the
other filters are checked right after joining, and the bot leaves a lobby that doesn't match.
The panel listens on 127.0.0.1 only and every API call needs the token in the page.

It says in the chat that it is an AI (text in the settings) and plays under the bot's account
name. When the session ends it logs in again; a renewed token or a banned owner stops it with
an error.
