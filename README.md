# Hochciv-KI

Neural-network AI for HochCiv (Python + PyTorch). See [PLAN.md](PLAN.md).

## Setup (Windows, cmd.exe)

```
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
```

Node.js must be on the PATH. The rules engine is loaded read-only from
`..\multiplayer-server\HochCiv\js`; set `HOCHCIV_ENGINE_DIR` to use another copy. That folder
changes with the branch checked out there, so long runs use a frozen copy (`git archive` into
`runs\engines\<version>`). The online bot needs no setting: it plays on the engine the server
runs (see *Playing online*).

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

`--opp` also takes networks, and a comma-separated list fills the opponent seats in turn, so a
new model can be checked head to head, not only against the KI (in `eval` and `eval_modes`):

```
.venv\Scripts\python -m hochciv_ki.eval_modes --agent ckpt:checkpoints\ppo3\best.pt --opp ckpt:checkpoints\v2.pt
.venv\Scripts\python -m hochciv_ki.eval --agent ckpt:a.pt --opp ckpt:b.pt,ki:schwer --players 4
```

A network against itself scores exactly the fair share (every seed from every seat).

`search:<path>[:k]` plays a network with a one-step look-ahead (`hochciv_ki/search.py`,
DESIGN_SEARCH.md): the policy's top k candidates (default 4) are tried on copies of the game
and scored with the value head, "end" after the other seats' turns played by KI schwer. It works
in `eval`, `eval_modes` and the online client (`--agent search:checkpoints2.pt`, at most 15 s
of search per turn); about 0.1–0.6 s per decision.

`train_ppo --search-frac 0.25` lets the network learn from that search during training
(stage 3, `hochciv_ki/expert.py`): a quarter of the learner's decisions get a target from the
search, trained in a phase after the PPO update. `python -m hochciv_ki.expert` measures what it
costs and how often it changes the best move.

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

## v4 (production model since 2026-11-06)

`checkpoints/v4.pt` = `checkpoints/ppo9/best.pt`, iteration 20: first run with search stage 3
(DESIGN_SEARCH.md), from `ppo8`, `--lr 5e-5 --target-kl 0.05 --search-frac 0.25 --search-k 3
--search-tau 0.03`, no shaping. Plays without search. Full v2 suite on seeds not used to pick
it (`--base-seed 290000`, 798 games): against KI schwer 1.536 ± 0.145, head to head against v3
1.281 ± 0.137 (9 modes better, 2 equal, none worse). Hex suite against KI schwer 1.795.
Training evals swung between 1.18 and 1.48 over the run, so iteration 20 is a high point, not a
stable level.

## Engine v106 (2026-11-09/10)

v106 adds free claims (v103), the founding rule next to foreign claims (v104), copy ways
(v105) and limits Greece's free techs (v101). The network gets six new inputs per seat for
them (`claimsLeft`, three open copy ways, free techs left / available). `checkpoints/v4x.pt` is
v4 widened to these inputs (new weights zero, same decisions as v4). All numbers below are on
v106, v2 suite, `--base-seed 490000`, 798 games, checked with `hochciv_ki.promote`:

| candidate | against KI schwer (mean relative) | head to head against v4 | gate |
|---|---|---|---|
| v4 | 1.431 | – | production |
| ppo12 (v4 trained on v106, v2 mix) | 1.177, marathon 0.000 | 0.705, z = −6.2 | keep v4 |
| v4 with search (top 4) | 1.304, marathon 0.175 | 0.889, z = −2.9 | keep v4 |
| ppo13 (from v4x, mix v3, `--anchor-kl 0.5 --anchor-frac 0.5`) | **1.605**, no mode worse | 0.926, z = −0.6 | keep v4 (h2h not significant) |

| ppo14 (from ppo13, mix vs4, search) | collapsed in training (iteration 65: 0.81, 6p/8p/marathon 0) | – | stopped |
| **ppo15 = v5** (from ppo13, mix vs4b, no search, `--lr 1.5e-5`) | **1.642**, no mode worse | **1.087, z = 1.65** | **promote** |

ppo15 trains with about a quarter of the games against fixed v4x (`--fixed`, at most one v4 per
game) and picks best.pt by head to head against v4x (`--eval-vs`, `--min-ki 1.38`, iteration
90). ppo14 tried the same with half the games against v4, games full of v4 copies and search
distillation (whose look-ahead models the opponents as KI schwer) and fell apart. ppo15 head to
head against ppo13: 1.152. `checkpoints/v5.pt` = `checkpoints/ppo15/best.pt`.

The search assumes the opponents play like KI schwer and loses to plain v4 offline and online.
ppo13 beats KI schwer clearly more than v4 and ties v4 head to head; it runs as the second
bot (NeuraNew) for a live comparison, v4 stays the production bot.

## v3 (production model from 2026-11-03 to 2026-11-06)

`checkpoints/v3.pt` = run B' (`checkpoints/ppo4/best.pt`, iteration 80): PPO from v2 on the v2 mix
with reward shaping (DESIGN_SHAPING.md), `--lr 5e-5 --target-kl 0.05 --imit 0.03 --shape 1`.
Full v2 suite, same seeds: against KI schwer it ties v2 (mean relative score 1.184 each); head
to head against v2 it scores 1.15, at or above the fair share in all 11 modes.

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

`--continuous` keeps every worker playing: an iteration takes the next `--games` finished games,
long games carry over, and games go on during the update (about 1.4× faster than waiting for
the slowest game of every batch).

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
.venv\Scripts\python -m hochciv_ki.online.client --config online.env --agent ckpt:checkpoints\v4.pt --open
```

The engine is the one the server runs: the client reads `APP_VERSION` from the server's
`js/data.js` at the start and every 5 minutes, and plays new games on the local copy of that
version (`..\multiplayer-server\HochCiv-<version>\js`, `..\multiplayer-server\HochCiv\js` or
`runs\engines\<version>`). A deploy switches it over by itself; running Rundenspiele keep the
engine they were started on. If there is no local copy of the new version it stays on the old one
and logs an error.

It logs in with the token (`POST /api/auth/bot`) and opens a control panel on
<http://127.0.0.1:8765/> (`--port`, `--open` opens it in the browser, `--no-panel` runs with
the saved settings, `--device cpu|cuda` where a `ckpt:` network runs; the default takes the GPU
if there is one). The network needs no GPU: on the CPU a move takes about 4 ms, slightly less than
on the GPU. The panel shows the running games, the public lobbies and open
Rundenspiele with a *Join* button, the log, and the settings in a column on the right. Every
change is saved at once (`runs/online-settings.json`). The switches work together:

| Setting | What the bot does |
|---|---|
| Join games | Joins public lobbies (live games: on/off and how many at once) and open Rundenspiele (on/off, how many at once) that match the join filters: maps (Marathon also covers custom maps with the marathon rules), empires min–max (up to 8), wonders, event modes (none / easy / hard), KI seats, ranked/unranked, live only with a turn timer, games hosted by other bots |
| Host games | A list of setups: live lobby or Rundenspiel, map (also Marathon, 4 empires), empires (up to 8 on the tile map), KI seats and level, wonders, events (off / easy / hard), turn timer, nations, other player-bots, public, ranked (🔥: MMR counts, KI on Schwer); each setup is kept open or running × times. A live lobby starts when it is full, or after X minutes with at least Y players; free seats become KI |
| Matchmaking | Ranked queue (bot accounts only; it only meets players who opted in to bots), one game at a time |
| Chat greeting | Sent on entering a live lobby |

The game list's *Join* works for every game the bot can play, whatever the join filters say;
the reasons a game doesn't match are shown next to it (red ones block joining: full, no
player-bots, an older engine). The bot plays every mode – wonders, events, the marathon (it
drafts its two abilities like the JS KI) – with up to 8 empires, and always plays every game it
is in. The public lobby list shows only the map and the human seats: the
other filters are checked right after joining, and the bot leaves a lobby that doesn't match.
The panel listens on 127.0.0.1 only and every API call needs the token in the page.

It says in the chat that it is an AI (text in the settings) and plays under the bot's account
name. When the session ends it logs in again; a renewed token or a banned owner stops it with
an error.
