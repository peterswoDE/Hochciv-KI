# Hochciv-KI

Neural-network AI for HochCiv (Python + PyTorch). See [PLAN.md](PLAN.md).

This repo holds the code, the released models (`checkpoints/v*.pt`, each run's `best.pt`) and the
online bots (`deploy/`). Training runs and their data (snapshots, `latest.pt`, BC data, eval logs)
live in `../Hochciv-NeuralNet` (not a repo; its `nn.ps1` runs the modules here). Copy only the
models you want to keep back into `checkpoints/`.

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

## Models

The models are in the repository through Git LFS (`git lfs install` before cloning): the
releases `checkpoints/v*.pt` and the `best.pt` of every training run. Snapshots, `latest.pt`
and the run state stay local.

## Selftest

```
.venv\Scripts\python -m hochciv_ki.selftest
```

Plays complete games with a random agent against the JS KI and bots on tile, fixed and big maps.

## Layout

| Path | What |
|---|---|
| `bridge/worker.js` | Node worker: loads the engine, runs games, JSON lines over stdin/stdout |
| `bridge/rush.js` | Scripted slavery rusher (seat kind `rush`, opponent `rush`): training opponent and gate check |
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

## The rusher (engine v107, 2026-11-16)

Online, v4 loses most games against people and bots who open with slavery: research it in round
1, grow every city and sacrifice the new population (+7 coins each), buy cheap military techs
and stand armies next to the capital in round 2 – a two-turn siege takes it in round 3 to 5. Our
networks never sacrifice and field their first army in round 4: neither KI schwer nor self-play
ever shows this plan. `bridge/rush.js` scripts it (seat kind `rush`; opponent `rush` in `eval`,
the `rush` suite in `eval_modes`, rule 4 of `promote`, mix `rush` in `train_ppo`). The rest of
its turn (founding, research, defense) is the KI planner. It beats KI schwer about 13 to 3 in
duels (`python -m tests.test_rush`).

Rush suite, `--base-seed 490000`, 380 games (mean relative score, 1.0 = fair share):

| agent | 2p | 2p wonders | 2p events hard | 3p (+ KI) | 4p (+ 2 KI) | mean relative |
|---|---|---|---|---|---|---|
| KI schwer | 0.367 | 0.562 | 0.388 | 0.333 | 0.150 | 0.847 |
| v4 | 0.417 | 0.525 | 0.425 | 0.417 | 0.150 | 0.917 |
| v5 | 0.450 | 0.613 | 0.375 | 0.417 | 0.150 | 0.945 |

Against KI schwer v4 and v5 reach 1.43 and 1.64; against the rusher both are below the fair
share. ppo16 (`runs/queue11.sh`) trains v5 on v107 with a quarter of the games against it.

## Online games and the HardcodedBots rusher (engine v108, 2026-11-18)

ppo23 beat v5 offline (gate z 1.40) but lost online: about two thirds against v5 and most games
against ChuckBot, an early slavery rusher. Online games are mostly duels in "random" lobbies (tile
map, random nation and ability, no wonders or events). The rusher of HochCiv-HardcodedBots
(now `hardcoded/` in this repo, formerly `../HochCiv-HardcodedBots`; pinned copies in
`runs/hcbots/<commit>` via `HOCHCIV_HCBOTS_DIR`) is much closer to it than `bridge/rush.js`. `hochciv_ki/hcbot.py` plays it
as an external seat (seat kind `ext`, opponent `hc:rusher`, seat `H` in `train_ppo`): the worker
hands the state over (`external_state`), the bot plays its turn in its own worker, the state comes
back (`external_done`). Suite `online` in `eval_modes`, mix `online` in `train_ppo`.

Suite `online`, `--base-seed 490000`: v5 0.467 in duels against the rusher, 0.738 against KI
schwer, 0.450 in 3p (+ KI). ppo24 (`runs/queue18.sh`, from v5, mix online, 100 iterations) stayed
at 0.44–0.52 against the rusher and failed the gate.

`fork` in `bridge/worker.js` takes `keepSeats`: the other agent and external seats stay seats on
the copy (an external bot keeps playing them), for counterfactual rollouts against the HC rusher.

## Engine v110 (2026-11-19)

v110 pays 4 coins per sacrificed population instead of 7 (v109 only changed the UI and added
`unspent`, the end-turn check of human players). Nothing in the bridge or the features depends
on the number. `runs/engines/v110` is the server's copy, identical to the engine repo at 524284b.
The online client follows the server's version by itself (`check_engine`, every 5 minutes) and
finds the engine in `multiplayer-server/HochCiv/js` or `runs/engines/v110`. `bridge/rush.js` wins
7 of 16 against KI schwer on v110 (v108: 10), `tests/test_rush.py` now asks for a third.
ppo25 (`runs/queue19.sh`) fine-tunes v5 on v110.

## Engine v112 (2026-11-23)

v111/v112 bring eight experimental nations (`EXP_CIVS`: Rome with the Legion, +50 % power per
adjacent own army, and Via Appia; Venice, no cities, free growth, 3 free claims per round; Aztecs
and Egypt with wonders; Germany, Maya, China with `collectIncomeNow`, USA), 5 coins per sacrificed
population, and tile maps that ignore the hex option (2 players: tri8 map, fixed seats). Online,
tile maps start with a placement phase and `S.seed` is no longer sent. The server only offers the
experimental nations on tile and custom maps, not in marathon or ranked games (`expCivsOn`).

- `bridge/features.js`: the civ and ability one-hots come from a fixed list (all 12 nations), with
  an "other" slot for unknown ones, plus `armyCost`, `incomeNowOpen`, `noFound`, `growFree`. GLOB_N
  1651 → 1835. The layout is the same on v107–v112, so the online client can serve old Rundenspiele
  with the same net (`Brain.on` checks it).
- `checkpoints/v5x.pt`: v5 widened (`hochciv_ki.widen`), new inputs at zero, so it plays like v5.
- `bridge/worker.js`: `expCivs` option (draws from `civKeysFor`, not on fixed maps or marathon).
  `eval.py --exp-civs`, suite `exp` in `eval_modes`, `train_ppo --exp-share`.
- Online client and panel: host setups with "Experimental nations", join filter for them.

v5x on v112 against KI schwer (suite v2, scale 2, `--base-seed 490000`), v5 on v110 for comparison:

| mode | v5 on v110 | v5x on v112 | fair |
|---|---|---|---|
| 2p, plaettchen, wonders | 0.692 | 0.742 | 0.500 |
| 2p, plaettchen, events hard | 0.642 | 0.679 | 0.500 |
| 2p, original, wonders, events easy | 0.725 | 0.762 | 0.500 |
| 3p, customgen, wonders, events hard | 0.533 | 0.508 | 0.333 |
| 4p, plaettchen, wonders, events hard | 0.450 | 0.492 | 0.250 |
| 6p, plaettchen | 0.283 | 0.250 | 0.167 |
| 8p, customgen, wonders | 0.271 | 0.271 | 0.125 |
| 4p, marathon | 0.525 | 0.450 | 0.250 |
| 3p, customgen-marathon, wonders | 0.867 | 0.850 | 0.333 |
| 2p, plaettchen | 0.650 | 0.763 | 0.500 |
| 4p, customgen | 0.417 | 0.392 | 0.250 |
| mean relative | 1.732 | 1.724 | |

Suite `online` (rusher of the HardcodedBots main tree, still v110 code): v5 on v110 / v5x on v112
0.567 / 0.613 in duels against the rusher, 0.738 / 0.721 against KI schwer, 0.500 / 0.444 in 3p
(+ KI); mean relative 1.369 / 1.333, all within the 95 % intervals. Suite `exp` (experimental
nations in the draw), v5x: 0.683 in duels, 0.681 with wonders, 0.475 in 3p custom, 0.483 in 4p;
mean relative 1.522, below the 1.72 of the normal modes.

ppo26 (`runs/queue20.sh`) fine-tunes v5x on v112, mix online with the experimental nations in
35 % of the games, gate against v5x. `checkpoints/ppo25/best.pt` (v110, gate failed) is kept.

### v6 (production model since 2026-11-24)

`checkpoints/v6.pt` = `checkpoints/ppo26/best.pt`, iteration 100 (v5x fine-tuned on v112, mix
online, experimental nations in 35 % of the games, `--lr 3e-5`, 120 iterations). Gate against v5x
on v112 (`runs/gate_ppo26_v112.json`), core modes:

| | ppo26 = v6 | v5x |
|---|---|---|
| head to head, 1240 games | **1.096, z = 2.67** | – |
| KI schwer (core) | **1.666** | 1.564 |
| suite online (hc:rusher) | **1.510** | 1.335 |
| suite exp | **1.653** | 1.522 |

Only loss: 3p custom marathon 0.817 against 0.850 (15 games). Since 2026-11-24 both online
clients play v6.

### Next steps (queue21, 2026-11-24)

- **Nation choice** (`hochciv_ki.civ_eval`, `runs/civ_main_v112.json`, ppo26, 48 seeds, mean rel
  over duel / 4p vs KI schwer / duel vs hc:rusher): england:gruenden 2.32, usa 2.25,
  england:kuestenstaedte 2.22, maya 2.17, russland 2.14 … rom:basis 1.50, venedig 0.93. The
  network plays Venice (no cities) and the Roman legion badly. The clients still join with a
  random nation.
- **Throughput:** 14 / 28 / 42 workers collect 80 / 74 / 81 decisions/s. More workers than cores
  do not help; the GPU wait measured beside a running training was the training's.
- **exploit2** (from ppo26, `--lr 1.5e-4 --target-kl 0.1`, mix exploit): 1.09–1.25 against
  ppo26 in training, 0.504 in duels (suite online). Like exploit1, no hole found. It reached
  1.80–1.96 against KI schwer (ppo26: 1.67).
- **ppo27** (online_league with exploit2, `--search-frac 0.25`) collapsed: against ppo26 0.84 →
  0.66 from iteration 10 to 40, 3p against the rusher 0.54 → 0.16. Stopped at iteration 41.
  Like ppo14, the search phase together with games against a fixed network breaks the policy.

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
