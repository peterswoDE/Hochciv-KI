# Hochciv-KI – Plan

A neural-network AI for HochCiv, written in Python, trained on this PC and able to play
online on hochciv.de (public lobbies, Rundenspiele, later matchmaking).

## Decisions (2026-10-30)

| Topic | Decision |
|---|---|
| Training hardware | This PC: RTX 5070 (12 GB), 16 threads. Long runs (overnight / multi-day) are fine, so checkpoints must allow stop and resume |
| Scope v1 | 1v1 and 3–4 players. Standard/random maps, custom maps, tile map with the placement phase. Events, wonders and marathon are **off** in v1 |
| Strength goal | v1: beat the best JS KI (`ki.js`, level `schwer`). Long term: stronger than strong human players |
| Training data | **No human data.** Only imitation of the JS KI plus self-play |
| Engine | Always the current engine, loaded read-only from `../multiplayer-server/HochCiv/js` (exactly what the server runs). Never edited from here |
| Online identity | A regular account at first, with a name that marks it as a bot (e.g. `HochCiv-NN [KI]`). A dedicated bot account type in prod comes later |
| Online activities | Join public lobbies, Rundenspiele, matchmaking. **Matchmaking stays disabled** until the bot account type exists, so human MMR is not affected |
| Language | English (code, comments, docs) |
| Other projects | Read-only. Changes to Hochciv, multiplayer-server or HochCiv-Android only after asking |

## Architecture

```
 Python (training / play)                         Node.js (rules)
┌───────────────────────────────┐   JSON lines  ┌────────────────────────────────┐
│ hochciv_ki.env   gym-like env │◀────stdio────▶│ bridge/worker.js               │
│ hochciv_ki.model PyTorch net  │   N workers   │  loads multiplayer-server/      │
│ hochciv_ki.train BC + PPO     │               │  HochCiv/js/* in a vm sandbox   │
│ hochciv_ki.eval  paired seeds │               │  (like engine-adapter.js)       │
│ hochciv_ki.online socket.io   │               │  candidates, step, observe      │
└───────────────────────────────┘               └────────────────────────────────┘
```

### 1. Simulator bridge (`bridge/`)
- `worker.js` loads the game files in the same order as `engine-adapter.js` and runs games
  with the real engine, so rules are always exact and a new game version needs no port.
- Protocol: one JSON request per line: `reset(cfg, seed)`, `candidates()`, `step(action)`,
  `observe(pi)`, `ki_move()` (the JS KI's choice, used as the teacher), `snapshot` / `restore`.
- Python runs a pool of workers (about 14) in parallel. Simulation speed is the main
  bottleneck, so we profile it early and batch requests.
- A version check reads `APP_VERSION` and stores it in every checkpoint, so a model always
  knows which engine version it was trained on.

### 2. Observation and action space
- **Map:** the hex grid as a 2D tensor (axial coordinates, padded to a maximum size). One
  channel each for terrain, yields, owner, cities (size, capital), armies (owner, strength),
  roads, colonies, threat or range, and so on.
- **Global:** per player (rotated so that "me" always comes first): resources, income,
  power, techs (multi-hot), availability, victory progress, turn order and the round.
- **Actions:** a turn is a sequence of single engine actions (`research`, `foundCity`,
  `growCity`, `buildArmy`, `moveArmy`, `buyPower`, `buildRoad`, `buyTile`, `coverPop`,
  `sacrifice`, `copyTech`, `freeTech`, `backPick`, `nuke`, `endTurn`). The bridge lists every
  legal concrete action (derived from the candidate generation in `ki.js`), each with a
  feature vector. The network scores the candidates (pointer/attention head), so there is
  no fixed action space and no illegal moves.
- **Placement phase / tile map:** v1 starts with `kiPlaceSeat` from `ki.js`, and a learned
  placement policy follows once the main policy is strong.

### 3. Network (`hochciv_ki/model.py`)
- Map encoder (residual CNN over hexes) plus an MLP over player vectors, fused into a
  state embedding.
- Candidate encoder: action features plus map features gathered at the action's
  coordinates. Attention over the candidates gives the policy logits.
- Value head: win probability per seat, so it works for 2–4 players.
- Exported to ONNX as well as PyTorch, which keeps inference light for the online client.

### 4. Training (`hochciv_ki/train/`)
1. **Behaviour cloning:** generate many games of KI `schwer` against itself and against
   bots, and train the policy to predict its actions. Target: about the strength of KI
   `mittel` or better, quickly.
2. **PPO self-play** against a league: past snapshots, KI `schwer`/`mittel`/`leicht` and
   bots. Mixed 1v1 and 3–4 player games. Rewards: win or loss at the end, plus small shaped
   rewards early on (score or victory progress) that are phased out over time.
3. **Curriculum:** 1v1 on standard maps first, then tile maps with placement, then custom
   maps and 3–4 players.
4. Checkpoints with optimizer and league state, plus TensorBoard logs. `train --resume`
   continues a run.

### 5. Evaluation (`hochciv_ki/eval.py`)
- Paired seeds like `tools_ki.js stufen`: every seed is played twice with the seats
  swapped, which cancels out the start player advantage.
- Reports win rate with a confidence interval against KI `schwer`, plus victory types and
  the length of games. Run automatically for every checkpoint.
- **v1 is done** when the model wins more than 55% against KI `schwer` (95% CI above 50%)
  in 1v1 and in 4-player games.

### 6. Online client (`hochciv_ki/online/`)
- Python `python-socketio` client: logs in (`POST /api/auth/login`, cookie jar), sends no
  `Origin` header (scripts are allowed through), reads `state:update`, sends
  `action {type, params}` and answers placement.
- **Lobbies:** polls `GET /api/public-sessions`, joins lobbies that fit the scope (player
  count, no events/wonders/marathon, current version), writes a short chat line saying it
  is an AI, and plays within the turn timer.
- **Rundenspiele:** REST list/join plus `async:open`/`async:action`/`async:place`.
- **Matchmaking:** implemented behind a switch that is **off** until the bot account type
  exists.
- Version guard: if the server's `APP_VERSION` differs from the checkpoint's version, the
  client logs a warning and plays anyway ("always current").
- Tested on **staging** (staging.hochciv.de, its own bot account) before production.
  Credentials go in a git-ignored file.

## Milestones

| # | Milestone | Result |
|---|---|---|
| M0 | Repo skeleton: venv, PyTorch + CUDA, Node bridge loads the engine | `python -m hochciv_ki.selftest` plays a random legal game |
| M1 | Bridge complete: candidates, observation, KI teacher, worker pool | Throughput benchmark (games/hour) |
| M2 | Baseline agents (random, JS KI) + evaluation harness | Reproduces KI-vs-bot numbers from ANNAHMEN.md |
| M3 | Network + behaviour cloning | NN plays complete games and beats `leicht`/`mittel` |
| M4 | PPO self-play with league, 1v1 | Beats KI `schwer` in 1v1 |
| M5 | 3–4 players, tile map/custom maps | **v1 goal:** beats `schwer` in all v1 modes |
| M6 | Online client on staging (lobbies, Rundenspiele) | Plays full games on staging |
| M7 | Production with the bot account | Joins public lobbies on hochciv.de |
| later | Learned placement, events/wonders/marathon, matchmaking with bot account type, stronger search (e.g. MCTS over the engine) | Superhuman strength |

## Risks
- **Simulation speed:** `ki.js` itself simulates many copies, so it is slow as a teacher.
  Mitigation: cache teacher data and use a pool of parallel workers. The NN agent itself
  only needs `candidates` and `step`.
- **Engine changes ("always current"):** observation features read engine state fields
  directly, so a renamed field breaks them. The bridge selftest runs on every new version.
- **Hex geometry / map size:** maps vary in size. Padding to a maximum size plus masks.

## Status (2026-10-31)

| # | Status |
|---|---|
| M0 | Done: venv (Python 3.13, torch 2.11 + CUDA 12.8 on the RTX 5070), Node bridge loads the server's engine (v98), selftest |
| M1 | Done: features (map planes, seat vectors, candidates evaluated on copies), teacher label, worker pool (~6 games/s with observations) |
| M2 | Done: paired evaluation; the teacher (one kiPlan step) plays like KI schwer: 0.525 vs the KI's own 0.500 |
| M3 | Done: BC on 730k samples, 0.540 in 1v1 and 0.283 in 4p against KI schwer (teacher level) |
| M4 | Done: PPO 1v1, 0.559 ± 0.040 on 600 fresh games (BC on the same games 0.523) |
| M5 | Done (v1 goal): `checkpoints/v1.pt` beats KI schwer in every v1 mode – 1v1 tile map 0.610 ± 0.055, 4p tile map 0.310 ± 0.059 (fair 0.25), 3p 0.472 (fair 0.333); details in README. PPO keeps training |
| M6 | Live games done and tested on staging (guest host + bot, no rejected actions); Rundenspiele implemented but untested: they need a second human account on staging |
| M7 | Waiting for the bot account on hochciv.de |

v1 criterion for more than two empires: the fair share is 1/n, so "beats KI schwer" means a
win share significantly above 1/n (95 % CI above 0.25 in 4p), aiming at 10 % above it.
