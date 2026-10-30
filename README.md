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
| `hochciv_ki/selftest.py` | M0 selftest |
