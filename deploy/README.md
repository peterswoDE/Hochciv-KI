# Bots in Docker, eine Kontrollseite

Alle Online-Bots (Hochciv-KI und HochCiv-HardcodedBots) laufen als Container. Eine gemeinsame
Seite zeigt sie alle und schaltet zwischen ihren Panels um: **http://127.0.0.1:8700**

| Bot      | Code          | Panel-Port (intern) | Ordner           | Secret                 |
|----------|---------------|---------------------|------------------|------------------------|
| NeuraBot | Hochciv-KI    | 8765                | `bots/neurabot/` | `secrets/neurabot.env` |
| NeuraNew | Hochciv-KI    | 8766                | `bots/neuranew/` | `secrets/neuranew.env` |
| StatiBot | HardcodedBots | 8780                | `bots/statibot/` | `secrets/statibot.env` |

Ein Ordner pro Bot, der Ordnername ist die Bot-ID (= Service-Name in `docker-compose.yml`):

```
deploy/
  bots/<id>/bot.json     Name, Panel-Port, Art (im Repo; daraus baut der Hub seine Liste)
  bots/<id>/data/        was der Bot schreibt: settings.json (git-ignoriert)
  secrets/<id>.env       Zugangsdaten (git-ignoriert)
```

Die Secrets liegen bewusst nicht in `bots/`: der Hub liest `bots/` und soll sie nicht sehen.

## Start

Voraussetzung: Docker Desktop (für die GPU: WSL2-Backend mit NVIDIA-Treiber), die Repos
nebeneinander (`Hochciv-KI`, `HochCiv-HardcodedBots`, `multiplayer-server`).

```
cd Hochciv-KI/deploy
docker compose up -d --build      # bauen und starten (auch nach git pull)
docker compose ps
docker compose logs -f neurabot
docker compose restart neuranew
docker compose down               # alles stoppen
```

Die Container starten nach einem Windows-Neustart von selbst wieder, sobald Docker Desktop läuft
(`restart: unless-stopped`; in Docker Desktop „Start when you sign in" einschalten).

Wichtig: Ein Bot-Konto darf nur einmal laufen. Vorher die Bots außerhalb von Docker beenden.

## Secrets

Jeder Bot hat eine eigene Datei `secrets/<bot>.env` (git-ignoriert), z. B.:

```
HOCHCIV_URL=https://hochciv.de
HOCHCIV_TOKEN=hcbot_...
```

Der Container bekommt nur seine eigene Datei, als `/run/secrets/bot.env`. Die Werte stehen in
keiner Umgebungsvariable und keinem Image.

## Wie es zusammenhängt

- `net` hält das gemeinsame Netz. Bots und Hub hängen daran (`network_mode: service:net`), so
  bleibt jedes Bot-Panel auf `127.0.0.1:<Port>` wie ohne Docker. Nur der Hub ist nach außen
  veröffentlicht, und nur auf `127.0.0.1` des Rechners.
- `hub/hub.py` liest `bots/*/bot.json`, zeigt die Übersicht und reicht Seite und API des gewählten Bots
  durch. Den Panel-Token der Bots behält er für sich; der Browser sieht nur den des Hubs.
- Die Bots selbst sind unverändert: dieselben Clients, dieselben Panels.
- Gemountet (nur lesend): `checkpoints/`, `multiplayer-server` (Engine-Kopien). Schreibend:
  `bots/<bot>/data/` (Panel-Einstellungen) und bei den KI-Bots `runs/online_positions` (Stellungen
  verlorener Partien für `train_ppo --loss-dir`).

## Neuer Bot

1. Ordner `bots/<id>/` mit `bot.json` anlegen (eigener, freier `port`).
2. `secrets/<id>.env` anlegen.
3. In `docker-compose.yml` einen Service `<id>` anlegen (Vorlage: `neuranew` oder `statibot`) mit
   demselben `--port`, dem Secret und `./bots/<id>/data:/data`; das Secret unten unter
   `secrets:` eintragen.
4. `docker compose up -d --build`

## Ohne GPU

Den `deploy:`-Block bei `x-ki` entfernen und mit
`docker compose build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu` bauen; die
Bots nehmen dann die CPU (`--device auto`).
