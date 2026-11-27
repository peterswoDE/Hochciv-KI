# Bots in Docker, eine Kontrollseite

Alle Online-Bots (KI-Bots aus `hochciv_ki/`, fest programmierte aus `hardcoded/`) laufen als Container. Eine gemeinsame
Seite zeigt sie alle und schaltet zwischen ihren Panels um: **http://127.0.0.1:8700**

| Bot      | Code          | Panel-Port (intern) | Ordner           |
|----------|---------------|---------------------|------------------|
| NeuraBot | Hochciv-KI    | 8765                | `bots/neurabot/` |
| NeuraNew | Hochciv-KI    | 8766                | `bots/neuranew/` |
| StatiBot | `hardcoded/`  | 8780                | `bots/statibot/` |

Ein Ordner pro Bot mit allem, was zu ihm gehört; der Ordnername ist die Bot-ID (= Service-Name in
`docker-compose.yml`):

```
deploy/bots/<id>/
  bot.json        Name, Panel-Port, Art (im Repo; daraus baut der Hub seine Liste)
  bot.env         Zugangsdaten (git-ignoriert)
  settings.json   Panel-Einstellungen, schreibt der Bot selbst (git-ignoriert)
```

Jeder Bot sieht nur seinen eigenen Ordner (als `/data`). Der Hub bekommt nur die `bot.json`-Dateien,
keine `bot.env`.

## Start

Voraussetzung: Docker Desktop (für die GPU: WSL2-Backend mit NVIDIA-Treiber), die Repos
nebeneinander (`Hochciv-KI`, `multiplayer-server`).

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

Jeder Bot hat eine eigene Datei `bots/<id>/bot.env` (git-ignoriert), z. B.:

```
HOCHCIV_URL=https://hochciv.de
HOCHCIV_TOKEN=hcbot_...
```

Der Container sieht nur seine eigene Datei. Die Werte stehen in keiner Umgebungsvariable und keinem
Image.

## Wie es zusammenhängt

- `net` hält das gemeinsame Netz. Bots und Hub hängen daran (`network_mode: service:net`), so
  bleibt jedes Bot-Panel auf `127.0.0.1:<Port>` wie ohne Docker. Nur der Hub ist nach außen
  veröffentlicht, und nur auf `127.0.0.1` des Rechners.
- `hub/hub.py` liest `bots/*/bot.json`, zeigt die Übersicht und reicht Seite und API des gewählten Bots
  durch. Den Panel-Token der Bots behält er für sich; der Browser sieht nur den des Hubs.
- Die Bots selbst sind unverändert: dieselben Clients, dieselben Panels.
- Gemountet (nur lesend): `checkpoints/`, `multiplayer-server` (Engine-Kopien). Schreibend:
  `bots/<id>/` (Panel-Einstellungen) und bei den KI-Bots `runs/online_positions` (Stellungen
  verlorener Partien für `train_ppo --loss-dir`).

## Neuer Bot

1. Ordner `bots/<id>/` mit `bot.json` (eigener, freier `port`) und `bot.env` anlegen.
2. In `docker-compose.yml` einen Service `<id>` anlegen (Vorlage: `neuranew` oder `statibot`) mit
   demselben `--port` und `./bots/<id>:/data`; beim `hub` die Zeile für `bots/<id>/bot.json`
   ergänzen.
3. `docker compose up -d --build`

## Ohne GPU

Den `deploy:`-Block bei `x-ki` entfernen und mit
`docker compose build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu` bauen; die
Bots nehmen dann die CPU (`--device auto`).
