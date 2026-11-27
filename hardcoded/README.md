# HochCiv – fest programmierte Bots

Bots für HochCiv mit fester, in Python geschriebener Logik. Gegner und Maßstab ist die
eingebaute **KI auf Stufe „schwer"** (`ki.js`). Ziel: sie verlässlich schlagen – im Duell auf
der Plättchenkarte, zu dritt und zu viert, auf der Originalkarte und auf eigenen Karten.
Wunder, Ereignisse und Marathon spielen keine Rolle.

Dieser Ordner war früher das eigene Repo `HochCiv-HardcodedBots` (Historie übernommen). Befehle
unten laufen aus `hardcoded/`; online läuft der Bot (Statibot) über `../deploy/` in Docker.

## Grundsätze

- **Alle Entscheidungen fallen in Python.** Die Regelmaschine ist nur Orakel: Sie beantwortet
  Regelfragen (Kosten, Reichweiten, Angriffs- und Verteidigungswerte, Einkommen …) und führt
  Aktionen aus. Aktionen auf einer Kopie des Spiels auszuprobieren ist erlaubt (`fork`,
  `trials`).
- **Nie** `ki*`-, `bot*`- oder `rush*`-Funktionen der Regelmaschine aufrufen – das wäre die KI
  selbst. `ki.js` darf man lesen, um den Gegner zu verstehen.
- Die Regelmaschine liegt schreibgeschützt in `multiplayer-server/HochCiv/js` (v112, Ordner neben
  `Hochciv-KI`; gesucht wird im nächsten Ordner darüber, der ihn hat) und wird hier **nicht**
  verändert. Anderer Ort: Umgebungsvariable `HOCHCIV_ENGINE_DIR`.

## Aufbau

```
template/             der Rahmen, für alle Bots gleich
  engine/worker.js    lädt die Regelmaschine in einer Node-VM, spricht JSON-Zeilen über stdin/stdout
  hcbots/bridge.py    startet den Worker, schickt Anfragen
  hcbots/turn.py      Turn: Sicht und Werkzeuge für einen Zug (s. u.)
  hcbots/runner.py    spielt eine ganze Partie offline (Python-Sitze gegen KI-Sitze)
  hcbots/hexgrid.py   Hex-Abstände, Nachbarn, Distanzfelder
  example_bot.py      der kleinste Bot – Vorlage zum Kopieren
bots/<name>/bot.py    je Bot ein Ordner; die Klasse hat name = "<name>"
evaluate.py           Auswertung gegen die KI über viele Seeds und alle Sitze
online/client.py      Online-Client: spielt mit einem Bot-Konto auf hochciv.de / staging
online/panel.py       Control-Board (lokale Webseite) zum Steuern des Online-Clients
tests/online_sim.py   die Online-Zugschleife gegen einen simulierten Server testen
```

## Ein Bot

Eine Klasse mit `name` und zwei Methoden:

- `place(seat, plan)` → `{"o", "cell"}` oder `None`: Startplättchen auf Plättchenkarten.
  `seat["choices"][o][cell]` ist das Einkommen einer Hauptstadt dort.
- `turn(t)`: einen Zug spielen. Den Zug **nicht** beenden, das macht der Runner.

Werkzeuge in `t` (`template/hcbots/turn.py`):

| | |
|---|---|
| `t.S`, `t.pi`, `t.me`, `t.round` | Spielstand, eigener Sitz, eigener Spieler, Runde |
| `t.cities(pi)`, `t.armies(pi)`, `t.capital(pi)`, `t.city_at(r,c)`, `t.army_at(r,c)`, `t.enemies()` | Abfragen |
| `t.q(fn, *args)` / `t.call([(fn, *args), …])` | Funktion der Regelmaschine aufrufen (eine / viele in einem Rundlauf). `"$S"` steht für den Spielstand, `{"$city": id}`, `{"$player": pi}` für Objekte; `("$player", pi)` liefert den Spieler selbst |
| `t.act(type, **params)` | Aktion ausführen; gibt `None` oder die Fehlermeldung zurück |
| `t.fork()` | Kopie des Spiels als eigener `Turn` (mit `close()` freigeben) |
| `t.trials([(acts, calls), …])` | viele Versuche auf je einer frischen Kopie in einem Rundlauf: je Versuch `None` (Aktion scheiterte) oder die Werte der `calls` danach |

Zufall in Kopien hat einen eigenen Seed – was dort klappt, muss im echten Spiel nicht genauso
ausgehen.

## Auswerten

```
python evaluate.py rusher                                  # alle Modi, 12 Seeds je Modus
python evaluate.py rusher --modes tile2 --seeds 64         # Duell auf der Plättchenkarte
python evaluate.py rusher --modes tile2 --seeds 64 --losses --opt soft_share=0.5 def_w=1
python evaluate.py rusher --seed0 20 --seeds 1 -v          # eine Partie, ausführlich
```

Modi: `tile2 tile3 tile4` (Plättchenkarte), `orig4` (Originalkarte zu viert),
`custom2 custom3 custom4` (erzeugte eigene Karten). Der Bot sitzt reihum auf jedem Platz;
„fair" ist 1/n. `--opt key=value` reicht Konstruktor-Argumente an den Bot durch.

Die Streuung ist groß: bei 64 Partien etwa ±6 Prozentpunkte. Vergleiche daher mit 64 Seeds
oder mehr, eine Partie dauert 20–40 s je Prozess.

## Regeln, die für Bots zählen (v112)

- **Eroberung:** zwei Treffer in Folge, je am Ende eines eigenen Zuges (Angriff > Verteidigung).
  Ein Fehlschlag setzt den Zähler zurück. Wer eine Hauptstadt erobert, gewinnt.
- **Angriff** je Armee in Reichweite: Macht (+3 mit Belagerung, ×2 mit Dynamit). Rom (Legion):
  +50 % Macht je angrenzender eigener Armee, je Armee einzeln (`attackValue` mit Armeeliste).
- **Verteidigung:** Bevölkerung × (1 + 1 in der Hauptstadt + MG) + 5 mit Stadtmauern + Macht mit
  Burgenbau + Macht × eigene Armeen in Reichweite. Ohne Helfer zählt Macht nicht.
- **Flankieren:** zwei eigene Stellungen in Reichweite, gegenüber (mit Taktik beliebig), und
  mehr Macht als der Gegner.
- **Macht** verfällt zu Beginn des eigenen Zuges (`powerDecay`). Übrige Ressourcen verfallen am
  Zugende.
- **Armeekosten:** Faktor × (Zahl der Armeen + 1); Faktor 5, mit Demokratie 4, mit
  Nationalismus 2.
- **Sklaverei:** eine Bevölkerung opfern → `SLAVERY_COINS` Münzen (seit v111: 5, v110: 4, v84–v109: 7).
  Konstanten der Regelmaschine liest `t.const(name)`.
- **Wachstum:** ein bezahlter Schritt je Stadt und Runde (+1 mit Keramik, +1 mit Dampfmaschine);
  kostet Nahrung und Münzen in Höhe der Bevölkerung.
- **Experimentelle Nationen (seit v111):** Rom, Venedig, Azteken, Ägypten, Deutschland, Maya,
  China, USA – in Lobbys, Rundenspielen und Matchmaking wählbar (nicht in Ranglistenspielen und
  Marathon). Im Bridge-Worker mit `expCivs: true` (Azteken und Ägypten nur mit `wonders`).
  Venedig gründet keine Städte, wächst kostenlos und beansprucht je Runde 3 herrenlose Felder
  (`claimTile`, wie Kolonialismus). China kann einmal je Partie das Einkommen sofort einsammeln –
  dafür gibt es online keine Server-Aktion, der Bot nutzt es nicht.
- **Plättchenkarten** beginnen online mit einer Platzierungsphase (`placement:start`, dann
  `game:start`); `S.seed` ist je Partie zufällig und wird nicht an Clients geschickt.
- **Siege:** Hauptstadt erobern, Wirtschaft (Anteil an der Weltbevölkerung, `victoryOption`),
  Forschung (Singularität).

## Was die KI „schwer" tut (zum Verstehen, aus `ki.js`)

Sie bewertet Spielstände (Einkommen über einen Horizont, Bevölkerung, Techs, Siegfortschritt,
militärisches Risiko) und wählt gierig die beste Aktion je Ressource. Einen ersten Treffer
setzt sie, wenn ihre nächste Welle die beste Antwort des Verteidigers übertrifft
(`kiOffense`, `kiDefensePotential`: Münzeinkommen in Macht × eigene Armeen in Reichweite und
einen Zug entfernt, plus Wachstum). Python-Sitze laufen als `human` – gegen sie rechnet die
KI bei laufender Belagerung mit dem entschlossenen Angreifer. Sie greift früh an (Runde 3–5),
gern kleine Hauptstädte ohne Armeen daneben.

## Bots

### rusher (`bots/rusher`)

Sklavenwirtschaft und eine frühe Belagerung der nächsten feindlichen Hauptstadt.

1. Läuft eine eigene Belagerung, die diesen Zug gewinnt: zuerst zuschlagen.
2. Verteidigung von Städten mit einem Treffer (Wachstum, Helfer, Macht – Pläne auf Kopien
   durchgerechnet).
3. Angriff, wenn eine gewinnende Belagerung bezahlbar ist.
4. Wirtschaft (`econ.py`): je Schritt alle Kandidaten (Forschung, Freiwahlen, Wachstum,
   Gründung, Straßen) auf Kopien ausprobieren, den besten Gewinn je ausgegebener Ressource
   ausführen. Weicht der größte Gewinn davon ab, spielt der Bot beide Varianten auf einer Kopie
   bis zum Zugende durch und nimmt die bessere (`lookahead`) – sonst wächst er oft die
   Hauptstadt und hat danach die Nahrung für eine Gründung nicht mehr.
5. Freie Felder beanspruchen (Kolonialismus, Venedig): die mit dem besten Ertrag.
6. Opfern (Sklaverei), Armeen sammeln, Rest in Macht.

Mit Rom rechnet der Angriff die Legion je Armee über die Regelmaschine; gegen ein feindliches Rom
schätzt die Bedrohung bis zu +100 % je Angreifer (zwei Nachbarn).

Angriffe und Verteidigung rechnen mit verdeckten Reserven: Opfern (Sklaverei), Alchemie
(Wissenschaft → Münzen) und noch nicht erforschte Belagerung/Dynamit bzw. Stadtmauern/
Burgenbau. Die KI-Schätzung zählt nur bereits erforschte Angriffstechniken, deshalb hält
der Bot Belagerung und Dynamit zurück, bis der Angriff steht (`hold_siege`). Die Bedrohung
durch Feinde wird als schlimmster Fall über deren Forschungsmöglichkeiten geschätzt
(`full_threat`). Wie viel der Feind im eigenen Zug noch wachsen kann, schätzt
`enemy_growth` aus seinem Einkommen (Wachstum und Gründungen, gierig nach Preis).

Meldet ein Gegner einen Sieg an (Wirtschaft, Singularität), plant der Bot den Rest der Runde
mit `ClaimPlanner`: eigene Anmeldung erreichen oder den Punktestand (Bevölkerung + Techniken)
übertreffen (`answer_claims`).

Späte Techniken werden wie bei der KI bewertet (`late_bonus`: Eisenbahn, Computertechnik,
Verbundwerkstoffe, Massenmedien/Gentechnik); mit Eisenbahn baut der Bot die Wege zu seinen
Städten zu Schienen aus (`rails`, +4 statt +1 Handelsertrag je Stadt). Wie die KI nimmt er
die Wissenschaftliche Methode vor jeder anderen bezahlten Technik (`method_first`).

Singularitätsrennen: Liegt ein Reich (er selbst oder ein Gegner) höchstens `race_turns` Runden
vor der Singularität, zählt jede Wissenschaft, die vom restlichen Weg dorthin fehlt, mit
`race_w` gegen den Plan – Ressourcen verfallen am Rundenende, also zieht der Bot Techniken auf
dem Weg zur Singularität dem Wachstum vor (`race_mode=hold`; `progress` bewertet nur den
Fortschritt und war schwächer). Reicht es in einem Zug,
erforscht er die ganze Kette auf einmal (`sing_path`).

Optionen (`--opt`): `stage_armies`, `from_round`, `defend`, `soft_share` (vorbeugende
Verteidigung), `garrison_max`/`garrison_k`/`garrison_share` (Armeen an der Hauptstadt),
`routine_sac`, `slav_scale` (Wert der Sklaverei; Standard Münzen/7, da auf 7 Münzen
abgestimmt), `sac_from`/`slavery_from` (erst ab dieser Runde opfern bzw. Sklaverei
nehmen), `strike_sac`, `strike_alch`, `defend_prep`, `def_techs`, `counter_flank`,
`full_threat`, `hold_siege`, `hold_stahl`, `sac_safe`, `answer_claims`, und für die Wirtschaft
(`econ`) `threat_w`, `threat_exp`, `grow_est`, `grow_margin`, `grow_step`, `coin_price`, `def_w`,
`mil_scale`, `mil_from`, `chunk`, `chunk_k`, `chunk_r`, `w_city`, `hold`, `late_bonus`, `rails`,
`method_first`, `sing_path`, `lookahead`, `lookahead_k` (mehr Kandidaten durchspielen), `race_w`, `race_turns`, `race_mode`, `eta_h` (Horizont wie die KI
nach der schnellsten Singularität kürzen; schwächer, daher aus).

Stand gegen KI schwer mit v110 (192 Seeds ab 1001 je Modus; fair = 1/Spielerzahl):

| Modus | tile2 | custom2 | tile3 | tile4 | orig4 | custom3 | custom4 |
|---|---|---|---|---|---|---|---|
| Siege | 61 % | 52 % | 41 % | 40 % | 34 % | 30 % | 29 % |
| fair | 50 % | 50 % | 33 % | 25 % | 25 % | 33 % | 25 % |

Auf custom2 gewinnt die KI gegen sich selbst als Sitz 1 rund 62 %, als Sitz 0 rund 38 %; manche
Starts sind für beide aussichtslos (kaum Nahrung). Wer zuerst eine Hauptstadt trifft, gewinnt
meist. Offene Baustellen: Ersttreffer als zweiter Spieler, die nicht zur Eroberung führen, und
Singularitätsrennen, in denen die KI zuerst ankommt.

## Einrichten

Node.js (≥ 18) und Python (≥ 3.10).

```
pip install -r requirements.txt
python -m unittest discover tests
```

Zugangsdaten (`hcbot_…`-Tokens) und `*.env`-Dateien gehören nie ins Repository.

## Online spielen (Control-Board)

Der Online-Client spielt mit einem **Bot-Konto** als Spieler-Bot (🤖) auf einem HochCiv-Server –
Live-Lobbys, Rundenspiele und Matchmaking. Gesteuert wird er über ein Control-Board im Browser.

1. Bot-Konto anlegen: auf der Website Kontoseite → Reiter **Meine Bots** → Name → **Bot erstellen**
   (braucht die Rolle *Bot-Ersteller*). Das Token `hcbot_…` wird nur einmal gezeigt.
2. `online.env` in `hardcoded/` anlegen (für Docker stattdessen `deploy/bots/<id>/bot.env`) (wird von `.gitignore` ignoriert, nie committen):
   ```
   HOCHCIV_URL=https://staging.hochciv.de
   HOCHCIV_TOKEN=hcbot_...
   ```
   Alternativ als Umgebungsvariablen `HOCHCIV_URL` / `HOCHCIV_TOKEN`.
3. Starten:
   ```
   python -m online.client --open                      # Rusher mit Standardoptionen
   python -m online.client --bot rusher --opt garrison_max=1 --port 8780
   ```
   Das Control-Board läuft auf `http://127.0.0.1:8780/` (nur lokal, jede API-Anfrage braucht das
   Token, das in die Seite geschrieben wird).

Im Control-Board:

| Bereich | Was es tut |
|---|---|
| **Join** | öffentlichen Lobbys / Rundenspielen beitreten, die zu den Filtern passen (Karten, 2–4 Reiche, KI-Sitze, gewertet, Timer). Standard: keine Wunder, keine Ereignisse – dafür spielt der Bot nicht. |
| **Host** | eigene Lobbys / Rundenspiele nach Vorlagen offen halten (Karte, Reiche, KI-Stufe, Timer, Anzahl) |
| **Matchmaking** | gewertete Warteschlange (nur mit Spielern, die Bots zulassen) |
| **Spielliste** | einzelne Lobbys / Rundenspiele gezielt beitreten oder verlassen |
| **Log** | jeder Zug: Runde, gesendete Aktionen, Dauer, lokal abgelehnte Züge |

Die Einstellungen werden in `runs/online-settings.json` gespeichert. Marathon-Spiele (Fähigkeiten-
Draft) spielt der Bot nicht. Startplättchen legt er selbst (`Rusher.place`).

**Wie ein Zug online läuft:** Der Server schickt den vollen Zustand, aber ohne seine Würfel
(`S.seed`). Der Client lädt ihn in die lokale Engine und lässt den Bot wie offline entscheiden.
Jede Aktion prüft zuerst die lokale Engine; erst dann geht sie als `action` an den Server, und
der neue Serverzustand ersetzt sofort den lokalen (`OnlineTurn` in `online/client.py`). So
entscheidet der Bot immer auf dem echten Stand, auch wenn Kampf- oder Forschungswürfe anders
fallen als lokal. Probieren (`fork`, `trials`) bleibt lokal und kostet keine Server-Aufrufe.
`python tests/online_sim.py --mode tile2 --seeds 20` testet genau diese Schleife gegen einen
simulierten Server (zweiter Engine-Worker mit eigenen Würfeln).

Spielt der Server eine neue Engine-Version, nimmt der Client die passende lokale Kopie aus
`multiplayer-server/HochCiv-<version>/js` (laufende Rundenspiele behalten ihre Version).
