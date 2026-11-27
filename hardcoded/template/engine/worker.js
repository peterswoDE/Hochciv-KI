/* HochCiv-HardcodedBots engine worker – the rules oracle for the Python bots.

   Loads the HochCiv rules engine (read-only, by default ../../multiplayer-server/HochCiv/js, the
   version the server runs) into a vm sandbox like the server's engine-adapter.js and serves
   requests over stdin/stdout: one JSON request per line, one JSON response per line.

   Request:  { "id": 1, "cmd": "reset", ...args }
   Response: { "id": 1, "ok": true, ...result }  or  { "id": 1, "ok": false, "error": "..." }

   Seats: "py" (controlled from Python; to the engine a human seat), "ki" (the JS KI, js/ki.js,
   level `kiLevel`, default "schwer") and "bot" (js/bots.js). The worker plays every ki/bot seat
   itself, so after reset and after every ended turn the game is over or waits for a py seat.

   The Python bots make every decision themselves. The worker only answers rule questions
   (`call`: any engine function except the KI's and the bots' own planners), applies actions in
   the server's format (`act`, same as engine-adapter applyAction) and copies games so a bot can
   try actions out (`fork`).                                                                     */
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const readline = require('readline');

// default: the server repo in the nearest folder above that has one (next to Hochciv-KI)
function defaultEngineDir() {
  for (let d = __dirname; ; d = path.dirname(d)) {
    const js = path.join(d, 'multiplayer-server', 'HochCiv', 'js');
    if (fs.existsSync(js)) return js;
    if (path.dirname(d) === d) return path.resolve(__dirname, '..', '..', '..', 'multiplayer-server', 'HochCiv', 'js');
  }
}
const ENGINE_DIR = process.env.HOCHCIV_ENGINE_DIR || defaultEngineDir();
// Same order as engine-adapter.js (index.html minus the DOM code)
const FILES = ['data.js', 'civs.js', 'i18n.js', 'hex.js', 'tiles.js', 'engine.js', 'expansion.js',
  'bots.js', 'ki.js', 'marathon.js'];

const sandbox = {
  console: { log() {}, info() {}, warn() {}, error: (...a) => process.stderr.write(a.join(' ') + '\n') },
  Math, JSON, Set, Map, Array, Object, Number, String, Infinity, parseInt, parseFloat, isNaN,
  isFinite, Error, TypeError, RangeError, RegExp, Date, Symbol,
  T: (str, ...args) => {
    let out = str;
    args.forEach(a => { if (out && out.replace) out = out.replace('%s', a); });
    return out;
  },
};
vm.createContext(sandbox);
for (const f of FILES) {
  vm.runInContext(fs.readFileSync(path.join(ENGINE_DIR, f), 'utf8'), sandbox, { filename: f });
}
const E = sandbox;
// top-level const/let of the engine files are not properties of the sandbox – read them by name
const konst = name => vm.runInContext(`typeof ${name} !== 'undefined' ? ${name} : undefined`, sandbox);
const VERSION = konst('APP_VERSION');
const CIVS = konst('CIVS');
// with the experimental nations (v111: EXP_CIVS, in games with the expCivs option)
const ALL_CIVS = konst('ALL_CIVS') || CIVS;
const CIV_BY_KEY = Object.fromEntries(ALL_CIVS.map(c => [c.k, c]));
/* Nations a game draws from: the base ones, with exp also the experimental ones (those that need
   wonders only with wonders) – as the server's lobby offers them (engine civKeysFor). */
function civPool(exp, wo) {
  const f = konst('civKeysFor');
  return f ? f({ exp: !!exp, wo: !!wo }) : CIVS.map(c => c.k);
}
/* expCivs as the server applies it: tile and custom maps only (no marathon here at all). */
const expOpts = (a, fixed) => ({ exp: !!a.expCivs && !fixed, wo: !!a.wonders });

/* ------------------------------------------------------------------ games */
const games = new Map();   // gid -> { S, py: Set<pi>, sandboxed?: true }

function mulberry(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

/* Seats: [{ kind: 'py'|'ki'|'bot', civ?, ability?, kiLevel? }]. Missing civ/ability are drawn
   from the seed: tile and custom maps allow any nation (also twice), the original map needs
   distinct ones. opts: { exp, wo } – the experimental nations in the draw (expOpts). */
function makePlayers(seats, seed, distinct, opts = {}) {
  const rnd = mulberry(seed ^ 0x5bd1e995);
  const used = new Set(seats.filter(s => s.civ).map(s => s.civ));
  return seats.map(s => {
    let civ = s.civ;
    if (!civ) {
      const pool = civPool(opts.exp, opts.wo).filter(k => !distinct || !used.has(k));
      civ = pool[Math.floor(rnd() * pool.length)];
      used.add(civ);
    }
    if (!CIV_BY_KEY[civ]) throw new Error(`engine ${VERSION} has no nation ${civ}`);
    const abilities = CIV_BY_KEY[civ].abilities || [{ k: 'basis' }];
    const ability = s.ability || abilities[Math.floor(rnd() * abilities.length)].k;
    const kind = s.kind === 'py' ? 'human' : s.kind;
    const p = { civ, ability, kind, diff: s.diff || 'prinz' };
    if (s.kind === 'ki') p.kiLevel = s.kiLevel || 'schwer';
    return p;
  });
}

/* A random custom map (as the map editor stores one) for `n` empires – same generator as in
   Hochciv-KI, so seeds mean the same maps in both repos. */
function genCustomMap(seed, n) {
  const rnd = mulberry(seed ^ 0x2545F491);
  for (let attempt = 0; attempt < 20; attempt++) {
    const H = 8 + Math.floor(rnd() * 13);
    const W = 10 + Math.floor(rnd() * 21);
    let land = Array.from({ length: H }, () => Array.from({ length: W }, () => rnd() < 0.6));
    for (let it = 0; it < 3; it++) {
      land = land.map((row, r) => row.map((v, c) => {
        let k = 0, m = 0;
        for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
          const rr = r + dr, cc = c + dc;
          if (rr < 0 || rr >= H || cc < 0 || cc >= W) continue;
          m++; if (land[rr][cc]) k++;
        }
        return k * 2 > m || (k * 2 === m && v);
      }));
    }
    const LAND = 'GGGGGGGGWWWWBBBFFFIV';
    const off = rnd() < 0.3;
    const rows = land.map(row => row.map(v => v ? LAND[Math.floor(rnd() * LAND.length)] : (off && rnd() < 0.5 ? 'X' : 'M')).join(''));
    const cells = [];
    rows.forEach((row, r) => [...row].forEach((t, c) => { if ('GWBFI'.includes(t)) cells.push([r, c]); }));
    for (let i = cells.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [cells[i], cells[j]] = [cells[j], cells[i]]; }
    const want = Math.min(8, n + Math.floor(rnd() * 3));
    for (const minD of [6, 5, 4, 3]) {
      const spawns = [];
      for (const p of cells) {
        if (spawns.length >= want) break;
        if (spawns.every(q => E.hexDistance(p[0], p[1], q[0], q[1]) >= minD)) spawns.push(p);
      }
      if (spawns.length >= n) return { name: 'Zufallskarte ' + seed, rows, spawns };
    }
  }
  throw new Error('no custom map for ' + n + ' players');
}

/* The placement choices of a seat on a tile map: per orientation o and cell n the first-turn
   income of a capital there (placeYieldTable, the numbers the lobby shows), or null. */
function placeChoices(plan, seat, player) {
  return E.placeYieldTable(plan, seat, { civ: player.civ, ability: player.ability || 'basis' });
}

/* Step 1 on tile maps: the plan the seed gives and, for every py seat, its placement choices.
   Python picks { o, cell } per seat and passes them to reset as `placements`. */
function tilePlanFor(a) {
  const seed = (a.seed ?? 0) | 0;
  const players = makePlayers(a.players, seed, false, expOpts(a, a.map === 'original'));
  const plan = E.tilePlan(players.map(p => p.civ), seed, { hex: a.map === 'plaettchen-hex' });
  if (!plan) throw new Error('no tile map for ' + players.length + ' players');
  const seats = plan.seats.filter(s => a.players[s.idx].kind === 'py')
    .map(s => ({ idx: s.idx, slot: s.slot, civ: players[s.idx].civ, ability: players[s.idx].ability,
      choices: placeChoices(plan, s, players[s.idx]) }));
  return { plan, seats };
}

/* map: 'plaettchen' (tile map with placement), 'plaettchen-hex', 'original', 'customgen' (a random
   custom map, genCustomMap) or a custom map object as stored by the map editor. Since v111 the
   engine has one tile map per player count ('plaettchen-hex' is the same). expCivs: the
   experimental nations in the draw (not on the original map).
   placements: { seatIdx: { o, cell } } for the py seats on tile maps (tilePlanFor); a py seat
   without one is placed like a bot. ki seats place with the JS KI, bots like bots. */
function reset(a) {
  const seed = (a.seed ?? Math.floor(Math.random() * 2 ** 31)) | 0;
  const map = a.map === 'customgen' ? genCustomMap(seed, a.players.length) : (a.map || 'plaettchen');
  const fixed = map === 'original';
  const players = makePlayers(a.players, seed, fixed, expOpts(a, fixed));
  const cfg = {
    seed, players, duel: players.length === 2, events: !!a.events,
    eventMode: a.eventMode === 'easy' ? 'easy' : 'hard', wonders: !!a.wonders, startPlayer: a.startPlayer ?? 0,
  };
  if (map === 'plaettchen' || map === 'plaettchen-hex') {
    const plan = E.tilePlan(players.map(p => p.civ), seed, { hex: map === 'plaettchen-hex' });
    if (!plan) throw new Error('no tile map for ' + players.length + ' players');
    const setup = E.rollSetup(Object.assign({}, cfg));
    const rnd = E.mapRng(seed + 12345);
    const placements = a.placements || {};
    plan.seats.forEach(seat => {
      const pl = players[seat.idx], want = placements[seat.idx];
      let err = 'none';
      if (want && a.players[seat.idx].kind === 'py') err = E.placeSeat(plan, seat, want.o, want.cell);
      else if (pl.kind === 'ki') err = E.kiPlaceSeat(plan, seat, pl, setup);
      if (err) E.botPlaceSeat(plan, seat, rnd);
    });
    Object.assign(cfg, { map: E.tileMap(plan), avail: setup.avail, wpool: setup.wpool });
  } else if (fixed) {
    cfg.map = JSON.parse(JSON.stringify(konst('DEFAULT_MAP')));
  } else {
    const r = E.checkCustomMap(map);
    if (r.error) throw new Error('custom map: ' + r.error);
    cfg.map = E.mapForPlayers(r.map, players.length, seed);
    if (!cfg.map) throw new Error('custom map has too few start points');
  }
  const S = E.newGame(cfg);
  const py = new Set(S.players.map((p, i) => [p, i]).filter(([p]) => p.kind === 'human').map(([, i]) => i));
  const gid = a.gid ?? nextGid();
  const g = { S, py };
  games.set(gid, g);
  runOthers(g);
  return Object.assign({ gid }, status(g));
}
const nextGid = () => (games.size ? Math.max(...games.keys()) + 1 : 1);

function status(g) {
  const S = g.S;
  const out = { cur: S.cur, round: S.round, over: !!S.over, py: [...g.py] };
  if (S.over) {
    out.winners = S.over.winners || (S.over.winner != null ? [S.over.winner] : []);
    out.how = S.over.how;
  }
  return out;
}

/* Play the ki and bot seats until a py seat is on turn or the game is over. */
function runOthers(g) {
  const S = g.S;
  if (g.sandboxed) return;
  for (let guard = 0; !S.over && !g.py.has(S.cur) && guard < 400; guard++) {
    const pi = S.cur, p = S.players[pi];
    if (p.kind === 'ki') E.kiTurn(S, pi); else E.botTurn(S, pi);
    if (S.over) break;
    E.endTurn(S);
  }
}

/* A server action applied like engine-adapter.js applyAction. Returns the error text or null.
   After endTurn the ki and bot seats move (not in a forked sandbox). */
function serverAction(g, type, params) {
  const S = g.S, pi = S.cur;
  if (S.over) return 'game over';
  const city = () => S.cities.find(c => c.id === params.cityId);
  const need = x => x || (() => { throw new Error('not found'); })();
  switch (type) {
    case 'research': return E.doResearch(S, pi, params.tech);
    case 'freeTech': return E.useFreeTech(S, pi, params.tech);
    case 'freePick': return E.useFreePick(S, pi, params.tech);
    case 'backPick': return E.useBackPick(S, pi, params.tech);
    case 'copyTech': return E.copyTech(S, pi, params.tech, params.mode);
    case 'foundCity': return E.foundCity(S, pi, params.r, params.c);
    case 'growCity': return E.growCity(S, pi, need(city()), params.mode);
    case 'sacrifice': return E.sacrifice(S, pi, need(city()));
    case 'buildArmy': return E.buildArmy(S, pi, need(city()));
    case 'moveArmy': {
      const a = S.armies.find(x => x.id === params.armyId);
      if (!a || a.owner !== pi) return 'army';
      return E.moveArmy(S, a, params.r, params.c);
    }
    case 'buyPower': return E.buyPower(S, pi, params.n || 1);
    case 'buildRoad': return E.buildRoad(S, pi, params.r, params.c, params.target);
    case 'buyTile': return E.buyTile(S, pi, params.r, params.c);
    case 'claimTile': return E.claimTile(S, pi, params.r, params.c);
    case 'coverPop': return E.coverPop(S, pi, params.kind, params.amount);
    case 'uncoverPop': return E.uncoverPop(S, pi, params.kind, params.amount);
    case 'buildWonder': return E.buildWonder(S, pi, need(city()), params.wonder);
    case 'nuke': return E.nuke(S, pi, params.r, params.c);
    case 'endTurn': {
      const stop = E.blockingIssues(S, pi);
      if (stop && stop.length) return String(stop[0]);
      E.endTurn(S);
      runOthers(g);
      return null;
    }
    default: return 'unknown action ' + type;
  }
}

/* ------------------------------------------------------------------ rules oracle
   `call`: { fn, args } – any engine function, arguments as JSON with references:
     "$S" the game state, {"$city": id}, {"$army": id}, {"$player": pi}, also inside lists.
   The planners of the JS KI and the bots are off limits: the decisions are made in Python. */
const FORBIDDEN = /^(ki|bot|rush)/i;
function resolve(S, v) {
  if (v === '$S') return S;
  if (Array.isArray(v)) return v.map(x => resolve(S, x));     // lists of armies etc. (attackValue)
  if (v && typeof v === 'object' && !Array.isArray(v)) {
    if ('$city' in v) return S.cities.find(c => c.id === v.$city) || null;
    if ('$army' in v) return S.armies.find(a => a.id === v.$army) || null;
    if ('$player' in v) return S.players[v.$player] || null;
  }
  return v;
}
function plain(x, depth = 0) {
  if (x instanceof Map) return [...x.entries()].map(([k, v]) => [k, plain(v, depth + 1)]);
  if (x instanceof Set) return [...x].map(v => plain(v, depth + 1));
  if (x === Infinity) return 1e9;
  if (Array.isArray(x)) return x.map(v => plain(v, depth + 1));
  if (x && typeof x === 'object') {
    if (depth > 6) return null;
    const o = {};
    for (const k of Object.keys(x)) if (typeof x[k] !== 'function') o[k] = plain(x[k], depth + 1);
    return o;
  }
  return x;
}
function callFn(S, fn, args) {
  if (FORBIDDEN.test(fn)) throw new Error('not allowed: ' + fn);
  // plain state reads for trial copies: '$player' pi -> the player object
  if (fn === '$player') return plain(S.players[args[0]]);
  let f = E[fn];
  if (typeof f !== 'function') f = konst(fn);
  if (typeof f !== 'function') throw new Error('no engine function ' + fn);
  return plain(f.apply(null, (args || []).map(v => resolve(S, v))));
}
function constant(name) {
  if (!/^[A-Z_][A-Z0-9_]*$/.test(name)) throw new Error('bad constant ' + name);
  return plain(konst(name));
}

/* The state without the log (large and only text). */
function stateOf(g) {
  const out = {};
  for (const k in g.S) if (k !== 'log' && k !== 'kiSim') out[k] = g.S[k];
  return out;
}

/* A copy of a game to try actions on. Nothing moves by itself in it (no ki/bot turns after
   endTurn), and it has no hidden randomness of its own beyond the copied state. */
function fork(a) {
  const g = game(a);
  const S = JSON.parse(JSON.stringify(Object.assign({}, g.S, { log: [] })));
  S.seed = trialSeed(S.seed);
  const gid = nextGid();
  games.set(gid, { S, py: new Set(g.py), sandboxed: true });
  return { gid };
}

/* Copies get their own dice (as the KI's planning copies do): what a try rolls is not what the
   real game will roll, so trying an action reveals nothing about future dice. */
let trialCount = 0;
function trialSeed(seed) {
  trialCount = (trialCount + 1) | 0;
  return Math.imul((seed | 0) ^ 0x5bd1e995, 2654435761 + trialCount) | 0;
}

/* Many throwaway tries in one round trip: each trial runs its actions on a fresh copy of the
   game and then asks its rule questions on that copy. trials: [{ acts: [{type, params}],
   calls: [{fn, args}] }] -> results: [{ err } | { err: null, results: [value | {error}] }] */
function tryMany(a) {
  const g = game(a);
  const base = JSON.stringify(Object.assign({}, g.S, { log: [] }));
  return { results: a.trials.map(tr => {
    const S = JSON.parse(base);
    S.seed = trialSeed(S.seed);
    const tg = { S, py: new Set(g.py), sandboxed: true };
    for (const x of tr.acts || []) {
      let err;
      try { err = serverAction(tg, x.type, x.params || {}); } catch (e) { err = String(e && e.message || e); }
      if (err) return { err: String(err) };
    }
    return { err: null, results: (tr.calls || []).map(c => {
      try { return { v: callFn(S, c.fn, c.args) }; } catch (e) { return { error: String(e && e.message || e) }; }
    }) };
  }) };
}

/* ------------------------------------------------------------------ online helpers */
/* Load a server state (online play) as a game; nothing moves by itself in it. */
function load(a) {
  const gid = a.gid ?? nextGid();
  games.set(gid, { S: a.state, py: new Set([a.pi]), sandboxed: true });
  return Object.assign({ gid }, status(games.get(gid)));
}

/* ------------------------------------------------------------------ protocol */
function game(a) {
  const g = games.get(a.gid);
  if (!g) throw new Error('unknown game ' + a.gid);
  return g;
}
const handlers = {
  info: () => ({ version: VERSION, engineDir: ENGINE_DIR, civs: CIVS.map(c => c.k), expCivs: ALL_CIVS.filter(c => c.exp).map(c => c.k) }),
  tile_plan: a => tilePlanFor(a),
  // placement choices on a server plan (online): { plan, seatIdx, player: {civ, ability} }
  place_choices: a => {
    const seat = a.plan.seats.find(s => s.idx === a.seatIdx);
    if (!seat) throw new Error('no seat ' + a.seatIdx);
    return { choices: placeChoices(a.plan, seat, a.player) };
  },
  reset: a => reset(a),
  load: a => load(a),
  fork: a => fork(a),
  try: a => tryMany(a),
  close: a => { games.delete(a.gid); return {}; },
  status: a => status(game(a)),
  state: a => Object.assign({ state: stateOf(game(a)) }, status(game(a))),
  // one action; with `state: true` the new state comes along
  act: a => {
    const g = game(a);
    let err;
    try { err = serverAction(g, a.type, a.params || {}); } catch (e) { err = String(e && e.message || e); }
    const out = Object.assign({ err: err ? String(err) : null }, status(g));
    if (a.state) out.state = stateOf(g);
    return out;
  },
  // several rule questions in one round trip: calls [{ fn, args }] -> results [value | {error}]
  call: a => {
    const S = game(a).S;
    return { results: a.calls.map(c => {
      try { return { v: callFn(S, c.fn, c.args) }; } catch (e) { return { error: String(e && e.message || e) }; }
    }) };
  },
  consts: a => ({ consts: Object.fromEntries(a.names.map(n => [n, constant(n)])) }),
  log: a => ({ log: game(a).S.log.slice(-(a.n || 50)) }),
};

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
rl.on('line', line => {
  if (!line.trim()) return;
  let req;
  try { req = JSON.parse(line); } catch (e) {
    process.stdout.write(JSON.stringify({ ok: false, error: 'bad json' }) + '\n');
    return;
  }
  let res;
  try {
    const h = handlers[req.cmd];
    if (!h) throw new Error('unknown cmd ' + req.cmd);
    res = Object.assign({ id: req.id, ok: true }, h(req));
  } catch (e) {
    res = { id: req.id, ok: false, error: String(e && e.stack || e) };
  }
  process.stdout.write(JSON.stringify(res) + '\n');
});
