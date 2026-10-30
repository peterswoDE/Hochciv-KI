/* Hochciv-KI bridge worker.

   Loads the HochCiv rules engine (read-only, by default from ../multiplayer-server/HochCiv/js,
   the version the server runs) into a vm sandbox, like the server's engine-adapter.js, and
   serves games over stdin/stdout: one JSON request per line, one JSON response per line.

   Request:  { "id": 1, "cmd": "reset", ...args }
   Response: { "id": 1, "ok": true, ...result }  or  { "id": 1, "ok": false, "error": "..." }

   Seats are "agent" (controlled from Python), "ki" (js/ki.js) or "bot" (js/bots.js). The
   worker plays every non-agent seat itself, so after reset and after each ended turn the game
   is either over or waiting for an agent.

   Action space (v1): the candidate list the JS KI builds for itself (kiCandidates, every entry
   a short sequence of real engine actions), plus "end". "end" lets the JS KI position the
   remaining armies and use open free picks (kiPositionArmies, kiUsePicks), then ends the
   turn. Learned army positioning comes later.                                              */
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const readline = require('readline');

const ENGINE_DIR = process.env.HOCHCIV_ENGINE_DIR ||
  path.resolve(__dirname, '..', '..', 'multiplayer-server', 'HochCiv', 'js');
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
  const p = path.join(ENGINE_DIR, f);
  vm.runInContext(fs.readFileSync(p, 'utf8'), sandbox, { filename: f });
}
const E = sandbox;
const konst = name => vm.runInContext(`typeof ${name} !== 'undefined' ? ${name} : undefined`, sandbox);
const VERSION = konst('APP_VERSION');
const CIVS = konst('CIVS');
const CIV_BY_KEY = Object.fromEntries(CIVS.map(c => [c.k, c]));
const armiesOf = konst('armiesOf');
const FEAT = require('./features').makeFeatures(E, konst);

/* ------------------------------------------------------------------ games */
const MAX_TURN_STEPS = 60;
const games = new Map();   // gid -> { S, agents: Set<pi>, turn: per-turn KI context or null }

function mulberry(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

/* Seats: [{ kind: 'agent'|'ki'|'bot', civ?, ability?, kiLevel?, diff? }]. Missing civ/ability
   are drawn from the seed: tile and custom maps allow any nation (also twice), fixed maps need
   distinct ones. */
function makePlayers(seats, seed, distinct) {
  const rnd = mulberry(seed ^ 0x5bd1e995);
  const used = new Set(seats.filter(s => s.civ).map(s => s.civ));
  return seats.map(s => {
    let civ = s.civ;
    if (!civ) {
      const pool = CIVS.map(c => c.k).filter(k => !distinct || !used.has(k));
      civ = pool[Math.floor(rnd() * pool.length)];
      used.add(civ);
    }
    const abilities = CIV_BY_KEY[civ].abilities || [{ k: 'basis' }];
    const ability = s.ability || abilities[Math.floor(rnd() * abilities.length)].k;
    const p = { civ, ability, kind: s.kind === 'agent' ? 'human' : s.kind, diff: s.diff || 'prinz' };
    if (s.kind !== 'bot') p.kiLevel = s.kiLevel || 'schwer';
    return p;
  });
}

/* A random custom map (as the map editor would store one) for `n` empires: land masses from
   smoothed noise, the land types in about the shares of the tile maps, start points spread
   out. Training on these keeps the network from learning only the built-in maps. */
function genCustomMap(seed, n) {
  const rnd = mulberry(seed ^ 0x2545F491);
  for (let attempt = 0; attempt < 20; attempt++) {
    const H = 8 + Math.floor(rnd() * 13), W = 10 + Math.floor(rnd() * 21);
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
    const off = rnd() < 0.3;   // some maps without a sea border: "Kein Feld" around them
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

/* map: 'plaettchen' (tile map with placement, also the lobby's "random"), 'original', 'gross',
   'customgen' (a random custom map, genCustomMap),
   or a custom map object as stored by the map editor ({ name, rows, spawns, ... }). Agent seats
   place their start tile with the JS KI (kiPlaceSeat) in v1. */
function reset(a) {
  const seed = (a.seed ?? Math.floor(Math.random() * 2 ** 31)) | 0;
  const map = a.map === 'customgen' ? genCustomMap(seed, a.players.length) : (a.map || 'plaettchen');
  const fixed = map === 'original' || map === 'gross';
  const players = makePlayers(a.players, seed, fixed);
  const cfg = {
    seed, players, duel: players.length === 2, events: !!a.events, eventMode: 'hard',
    wonders: !!a.wonders, startPlayer: a.startPlayer ?? 0,
  };
  if (map === 'plaettchen') {
    const plan = E.tilePlan(players.map(p => p.civ), seed);
    if (!plan) throw new Error('no tile map for ' + players.length + ' players');
    const setup = E.rollSetup(Object.assign({}, cfg));
    const rnd = E.mapRng(seed + 12345);
    plan.seats.forEach(seat => {
      const pl = players[seat.idx];
      if (pl.kind === 'bot' || E.kiPlaceSeat(plan, seat, pl, setup)) E.botPlaceSeat(plan, seat, rnd);
    });
    Object.assign(cfg, { map: E.tileMap(plan), avail: setup.avail, wpool: setup.wpool });
  } else if (fixed) {
    cfg.map = JSON.parse(JSON.stringify(konst(map === 'gross' ? 'MAP_GROSS' : 'DEFAULT_MAP')));
  } else {
    const r = E.checkCustomMap(map);
    if (r.error) throw new Error('custom map: ' + r.error);
    cfg.map = E.mapForPlayers(r.map, players.length, seed);
    if (!cfg.map) throw new Error('custom map has too few start points');
  }
  const S = E.newGame(cfg);
  // newGame sorts into turn order and keeps the seat as `slot`
  const agents = new Set(S.players.map((p, i) => [p, i]).filter(([p]) => p.kind === 'human').map(([, i]) => i));
  /* newGame keeps kiLevel only for KI seats. Agent seats need it too: the KI context (threat,
     strike plans), the teacher and kiTurn read it, and would fall back to KI_DEFAULT_LEVEL. */
  for (const pi of agents) S.players[pi].kiLevel = players[S.players[pi].slot].kiLevel;
  const gid = a.gid ?? (games.size ? Math.max(...games.keys()) + 1 : 1);
  const g = { S, agents, turn: null };
  games.set(gid, g);
  runOthers(g);
  return Object.assign({ gid }, status(g));
}

function status(g) {
  const S = g.S;
  const out = { cur: S.cur, round: S.round, over: !!S.over, agents: [...g.agents] };
  if (S.over) {
    out.winners = S.over.winners || (S.over.winner != null ? [S.over.winner] : []);
    out.how = S.over.how;
  }
  return out;
}

/* Play bot and KI seats until an agent is on turn or the game is over. */
function runOthers(g) {
  const S = g.S;
  for (let guard = 0; !S.over && !g.agents.has(S.cur) && guard < 400; guard++) {
    const pi = S.cur, p = S.players[pi];
    if (p.kind === 'ki') E.kiTurn(S, pi); else E.botTurn(S, pi);
    if (S.over) break;
    E.endTurn(S);
  }
  g.turn = null;
}

/* The KI context of the agent's current turn, made once per turn like kiTurn does. */
function turnCtx(g) {
  const S = g.S, pi = S.cur;
  if (!g.turn || g.turn.pi !== pi || g.turn.round !== S.round) {
    E.kiSeedRng(S, pi);
    g.turn = { pi, round: S.round, K: E.kiContext(S, pi), memo: {}, failed: new Set(), cands: null, steps: 0 };
  }
  return g.turn;
}

function candidates(g) {
  const S = g.S, pi = S.cur;
  if (S.over || !g.agents.has(pi)) return [];
  const t = turnCtx(g);
  E.kiStepMemo(t.memo, S, pi);
  const list = E.kiCandidates(S, pi, t.K, t.memo, false).filter(c => !t.failed.has(c.tag));
  t.cands = list;
  return list.map(c => c.tag).concat(['end']);
}

/* Apply candidate `i` of the last candidates() call (index len = "end"). */
function step(g, i) {
  const S = g.S, pi = S.cur;
  if (S.over) return { err: 'game over' };
  if (!g.agents.has(pi)) return { err: 'not an agent turn' };
  const t = turnCtx(g);
  if (!t.cands) candidates(g);
  const finish = () => (g.online ? endOnline(g) : endAgentTurn(g));
  if (i === t.cands.length) return finish();
  const c = t.cands[i];
  if (!c) return { err: 'bad candidate index ' + i };
  t.cands = null;
  const military = /^(strike|defend|flank)/.test(c.tag);
  const before = military ? new Map(armiesOf(S, pi).map(a => [a.id, a.r + ',' + a.c])) : null;
  let err;
  try { err = c.run(S); } catch (e) { err = String(e && e.message || e); }
  if (err) { t.failed.add(c.tag); return { err: String(err), tag: c.tag }; }
  if (military) {   // same bookkeeping as kiPlan: armies with a task stay where they are
    for (const a of armiesOf(S, pi)) if (before.get(a.id) !== a.r + ',' + a.c) t.memo.committed.add(a.id);
  }
  // like kiPlan, at most MAX_TURN_STEPS actions per turn – then the turn ends by itself
  if (++t.steps >= MAX_TURN_STEPS && !S.over) return Object.assign(finish(), { tag: c.tag, forcedEnd: true });
  return { tag: c.tag };
}

function endAgentTurn(g) {
  const S = g.S, pi = S.cur, t = turnCtx(g);
  if (!S.over) E.kiPositionArmies(S, pi, t.K, t.memo);
  if (!S.over) E.kiUsePicks(S, pi, t.K);
  if (!S.over) E.endTurn(S);
  if (!S.over) runOthers(g);
  g.turn = null;
  return { tag: 'end' };
}

/* Observation of the agent on turn: every candidate is tried on a copy (features.js), failing
   ones are dropped, and the following step() indexes into this filtered list. With
   `grid: false` the map planes are left out (they only change after an action). */
const b64 = arr => Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength).toString('base64');
function observe(g, wantGrid = true) {
  const S = g.S, pi = S.cur;
  if (S.over || !g.agents.has(pi)) throw new Error('no agent on turn');
  const t = turnCtx(g);
  E.kiStepMemo(t.memo, S, pi);
  const all = E.kiCandidates(S, pi, t.K, t.memo, false).filter(c => !t.failed.has(c.tag));
  const ev = FEAT.evaluate(S, pi, t, all);
  t.cands = ev.keep.map(j => all[j]);
  const tags = t.cands.map(c => c.tag);
  const { cand, pos } = FEAT.encodeCands(S, pi, tags, ev);
  const out = {
    pi, round: S.round, tags: tags.concat(['end']), teacher: ev.teacher,
    glob: b64(FEAT.glob(S, pi)), cand: b64(cand), pos: b64(pos),
  };
  if (wantGrid) { const gr = FEAT.grid(S, pi); out.grid = b64(gr); out.gshape = [gr.h, gr.w]; }
  return out;
}

/* Let the JS KI play the whole current turn of an agent seat (baseline). */
function kiTurnFull(g) {
  const S = g.S, pi = S.cur;
  E.kiTurn(S, pi);
  if (!S.over) E.endTurn(S);
  if (!S.over) runOthers(g);
  g.turn = null;
}

/* ------------------------------------------------------------------ online play
   A game loaded from the server's state (`load`). Candidates run on the local copy as usual,
   while every engine action they call on that copy is recorded and translated into the
   server's `action { type, params }` (engine-adapter.js applyAction). Calls on the KI's
   evaluation copies are not recorded (they are other objects), nested engine calls neither.
   The client sends the recorded actions and loads the server's next state. */
const SERVER_ACTION = {
  doResearch: (S, pi, tech) => ['research', { tech }],
  useFreeTech: (S, pi, tech) => ['freeTech', { tech }],
  useFreePick: (S, pi, tech) => ['freePick', { tech }],
  useBackPick: (S, pi, tech) => ['backPick', { tech }],
  copyTech: (S, pi, tech, mode) => ['copyTech', { tech, mode }],
  foundCity: (S, pi, r, c) => ['foundCity', { r, c }],
  growCity: (S, pi, city, mode) => ['growCity', { cityId: city.id, mode }],
  sacrifice: (S, pi, city) => ['sacrifice', { cityId: city.id }],
  buildArmy: (S, pi, city) => ['buildArmy', { cityId: city.id }],
  moveArmy: (S, army, r, c) => ['moveArmy', { armyId: army.id, r, c }],
  buyPower: (S, pi, n) => ['buyPower', { n: n || 1 }],
  buildRoad: (S, pi, r, c, target) => ['buildRoad', { r, c, target }],
  buyTile: (S, pi, r, c) => ['buyTile', { r, c }],
  coverPop: (S, pi, kind, amount) => ['coverPop', { kind, amount }],
  uncoverPop: (S, pi, kind, amount) => ['uncoverPop', { kind, amount }],
  buildWonder: (S, pi, city, wonder) => ['buildWonder', { cityId: city.id, wonder }],
  nuke: (S, pi, r, c) => ['nuke', { r, c }],
};
const rec = { S: null, list: null, depth: 0 };
for (const name of Object.keys(SERVER_ACTION)) {
  const orig = sandbox[name];
  if (typeof orig !== 'function') continue;
  sandbox[name] = function (...args) {
    const top = rec.list && rec.depth === 0 && args[0] === rec.S;
    rec.depth++;
    let res;
    try { res = orig.apply(this, args); } finally { rec.depth--; }
    if (top && !res) rec.list.push(SERVER_ACTION[name](...args));
    return res;
  };
}
function recording(g, fn) {
  rec.S = g.S; rec.list = []; rec.depth = 0;
  try { const r = fn(); return Object.assign(r, { actions: rec.list }); } finally { rec.S = null; rec.list = null; }
}

/* The server's state for the agent at state index `pi`. Old turn context is kept while the
   same turn goes on (failed candidates, committed armies). */
function load(a) {
  const S = a.state;
  const pi = a.pi;
  S.players[pi].kiLevel = a.kiLevel || 'schwer';
  const gid = a.gid ?? (games.size ? Math.max(...games.keys()) + 1 : 1);
  const old = games.get(gid);
  const g = { S, agents: new Set([pi]), turn: null, online: true };
  if (old && old.turn && old.turn.pi === S.cur && old.turn.round === S.round) {
    g.turn = old.turn;
    g.turn.K = E.kiContext(S, S.cur);   // armies and resources changed
  }
  games.set(gid, g);
  return Object.assign({ gid }, status(g));
}

/* Online "end": position armies and use picks (recorded), then the server's endTurn. */
function endOnline(g) {
  const S = g.S, pi = S.cur, t = turnCtx(g);
  if (!S.over) E.kiPositionArmies(S, pi, t.K, t.memo);
  if (!S.over) E.kiUsePicks(S, pi, t.K);
  rec.list.push(['endTurn', {}]);
  g.turn = null;
  return { tag: 'end' };
}

/* Start tile of the agent: the JS KI's choice (kiPlaceSeat) on the server's plan.
   Returns { o, cell } for placement:action / async:place. */
function place(a) {
  const plan = a.plan;
  const seat = plan.seats.find(s => s.idx === a.seatIdx);
  if (!seat) throw new Error('no seat ' + a.seatIdx);
  const pl = a.players[a.seatIdx];
  let err = E.kiPlaceSeat(plan, seat, { civ: pl.civ, ability: pl.ability || 'basis' }, a.setup || null);
  if (err) err = E.botPlaceSeat(plan, seat, E.mapRng((a.seed || 1) + 12345));
  if (err || seat.cell == null) throw new Error('no placement: ' + err);
  return { o: seat.o, cell: seat.cell };
}

/* A server action applied like engine-adapter.js applyAction – for testing the online path
   without a server (tests/test_online_offline.py). After endTurn the other seats move. */
function serverAction(g, type, params) {
  const S = g.S, pi = S.cur;
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
    case 'coverPop': return E.coverPop(S, pi, params.kind, params.amount);
    case 'uncoverPop': return E.uncoverPop(S, pi, params.kind, params.amount);
    case 'buildWonder': return E.buildWonder(S, pi, need(city()), params.wonder);
    case 'nuke': return E.nuke(S, pi, params.r, params.c);
    case 'endTurn': {
      const stop = E.blockingIssues(S, pi);
      if (stop && stop.length) return stop[0];
      E.endTurn(S);
      runOthers(g);
      return null;
    }
    default: return 'unknown action ' + type;
  }
}

/* ------------------------------------------------------------------ protocol */
const handlers = {
  server_action: a => { const g = game(a); return Object.assign({ err: serverAction(g, a.type, a.params || {}) || null }, status(g)); },
  load: a => load(a),
  place: a => place(a),
  // online step: like step, but returns the server actions instead of continuing the game
  step_online: a => {
    const g = game(a);
    return recording(g, () => step(g, a.i));
  },
  rollSetup: a => ({ setup: E.rollSetup(a.cfg) }),
  info: () => ({ version: VERSION, engineDir: ENGINE_DIR, civs: CIVS.map(c => c.k), dims: FEAT.dims }),
  observe: a => observe(game(a), a.grid !== false),
  reset: a => reset(a),
  candidates: a => ({ cands: candidates(game(a)) }),
  step: a => Object.assign(step(game(a), a.i), status(game(a))),
  ki_turn: a => { kiTurnFull(game(a)); return status(game(a)); },
  state: a => ({ state: game(a).S }),
  status: a => status(game(a)),
  close: a => { games.delete(a.gid); return {}; },
};
function game(a) {
  const g = games.get(a.gid);
  if (!g) throw new Error('unknown game ' + a.gid);
  return g;
}

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
