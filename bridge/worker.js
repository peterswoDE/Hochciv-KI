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

/* ------------------------------------------------------------------ games */
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
    if (s.kind === 'ki') p.kiLevel = s.kiLevel || 'schwer';
    return p;
  });
}

/* map: 'plaettchen' (tile map with placement, also the lobby's "random"), 'original', 'gross',
   or a custom map object as stored by the map editor ({ name, rows, spawns, ... }). Agent seats
   place their start tile with the JS KI (kiPlaceSeat) in v1. */
function reset(a) {
  const seed = (a.seed ?? Math.floor(Math.random() * 2 ** 31)) | 0;
  const map = a.map || 'plaettchen';
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
    g.turn = { pi, round: S.round, K: E.kiContext(S, pi), memo: {}, failed: new Set(), cands: null };
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
  if (i === t.cands.length) return endAgentTurn(g);
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

/* Let the JS KI play the whole current turn of an agent seat (baseline; imitation comes in M1). */
function kiTurnFull(g) {
  const S = g.S, pi = S.cur;
  E.kiTurn(S, pi);
  if (!S.over) E.endTurn(S);
  if (!S.over) runOthers(g);
  g.turn = null;
}

/* ------------------------------------------------------------------ protocol */
const handlers = {
  info: () => ({ version: VERSION, engineDir: ENGINE_DIR, civs: CIVS.map(c => c.k) }),
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
