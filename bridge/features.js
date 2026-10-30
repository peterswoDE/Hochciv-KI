/* Observation features for the network, computed in the worker (the state never leaves Node).

   observe(g) evaluates every candidate of the current agent turn on a copy of the state, the
   way the JS KI's planner does (kiPlan): run the candidate, play the combat phase, score with
   kiValue. Candidates that fail on the copy are dropped, so the network only sees legal
   actions. The same evaluation gives the teacher label: the candidate the KI planner would
   take next (one step of kiPlan at level "schwer"), or "end" when nothing is worth it.

   Layout (all float32, sent base64-encoded):
     grid   [GRID_C, h, w]            map planes as big as the map (custom maps up to 48 × 48),
                                      seats relative to the observer
     glob   [GLOB_N]                  game and per-seat vectors
     cand   [n, CAND_F]               one row per candidate, the last row is "end"
     pos    [n, 2] int32              target cell of a candidate (r, c) or (-1, -1)
   Seats: slot 0 is the observer, then the others in turn order; with more than NSEAT empires
   the rest share the last slot.                                                            */
'use strict';

const NSEAT = 4;
const TERRAINS = ['G', 'W', 'B', 'F', 'M', 'I', 'V', 'X'];
const CAND_TYPES = ['tech', 'free', 'back', 'pick', 'copyf', 'copyp', 'gfree', 'grow', 'found',
  'road1', 'road2', 'slave', 'col', 'wonder', 'power', 'strike', 'defend', 'flank', 'end'];

function makeFeatures(E, konst) {
  const fn = name => (typeof E[name] === 'function' ? E[name] : konst(name));
  const citiesOf = fn('citiesOf'), armiesOf = fn('armiesOf'), popOf = fn('popOf'),
    worldPop = fn('worldPop'), roadLevel = fn('roadLevel'),
    capitalsTaken = fn('capitalsTaken');
  const TECHS = konst('TECHS');
  const TECH_KEYS = Array.isArray(TECHS) ? TECHS.map(t => t.k) : Object.keys(TECHS);
  const TECH_IDX = Object.fromEntries(TECH_KEYS.map((k, i) => [k, i]));
  const NT = TECH_KEYS.length;
  const ABILS = [];
  for (const c of konst('CIVS')) for (const a of (c.abilities || [])) ABILS.push(c.k + ':' + a.k);
  const CIV_KEYS = konst('CIVS').map(c => c.k);
  const KI_W = konst('KI_W'), KI_RESEARCH_TAG = konst('KI_RESEARCH_TAG');

  // ---------------------------------------------------------------- grid
  const CH = {
    terrain: 0,                       // 8 one-hot
    yield: 8,                         // 3: sci, food, coins for the observer
    ctrl: 11,                         // NSEAT: controlled by seat
    city: 15,                         // NSEAT: city pop / 5
    cap: 19,                          // capital
    army: 20,                         // NSEAT: armies on the cell
    mp: 24,                           // observer's army: movement left / 4
    road: 25,                         // road level / 2
    bought: 26,                       // NSEAT: colony bought by seat
    siege: 30,                        // siege counter on a city / 2
    valid: 31,                        // inside the map
  };
  const GRID_C = 32;

  const slotOf = (S, me) => {
    const n = S.players.length, m = new Array(n);
    for (let k = 0; k < n; k++) m[(me + k) % n] = Math.min(k, NSEAT - 1);
    return m;
  };

  function grid(S, me) {
    const GRID_H = S.map.rows.length, GRID_W = Math.max(...S.map.rows.map(r => r.length));
    const out = new Float32Array(GRID_C * GRID_H * GRID_W);
    out.h = GRID_H; out.w = GRID_W;
    const at = (ch, r, c) => (ch * GRID_H + r) * GRID_W + c;
    const slot = slotOf(S, me);
    const rows = S.map.rows;
    const pre = { y: {}, cap: fn('capitalOf')(S, me) };
    for (let r = 0; r < Math.min(rows.length, GRID_H); r++) {
      for (let c = 0; c < Math.min(rows[r].length, GRID_W); c++) {
        const t = rows[r][c];
        const ti = TERRAINS.indexOf(t);
        if (ti >= 0) out[at(CH.terrain + ti, r, c)] = 1;
        if (t === 'X') continue;
        out[at(CH.valid, r, c)] = 1;
        const y = E.tileYieldAt(S, me, r, c, pre);
        for (let k = 0; k < 3; k++) out[at(CH.yield + k, r, c)] = y[k] / 3;
        out[at(CH.road, r, c)] = roadLevel(S, r, c) / 2;
      }
    }
    const put = (ch, r, c, v) => { if (r >= 0 && r < GRID_H && c >= 0 && c < GRID_W) out[at(ch, r, c)] += v; };
    S.players.forEach((p, pi) => {
      if (p.dead) return;
      for (const k of E.controlledTiles(S, pi)) { const [r, c] = k.split(',').map(Number); put(CH.ctrl + slot[pi], r, c, 1); }
      for (const k of (S.bought[pi] || [])) { const [r, c] = k.split(',').map(Number); put(CH.bought + slot[pi], r, c, 1); }
    });
    for (const ct of S.cities) {
      put(CH.city + slot[ct.owner], ct.r, ct.c, ct.pop / 5);
      if (ct.cap) put(CH.cap, ct.r, ct.c, 1);
    }
    for (const a of S.armies) {
      put(CH.army + slot[a.owner], a.r, a.c, 1);
      if (a.owner === me) put(CH.mp, a.r, a.c, (a.mp || 0) / 4);
    }
    for (const sk in S.sieges) {
      const [, cid] = sk.split('|');
      const ct = S.cities.find(x => x.id === +cid);
      if (ct) put(CH.siege, ct.r, ct.c, S.sieges[sk] / 2);
    }
    return out;
  }

  // ---------------------------------------------------------------- global
  const SEAT_F = 22 + CIV_KEYS.length + ABILS.length + 2 * NT;
  const GAME_F = 8;
  const GLOB_N = GAME_F + NSEAT * SEAT_F;

  function seatVec(S, pi, me, out, o) {
    const p = S.players[pi];
    const n = S.players.length;
    const inc = E.income(S, pi);
    const w = worldPop(S) || 1;
    const opt = E.victoryOption(S, p);
    let i = o;
    out[i++] = 1;                                     // present
    out[i++] = p.dead ? 1 : 0;
    out[i++] = S.cur === pi ? 1 : 0;
    out[i++] = ((pi - (S.startIdx || 0) + n) % n) / n; // position in the round
    out[i++] = p.res.sci / 20; out[i++] = p.res.food / 20; out[i++] = p.res.coins / 20;
    out[i++] = inc.sci / 20; out[i++] = inc.food / 20; out[i++] = inc.coins / 20;
    out[i++] = (p.power || 0) / 20;
    out[i++] = popOf(S, pi) / 20;
    out[i++] = popOf(S, pi) / (w * opt.frac);          // economic victory progress
    out[i++] = citiesOf(S, pi).length / 10;
    out[i++] = armiesOf(S, pi).length / 10;
    out[i++] = Object.keys(p.techs).length / 30;
    out[i++] = capitalsTaken(p);
    out[i++] = (p.freeArmies || 0) / 3;
    out[i++] = (p.backPicks || []).length / 3;
    out[i++] = (S.wonders || []).filter(x => x.owner === pi).length / 3;
    out[i++] = E.powerOf ? E.powerOf(S, pi) / 20 : 0;
    out[i++] = (S.claims || []).some(c => c.pi === pi) ? 1 : 0;
    const ci = CIV_KEYS.indexOf(p.civ); if (ci >= 0) out[i + ci] = 1; i += CIV_KEYS.length;
    const ai = ABILS.indexOf(p.civ + ':' + p.ability); if (ai >= 0) out[i + ai] = 1; i += ABILS.length;
    for (const k in p.techs) if (k in TECH_IDX) out[i + TECH_IDX[k]] = 1; i += NT;
    for (const k in p.avail) if (k in TECH_IDX && p.avail[k]) out[i + TECH_IDX[k]] = 1; i += NT;
  }

  function glob(S, me) {
    const out = new Float32Array(GLOB_N);
    out[0] = S.round / 15;
    out[1] = S.players.length / 8;
    out[2] = S.duel ? 1 : 0;
    out[3] = S.marathon ? 1 : 0;
    out[4] = S.wo ? 1 : 0;
    out[5] = S.ev ? 1 : 0;
    out[6] = (S.claims || []).length ? 1 : 0;
    out[7] = worldPop(S) / 40;
    // v1 plays up to NSEAT empires; with more, the seats after the first NSEAT are left out
    const n = S.players.length;
    for (let k = 0; k < Math.min(n, NSEAT); k++) seatVec(S, (me + k) % n, me, out, GAME_F + k * SEAT_F);
    return out;
  }

  // ---------------------------------------------------------------- candidates
  const NUM_F = 24;
  const CAND_F = CAND_TYPES.length + NT + NUM_F;

  const typeOf = tag => {
    const t = tag.split(':')[0];
    return t === 'road' ? 'road1' : t;
  };
  // target cell of a candidate, for gathering map features
  function targetOf(S, tag) {
    const parts = tag.split(':');
    const t = parts[0];
    if (t === 'found' || t === 'col') { const [r, c] = parts[1].split(',').map(Number); return [r, c]; }
    if (/^(grow|gfree|slave|strike|defend|road1|road2)$/.test(t)) {
      const ct = S.cities.find(x => x.id === +parts[1]);
      if (ct) return [ct.r, ct.c];
    }
    if (t === 'flank') {
      const a = S.armies.find(x => x.id === +parts[1]);
      if (a) return [a.r, a.c];
    }
    return [-1, -1];
  }

  function summary(S, pi) {
    const p = S.players[pi];
    let ePop = 0, eCities = 0, eArmies = 0;
    S.players.forEach((q, j) => {
      if (j === pi) return;
      ePop += popOf(S, j); eCities += citiesOf(S, j).length; eArmies += armiesOf(S, j).length;
    });
    const inc = E.income(S, pi);
    return {
      res: [p.res.sci, p.res.food, p.res.coins], inc: [inc.sci, inc.food, inc.coins],
      pop: popOf(S, pi), cities: citiesOf(S, pi).length, armies: armiesOf(S, pi).length,
      power: p.power || 0, techs: Object.keys(p.techs).length, ePop, eCities, eArmies,
    };
  }

  function candRow(out, row, tag, ev, base, b0) {
    const o = row * CAND_F;
    const ty = CAND_TYPES.indexOf(typeOf(tag));
    if (ty >= 0) out[o + ty] = 1;
    const parts = tag.split(':');
    if (parts[1] && parts[1] in TECH_IDX) out[o + CAND_TYPES.length + TECH_IDX[parts[1]]] = 1;
    let i = o + CAND_TYPES.length + NT;
    const num = parts.slice(1).map(Number).filter(x => Number.isFinite(x));
    out[i++] = parts[0] === 'power' ? num[0] / 10 : 0;
    out[i++] = parts[0] === 'strike' ? (num[1] || 0) / 5 : 0;
    out[i++] = parts[0] === 'defend' ? (num[1] || 0) / 5 : 0;
    if (!ev) return;                                   // "end": no change
    out[i++] = ev.dv / 20;
    out[i++] = ev.spent / 20;
    out[i++] = ev.ratio;
    out[i++] = ev.dv > 0.05 ? 1 : 0;
    const a = ev.after;
    for (let k = 0; k < 3; k++) out[i++] = (a.res[k] - b0.res[k]) / 10;
    for (let k = 0; k < 3; k++) out[i++] = (a.inc[k] - b0.inc[k]) / 5;
    out[i++] = (a.pop - b0.pop) / 3;
    out[i++] = (a.cities - b0.cities);
    out[i++] = (a.armies - b0.armies);
    out[i++] = (a.power - b0.power) / 10;
    out[i++] = (a.techs - b0.techs);
    out[i++] = (a.ePop - b0.ePop) / 3;
    out[i++] = (a.eCities - b0.eCities);
    out[i++] = (a.eArmies - b0.eArmies);
    out[i++] = ev.win ? 1 : 0;
    out[i++] = ev.lose ? 1 : 0;
    out[i++] = (ev.v - base) / 50;
    // stored as float16 on disk: keep every value in its range
    for (let k = o; k < o + CAND_F; k++) out[k] = Math.max(-30000, Math.min(30000, out[k]));
  }

  /* Evaluate the candidate list `cands` (kiCandidates entries) for player pi.
     Returns { keep: indices into cands that ran without error, evs, base, teacher } where
     teacher is an index into keep, or keep.length for "end". */
  function evaluate(S, pi, t, cands) {
    const p = S.players[pi];
    const seedOf = () => Math.floor(E.kiRandom(p) * 2147483647);
    const base = E.kiScore(E.kiClone(S, seedOf()), pi, t.K);
    const b0 = summary(S, pi);
    const keep = [], evs = [];
    for (let j = 0; j < cands.length; j++) {
      const c = cands[j];
      const X = E.kiClone(S, seedOf());
      let err;
      try { err = c.run(X); } catch (e) { err = String(e && e.message || e); }
      if (err) { t.failed.add(c.tag); continue; }
      const v = E.kiScore(X, pi, t.K);
      const dv = v - base;
      const spent = E.kiSpent(p.res, X.players[pi].res,
        t.K.emergency || KI_RESEARCH_TAG.test(c.tag) ? 1 : KI_W.sciReserve);
      const winners = X.over ? (X.over.winners || [X.over.winner]) : [];
      keep.push(j);
      evs.push({ dv, v, spent, ratio: dv / (Math.max(0, spent) + 2), after: summary(X, pi),
        win: winners.includes(pi), lose: !!X.over && !winners.includes(pi), c });
    }
    // teacher: one planning step of kiPlan at level schwer
    const scored = evs.map((ev, k) => ({ ev, k, c: ev.c, dv: ev.dv, ratio: ev.ratio })).filter(x => x.dv > 0.05);
    let teacher = keep.length;
    if (scored.length) {
      scored.sort((a, b) => b.ratio - a.ratio);
      let pick = scored[0];
      const top = scored.reduce((a, b) => b.dv > a.dv ? b : a);
      if (top !== pick && top.dv > pick.dv * 2.5 && top.ratio > pick.ratio * 0.45) pick = top;
      const methode = E.kiMethodeZuerst(S, pi, scored, pick);
      if (methode) pick = methode;
      teacher = pick.k;
    }
    return { keep, evs, base, b0, teacher };
  }

  function encodeCands(S, pi, tags, ev) {
    const n = tags.length + 1;
    const cand = new Float32Array(n * CAND_F);
    const pos = new Int32Array(n * 2);
    tags.forEach((tag, k) => {
      candRow(cand, k, tag, ev.evs[k], ev.base, ev.b0);
      const [r, c] = targetOf(S, tag);
      pos[2 * k] = r; pos[2 * k + 1] = c;
    });
    candRow(cand, tags.length, 'end', null, ev.base, ev.b0);
    pos[2 * tags.length] = -1; pos[2 * tags.length + 1] = -1;
    return { cand, pos };
  }

  return {
    grid, glob, evaluate, encodeCands,
    dims: { GRID_C, GLOB_N, CAND_F, NSEAT, NT, CAND_TYPES, TECH_KEYS },
  };
}

module.exports = { makeFeatures, NSEAT, CAND_TYPES };
