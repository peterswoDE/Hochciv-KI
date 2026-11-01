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
   Seats: slot 0 is the observer, then the other empires in turn order (up to MAX_PLAYERS = 8).
   The barbarians (events) are no seat: they have their own map planes.
   Every input has a name (dims.names); hochciv_ki/widen.py carries trained weights over to a
   new layout by these names, new inputs start at zero weight.                              */
'use strict';

const NSEAT = 8;
const TERRAINS = ['G', 'W', 'B', 'F', 'M', 'I', 'V', 'X'];
const CAND_TYPES = ['tech', 'free', 'back', 'pick', 'copyf', 'copyp', 'gfree', 'grow', 'found',
  'road1', 'road2', 'slave', 'col', 'wonder', 'power', 'strike', 'defend', 'flank', 'end'];

function makeFeatures(E, konst) {
  const fn = name => (typeof E[name] === 'function' ? E[name] : konst(name));
  const citiesOf = fn('citiesOf'), armiesOf = fn('armiesOf'), popOf = fn('popOf'),
    worldPop = fn('worldPop'), roadLevel = fn('roadLevel'),
    capitalsTaken = fn('capitalsTaken'), abilitiesOf = fn('abilitiesOf'),
    wonderCost = fn('wonderCost'), hasWonder = fn('hasWonder');
  const empireCount = S => S.players.filter(p => p.kind !== 'barbar').length;   // barbarians come last
  const TECHS = konst('TECHS');
  const TECH_KEYS = Array.isArray(TECHS) ? TECHS.map(t => t.k) : Object.keys(TECHS);
  const TECH_IDX = Object.fromEntries(TECH_KEYS.map((k, i) => [k, i]));
  const NT = TECH_KEYS.length;
  const ABILS = [];
  for (const c of konst('CIVS')) for (const a of (c.abilities || [])) ABILS.push(c.k + ':' + a.k);
  const CIV_KEYS = konst('CIVS').map(c => c.k);
  const KI_W = konst('KI_W'), KI_RESEARCH_TAG = konst('KI_RESEARCH_TAG');
  const WONDER_KEYS = konst('WONDERS').map(w => w.k);
  const WONDER_IDX = Object.fromEntries(WONDER_KEYS.map((k, i) => [k, i]));
  const NW = WONDER_KEYS.length;
  const EVENT_KEYS = konst('EVENT_ROWS').flat().map(e => e.k);
  const NE = EVENT_KEYS.length;

  // ---------------------------------------------------------------- grid
  // [name, planes, integer scale (a plane stores value / scale, the model keeps uint8)]
  const GRID_LAYOUT = [
    ['terrain', 8, 1],                // one-hot
    ['yield', 3, 3],                  // sci, food, coins for the observer
    ['ctrl', NSEAT, 1],               // controlled by seat
    ['city', NSEAT, 5],               // city pop / 5
    ['cap', 1, 1],                    // capital
    ['army', NSEAT, 1],               // armies on the cell
    ['mp', 1, 4],                     // observer's army: movement left / 4
    ['road', 1, 2],                   // road level / 2
    ['bought', NSEAT, 1],             // colony bought by seat
    ['siege', 1, 2],                  // siege counter on a city / 2
    ['barbCity', 1, 5],               // barbarian city pop / 5
    ['barbArmy', 1, 1],               // barbarian armies
    ['wonder', 1, 3],                 // wonders in the city / 3
    ['valid', 1, 1],                  // inside the map – must stay the last plane
  ];
  const CH = {}, GRID_NAMES = [], GRID_SCALE = [];
  for (const [k, n, sc] of GRID_LAYOUT) {
    CH[k] = GRID_NAMES.length;
    for (let j = 0; j < n; j++) { GRID_NAMES.push(n > 1 ? k + j : k); GRID_SCALE.push(sc); }
  }
  const GRID_C = GRID_NAMES.length;

  // empires in turn order relative to the observer; the barbarians get -1
  const slotOf = (S, me) => {
    const n = empireCount(S), m = new Array(S.players.length).fill(-1);
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
      if (p.dead || slot[pi] < 0) return;
      for (const k of E.controlledTiles(S, pi)) { const [r, c] = k.split(',').map(Number); put(CH.ctrl + slot[pi], r, c, 1); }
      for (const k of (S.bought[pi] || [])) { const [r, c] = k.split(',').map(Number); put(CH.bought + slot[pi], r, c, 1); }
    });
    for (const ct of S.cities) {
      put(slot[ct.owner] < 0 ? CH.barbCity : CH.city + slot[ct.owner], ct.r, ct.c, ct.pop / 5);
      if (ct.cap) put(CH.cap, ct.r, ct.c, 1);
    }
    for (const w of (S.wonders || [])) {
      const ct = S.cities.find(x => x.id === w.cityId);
      if (ct) put(CH.wonder, ct.r, ct.c, 1 / 3);
    }
    for (const a of S.armies) {
      put(slot[a.owner] < 0 ? CH.barbArmy : CH.army + slot[a.owner], a.r, a.c, 1);
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
  const SEAT_NAMES = ['present', 'dead', 'cur', 'turnPos', 'resSci', 'resFood', 'resCoins', 'incSci',
    'incFood', 'incCoins', 'power', 'pop', 'econProgress', 'cities', 'armies', 'techs', 'capitalsTaken',
    'freeArmies', 'backPicks', 'wonders', 'powerOf', 'claim',
    ...CIV_KEYS.map(k => 'civ:' + k), ...ABILS.map(k => 'abil:' + k),
    ...TECH_KEYS.map(k => 'tech:' + k), ...TECH_KEYS.map(k => 'avail:' + k),
    ...WONDER_KEYS.map(k => 'wonderOwn:' + k), 'wonderCost', 'cultureWin'];
  const SEAT_F = SEAT_NAMES.length;
  const GAME_NAMES = ['round', 'players', 'duel', 'marathon', 'wonders', 'events', 'claims', 'worldPop',
    'evEasy', 'nukeBan', 'barbarians', ...EVENT_KEYS.map(k => 'event:' + k),
    ...EVENT_KEYS.map(k => 'eventNext:' + k), ...WONDER_KEYS.map(k => 'wonderPool:' + k),
    ...WONDER_KEYS.map(k => 'wonderBuilt:' + k)];
  const GAME_F = GAME_NAMES.length;
  const GLOB_N = GAME_F + NSEAT * SEAT_F;
  const GLOB_NAMES = GAME_NAMES.concat(...Array.from({ length: NSEAT }, (_, k) => SEAT_NAMES.map(x => 'seat' + k + ':' + x)));

  function seatVec(S, pi, me, out, o) {
    const p = S.players[pi];
    const n = empireCount(S);
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
    // one ability, or the two drafted ones in the marathon
    for (const a of abilitiesOf(p)) { const ai = ABILS.indexOf(a); if (ai >= 0) out[i + ai] = 1; }
    i += ABILS.length;
    for (const k in p.techs) if (k in TECH_IDX) out[i + TECH_IDX[k]] = 1; i += NT;
    for (const k in p.avail) if (k in TECH_IDX && p.avail[k]) out[i + TECH_IDX[k]] = 1; i += NT;
    for (const x of (S.wonders || [])) if (x.owner === pi && x.k in WONDER_IDX) out[i + WONDER_IDX[x.k]] = 1;
    i += NW;
    out[i++] = S.wo ? wonderCost(S, pi) / 40 : 0;
    out[i++] = p.cultureWin != null ? 1 : 0;
  }

  function glob(S, me) {
    const out = new Float32Array(GLOB_N);
    const n = empireCount(S);
    out[0] = S.round / 15;
    out[1] = n / 8;
    out[2] = S.duel ? 1 : 0;
    out[3] = S.marathon ? 1 : 0;
    out[4] = S.wo ? 1 : 0;
    out[5] = S.ev ? 1 : 0;
    out[6] = (S.claims || []).length ? 1 : 0;
    out[7] = worldPop(S) / 40;
    out[8] = S.ev && S.ev.mode === 'easy' ? 1 : 0;
    out[9] = S.nukeBan ? 1 : 0;
    out[10] = S.players.some(p => p.kind === 'barbar' && !p.dead) ? 1 : 0;
    let i = 11;
    // this round's event, and the next one when the observer may see it (the oracle). Never
    // rolled here: peekNextEvent would roll the dice.
    const ek = e => EVENT_KEYS.indexOf(e.k);
    if (S.event && S.event.round === S.round && S.event.k) out[i + ek(S.event)] = 1;
    i += NE;
    if (S.evNext && S.evNext.k && hasWonder(S, me, 'orakel')) out[i + ek(S.evNext)] = 1;
    i += NE;
    if (S.wo) for (const lvl of [1, 2, 3]) for (const k of ((S.wpool && S.wpool[lvl]) || [])) if (k in WONDER_IDX) out[i + WONDER_IDX[k]] = 1;
    i += NW;
    for (const x of (S.wonders || [])) if (x.k in WONDER_IDX) out[i + WONDER_IDX[x.k]] = 1;
    for (let k = 0; k < Math.min(n, NSEAT); k++) seatVec(S, (me + k) % n, me, out, GAME_F + k * SEAT_F);
    return out;
  }

  // ---------------------------------------------------------------- candidates
  const NUM_NAMES = ['powerN', 'strikeN', 'defendN', 'dv', 'spent', 'ratio', 'gain',
    'dResSci', 'dResFood', 'dResCoins', 'dIncSci', 'dIncFood', 'dIncCoins', 'dPop', 'dCities', 'dArmies',
    'dPower', 'dTechs', 'dEPop', 'dECities', 'dEArmies', 'win', 'lose', 'dvBase'];
  const CAND_NAMES = [...CAND_TYPES.map(k => 'type:' + k), ...TECH_KEYS.map(k => 'tech:' + k), ...NUM_NAMES,
    ...WONDER_KEYS.map(k => 'wonder:' + k)];
  const CAND_F = CAND_NAMES.length;
  const NUM_O = CAND_TYPES.length + NT, WONDER_O = NUM_O + NUM_NAMES.length;

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
    if (parts[0] === 'wonder' && parts[1] in WONDER_IDX) out[o + WONDER_O + WONDER_IDX[parts[1]]] = 1;
    let i = o + NUM_O;
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
    dims: { GRID_C, GLOB_N, CAND_F, NSEAT, NT, CAND_TYPES, TECH_KEYS, GRID_SCALE,
      names: { grid: GRID_NAMES, glob: GLOB_NAMES, cand: CAND_NAMES } },
  };
}

module.exports = { makeFeatures, NSEAT, CAND_TYPES };
