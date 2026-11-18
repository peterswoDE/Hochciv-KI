/* Scripted "rusher" opponent, run inside the engine sandbox (bridge/worker.js loads it after the
   engine files, so every engine and KI function is in scope).

   Modelled on the human and bot players that beat our network in prod (sealgo, SeymourBot,
   November 2026 replays): research slavery first, grow every city and sacrifice the new
   population for SLAVERY_COINS each round, buy cheap military techs, then from round 2 on
   march armies next to the nearest enemy capital and buy enough power to win the two-turn
   siege. The JS KI never sacrifices and never strikes this early, and the network was trained
   against it and anchored to it, so it never learned to defend against this plan.

   rushTurn(S, pi) plays a whole turn like kiTurn: the rush part first (research, grow and
   sacrifice, strike or stage), then the KI planner with the rush armies committed (it founds
   cities, researches, defends) and the KI army positioning for the rest. */
'use strict';

// Techs the rusher buys first, in this order, whenever available and affordable.
const RUSH_TECHS = ['sklaverei', 'eisenverarbeitung', 'demokratie', 'belagerung', 'taktik',
  'nationalismus', 'gewehre', 'rittertum', 'militaergericht', 'dynamit'];
const RUSH_FROM_ROUND = 2;        // first round in which armies march
const RUSH_STAGE_ARMIES = 3;      // armies kept marching toward a target that is still out of reach
const RUSH_POWER_SHARE = 0.7;     // share of the coins left after a strike spent on extra power

// Paid from science only (pay() would take the rest from the coins, which buy armies) – except
// slavery, which pays for itself in the same turn.
function rushResearch(S, pi) {
  if (evActive(S, pi, 'dunkles_zeitalter')) return;
  const p = S.players[pi];
  for (let guard = 0; guard < 12; guard++) {
    const open = researchable(S, pi);
    const t = RUSH_TECHS.map(k => open.find(x => x.k === k)).find(x => x &&
      techCost(S, pi, x) <= (x.k === 'sklaverei' ? available(S, pi, 'sci') : p.res.sci));
    if (!t || doResearch(S, pi, t.k)) return;
  }
}

// Found cities first, on the KI's best sites: every city is one more sacrifice per round.
function rushFound(S, pi, K, memo) {
  for (let guard = 0; guard < 4; guard++) {
    const s = kiSites(S, pi, K, memo)[0];
    if (!s || foundCity(S, pi, s.r, s.c)) return;
  }
}

// With slavery: grow every city that can (paid growth costs one food and one coin per
// population), then sacrifice one population in every city that can, +SLAVERY_COINS each.
// Without slavery the KI planner decides on growth.
function rushGrowAndSacrifice(S, pi) {
  if (!slaveryUsable(S.players[pi])) return;
  for (const city of citiesOf(S, pi)) {
    if (freeGrowthAvailable(S, pi, city) && !growBlockReason(S, pi, city)) growCity(S, pi, city, 'free');
    if (!canGrowPaid(S, pi, city)) growCity(S, pi, city, 'paid');
  }
  for (const city of citiesOf(S, pi))
    if (!sacrificeError(S, pi, city)) sacrifice(S, pi, city);
}

// The rush target: the enemy capital nearest to the own capital (not the barbarians).
function rushTarget(S, pi) {
  const home = capitalOf(S, pi) || citiesOf(S, pi)[0];
  if (!home) return null;
  let best = null;
  for (const c of S.cities) {
    if (c.owner === pi || !c.cap) continue;
    const o = S.players[c.owner];
    if (o.dead || o.kind === 'barbar') continue;
    const d = hexDistance(home.r, home.c, c.r, c.c);
    if (!best || d < best.d) best = { c, d };
  }
  return best && best.c;
}

// Strike: of the KI's strike variants against the target, the one with the most armies.
// Returns true if one was run.
function rushStrike(S, pi, target, K, memo) {
  kiStepMemo(memo, S, pi);
  const plan = kiStrikePlan(S, pi, target, K, memo);
  if (!plan) return false;
  const n = v => +v.tag.split(':')[2];
  const pick = plan.reduce((a, b) => n(b) > n(a) ? b : a);
  return !pick.run(S);
}

// Out of reach: march the armies toward the target and keep up to RUSH_STAGE_ARMIES of them.
function rushStage(S, pi, target) {
  const near = (r, c) => hexDistance(r, c, target.r, target.c);
  const own = armiesOf(S, pi);
  if (own.length < RUSH_STAGE_ARMIES) {
    const cities = citiesOf(S, pi).filter(c => !armyAt(S, c.r, c.c))
      .sort((a, b) => near(a.r, a.c) - near(b.r, b.c));
    if (cities.length && available(S, pi, 'coins', payOpts(S, pi)) >= armyCost(S, pi))
      buildArmy(S, pi, cities[0]);
  }
  for (const a of armiesOf(S, pi)) {
    if (a.mp <= 0) continue;
    const tiles = kiArmyTiles(S, a).filter(t => !armyAt(S, t[0], t[1]) || (t[0] === a.r && t[1] === a.c));
    if (!tiles.length) continue;
    const best = tiles.reduce((x, y) => near(y[0], y[1]) < near(x[0], x[1]) ? y : x);
    if (best[0] !== a.r || best[1] !== a.c) moveArmy(S, a, best[0], best[1]);
  }
}

// With armies in range of the target, put most of the remaining coins into power: the
// defender buys power and armies in its own turn, and the siege needs a second winning turn.
function rushTopUp(S, pi, target) {
  if (!attackersOn(S, pi, target).length) return;
  const price = powerPrice(S, pi);
  const n = Math.floor(available(S, pi, 'coins', payOpts(S, pi)) * RUSH_POWER_SHARE / price);
  if (n > 0) buyPower(S, pi, n);
}

function rushTurn(S, pi) {
  const p = S.players[pi];
  kiSeedRng(S, pi);
  if (p.dead || S.over) return;
  rushResearch(S, pi);
  const K = kiContext(S, pi);
  const memo = kiStepMemo({}, S, pi);
  rushFound(S, pi, K, memo);
  rushGrowAndSacrifice(S, pi);
  const target = rushTarget(S, pi);
  if (target && S.round >= RUSH_FROM_ROUND && !K.emergency) {
    if (!rushStrike(S, pi, target, K, memo)) rushStage(S, pi, target);
    const live = S.cities.includes(target) && target.owner !== pi;
    if (live) rushTopUp(S, pi, target);
    // armies at the front stay there: the KI planner and positioning leave them alone
    for (const a of armiesOf(S, pi))
      if (!cityAt(S, a.r, a.c) && hexDistance(a.r, a.c, target.r, target.c) <= attackRange(S, pi) + 3)
        memo.committed.add(a.id);
  }
  if (S.over) return;
  // the rest like kiTurn: plan (found, research, defend), position, use free picks
  kiPlan(S, pi, kiContext(S, pi), null, memo);
  if (!S.over) kiPositionArmies(S, pi, kiContext(S, pi), memo);
  kiUsePicks(S, pi, kiContext(S, pi));
}
