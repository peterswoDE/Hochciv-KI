# Search at decision time – design

Status: stage 1 implemented (2026-11-02, `hochciv_ki/search.py`); first measurement 2026-11-03.
Stage 3 built 2026-11-05 (`hochciv_ki/expert.py`, `train_ppo.py --search-frac`), not measured yet.

Half v2 suite against KI schwer, same seeds for both (`runs/eval_v2_half.*`,
`runs/eval_search_v2_half.*`): **search (k = 4) 1.31 against plain v2 1.26** mean relative score.
Better in 6 modes, worse in 3, equal in 2; the biggest gains are 2p original map +0.20 (40 games)
and 4p marathon +0.10 (20), the biggest loss 8p custom map −0.08 (24). Leaning positive, not
proven (15–60 games per mode). It took about 6× as long as the plain network (2330 s against
about 400 s; 4p marathon alone 891 s).

Found while building: the fork is unbiased (its estimate of "end" 0.610 against the real value
at the next turn start 0.591, correlation 0.99 over 27 turn ends), and "end" usually scores far
below acting – unspent resources expire at the turn's end. So "end" is only searched when the
policy ranks it in its top k or gives it at least 10 %; that cut a 4-player game from 110 s to
26 s.

## Why

- **RL alone plateaued.** `ppo2` gave no gain over its start (`DESIGN_SHAPING.md`); run B tests
  shaping, but even if it helps, the network only ever sees the position in front of it.
- **We have the exact engine.** Every candidate can be tried on a copy of the game, which is
  how strong game AIs get past their teacher. The KI already does this one step deep with a
  hand-written score (`kiPlan` / `kiScore`).
- **Online there is time.** Turn timers are 60 s to 10 minutes or unlimited (Rundenspiele:
  hours). The network uses about 4 ms per decision and a turn has about 10 decisions, so the
  bot uses well under 1 % of even the shortest timer.
- **The value head is good.** Measured on v2 against KI schwer (55 seeds, every seat), at the
  first decision of each turn:

  | Log loss (lower is better) | 2p round 1 | 2p round 3 | 2p round 5 | 4p round 1 | 4p round 3 | 4p round 5 |
  |---|---|---|---|---|---|---|
  | constant (fair share) | 0.693 | 0.693 | 0.693 | 0.617 | 0.623 | 0.590 |
  | value head | **0.499** | **0.459** | **0.381** | **0.585** | **0.521** | **0.444** |
  | Φ (victory progress) | 0.693 | 0.659 | 0.671 | 0.617 | 0.627 | 0.526 |

  A search needs an evaluation of the positions it reaches; this one is far better than the
  hand-made Φ, already in round 1.

## Costs we work with

Measured with `profile_collect.py` (under load, so rough):

| Step | Time |
|---|---|
| `observe` (lists ~45 candidates, tries each on a copy, scores it) | ~25 ms |
| network forward, one position | ~4 ms (CPU, batch 1) |
| one KI turn (`kiTurn`, for an opponent) | ~75 ms |
| decisions per turn | ~10 |

So one extra position costs about 30 ms, and simulating the opponents to my next turn about
75 ms per opponent.

## Where the value head applies

The value head is trained on **decision points**: the agent on turn, before an action. Two
kinds of successors:

1. **An action inside my turn** (research, found, grow, move …): the next state is again my
   decision point, so `V(s')` comes straight from `observe` + forward on a copy.
2. **"end"**: the next states are the opponents' turns. The search plays them with a model of
   the opponents up to my next decision point, then takes `V`. The model is the KI schwer
   (`kiTurn`) at first: it is what the evaluation plays against, and a reasonable stand-in for
   people online. The combat phase at my turn's end and the opponents' replies are exactly
   where a one-step score misses things, so this branch matters most.

Games have randomness (`kiClone` takes a seed); the "end" branch is averaged over 2 seeds at
first.

## Stage 1: one-step look-ahead with the value head

At every decision:

1. The policy gives its probabilities over the candidates.
2. The top `k` candidates (default 4) plus "end" are each played on a copy of the game, and
   their successor gets `V` (for "end": after the simulated opponent turns).
3. The action with the best `V(s') + c · log p(a)` is taken (`c` small, as a tie-breaker that
   keeps the policy's judgement where the values are close).

Cost per decision: `k` × 30 ms + (opponents × 75 ms × 2 seeds) ≈ 0.1 s (2 players) to 1.2 s
(8 players); per turn about 1–12 s. That fits the shortest timer with margin; the search gets a
time budget (default: a quarter of the turn timer, at most 30 s per turn), and when the budget
runs out it falls back to the plain policy for the rest of the turn.

**Measured offline first**: the v2 suite against KI schwer and head to head against v2 without
search, paired seeds. A full suite takes `k`+ times longer (about 1.5–2 hours); tune `k` and
`c` on half suites.

## Stage 2: searching a whole turn

A turn is a sequence of actions that share resources: research first or grow first changes
what is possible afterwards. Stage 2 keeps a **beam** of `b` partial turns (default 3): every
step expands each partial turn by its top candidates, keeps the `b` best by `V`, and ends a
branch at "end", where it is scored after the simulated opponents. The best complete turn is
played.

Cost: about `b` × 10 decisions × `k` × 30 ms + `b` × opponents × 75 ms ≈ 4–15 s per turn. Only
worth building if stage 1 shows a gain; it reuses everything from stage 1.

## Stage 3 (later): learning from the search

The search's choices are better labels than the KI planner's. Training the policy on them
(expert iteration, as in AlphaZero) makes the network stronger, which makes the search
stronger. That needs search during training games, which multiplies the collection time by
`k` or more – only realistic after the throughput work (see below). Decided after stage 1/2.

Built 2026-11-05 (`hochciv_ki/expert.py`), ahead of stage 2: stage 1 search is enough to start,
and plain PPO stopped gaining (`ppo6` level with v3 head to head).

- **Only a share of the decisions is searched** (`--search-frac`, 25 %), the top 3 candidates,
  like playout cap randomization. The learner's seats only; KI seats stay as they are.
- **Target, not label:** `target(a) ∝ p(a) · exp((Q(a) − v) / τ)`, with `Q = v` for candidates
  not searched. These keep their prior, and one noisy value cannot force a label.
- **Moves stay sampled from the policy,** so the game is on-policy and PPO's data stays valid.
- **Own training phase:** after the PPO update, `search_phase` runs one epoch on the searched
  decisions, stopped at `--search-kl` from the policy that played (0.05, first run 0.075: where
  PPO's own brake sits at `--target-kl 0.05`). `PPO.update` is unchanged.
- **τ = 0.03:** on `ppo6` the best move changes in 9.1 % of the searched decisions (mean
  KL(target ‖ policy) 0.060, median 0.004); at τ = 0.1 in 4.4 %. Measured beside a running
  training, so the cost of 0.7× and 4.6× for the same setup says nothing; measure it alone.

## Code changes (stage 1)

| Where | Change |
|---|---|
| `bridge/worker.js` | `fork` (copy a game to a new `gid`, with its agent seats and turn context), `close` for forks; forks share nothing with the original |
| `hochciv_ki/env.py` | agents that need the game get `(bridge, gid)`: `play_game` passes it to agents with `needs_game = True` |
| `hochciv_ki/search.py` (new) | `SearchAgent(net, k, c, seeds, budget)`: policy → top `k` → forks → `V` → pick |
| `hochciv_ki/eval.py` | agent spec `search:<path>[:k]`, so `eval` / `eval_modes` measure it like any agent |
| `hochciv_ki/online/client.py` | the Brain already keeps a local engine copy per decision (`load`); the search forks it and reads the game's turn timer for its budget |

No change to the network or the training.

## Risks

- **Opponent model:** people don't play like the KI. The value head was trained against KI and
  league opponents, so the search assumes KI-like replies. The online results will show it.
- **Value errors get exploited.** A search picks the action whose value is highest, including
  where the value is wrong. One step deep this stays small; in stage 2 the policy term (`c`)
  and the beam keep it in check.
- **Time.** The turn time must never run out: the budget is checked before every expansion,
  and the fallback is the plain policy.

## Related finding for the shaping

The value head predicts the winner much better than Φ, especially early. If run B shows
that shaping helps but Φ is the weak point, a stronger potential is `Φ = V_frozen(s)` from a
fixed copy of v2's value head – still potential-based, so still safe.
