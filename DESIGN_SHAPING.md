# Reward shaping for PPO – design

Status: draft (2026-11-02), not implemented.

## Why

The `ppo2` run (154 iterations, about 2.3M decisions, from `v1w`) gave no gain. On the full
v2 suite the mean score relative to the fair share was 1.22 for `v1w`, 1.19 for iteration 100
and 1.13 for iteration 150 (`runs/eval_snap*_v2.log`). Three reasons, in order of how much
they probably matter:

1. **Sparse reward.** The only reward is the final score (1 / 1/k / 0) after 50–100 learner
   decisions. In 4–8 player games the learner wins rarely, so most games say "0" whatever it
   did. The value head (one sigmoid, the learner's win chance) has to learn everything from
   that signal before the advantages become useful.
2. **The anchor.** The imitation term (`--imit 0.1`, still 0.05 at the end) and the small
   learning rate (`3e-5`) keep the policy close to the KI planner.
3. **Measurement.** The evaluation during training (6–30 games per mode) can't tell a real
   change from noise, so `best.pt` was picked by luck.

This design is about 1. Changes 2 and 3 are flags and small code changes; they go into the
same experiment (see *Experiment*).

## Approach: potential-based shaping

A shaped reward `F = Φ(s') − Φ(s)` between consecutive learner decisions, where `Φ` is a
hand-made estimate of "how well am I doing". With gamma = 1 the shaping terms telescope: their
sum over a game is `Φ(end) − Φ(start)`. With `Φ(end) = 0` it is `−Φ(start)`, which doesn't
depend on the policy. So **shaping can't change which policy is best** (Ng, Harada, Russell
1999); it only gives each decision earlier feedback. This rules out the usual failure of
shaped rewards: the agent farming the shaping signal instead of winning.

Non-potential bonuses ("+0.1 per captured capital", "+0.01 per tech") are rejected for that
reason: they change the objective, and the network would learn to optimise them.

### Where it enters

The value head stays as it is: the learner's chance to win, trained on the λ-return of the
**unshaped** reward, so the online client's value display keeps its meaning and the returns
stay in [0, 1] for the sigmoid. Shaping only enters the advantage:

```
delta_t = (Φ(s_{t+1}) − Φ(s_t)) * c  +  r_t + V(s_{t+1}) − V(s_t)     # Φ(terminal) = 0
A_t     = sum_k λ^k delta_{t+k}                                       # as in gae() today
```

`c` is the shaping weight (flag `--shape`, 0 = off). `s_{t+1}` is the learner's next
decision, so the opponents' turns in between count: if an opponent takes a city, the
learner's last decision before it sees a drop.

### The potential Φ

Per seat `i`, a **victory progress** in [0, 1]: the furthest it is along any victory path.

| Path | Progress | Engine source |
|---|---|---|
| Economic | `min(1, popOf(i) / worldPop / frac)`, `frac` from `victoryOption` (duel, UN, theology) | `checkVictory` |
| Science | fields with an age-3 tech / number of fields (`singularityReady`), plus a bit per age reached | `applyTech`, `singularityReady` |
| Culture (wonders on) | wonder levels toward a level-3 wonder, 1 with `cultureWin` | `wonderCounts`, `checkCultureVictory` |
| Military (marathon) | `capitalsTaken / capitalsToWin` (outside the marathon one capture ends the game, so 0) | `captureCity` |
| Claim | 1 once the seat has a claim this round | `claimVictory` |

`prog_i = max` over the paths; dead seats and barbarians count as 0 and can't win.

From the progress of all seats, a **soft share of the win**:

```
w_i  = exp(β · prog_i²) / Σ_j exp(β · prog_j²)        # over living, winnable seats
Φ(s) = w_me − 1 / n
```

It sums to 1 over the seats like the real score, starts near 0 for everybody (fair share), and
scales with the number of empires. The square makes progress count more near the goal, like the
KI's own `40 · q⁶` term in `kiVictory`. The exact science and culture formulas get settled
while implementing, against the engine's rules.

### Calibrating β and checking Φ

Before any training: record about 2000 games KI schwer vs KI schwer over the v2 modes and log
Φ at every turn together with the outcome.

- β is fitted so that `w_i` predicts the winner best (log loss), separately checked per
  player count.
- Φ is only worth using if it predicts the winner clearly better than chance well before the
  end, say by the middle of the game. If it doesn't, the shaping would only add noise, and we
  revise Φ before spending GPU hours.

### Code changes

| Where | Change |
|---|---|
| `bridge/worker.js` `observe` | adds `prog`: the victory progress per seat (cheap: pop, techs, wonders, capitals, claims – no simulation) |
| `bridge/features.js` | the progress function, next to `seatVec` (which already has the economic share) |
| `hochciv_ki/env.py` | `Obs.prog`; `play_game` records the final progress too |
| `hochciv_ki/train_ppo.py` | `phi()` from `prog`, `gae()` with the shaping term, flags `--shape` (c) and `--shape-beta` (β); logs the mean shaping reward |
| `hochciv_ki/calibrate_phi.py` (new) | records KI-vs-KI games, fits β, prints how well Φ predicts the winner by game phase |

No change to the network or the observation layout, so all existing checkpoints keep working.

## Changes 2 and 3 (same experiment)

- **Less anchoring:** `--lr 1e-4 --imit 0.03` (half-life 150 as before). `--target-kl 0.05`
  stays as the brake.
- **Better evaluation during training:** half of the v2 suite (about 400 games, ±0.13 per
  2-player mode) every 20 iterations instead of a quarter every 10. Same compute per
  iteration, half the noise. `best.pt` is picked by the mean relative score as today. The
  final comparison keeps using the full suite on its own seeds (base seed 90 000).

## Experiment

Two runs from `checkpoints/v2.pt`, one after the other (the PC can't run both at full speed):

| Run | Flags | Purpose |
|---|---|---|
| A | `--lr 1e-4 --imit 0.03`, no shaping | Does less anchoring alone help? |
| B | as A, plus `--shape 1` | Does the shaping help on top? |

Each 80 iterations, about 7–8 hours. Afterwards the full suite on both `best.pt` files and on
`v2`. A model replaces `v2` only if its mean relative score is higher and no mode is clearly
worse (beyond its CI).

If run A already gets worse than `v2` quickly (two half-suite evaluations in a row clearly
below), it is stopped early: the learning rate is then too high, and B gets `--lr 5e-5`.

## Later, not in this round

- **Value per seat:** a value head that predicts every seat's share of the win (softmax over
  the seats) instead of one sigmoid. More signal per game in 4–8 player games.
- **The KI's evaluation as Φ:** `kiValue` is richer (incomes, risk, threats), but it would pull
  the policy toward the KI's judgement again – the opposite of what PPO is for here.
- **Search:** short look-ahead over the engine at decision time (planned in PLAN.md as
  "stronger search").
