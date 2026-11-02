# Reward shaping for PPO – design

Status: implemented (2026-11-02); run B (`checkpoints/ppo3`) started the same day.

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
| Science | `0.9 · mean over the 4 fields of (highest age reached + 1) / 4`: one age-3 tech in every field makes the Singularity researchable | `applyTech`, `singularityReady` |
| Culture (wonders on) | `0.9 · (min(level 1, 3) + min(level 2, 2)) / 5` (the pyramid rule needs 3 + 2 before a level 3), 1 with a level-3 wonder | `wonderCounts`, `wonderLevelOk`, `checkCultureVictory` |
| Military | `(capitalsTaken + best running siege of an enemy capital / 2) / capitalsToWin`: a capture takes two successful siege turns in a row | `captureCity`, `combatPhase` (`S.sieges`) |
| Claim | 1 once the seat has a claim this round | `claimVictory` |

`prog_i = max` over the paths; dead seats and barbarians can't win and are left out.

From the progress of all seats, a **soft share of the win**:

```
w_i  = exp(β · prog_i²) / Σ_j exp(β · prog_j²)        # over living, winnable seats
Φ(s) = w_me − 1 / n                                   # n: seats that could win at the start
β    = min(8, round − 1)                              # fitted, see below
```

It sums to 1 over the seats like the real score, starts near 0 for everybody (fair share), and
scales with the number of empires. The square makes progress count more near the goal, like the
KI's own `40 · q⁶` term in `kiVictory`. β depends on the round, which is part of the state, so
Φ is still a potential and the shaping still telescopes.

### Calibrating β and checking Φ

`calibrate_phi.py` plays KI schwer against itself over the v2 modes (player counts 2–8, the
training maps and options), records every seat's progress at the start of each round and the
winners, and fits β by log loss. Φ is only worth using if it predicts the winner better than
chance well before the end.

Result on 1000 games (`runs/calib_phi.log`): the games are short (median 6 rounds, 90 % within
9) and end by military victory 43 %, science 32 %, economy 22 %, culture 3 %. Without the siege
term the military path was invisible outside the marathon. A single β (best: 4) was
overconfident early – worse than the uniform guess in the first quarter of a game. Fitted per
round, the best β grows by about one per round:

| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8+ |
|---|---|---|---|---|---|---|---|---|
| best β | 0 | 1 | 2 | 3 | 4 | 6 | 6 | 8 |
| log loss uniform → Φ | 1.23 → 1.23 | 1.23 → 1.22 | 1.23 → 1.19 | 1.23 → 1.14 | 1.20 → 1.05 | 1.16 → 0.90 | 1.19 → 0.85 | 1.28 → 0.79 |

With `β = min(8, round − 1)`, Φ is never worse than the uniform guess and better for every
player count (log loss uniform → Φ: 2p 0.69 → 0.63, 3p 1.10 → 0.94, 4p 1.39 → 1.21, 6p
1.79 → 1.68, 8p 2.08 → 1.89). The seat with the highest Φ wins 43 % of the time in the second
quarter of a game, 55 % in the third and 69 % in the last (uniform: 17–50 % by player
count). In round 1, Φ is exactly 0 for everybody, so the noisy start adds nothing.

### Code changes

| Where | Change |
|---|---|
| `bridge/features.js` | `progress(S)`: every seat's progress by path (cheap: pop, techs, wonders, capitals, sieges, claims – no simulation) |
| `bridge/worker.js` | `observe` sends `prog` (per player index, −1 for seats that can't win); command `progress` for the calibration |
| `hochciv_ki/env.py` | `Obs.prog` |
| `hochciv_ki/train_ppo.py` | `phi()`, `gae()` with the shaping term, flags `--shape` (c) and `--shape-slope` (β per round); logs `shape_std`, how much the shaping moves the advantages |
| `hochciv_ki/calibrate_phi.py` | plays KI-vs-KI games, fits β, prints how well Φ predicts the winner by phase, player count and round |

No change to the network or the observation layout, so all existing checkpoints keep working.

## Changes 2 and 3 (same experiment)

- **Less anchoring:** `--lr 1e-4 --imit 0.03` (half-life 150 as before). `--target-kl 0.05`
  stays as the brake.
- **Better evaluation during training:** half of the v2 suite (about 400 games, ±0.13 per
  2-player mode) every 20 iterations instead of a quarter every 10. Same compute per
  iteration, half the noise. `best.pt` is picked by the mean relative score as today. The
  final comparison keeps using the full suite on its own seeds (base seed 90 000).

## Experiment

Decided 2026-11-02: only run B (shaping plus the changes 2 and 3), from `checkpoints/v2.pt`,
80 iterations (about 7–8 hours):

```
python -m hochciv_ki.train_ppo --init checkpoints/v2.pt --out checkpoints/ppo3 --mix v2 --iters 80 --lr 1e-4 --target-kl 0.05 --imit 0.03 --imit-half 150 --shape 1 --eval-every 20 --eval-seeds 120
```

Afterwards the full suite on its `best.pt` against `v2`. It replaces `v2` only if its mean
relative score is higher and no mode is clearly worse (beyond its CI). If it gets worse than
`v2` quickly (two half-suite evaluations in a row clearly below), it is stopped early and
restarted with `--lr 5e-5`. Without a run A, a gain can't be split between the shaping and the
looser anchor.

## Later, not in this round

- **Value per seat:** a value head that predicts every seat's share of the win (softmax over
  the seats) instead of one sigmoid. More signal per game in 4–8 player games.
- **The KI's evaluation as Φ:** `kiValue` is richer (incomes, risk, threats), but it would pull
  the policy toward the KI's judgement again – the opposite of what PPO is for here.
- **Search:** short look-ahead over the engine at decision time (planned in PLAN.md as
  "stronger search").
