# rl-004 line: findings, for review

Written 26 September 2026 at the close of the rl-004 session, for whoever reviews
this work and decides what comes next. `TRAINING.md` has the full record: "The
rl-004 line" in Results, and each run's entry with every decision taken during
the unattended run, what was seen and why. This page gathers the findings, the
likely reasons performance is not higher, and what is uncertain.

## Headline

| | Against RuleBot, per deal (2000-deal arena, default seed) |
|---|---|
| rl-003's iteration 110, old scoring (the previous best) | +23.5 ± 9.1 |
| rl-003's iteration 110, new scoring | +18.2 ± 9.5 |
| **rl-004d's iteration 10** (the best now) | **+56.5 ± 15.7**, +38.2 ± 14.2 over rl-003's 110 (paired) |

The duplicate score per deal has a standard deviation of about 360 points, so
+56 is about a sixth of a standard deviation per deal. It was also the best of
eight candidates on 4000 other fresh deals (+52.7 ± 10.8, +26.5 ± 9.8 over
rl-003's 110). Since it was chosen as the best of several on both sets, its
true level is probably a little lower. rl-004c's 150 (+53.7) and rl-004d's 100
(+49.0) are level with it: **the line's best is about +50 per deal.**

## What was run

§4's command (`--explore-bids 0.15 --explore-levels 0.1 --stake-scaling
--magnet 0.1`, 1024 deals and one PPO epoch per iteration, a 618k-parameter
transformer: width 128, 4 layers, 4 heads). Each run started from the best
checkpoint of the one before, with one change:

| Run | Start | Change | Iterations | Stopped because |
|---|---|---|---|---|
| rl-004 | rl-003's 110 | the planned settings | 116 | its gains at 80 were gone by 110 |
| rl-004b | rl-004's 80 | PPO's correction for exploration fixed | 98 | slid from its peak at 30–50 |
| rl-004c | rl-004b's 40 | `--magnet 0.3` | 236 | held about 120 iterations, then slid back |
| rl-004d | rl-004c's 150 | `--magnet 1.0` | 108 | held the best level, stopped improving |

That is 558 iterations, about 571,000 deals and 8.6 hours of training; all
self-play so far (rl-001 to rl-004d) is about 1.17 million deals.

## Findings

1. **A bug biased every run that explored (rl-003, rl-004), now fixed** (commit
   89034e8, with tests). The learner recorded a bid's chance under the explored
   odds, and PPO clipped its ratio against that, not against the policy's own.
   A favourite bid's ratio then started above the clip range (good results gave
   no gradient), an explored bid's below it (bad results gave none), so updates
   moved chance to explored bids whatever they scored: 11% of auction decisions
   started outside the clip range. It explains rl-003's auction entropy rising
   from 0.27 to 1.3 nats and rl-004's drift to hand-blind jumps to level 9. Now
   PPO clips against the policy that collected the move and weighs each step by
   old/played odds (decoupled PPO). After the fix the auction sharpened (entropy
   1.63 → 1.32).
2. **Higher contracts did not start to pay**, which was rl-004's purpose.
   Among RuleBots, contracts one level above its bid are made 32% of the time
   (rl-003: 43%), and "+1 − bid" is −203 ± 23 (−46 ± 12); no strength of hand
   gains from a level more. Exploring levels for 558 iterations did not teach
   it to make them.
3. **The bidding ran ahead of the play.** It bids higher anyway: level 8.84
   (8.18), mostly 9; 61% of its contracts are Flip (18%); it declares in 57% of
   seats (45%) and makes 49% (56%), yet earns more per contract (+243 vs +198).
   It bids in 91% of seats with no ace or Joker; its highest bid climbs only a
   little with the hand (8.53 → 8.94). Against this passive field that costs
   little: with no ace or Joker its first bid is worth −19 ± 37 against a pass.
4. **Self-play drifts towards beating itself, not RuleBot.** With the bug gone
   the policy still moved steadily to Flip and to declaring more. In sampled
   self-play that made it a better declarer (+228 → +292 per contract), while
   among RuleBots it declared too often and lost. A stronger magnet slowed the
   drift (the peak held about 40 iterations at 0.1, 120 at 0.3, and at 1.0 the
   policy held but stopped improving). The gains came from restarting at the
   best checkpoint, chosen on fresh deals against RuleBot.
5. **Defending is the weak spot, and it is exploitable.** As defender it scores
   90 per deal below RuleBot in that seat (arena), and in-run it fell from −69
   to −121 as self-play moved to Flip. An exploiter started from the policy
   itself found +20.3 ± 20.1 per deal in 100 iterations, nearly all by
   declaring against three copies of it. An exploiter from the RuleBot
   imitation found nothing (−120.2 ± 21.8).
6. **Search adds nothing yet**: +47.5 ± 47.0 against the plain policy's +63.3
   ± 48.0 on the same 200 deals, as for rl-003.
7. **The in-run curve misled.** A checkpoint chosen as the best on the
   evaluation deals is biased upwards there, so a run started from it looked
   worse than its start; rl-004d looked flat in-run while fresh deals showed it
   at the line's best. Checkpoints differ by 5–10 points, which takes 4000 or
   more paired deals to resolve.

## Why performance is not higher

Ranked by the evidence for each, with a test that would settle it.

1. **It barely trains against the opponent it is judged against.** In 25% of
   deals 1–3 seats go to random opponents drawn from RuleBot and eight recent
   snapshots, so only about 5% of training deals contain a RuleBot
   (`choose_lineups`, `Settings.opponent_share`). 95% of the signal is
   self-play, which rewards beating its own weaknesses (finding 4). *Test:*
   expose `opponent_share` (or a RuleBot share) as a flag and train with, say,
   30–50% of deals holding RuleBots; watch the RuleBot score and the defender
   role.
2. **Its defence is weak** (finding 5): about −90 per deal in a quarter of its
   seats, roughly −23 per deal overall. Self-play rarely shows it the plain and
   Clubs contracts RuleBot bids. *Test:* the same broader field; a longer
   exploiter; the defender role in `vs_rulebot_roles`.
3. **The policy moves slowly.** approx_kl is 0.003–0.007 per iteration and the
   clip fraction 0.03–0.06 since the fix (policy learning rate 1e-4, one PPO
   epoch, batch 512). Card play agrees with RuleBot on 61–62% of its decisions
   in every checkpoint since rl-003, and its contracts are made less often, not
   more. The learning signal for card play may be too weak for 1.17 million
   deals. *Test:* a higher learning rate (3e-4) or two PPO epochs, aiming at
   approx_kl 0.01–0.02; measure play on fixed contracts (the same hand and
   contract, the policy against RuleBot as declarer).
4. **The scale is small.** About 1.17 million self-play deals on a laptop,
   and every run was stopped or restarted within 100–240 iterations. Published
   self-play systems for card games trained on orders of magnitude more games.
   *Test:* a long run at settings that do not drift (below), judged on fresh
   deals.
5. **Noisy decisions and restarts.** ±15 per 2000 deals, and each restart
   began with a fresh optimiser and snapshot pool (only `--resume` keeps them)
   and two critic-only iterations. Stops and restarts in this session were judgement calls on
   noisy evidence (see caveats). *Test:* choose on 4000+ paired fresh deals.
6. **Two settings may not help.** `--explore-levels` did not make higher
   contracts pay (finding 2), and `--stake-scaling` was never tested on its own
   (rl-004 changed four settings at once). *Test:* one run without each, from
   rl-004d's 10.

## Caveats about this session

- The reported +56.5 is the best of three finalists on those deals, and the
  finalists were the best of eight on 4000 others: expect about +50.
- rl-003's +23.5 was measured under the old scoring and with the exploration
  bug. Under the new scoring it is +18.2 on the same deals.
- The stop and restart decisions (rl-004 at 116, rl-004b at 98, rl-004c at
  236, rl-004d at 108) were judgement calls under the user's rules; the reasons
  are in each run's entry. rl-004b and rl-004c might have recovered if left.
- The averaged-weights test (rl-004's entry: is the drift only greedy noise?)
  was done on checkpoints trained with the bug, and may come out differently
  now.
- Explanations 1–3 above are hypotheses from the evidence, not tested.

## Reproduce

- Checkpoints, resumable state and arena records are in
  `~/danish-wist-training/runs/` on the laptop (not in git). The best policy is
  `runs/rl-004d/checkpoints/policy-0010.{pt,npz}`.
- Each run's `run.json`, `settings.json`, `log.jsonl` and `evals.jsonl` are in
  `results/rl-004*/` and `results/x-004*/`; `python -m learn.curve results/<run>`
  shows a curve.
- Arena and reports: `results/arena/rl-004*`. Margins, contracts and screens:
  `results/rl-004*/`.
- Scripts used for the analyses (`results/rl-004/scripts/`):
  - `screen.py`: several policies on the same deals from several seeds, pooled
    and paired against the first. `--seeds 12345` plays the in-run
    evaluation's deals; `--seeds 21 23` was the 4000-deal screen.
  - `clipped.py`: how many auction decisions start outside PPO's clip range
    under the old loss.
  - `avg.py`: averages checkpoints' weights into an `.npz`.
