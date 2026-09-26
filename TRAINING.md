# Training — brief for the `training` branch

**Goal:** run and look after self-play training of the learned bot on the
MacBook Pro (M1 Pro, 16 GB), and report what happened. Read `LEARNING.md` for
the method and `CLAUDE.md` for how the project works. Agree with the user
before starting a long run.

**Where things stand** (26 September 2026): the best policy so far is rl-003's
iteration 110, **+23.5 ± 9.1** points per deal against a RuleBot field over
2000 fresh deals. The next run, rl-004, is prepared but not started: see
"Next" and "Handoff" at the end. In the first session the user also asked the
training agent to review and improve `learn/`; those changes (an unbiased
critic, a faster update, exact resume, exploration of Flip and Halves, and
the `learn.contracts` and `learn.curve` tools) are on this branch with tests.

Later on 26 September the rules changed: a Plain contract played in clubs now
scores as Clubs (`RULES.md` §9). rl-001 to rl-003 trained under the old
scoring. A probe of rl-003's bids (`learn.margins`, "How the bidding fits its
play" below) showed that they fit its own card play: the limit is playing
higher contracts, which it has rarely done. So rl-004 was prepared again
around that. It explores higher contracts (`--explore-levels`) and weighs card
play evenly across stakes (`--stake-scaling`).

## 1. Set up

```sh
git checkout training && git merge origin/main
uv venv -p 3.13 --managed-python && source .venv/bin/activate
uv pip install -e ".[learn,dev]"
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
pytest -q                                   # everything should pass
```

## 2. Smoke test (a few minutes)

```sh
python -m learn.selfplay --iterations 3 --deals 64 --eval-every 3 --eval-deals 20 --out runs/smoke
```

Check it runs on `mps` without errors (if MPS fails, retry with
`--device cpu` and report the error), and note `collect_s` and `update_s`
per iteration. Try `--workers 6` and `8`, and keep the faster one (on the M1
Pro they are the same: playing is under a tenth of each iteration).

## 3. Imitation start (about half an hour)

```sh
python -m learn.imitate --deals 20000 --epochs 3 --explore 0.1 --out runs/bc-explore.pt
```

Expect about 90–97% agreement with RuleBot, and a duplicate score against a
RuleBot field near zero. This writes `runs/bc-explore.pt` and `.npz`.

`--explore 0.1` matters: RuleBot never bids Flip or Halves, so a plain copy
of it gives them about 0.03% probability, and self-play (which learns only
from what it samples) never finds out whether they pay. With it, a tenth of
each bid's target goes to the other kinds of contract at the same level, and
a tenth of each contract decision's (ace, trumps, Flip, exchange, fucdic) to
its other choices. Check it with `python -m learn.contracts --phases
runs/bc-explore.npz` (the flip and halves columns).

## 4. The main run

Size it from a timing run at the real settings (the smoke test is too small
to show them) to take roughly 4–8 hours. On the M1 Pro, one iteration of
1024 deals takes about 56 s on MPS (5–7 s playing, 48–51 s updating, and
6–12 s for a 1000-deal evaluation every 10 iterations), so 450 iterations
take about 7 h. The next run (see "Next" below for why):

```sh
caffeinate -i nohup python -m learn.selfplay \
    --init runs/rl-003/checkpoints/policy-0110.pt \
    --init-critic runs/rl-003/checkpoints/critic-0110.pt \
    --explore-bids 0.15 --explore-levels 0.1 --stake-scaling \
    --magnet 0.1 --critic-warmup 2 \
    --iterations 450 --deals 1024 --ppo-epochs 1 \
    --eval-every 10 --eval-deals 2000 --out runs/rl-004 > runs/rl-004.out 2>&1 &
```

A run from scratch starts from the imitation instead: `--init
runs/bc-explore.pt --critic-warmup 5`. `--init` also takes a run's
directory, for its latest policy and critic.

To stop it, `kill` the Python process (not `caffeinate`, which is its child):
it stops the workers too, and `--resume` carries on later.

`caffeinate -i` stops the Mac sleeping while it is open and on power; closing
the lid still sleeps it. With `--critic-warmup N` the first N iterations
train only the critic (`warmup` in the log): 5 after imitation, which never
trains a value head, and 1 when the critic comes from a checkpoint.

`--explore-bids 0.15` moves 15% of each bid's chance, while playing, to the
other kinds of contract at the same level, so that Flip and Halves stay in
play while their follow-up decisions are learned (see rl-002 and rl-003
below). Both the chance the move was played with and the policy's own are
recorded: PPO clips its ratio against the policy's own, and weighs each step
by the two chances' ratio (before commit 89034e8 it clipped against the
played chance, which biased the update towards explored bids; see rl-004).
`--explore-levels 0.1` does the same across levels. It moves
10% of each bid's chance to the same kind of contract one and two levels up
(half each), so that higher contracts are played, and their play learned,
before the policy would bid them. `--stake-scaling` weighs every decision from
the exchange on, once the stake can no longer change, by the batch's mean stake
over its deal's. Card play then learns from every deal evenly, not mostly from
the few with the highest stakes. Bids, the call and trumps keep their weight in
points. `--magnet` is the weight of KL(policy || magnet), 0.02 by default.

If the run stops, run `python -m learn.selfplay --resume runs/rl-004`. It
carries on from the last finished iteration with everything the run had (the
networks, optimisers, magnet, snapshot pool and random state, from
`state.pt`) and its own settings from `run.json`; only `--iterations` (the
total), `--workers` and `--device` can change. (With several workers, games
finish in a varying order, so no two runs are bit-identical anyway.) A new
run refuses a directory that already holds one.

## 5. Watching it

Every 30–60 minutes read `runs/rl-004/log.jsonl`, and check that it changed
in the last few minutes (a hang shows no error). `python -m learn.curve
runs/rl-004` prints every evaluation with its paired change from the one
before and the falls in a row; with `--follow` it keeps watching and reports
new evaluations, non-finite values, skipped steps and a stalled log. One log
line per iteration:

| Field | Healthy |
|---|---|
| `vs_rulebot` ± `vs_rulebot_ci95` (every 10 iterations) | Rising above 0 and staying there. **The number that matters.** `vs_rulebot_roles` splits it by declarer, partner and defender. |
| `value_ev` | The share of the final scores' variance the critic explains: rising from 0 |
| `value_loss` | Falling, then flat (a cross-entropy over the critic's 255 value bins, not a squared error) |
| `belief_loss` | Falling, then flat (1.39 is uniform guessing over four places) |
| `entropy` | Starts low (the imitation policy is sharp); not collapsing towards 0 |
| `magnet_kl` | Small (under about 0.2); it grows between magnet refreshes every 10 iterations |
| `clip_fraction`, `approx_kl` | About 0.05–0.3, and about 0.01–0.03 |
| `skipped_steps` | 0 (steps skipped for a non-finite gradient) |
| `declared`, `level`, `made` | The contracts the learner declares in its own deals: the share of each kind, their mean level and how often they are made. Watch Flip and Halves: rising means self-play finds they pay |
| `collect_s`, `update_s` | Steady |

`decisions` counts only real choices: forced moves are played but not
recorded. **Stop and report** if: any value is NaN; `vs_rulebot` falls for
three evaluations in a row, well outside its confidence interval; or entropy
collapses. Every evaluation uses the same deals (the first 1000 of them are the
same in every run), so compare evaluations as paired differences, as
`learn.curve` does (`evals.jsonl` keeps each one's per-deal results): a single
evaluation's interval is about ±13 at 1000 deals, but a paired change's is
about ±9. Don't change settings in code to "fix" it. Report the evidence.

## 6. After the run

`runs/rl-004/checkpoints/` keeps the policy and critic from every 10th
iteration. If the in-run curve peaked before the end, compare the best few
checkpoints on fresh deals before choosing one (the in-run maximum over the
same 1000 deals is biased upwards):

```sh
python -m learn.arena --candidate runs/rl-004/policy.npz --field rule --deals 2000 \
    --record results/rl-004/vs-rule.jsonl
python -m learn.arena --candidate runs/bc-explore.npz --field rule --deals 2000
python -m learn.arena --candidate search:runs/rl-004/policy.npz --field rule --deals 200
python -m learn.report results/rl-004/vs-rule.jsonl       # how it bids and plays
python -m learn.margins runs/rl-004/policy.npz --deals 2000 --workers 8   # do its bids fit its play?
python -m learn.margins runs/rl-004/policy.npz --field rule --deals 2000 --workers 8
python -m learn.selfplay --init runs/bc-explore.pt --exploit runs/rl-004/policy.pt \
    --iterations 100 --deals 1024 --critic-warmup 5 --eval-every 10 --out runs/x-004
```

The arena uses all cores but two by default (`--workers`); belief-sampled
search costs about 90 s per deal on one core, so its 200 deals take about 40
minutes. Put the report's output in the results write-up: it shows how the
network's bidding and results differ from RuleBot's. The last command is the
exploitability test: a fresh learner in one seat against the policy in the
other three.

To see how the bidding changed over the run, compare the checkpoints' contracts in
self-play and among RuleBots, and their choices by phase (including how likely
each is to bid Flip or Halves, which only sampling can discover):

```sh
python -m learn.contracts --phases runs/bc-explore.npz runs/rl-004/checkpoints/policy-*0.npz
python -m learn.contracts --sample runs/bc-explore.npz runs/rl-004/policy.npz  # as in training
python -m learn.contracts --field rule rule runs/bc-explore.npz runs/rl-004/policy.npz
```

Then play a few deals yourself against it for the user to try:
`python -m web.server --bot runs/rl-004/policy.npz`.

## 7. Reporting

Add a "Results" section below, one entry per run: date, commit, command,
workers and device, time per iteration, the `vs_rulebot` curve (a short
table), the final arena numbers with confidence intervals, the exploiter's
`vs_target`, and anything odd. Copy each run's `run.json`, `settings.json`,
`log.jsonl` and `evals.jsonl` to `results/<run>/` (they are small;
checkpoints stay in the ignored `runs/`). Commit to `training`, merge `main`
in, and open a pull request to `main`.

## Results

All on the MacBook Pro M1 Pro, update on MPS with 8 workers, 1024 deals per
iteration (about 47k real decisions), one PPO epoch; about 56 s per
iteration. "Against RuleBot" is the duplicate advantage in points per deal,
with a 95% confidence interval; the in-run evaluations use 1000 fixed deals,
the arena 2000 other ones. Each run's `run.json`, `settings.json`,
`log.jsonl` and `evals.jsonl` are in `results/<run>/`; `python -m learn.curve
results/<run>` shows its curve.

### Summary

| Policy | Against RuleBot (arena, 2000 deals) |
|---|---|
| Imitation of RuleBot with exploration (`bc-explore`) | −8.6 ± 3.8 |
| rl-002 at iteration 351 (its last) | +6.4 ± 9.9 |
| **rl-003 at iteration 110** | **+23.5 ± 9.1** (declarer +60.7, partner +3.7, defender −18.6) |
| rl-003 at iteration 150 | +21.4 ± 8.9 |
| rl-003 at iteration 180 (its last) | +2.5 ± 9.8 |

rl-003's iteration 110 is the best policy (`runs/rl-003/checkpoints/policy-0110.npz`
on the laptop). It gains mostly as declarer: it declares in 48.5% of seats
(RuleBot 24%), with Flip and Halves among its main contracts.

**Exploitability** (`results/x-003/`): a fresh learner, from `bc-explore`,
trained for 100 iterations alone in one seat against three copies of
iteration 110, improved from −60.8 to **−26.5 ± 13.5** per deal against the
policy's own result in that seat, so it found no way to beat it. It was still
improving (+34 over the 100 iterations), so this bounds exploitability only
weakly; a longer exploiter, or one started from the policy itself, is the
stronger test, above all of the hand-blind bidding described below.

**Search** (`results/arena/rl-003-0110-search.out`): belief-sampled search
over iteration 110 (8 worlds, card play only) scored +31.4 ± 28.6 over 200
deals, where the plain policy scored +38.5 ± 28.3 on the same 200 deals (from
the recorded arena run). So search adds nothing yet, as LEARNING.md expected
until the belief head is fitted to the policy. It took 62 minutes on 6 cores.

### Imitation

| Start | Command | Time | Agreement with RuleBot | Against RuleBot |
|---|---|---|---|---|
| `bc` | `learn.imitate --deals 20000 --epochs 3` | 28 min on CPU | 96.9% | −7.2 ± 14.7 (200 deals) |
| `bc-explore` | the same with `--explore 0.1` (MPS) | 12 min | 96.7% | −17.9 ± 12.8 (200 deals), −8.6 ± 3.8 (arena) |

`bc` gives Flip and Halves bids about 0.03% of the chance in the auction;
`bc-explore` 2.0% each, with card play unchanged.

### rl-001 (25 September, commit 6cb33db): stopped at iteration 54 of 450

From `bc`, `--critic-warmup 5`. Against RuleBot every 10 iterations: +3 ± 8,
+5 ± 9, +4 ± 10, −9 ± 12, +9 ± 11. Its bidding grew aggressive at once: in
self-play the mean contract level rose from 8.34 to 9.12 by iteration 30, the
share made fell from 75% to 58%, and Clubs grew from a third to half of the
contracts. But it could never learn Flip or Halves: it tried about 1.6 Flip
bids per iteration in some 6,000 auction decisions. Stopped to restart with
exploration.

### rl-002 (26 September, commit b5e2a89): stopped at iteration 351 of 450

From `bc-explore`, `--critic-warmup 5`. Level with RuleBot throughout:

| Iteration | 10 | 50 | 100 | 150 | 200 | 250 | 300 | 350 |
|---|---|---|---|---|---|---|---|---|
| Against RuleBot | −7 ± 6 | −8 ± 12 | −8 ± 15 | −2 ± 13 | −2 ± 13 | +1 ± 15 | −3 ± 17 | +2 ± 14 |

- PPO cut Flip and Halves from about 5% of the learner's contracts to about
  0.5% by iteration 50. As first played they lost: in sampled self-play the
  declarer averaged −109 per Flip contract and −260 per Halves contract,
  against +218 for Clubs, because their follow-up decisions (the Flip choice,
  trumps named by the partner) had never been trained.
- Self-play then found Halves by itself: from about 0.2% of contracts to
  31–47% from iteration 280. Flip only reached about 2%.
- Defending stayed its weak spot (about −100 per deal), partner its strong one.
- Stopped (at 351, not 60 as intended: a stale reading) to start rl-003 from
  its networks with sustained exploration.

### rl-003 (26 September, commit 6421a04): stopped at iteration 180 of 450 by the stop rule

From rl-002's latest policy and critic, `--explore-bids 0.15 --critic-warmup 2`.

| Iteration | 10 | 30 | 50 | 70 | 90 | 110 | 130 | 150 | 170 | 180 |
|---|---|---|---|---|---|---|---|---|---|---|
| Against RuleBot | −4 ± 14 | +13 ± 16 | +16 ± 14 | +17 ± 14 | +13 ± 14 | +35 ± 13 | +14 ± 13 | +28 ± 13 | +8 ± 14 | +1 ± 14 |

- With Flip kept in play the policy took it up itself: its own chance of
  bidding Flip on RuleBot's auction positions rose from 0.3% to 6.1% by
  iteration 30, and Flip grew to 28–46% of the contracts played.
- Paired against iteration 10 it gained +39.2 ± 13.4 by iteration 110 and
  +31.9 ± 13.3 by 150, then fell three evaluations running, 150 → 180
  (−26.6 ± 11.3), back to +5.3 ± 13.5.
- The falls come from cycling between kinds of contract. Among RuleBots
  (`results/analysis/among-rulebots-150-170.txt`), iteration 150's contracts are 66% Flip, 23% Halves and 8% Clubs (58% made,
  declarer +156); iteration 170's are 21% Flip, 43% Halves and 28% Clubs,
  and its Clubs contracts make only 41% (+30). The in-run curve swings by
  15–25 points between evaluations.

### How the bidding changed

From `learn.contracts` over 1000 deals (`results/analysis/`). In self-play,
greedily, the contracts declared:

| Policy | Mean level | Made | Flip | Halves | Clubs | Declarer's mean score |
|---|---|---|---|---|---|---|
| `bc-explore` | 8.30 | 76% | 0% | 0% | 30% | +246 |
| rl-002, iteration 351 | 9.30 | 50% | 1% | 19% | 25% | +228 |
| rl-003, iteration 50 | 9.09 | 61% | 14% | 21% | 25% | +334 |
| rl-003, iteration 110 | 8.66 | 62% | 33% | 18% | 43% | +313 |
| rl-003, iteration 150 | 8.59 | 59% | 19% | 15% | 61% | +257 |
| rl-003, iteration 180 | 8.90 | 55% | 73% | 15% | 5% | +222 |

Among RuleBots (the policy in one seat), RuleBot itself declares in 24.8% of
seats (level 8.36, 75% made, +245 per contract). rl-003's iteration 110
declares in 47.8% of seats (level 8.18; 53% Halves, 18% Flip, 18% Clubs; 56%
made; +151), and rl-002's last policy in 62.9% (mostly plain, 50% made, +80).

On RuleBot's own decisions, by phase: the policy's entropy in the auction
rose from 0.27 (rl-002) to about 1.3 nats (rl-003 from iteration 110). It
gives plain, Clubs, Flip and Halves similar chances there (15%, 22%, 24% and
28% at iteration 110), so small updates flip its greedy choice between them:
the cycling above. Its card play now agrees with RuleBot's on only 56–61% of
decisions (98% at the start).

**Its bidding hardly reads its hand.** `learn.report` on the recorded arena
deals (`results/arena/rl-003-0110-report.txt`; the 15 MB record itself stays
in `runs/arena/`): iteration 110 bids in 86% of its seats even with no ace or
Joker (RuleBot 12%), and its highest bid averages 8.02 with none and 8.18
with four (RuleBot 7.33 and 8.81). It declares in 45% of seats, mostly
Halves at level 8, and makes 56% of them; alone it loses 370 per contract. Much of
its gain over RuleBot comes from outbidding a passive field, not from judging
hands, which is a likely weakness for an exploiter or a stronger field. This
came from self-play itself: rl-002, without `--explore-bids`, was the same
(`results/arena/rl-002-report.txt`: it bid in 88% of seats with no top card,
and its highest bid even fell slightly with stronger hands), and made only
49.5% of its contracts.
But the next section shows that these bids fit its own card play.

### How the bidding fits its play

`learn.margins` plays each deal four times per seat, the same except for one
of that seat's auction decisions: its own bid, a pass, or the same kind of
contract one or two levels higher. Everything else is played as before, so
the differences come from that one bid. rl-003's iteration 110, at each
seat's first bid, over 2000 deals under the new scoring
(`results/analysis/margins-rl-003-0110.txt`), in points per seat (self-play /
among RuleBots):

| Aces + Jokers | Bids | Bid − pass | One level higher − bid |
|---|---|---|---|
| 0 | 53% / 88% | +11 ± 40 / −45 ± 21 | −30 ± 37 / −92 ± 28 |
| 1 | 59% / 90% | +116 ± 29 / +41 ± 17 | −48 ± 33 / −78 ± 20 |
| 2 | 63% / 95% | +130 ± 32 / +109 ± 18 | −41 ± 35 / −39 ± 22 |
| 3 | 68% / 98% | +137 ± 46 / +188 ± 25 | −13 ± 53 / −1 ± 30 |
| 4+ | 69% / 100% | +200 ± 96 / +216 ± 60 | +42 ± 127 / +95 ± 69 |

- **Bidding beats passing, the more so the better the hand.** With no ace or
  Joker it is about even in self-play. Among RuleBots it now loses 45; under
  the old scoring it was even (+8 ± 16, over 3000 deals:
  `results/analysis/margins-rl-003-0110-old-scoring.txt`), so there the policy
  has something to relearn.
- **A level higher loses, because those contracts are not made.** Its
  contracts are made 61% of the time as bid, 45% one level up and 29% two
  levels up (self-play; 56%, 43% and 28% among RuleBots). Two levels higher
  loses 190 to 580 points. Only the strongest hands, among RuleBots, would gain
  from a level more.
- **Its second decision** (the duel that follows) shows the same, among
  RuleBots and, under the old scoring, in self-play.

So the bidding is about as bold as its play allows. The scoring invites bold
bidding, and that needs contracts made more often, above all higher ones,
which it has rarely played (4% of its contracts are at level 10 or above).
This is why rl-004 explores higher contracts rather than, as first planned,
freezing card play to train the bidding alone (LEARNING.md, "A bidding-first
curriculum").

**Where the update goes** (`results/analysis/update-shares-rl-003-0110.txt`).
Over 1024 deals of the same policy played as in training (sampled,
`--explore-bids 0.15`), bids were 15% of the recorded decisions and 29% of the
update (its share of the summed |advantage|): their advantages vary more than
card play's (74% of decisions, 60% of the update). Card play at stakes over
80 per trick was 8% of play decisions but 18% of play's update. With
`--explore-levels 0.1` added, play at stakes over 160 was 6% of play decisions
and 39% of play's update. `--stake-scaling` brings that to 12%, while bids
keep their share (25% before it, 28% with it). Normalising bids and card play
separately, as LEARNING.md §3.3 reads, would have cut the bids' share to 18%.

### rl-004 (26 September, commit 48d9748): stopped at iteration 116 of 450

The command in §4, started at 13:00 BST from rl-003's iteration 110, with 8
workers and the update on MPS. The first two iterations trained only the
critic (about 32 s each); from iteration 3 an iteration takes about 57 s (6 s
playing, 51 s updating). The critic explained 47% of the variance at
iteration 1 and 64% at iteration 3 (74% in rl-003's smoke test, before the
new scoring and the explored higher contracts).

Decisions during the run (what was seen, what was done, why):

- 13:00, start: the laptop was on battery (26%, about 4 hours left), not AC.
  Started anyway, as instructed. If it runs out, the run stops and is resumed
  with `--resume runs/rl-004`. By iteration 10 (13:11) it was on AC power,
  charging.
- Iterations 20 and 30: two significant falls in a row (+14.3 → −0.5 → −12.0;
  paired −14.8 ± 8.2 and −11.5 ± 6.7), all as declarer (+58 → +23 → −1).
  Among RuleBots (`learn.contracts --field rule`, 2000 deals,
  `results/rl-004/rl-004-contracts-rule-0010-0030.txt`), the greedy policy's
  contracts moved to Clubs: 22% → 48% → 67% of them, made only 41–42%, and
  its declarer score fell from +141 to +102 per contract. In sampled self-play
  (`--sample`, 1000 deals) iterations 10 and 30 declare the same kinds (about
  30% each of Clubs, Flip and Halves) and the declarer's score rose (+196 →
  +223). So the training policy barely moved while its greedy choice between
  nearly equal kinds flipped: rl-003's cycling again. Carried on (not yet the
  stop rule), with a restart from the best checkpoint ready if iteration 40
  fell too. It rose instead: +1.9, paired +13.9 ± 6.2.
- 13:45, the baseline. rl-003's iteration 110 had never been evaluated on
  these 2000 deals, or under the new scoring. `learn.arena --seed 12345`
  plays the in-run evaluation's deals (checked: it reproduces iteration 10's
  result deal for deal), and there it scores **+22.5 ± 10.0** (declarer +68,
  partner +101, defender −76). Every rl-004 evaluation so far is
  significantly below its own start: paired −8.2 ± 6.8, −23.0 ± 8.3,
  −34.5 ± 7.6 and −20.6 ± 7.5 at iterations 10–40.
- Why (`learn.contracts --phases`, `results/rl-004/rl-004-phases-0010-0040.txt`):
  on RuleBot's auction positions the greedy bid of iterations 10–40 agrees
  with the start's on only 37–56%, while contract and card-play decisions
  agree on 87–96%. The auction's chance of each kind stays nearly even
  (plain, clubs, flip, halves about 8–13%, 25%, 20–27%, 24–29%) and its
  entropy rose from 1.30 to 1.41–1.50 nats, so small updates reshuffle the
  greedy bid.
- Is it only greedy noise around a steady policy? An average of the weights
  of iterations 10–40 (an offline stand-in for the averaged policy TRAINING.md
  suggests; scratch script, `learn/` unchanged) scores −5.7 ± 10.5, −28.2 ±
  7.3 against the start: no better than its members. So the policy moves
  steadily, and each step moves its auction the same way: towards what pays
  in self-play, where its defence is weak (−75 per deal against RuleBot's), and
  away from what pays against RuleBot's defence.
- Decision: carry on to iteration 100. The stop rule has not fired, the swings
  are rl-003's, and what rl-004 is for (playing higher contracts, then bidding
  them) needs iterations and the `learn.margins` probe at 100. If at 100 it is
  still well below its start with no upward trend, stop and start rl-004b from
  rl-003's iteration 110 with one change.
- Iterations 50–90 were level with the start (+21.2, +15.8, +4.7, +27.4,
  +18.4; only 70 was below), so averaged over stretches the curve rose, from
  about +1 over 10–40 to about +15 over 50–100. Iteration 100 fell again
  (+5.0, the second significant fall in a row), and 110 to −1.7 (a fall of
  −6.7 ± 8.0, not significant, so the stop rule did not fire).
- `learn.margins` on iteration 100 (1000 deals, `results/rl-004/margins-0100*.txt`)
  went against the plan. Its own contracts were made 55% as bid, 31% one
  level up and 15% two up in self-play (rl-003: 61%, 45%, 29%), and 52%, 30%
  and 17% among RuleBots (56%, 43%, 28%). "+1 − bid" was −228 ± 57 in
  self-play and −221 ± 39 among RuleBots (rl-003: −33 and −46). So higher
  contracts paid less, not more. It opened at its first decision less often
  (41% in self-play, 59% among RuleBots; rl-003 61% and 93%), and as often
  with any hand (58–62% among RuleBots from no ace or Joker to four or more).
  Defending improved (−75 to −42 per deal).
- Fresh deals (`--seed 7`, 2000 deals, paired; `results/rl-004/rl-004-compare-seed7-*.txt`):
  iteration 80 scored **+35.6 ± 11.0**, **+12.9 ± 8.8 above rl-003's
  iteration 110** (+22.7 ± 10.1 there). Iterations 50 and 90 were level with
  rl-003's 110 (+25.9, +25.0). Then 100 scored +6.8 and 110 −3.7: from 80 to
  110 it lost 39.3 ± 10.8 there and 29.1 ± 10.6 on the evaluation deals,
  with partner falling steadily (+66 → +21 → −10).
- What changed (arena records of 1000 deals, `learn.report`,
  `results/rl-004/rl-004-0*-s7-report.txt`): at iteration 80 the policy bid in
  84–90% of seats, highest bid 8.07 with no ace or Joker to 8.76 with five,
  and its contracts were at levels 8, 9 and 10 in the ratio 1218 : 570 : 92,
  mostly Halves (52%) and Flip (30%). At iteration 110 it passed or jumped
  straight to 9 whatever its hand: it bid in only 28–60% of seats, its
  highest bid averaged 8.93–9.13 with any hand, its contracts were 28 : 995 :
  54 at levels 8, 9 and 10 (Flip 59%, plain 33%), made 49%, and alone it lost
  909 per contract. In self-play the learner's contracts rose from level 9.05
  to 9.45 and were made 37% of the time, down from 46%.
- 15:00, decision: stopped rl-004 at iteration 116. This is "cycling that
  wipes out the gains": a better policy than any before (iteration 80) was
  lost within 30 iterations, and the auction drifted to hand-blind jumps, the
  one direction `--explore-levels` pushes.
- **A bug in the exploration's correction.** The learner recorded each move's
  chance under the explored odds μ, and the loss clipped PPO's ratio π_new/μ.
  The policy's favourite bid keeps only 1 − 0.15 − 0.1 of its chance under μ,
  so its ratio can start near 1.33, above the clip (1.2): its good results gave
  no gradient and its bad ones a full one. An explored bid's ratio starts far
  below 0.8: its good results pushed it up with no clip, and its bad ones gave
  no gradient at all. So every update moved chance from the policy's favourite
  bids to the explored ones (other kinds, higher levels), whatever they scored.
  That fits rl-003's auction entropy rising from 0.27 to 1.3 nats with kinds
  near even, and rl-004's drift to higher levels. On iteration 80's own
  self-play (256 deals, `results/rl-004/rl-004-0080-clipped-auction.txt`), 11%
  of auction decisions started outside the clip range (3.8% favourites, 7.4%
  explored bids). Passes and most bids were unaffected, so the ratchet is
  modest per update but always in one direction. It is a likely cause of the
  drift, not a proven one. Fixed in commit 89034e8, with tests (decoupled PPO:
  the ratio is clipped against the policy's own old chance, and each step is
  weighed by old/played odds, at most 1/0.75 here). Without exploration
  nothing changes. This is the one code change made during the run: a real
  bug that kept training from learning which bids pay.
- Restarted as rl-004b from iteration 80 (the best checkpoint, on fresh
  deals), with rl-004's flags unchanged. The one change is the fixed
  correction, not one of the "what to do" list below. The fix removes the
  ratchet that the list's remedies (a stronger magnet, less exploration)
  would only have slowed. If rl-004b still drifts or cycles, the next run
  adds one of those.

### rl-004b (26 September, commit 89034e8): stopped at iteration 98 of 450

From rl-004's iteration 80 (policy and critic), started at 15:04 with rl-004's
command otherwise unchanged (`--init runs/rl-004/checkpoints/policy-0080.pt
--init-critic runs/rl-004/checkpoints/critic-0080.pt ... --out runs/rl-004b`).
Iteration 3 (the first update): clip fraction 0.067, approx_kl 0.007 (now
measured against the policy's own old chances; rl-004's 0.04 was inflated by
the explored odds), and the critic explained 77% of the variance.

Decisions during the run:

- Iterations 10–30: +23.5, +28.4, +30.9, with no significant change between
  evaluations (+4.8 ± 6.4, +2.5 ± 7.4) and level with both starts. With the
  fix, the auction sharpens instead of evening out
  (`results/rl-004b/rl-004b-phases-0010-0030.txt`): on RuleBot's auction
  positions its entropy fell from 1.63 nats (rl-004's iteration 80) to 1.52,
  1.42 and 1.41 (in rl-004 it rose from 1.30 to 1.50), Halves rose from 27% to
  40% of its chance and Clubs fell from 18% to 12%. Its greedy bid still
  changes (69%, 59% and 56% agreement with the start; rl-004 45%, 56%, 37%),
  but steadily in one direction. Carried on.
- Iterations 40 and 50: +30.0 and +30.8, flat (five evaluations within
  +23 to +31, no significant change between any two). Partner eased from
  +114 to +68 while declarer rose from +72 to +97. That may come from the
  policy declaring more rather than partnering worse, since roles follow the
  bidding, but partner also led rl-004's collapse, so it is watched.
- `learn.margins` at iteration 50 (1000 deals, `results/rl-004b/margins-*`),
  with rl-004's iteration 80 (its start) among RuleBots for comparison. The
  bidding is stable: it bids at its first decision in 87% of seats (start 89%),
  "bid − pass" is +114 ± 18 (+99 ± 17). But its contracts are made 55% as bid,
  34% one level up and 19% two up among RuleBots, the same as its start (54%,
  34%, 20%) and below rl-003's iteration 110 (56%, 43%, 28%). "+1 − bid" is
  −124 ± 28 (start −107 ± 26); only hands with four or more aces and Jokers
  would gain a level (+92 ± 155). In self-play: 57%, 34%, 18%, "+1 − bid"
  −117 ± 45. So far, exploring levels has not taught it to make higher
  contracts: one level up they were made less often after rl-004's 80
  iterations than before, and no more often since.
- Iterations 60–90: +27.8, +20.1, +19.2, +12.7. Each step was a fall too small
  to be significant (−3.1, −7.7, −0.9, −6.5), so the stop rule could not
  fire, but together 50 → 90 lost 18.2 ± 11.9, and iteration 90 was below its
  start (−14.7 ± 12.8). Declarer fell from +97 to +37. In self-play the drift
  was slow and steady, not rl-004's jump: level 9.27 → 9.40 over 80
  iterations, made steady at 44%, Flip 29% → 48% of the learner's
  contracts; entropy, clip fraction and magnet KL constant.
- Why (`learn.contracts`, `results/rl-004b/rl-004b-contracts-*-0050-0090.txt`):
  in sampled self-play iteration 90 is the *better* declarer (+292 per
  contract, 53% made; iteration 50 +228, 51%). Among RuleBots its greedy
  policy declares in 57% of seats (50: 45%), 55% Flip and 31% Clubs (50: 62%
  Halves), makes 48% (54%), and scores +133 per contract (+189). So with the
  ratchet gone, self-play itself moves the policy to declare more, in Flip
  and Clubs, against defenders as weak as itself (−86 per deal as defender,
  against RuleBot's), and RuleBot's defence punishes that.
- Fresh deals (`--seed 7`, 2000, `results/rl-004b/rl-004b-compare-seed7-a.txt`):
  **iteration 40 scored +46.5 ± 11.8, +10.8 ± 9.3 above rl-004's iteration
  80** (+35.6) and about 24 above rl-003's iteration 110 (+22.7 there); 30
  and 50 scored +44.1 and +40.9. Iteration 90 scored +25.4.
- 16:32, decision: stopped rl-004b at iteration 98 (cycling that wipes out
  the gains, as for rl-004, only slower) and started rl-004c from its
  iteration 40, the best policy so far, with one change: `--magnet 0.3`
  instead of 0.1. The drift that remains is self-play's own, towards what
  beats its own weak defence. The magnet (KL to a copy refreshed every 10
  iterations: magnetic mirror descent) is TRAINING.md's regulariser against
  such drift and cycling, and at 0.1 it let the policy lose about 20 points
  in 40 iterations. Less exploration would not touch a drift the policy
  now chooses for itself.

### rl-004c (26 September, commit 89034e8): running

From rl-004b's iteration 40 (policy and critic), started at 16:32 with
rl-004b's command except `--magnet 0.3` (`--init
runs/rl-004b/checkpoints/policy-0040.pt --init-critic
runs/rl-004b/checkpoints/critic-0040.pt ... --magnet 0.3 ... --out
runs/rl-004c`). Iteration 3: clip fraction 0.058, approx_kl 0.006, the critic
explained 73%.

Decisions during the run:

### Lessons

- A policy learns only from what it samples. A copy of RuleBot never tries
  Flip, and a policy whose first Flip contracts are badly played drops Flip
  before it learns to play it. `--explore` at the start and `--explore-bids`
  throughout keep it in play; the bid's own chance then follows its results.
- The in-run evaluation is noisy (±13 at 1000 deals) and swings when a few
  high-stakes bids change. Judge by paired changes (`learn.curve`) and choose
  checkpoints with the 2000-deal arena.
- Bidding can only be as bold as the play behind it. rl-003's bidding looked
  as if it ignored the hand, but `learn.margins` showed it fits its own play:
  with a weak hand, bidding lost little or nothing, and a level higher lost
  because those contracts were not made. Measure what the alternatives would have scored
  before blaming the bidding.
- Background jobs ignore Ctrl-C (SIGINT); `kill` (SIGTERM) now stops a run and
  its workers cleanly.

## Next: rl-004 (prepared, not started)

The command is in §4. It was smoke-tested (3 iterations of 64 deals, and a resume
of a fourth, which kept its settings) and differs from rl-003 in these ways, for
these reasons:

1. **Start from rl-003's iteration 110**, the best policy, with its own
   critic (`--init-critic`), which explained 74% of the variance from the
   first iteration of an earlier smoke test. `--critic-warmup 2` gives the
   critic two iterations to catch up with what changed under it: the new rule
   (a Plain contract in clubs scores double) and higher contracts.
2. **`--explore-levels 0.1`.** rl-003's bids fit its own play ("How the
   bidding fits its play" above). A level higher loses because its contracts
   one level up are made only about 44% of the time, and it has rarely played
   them. That is how Flip was first dropped (rl-002), and exploring kept Flip
   in play until its follow-up decisions were learned (rl-003). Exploring
   levels does the same for higher contracts.
3. **`--stake-scaling`.** Exploring levels brings in high-stakes deals. Without
   it, play at stakes over 160 per trick would be 6% of card-play decisions
   but 39% of card play's share of the update. With it, every deal teaches
   card play about equally, and bids keep their weight in points.
4. **Magnet 0.02 → 0.1.** The policy cycles between kinds of contract. The
   magnet, KL(policy || a copy refreshed every 10 iterations), is the
   regulariser meant to damp such cycling in self-play (DeepNash, magnetic
   mirror descent), and at 0.02 it is weak: the policy drifted about 0.06–0.08
   nats from it in each 10 iterations.
5. **`--eval-deals 2000`**, for a ±9 interval instead of ±13 (about 20 s every
   10 iterations). The first 1000 deals are the same as before, so the curves
   stay comparable, though the scoring changed (only Plain contracts in clubs
   score differently).

`--explore-bids 0.15` stays: the policy's own taste for Flip was still rising.
Several things change at once, so the curve alone will not say which helped.
Read the mechanisms instead, as below.

What to look for, and what to do:

- **Higher contracts start to pay.** Run `learn.margins` (§6) on a
  checkpoint every 50 iterations or so (2000 deals take about 10 minutes in
  self-play and 3 among RuleBots). Its contracts made one level up (rl-003: 44%) should
  rise, and "+1 - bid" should turn positive for the stronger hands. The log's
  `level` and `made` (the learner's own contracts, explored ones included)
  show the same from inside the run.
- **The bidding follows.** Once a level higher pays for strong hands, the
  policy's highest bid should climb with its hand (`learn.report` on a recorded
  arena run: bidding by top cards) and `learn.margins` should show it bidding
  them. If a level higher pays but the policy does not bid it, the bidding lags
  the play. That is when the bidding-first phase pays (LEARNING.md, "A
  bidding-first curriculum").
- The paired changes stay positive or flat, without 15–25-point swings: it
  works; carry on and pick the final policy with the arena.
- It still cycles: play an averaged policy (keep an exponential moving average
  of the policy's weights and evaluate and export that; in self-play the
  average converges where the latest policy cycles), or refresh the magnet
  less often (it is tied to `snapshot_every`, 10).
- It stops learning (paired changes near 0 for 50+ iterations, clip fraction
  under about 0.03): the magnet is too strong; try 0.05.
- Watch the auction's entropy (`learn.contracts --phases` on checkpoints). It
  rose to about 1.3 nats in rl-003, and the exploration may be part of why:
  explored bids that pay are raised. If it keeps rising, lower
  `--explore-bids` to 0.05–0.1.
- Defending is the weakest role in every run (−19 to −30 per deal at best,
  about −100 in rl-002). Worth a look in `vs_rulebot_roles` and with
  `learn.contracts --field rule`.

Other open work, from LEARNING.md and the review: fit the belief head to the
final policy before search relies on it; and an engine question, that the
FUCDIC phase is visible to every seat (a declined fucdic tells the others the
declarer holds at most one card of the called suit; training is not affected,
since only the declarer acts in it).

## Handoff

- **Code:** branch `training`, in the git worktree `~/danish-wist-training`
  (the main checkout `~/danish-wist` stays on `performance`). Python is
  `~/danish-wist-training/.venv/bin/python` (uv CPython 3.13.14, PyTorch 2.14
  with MPS).
- **`runs/` is not in git and lives only on this laptop**, under
  `~/danish-wist-training/runs/`:
  - `bc.pt`, `bc-explore.pt` and their `.npz`: the imitation starts.
  - `rl-001/`, `rl-002/`, `rl-003/`: each with `state.pt` (resumable with
    `--resume`), `checkpoints/` (policy, critic and `.npz` every 10
    iterations), logs and evaluations. rl-003's `checkpoints/policy-0110.*`
    and `critic-0110.pt` are the best and the start of rl-004.
  - `x-003/`: the exploiter against rl-003's iteration 110.
  - `arena/`, `analysis/`: the evaluations above (copied to `results/`).
- **Tools:** `learn.curve` (a run's curve and paired changes, `--follow` to
  watch), `learn.arena` (duplicate evaluation on all cores, `--record` to keep
  every deal), `learn.report` (from recorded deals: roles, contracts, and
  bidding by hand strength), `learn.contracts` (plays policies itself:
  contracts by kind, level and result, `--phases`, `--sample`, `--field
  rule`), and `learn.margins` (each bid against a pass and one or two levels
  higher, same cards: whether the bidding fits the play). `learn.report` and
  `learn.contracts` overlap and could be merged.
- **Recorded arena deals** of rl-003's iteration 110 and rl-002's last policy
  are in `runs/arena/*-vs-rule.jsonl` (15 MB each, too big for git).
- **Before a long run,** agree it with the user. The user expects Flip to prove
  strong and bidding to grow high and aggressive, since a made contract pays
  for every trick; both have held so far.
