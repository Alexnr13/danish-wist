# Training — brief for the `training` branch

**Goal:** run and look after self-play training of the learned bot on the
RTX 5090 workstation (until 27 September 2026, the MacBook Pro M1 Pro), and
report what happened. Read `LEARNING.md` for
the method and `CLAUDE.md` for how the project works. Agree with the user
before starting a long run.

**Review, 27 September 2026:** `results/review-2026-09/REVIEW.md` reviews
the rl-004 line and the method against the literature and holds **the todo list
for the next agent** and the steps for moving to the 5090 workstation. Its
main finding: all of the gain over RuleBot is bidding; on fixed contracts the
network's card play is level with RuleBot (+3.0 ± 3.4), and the by-role
arena split is not a measure of play. Start there.

**Where things stand** (29 September 2026, evening, on the workstation): "Next" of that afternoon is done. **The league now keeps at most 20 exploiters** (`--league-exploiters`, the oldest going first; with tests), and **rl-006 carried on to 6000 iterations** with it (1 hour 53 minutes, healthy, no stop rule): the league went from 40 exploiters and 9 past selves to 20 and 29, and the learner met its past selves in 43% of its league deals instead of 30%. **Its iteration 5400 is the best policy, a small step**: on the reporting seeds it is level with rl-006's 4000 in card play (+1.6 ± 1.7; **+26.3 ± 2.3** against RuleBot) and in the full game against RuleBot (+4.4 ± 13.9; **+185.9**), better against every learned field (+17.7 to +31.6, each just significant), and **its exploiter found nothing** (−18.3 ± 20.3; the 4000's took +23.4 ± 20.2). No checkpoint of the continuation beat the 4000 against RuleBot, which decides every choice, while the learner meets RuleBot in only about 1% of its deals. The progress graph (`results/progress/progress.html`) shows card play still creeping up (+28.1 at 6000, the line's best) and the full game against RuleBot swinging with the bidding. The web game's bot stays rl-005's 3600 (the user's choice). See "rl-006, to 6000" in Results. **Next (proposed, not agreed): RuleBot in a tenth of the deals, then rl-006 on to 8000; see "Next" near the end.**

**Where things stood** (29 September 2026, afternoon, on the workstation): the user asked for more training rather than more exploiter measurements, and **rl-006 carried on to 4000 iterations** (1 hour 53 minutes, healthy, no stop rule). **Its iteration 4000 is the best policy**: on the reporting seeds it beats rl-006's 2000 in the full game against every reference field (**+34.2 ± 14.3** against RuleBot, where it scores **+181.4**; +65 to +80 against the learned fields) and a little in card play (+1.4 ± 1.7; **+24.8 ± 2.2** against RuleBot), and its bidding reads the hand a little more. Its exploiter margin stays at about +20 (+23.4 ± 20.2, one exploiter). The progress graph (`results/progress/progress.html`) shows card play rising along the whole line to about +26 by rl-006's 3200 and level after, and the full game still rising at 4000. The web game's bot stays rl-005's 3600 (the user's choice). Runs live on the workstation only; `training`, `main` and `throughput` were rewritten to drop run files from history (CLAUDE.md; "What is committed under `runs/`"). See "rl-006, continued" in Results. Then the user agreed to cap the exploiters in the league and carry rl-006 on to 6000 (see above).

**Where things stood** (29 September 2026, night, on the workstation): "Next" steps 1 and 2 of 28 September are done. **The exploitability of rl-005's 3600 is confirmed**: a second exploiter (seed 1) takes +18.6 ± 15.4 from it, again as declarer, while rl-005's 2000 resists one trained twice as long (+10.9 ± 16.7). **rl-006** (2000 iterations from 3600 with 100-iteration exploiters in the league, 1 hour 53 minutes, healthy, no stop rule) made **its iteration 2000 the best policy**: on the reporting seeds its **card play is +23.4 ± 2.3** against RuleBot (3600: +19.9; paired **+3.5 ± 2.0**), and it beats 3600 in the full game against every reference field (**+18.1 ± 13.7** against RuleBot; +36.6, +39.5 and +60.3 against rl-005's 2000, rl-005's 3600 and x-005-3600), and for the first time its bidding follows the hand (among seats that bid, +0.65 a level per 10 high-card points; 3600: +0.25). **But it is no less exploitable**: a fresh exploiter takes +18.5 ± 17.7 from it (3600, same seed: +18.6 ± 15.4), and rl-006's league had thinned out its seeded references by iteration 441. The web game's bot is still 3600; replacing it is the user's call. See "Exploitability, confirmed or not" and "rl-006" in Results. Then (29 September, morning) the user asked for more training; see above.

**Where things stood** (28 September 2026, afternoon, on the
workstation): the four tasks agreed that morning are done. **rl-005 ran on
to 4000 iterations** (3.1 more hours, no stop rule, 146 million decisions
in all) and its **iteration 3600 is the best policy**: on the reporting
seeds its **card play on fixed contracts is +19.9 ± 2.1** against RuleBot
(rl-005's 2000: +13.3, paired **+6.7 ± 2.1**; rl-004d's 10: +3.9), and it
beats every reference field (the full game **+20.7 ± 10.6** over the 2000
against RuleBot, +61 to +95 against the learned fields). **But its
exploiter margin rose to +32.0 ± 15.4** (the 2000: +2.6 ± 14.3; one
exploiter each): a clone trained 100 iterations wins, mostly by declaring
more against it. The bidding
still ignores the hand but a level higher still loses, so no bidding
curriculum is indicated (T4.2); the alone contracts are the called ace in
the cat at chance rate, not bad calls (T4.3); entropy 0.03 stays (0.01 was
better against RuleBot only). See "rl-005, continued", "Bidding and alone
contracts" and "The entropy check" in Results. **Next (proposed, not
agreed): confirm the exploitability, then a run with longer exploiters in
the league; see "Next" near the end.** Since 28 September 3600 is also the
web game's default opponent: `python -m web.server` plays it from
`web/bot.npz` (`--bot rule` for RuleBots).

**Throughput** (28 September 2026, evening, branch `throughput`): an iteration at rl-005's settings takes **about 2.45 s instead of about 5.0 s** (collecting 2.3 to 1.0 s, updating 2.2 to 1.2 s; rl-005 resumed for 40 iterations against its own log), in 6 GB of GPU memory instead of 12. The update is compiled with a bfloat16 trunk, and the workers' networks run in the main process for all of them (`--worker-device` is gone; workers default to all cores but two). Time estimates in this file were made before and are now about twice what runs take. Collecting while updating would take it to about 1.3 s but changes the algorithm, so it is not done (see "Later"). Details and measurements: PERFORMANCE.md, "Training throughput".

**Where things stood** (28 September 2026, early morning, on the
workstation): the sweep and the long run agreed with the user were done.
The sweep ("The sweep" in Results) chose entropy 0.03 and dropped
`--stake-scaling`; the long run, **rl-005** (2100 iterations from rl-004d's
10 with the league, 3.5 hours), ran without drift or a stop rule. Its
iteration 2000 was then **the best policy**: on the reporting seeds (6000 deals)
its **card play on fixed contracts is +13.3 ± 2.1** against RuleBot (rl-004d's
10: +3.9 ± 1.9, paired **+9.4 ± 2.1**), the full game +108.4 ± 11.8 against
RuleBot (**+52.4 ± 10.2** over rl-004d's 10) and about +100 against every
learned reference field, and its exploiter margin is +2.6 ± 14.3 (rl-004d's
10: +15.5 ± 9.7). With critic search its card play reaches +20.2 ± 7.7 (500
deals). Its bidding is bolder and still ignores the hand. See "rl-005" in
Results. Then agreed with the user on 28 September: carry rl-005 on to
4000 iterations, a stronger reference set, the bidding and alone contracts,
and an entropy check (done; see above).

**Where things stood** (27 September 2026, evening): the review's Phase 0
and 1 were done (the workstation runs about twelve times faster, and the
measurement tools are in `learn/`), the league with exploiters and the EMA
magnet were built (Phase 2's code), and T3.1's search worked: over 2000
deals it lifts rl-004d's 10's card play from +3.0 ± 3.4 to +12.3 ± 4.0
against RuleBot.

**Where things stood** (26 September 2026, evening): the best policy is
**rl-004d's iteration 10, +56.5 ± 15.7** points per deal against a RuleBot
field over 2000 fresh deals, **+38.2 ± 14.2 above rl-003's iteration 110** on
the same deals (rl-003's +23.5 ± 9.1 was under the old scoring; under the new
it scores +18.2 ± 9.5 there). It came from the rl-004 line: four runs, each
started from the best checkpoint of the one before, after a bug in how PPO
corrected for exploration was found and fixed (commit 89034e8). **For review:
`results/rl-004/FINDINGS.md`** gathers the findings, why performance is not
higher, and the caveats. See "The rl-004 line" in Results for the detail, then
"Next" and "Handoff" at the end. In the first
session the user also asked the training agent to review and improve
`learn/`; those changes (an unbiased critic, a faster update, exact resume,
exploration of Flip and Halves, and the `learn.contracts` and `learn.curve`
tools) are on this branch with tests.

Later on 26 September the rules changed: a Plain contract played in clubs now
scores as Clubs (`RULES.md` §9). rl-001 to rl-003 trained under the old
scoring, the rl-004 line under the new.

## 1. Set up

```sh
git worktree add ~/danish-wist-training training     # from the main checkout
cd ~/danish-wist-training && git merge origin/main
uv venv -p 3.13 --managed-python && source .venv/bin/activate
uv pip install "torch==2.14.0" --index-url https://download.pytorch.org/whl/cu130  # the 5090
uv pip install -e ".[learn,dev]"
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
pytest -q                                   # everything should pass
```

(On the laptop: no CUDA index; PyTorch's default build has MPS.) If
pypi.nvidia.com times out, set `UV_HTTP_TIMEOUT=600`.

## 2. Smoke test (a few minutes)

```sh
python -m learn.selfplay --iterations 3 --deals 64 --eval-every 3 --eval-deals 20 --out runs/smoke
```

Check it runs without errors, on `cuda` (on the laptop, `mps`), and note
`collect_s` and `update_s` per iteration. The update and the networks that play run on `--device` (the GPU by default), in the main process; the `--workers` processes (all cores but two by default) play the games (PERFORMANCE.md, "Training throughput"). The first iteration includes compiling: up to a minute with an empty cache.

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

*rl-005's command, with the league, is in "Done: the sweep, then the long
run" near the end and in its Results entry. What follows is how the rl-004
line ran, and still applies.*

Size it from a timing run at the real settings (the smoke test is too small
to show them). On the workstation one iteration of 1024 deals takes about
4.5 s (1.3 s playing, 3 s updating, and under 2 s for a 2000-deal
evaluation), so 450 iterations take about 35 minutes; on the M1 Pro it took
56 s (7 h for 450). rl-004's command (see its entry below for why; on the
laptop it ran under `caffeinate -i`):

```sh
nohup python -m learn.selfplay \
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

To stop it, `kill` the Python process: it stops the workers too, and
`--resume` carries on later. (Workers also stop by themselves if the main
process dies some other way.)

With `--critic-warmup N` the first N iterations
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
total), `--workers` and `--device` can change. (Collecting repeats exactly with the same number of workers, but the GPU's update is not bit-identical from run to run, so no two runs are anyway.) A new run refuses a directory that already holds one.

## 5. Watching it

Every 30–60 minutes read `runs/rl-004/log.jsonl`, and check that it changed
in the last few minutes (a hang shows no error). `python -m learn.curve
runs/rl-004` prints every evaluation with its paired change from the one
before and the falls in a row; with `--follow` it keeps watching and reports
new evaluations, non-finite values, skipped steps and a stalled log. One log
line per iteration:

| Field | Healthy |
|---|---|
| `vs_rulebot` ± `vs_rulebot_ci95` (every 10 iterations) | Above 0. A sanity check, not the objective: judge a run by §6's measures. `vs_rulebot_roles` splits it by declarer, partner and defender, which is **not** a measure of play (§6) |
| `value_ev` | The share of the final scores' variance the critic explains: rising from 0 |
| `value_loss` | Falling, then flat (a cross-entropy over the critic's 255 value bins, not a squared error) |
| `belief_loss` | Falling, then flat (1.39 is uniform guessing over four places; the prior of the room left in each place is about 1.26: `learn.beliefs`) |
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

## 6. After the run: measuring and choosing

**The measures** (from 27 September 2026, REVIEW.md T1):

1. **Card play on fixed contracts**, the primary measure of play:
   `learn.arena --candidate play:<policy> --field rule`. RuleBot bids and
   sets up the contract, the policy plays the cards, so the stake is the
   baseline's; about ±3.5 over 2000 deals (the full arena: ±15).
2. **The reference set**: the full game against each of a few fields,
   with several candidates on the same deals, paired (`--candidate A B
   ...`). From 28 September: **RuleBot, rl-004d's 10, rl-005's 2000 and the
   latest exploiter, x-005-2000** ("The reference set" in Results). Until
   then rl-003's 110 and x-004d-0010 stood where rl-005's 2000 and
   x-005-2000 stand, but rl-005's 2000 beats them by about 95 to 160 per
   deal, so they no longer tell candidates apart. **For the next run the
   set becomes RuleBot, rl-005's 2000 (the older anchor), rl-005's 3600
   (the best policy) and x-005-3600 (the latest exploiter)**: 3600 beats
   rl-004d's 10 by 161 per deal and x-005-2000 by 88, so they drop out, as
   rl-003's 110 and x-004d-0010 did. rl-006 used that set (29 September).
   rl-006's continuation to 4000 used RuleBot, rl-005's 3600, rl-006's 2000 and x-006-2000-s1: rl-006's 2000 beat rl-005's 2000 by 116 per deal and x-005-3600 by 62, so they had dropped out. rl-006's continuation to 6000 used RuleBot, rl-006's 2000, rl-006's 4000 and x-006-4000-s1: rl-006's 4000 beat rl-005's 3600 by 107 per deal and x-006-2000-s1 by 56, so they had dropped out. **For the next run the set becomes RuleBot, rl-006's 4000 (the older anchor), rl-006's 5400 (the best policy) and x-006-5400-s1 (the latest exploiter of the best)**: rl-006's 5400 beats rl-006's 2000 by 95 per deal, so it drops out, and x-006-4000-s1 (beaten by 33) gives way to x-006-5400-s1, the learned field the 4000 does worst against (−18.0; "rl-006, to 6000").
3. **The exploiter margin**: `learn.exploit <policy.pt> --critic <critic.pt>`
   trains a clone of the policy for 100 iterations of 1024 deals against it,
   then plays it against the policy on 4000 fresh deals (4 minutes).
   Log it for every policy that becomes the best.
4. The **belief head**, before search relies on it: `learn.beliefs`.

Against a field that bids differently, the by-role split (declarer, partner,
defender) is not play quality: a candidate that pushes RuleBot a level
higher and then passes shows the cost as "defence" (REVIEW.md §2.3).
`hybrid_arena.py`'s other decompositions are `hybrid:<auction>,<contract>,<play>`.

**Seeds.** Choosing and reporting use different deals, since the best of
several noisy estimates is biased upwards:

| Seeds | Use |
|---|---|
| 12345 | The in-run evaluation (the curve only) |
| 7, 11, 13, 17, 21, 23 | Used to choose in the rl-004 line: spent |
| 31, 32 | Chose the sweep's settings, rl-005's 2000 and the entropy check (27–28 September): spent for choosing checkpoints; one-change checks paired with a shared start may still use them |
| 33, 34 | Chose rl-005's 3600 in its continuation (28 September): spent |
| 35, 36 | Chose rl-006's 2000 (29 September): spent |
| 37, 38 | Chose rl-006's 4000 (29 September): spent |
| 39, 40 | Chose rl-006's 5400 (29 September): spent |
| **43, 44** | **Choosing** in the next run (39 and 40 chose rl-006's 5400, so its results there are biased upwards) |
| **0, 41, 42** | **Reporting** (0 is where every earlier headline number was measured) |
| 101, 102 | `learn.exploit`'s margin |

**Choosing a checkpoint.** On the choosing seeds (the table above), 2000
deals each: the candidates' card play, and their full game against each
reference field, paired with the best policy so far (rl-006's 5400 from the evening of 29 September, rl-006's 4000 from that afternoon, rl-006's 2000 before; rl-005's 3600 from the afternoon of 28 September, rl-005's
2000 that morning, rl-004d's 10 before). Choose by the smallest of the reference
fields' paired results, and break ties within the paired interval by card
play. RuleBot's full-arena score is a sanity check (drop a run that falls
below its start there), not the objective. Then report the chosen one on
seeds 0, 41 and 42 with the same commands, and its exploiter margin.
`results/rl-005/choose.sh` runs the arena part (`SEEDS`, `START`, `FROM`,
`FIELDS`, `C`, `RUN` for another run's checkpoints, and `WORKERS` choose
what it measures; "Done: rl-005 carried on", tasks 1 and 2, has examples).
As run for rl-006, on seeds 35 and 36 (the next run uses 43 and 44, and the new reference set above):

```sh
REF="rule runs/rl-005/checkpoints/policy-2000.npz runs/rl-005/checkpoints/policy-3600.npz runs/x-005-3600/policy.pt"
BEST=runs/rl-005/checkpoints/policy-3600.npz
C="runs/rl-006/checkpoints/policy-0200.npz runs/rl-006/checkpoints/policy-0400.npz"
python -m learn.arena --field rule --seeds 35 36 --deals 2000 \
    --candidate play:$BEST $(for c in $C; do echo play:$c; done)
for field in $REF; do
    python -m learn.arena --field $field --seeds 35 36 --deals 2000 --candidate $BEST $C
done
python -m learn.exploit runs/rl-006/checkpoints/policy-0400.pt \
    --critic runs/rl-006/checkpoints/critic-0400.pt --out runs/x-006-0400
```

Each arena command takes seconds per candidate on the workstation. Then, to
see how the chosen policy bids and plays:

```sh
python -m learn.arena --candidate runs/rl-004/policy.npz --field rule --deals 2000 \
    --record results/rl-004/vs-rule.jsonl
python -m learn.arena --candidate search:runs/rl-004/policy.npz --field rule --deals 200
python -m learn.report results/rl-004/vs-rule.jsonl       # how it bids and plays
python -m learn.margins runs/rl-004/policy.npz --deals 2000   # do its bids fit its play?
python -m learn.margins runs/rl-004/policy.npz --field rule --deals 2000
python -m learn.beliefs runs/rl-004/policy.npz --deals 2000
python results/rl-005/scripts/alone.py results/rl-004/vs-rule.jsonl         # alone contracts
python results/rl-005/scripts/hand_reading.py results/rl-004/vs-rule.jsonl  # does it bid its hand?
```

The tools use all cores but two (`--workers`), or 12 when their networks are
on the GPU; belief-sampled search costs about 90 s per deal on one core, so
its 200 deals take several minutes. Put the report's output in the results
write-up: it shows how the network's bidding and results differ from
RuleBot's.

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
runs stay in the ignored `runs/`, on the workstation only; only standout weights are force-added, from 29 September: see "What is committed under `runs/`" in "Next"). Commit to `training`; merge `main` in and
open a pull request to `main` only when the user asks.

## Results

All on the MacBook Pro M1 Pro, update on MPS with 8 workers, 1024 deals per
iteration (about 47k real decisions), one PPO epoch; about 56 s per
iteration. "Against RuleBot" is the duplicate advantage in points per deal,
with a 95% confidence interval; the in-run evaluations use 1000 fixed deals,
the arena 2000 other ones. Each run's `run.json`, `settings.json`,
`log.jsonl` and `evals.jsonl` are in `results/<run>/`; `python -m learn.curve
results/<run>` shows its curve.

### Summary

The same 2000 deals (`learn.arena` with its default seed) for every row. The
rules changed between rl-003 and rl-004 (a Plain contract in clubs scores as
Clubs), so rl-003's best is shown under both.

| Policy | Scoring | Against RuleBot (arena, 2000 deals) |
|---|---|---|
| Imitation of RuleBot with exploration (`bc-explore`) | old | −8.6 ± 3.8 |
| rl-002 at iteration 351 (its last) | old | +6.4 ± 9.9 |
| rl-003 at iteration 110 | old | +23.5 ± 9.1 (declarer +60.7, partner +3.7, defender −18.6) |
| rl-003 at iteration 150 | old | +21.4 ± 8.9 |
| rl-003 at iteration 180 (its last) | old | +2.5 ± 9.8 |
| rl-003 at iteration 110 | new | +18.2 ± 9.5 |
| rl-004c at iteration 150 | new | +53.7 ± 16.2 (+35.4 ± 14.6 over rl-003's 110) |
| rl-004d at iteration 10 | new | +56.5 ± 15.7 (declarer +106, partner +121, defender −90; +38.2 ± 14.2 over rl-003's 110) |
| rl-004d at iteration 100 | new | +49.0 ± 16.3 (+30.8 ± 15.1 over rl-003's 110) |
| rl-005 at iteration 2000 (workstation, 28 September) | new | +105.1 ± 20.3 (+48.6 ± 17.9 over rl-004d's 10; `results/rl-005/summary-seed0.txt`) |
| rl-005 at iteration 3600 (workstation, 28 September) | new | +121.8 ± 24.5 (+65.3 ± 22.0 over rl-004d's 10; `results/rl-005/summary-seed0-3600.txt`) |
| rl-006 at iteration 2000 (workstation, 29 September) | new | +157.9 ± 29.0 (+101.4 ± 26.5 over rl-004d's 10; `results/rl-006/summary-seed0-2000.txt`) |
| rl-006 at iteration 4000 (workstation, 29 September) | new | +171.1 ± 31.4 (+114.6 ± 30.2 over rl-004d's 10; `results/rl-006/summary-seed0-4000.txt`) |
| **rl-006 at iteration 5400** (workstation, 29 September) | new | **+175.1 ± 30.9** (**+118.6 ± 31.8 over rl-004d's 10**; `results/rl-006/summary-seed0-5400.txt`) |

**rl-006's iteration 5400 is now the best policy** (`runs/rl-006/checkpoints/policy-5400.npz`, committed; chosen and reported as §6 says, in "rl-006, to 6000" below): level with rl-006's 4000 against RuleBot and in card play, better against every learned field (+18 to +32), and its exploiter found nothing (−18.3 ± 20.3; the 4000's took +23.4 ± 20.2). Before it rl-006's 4000 was the best (from the afternoon of 29 September), then rl-006's 2000 (from the night of 29 September), then rl-005's iteration 3600 (from the afternoon of 28 September), then rl-005's 2000, and until 28
September rl-004d's iteration 10
(`runs/rl-004d/checkpoints/policy-0010.npz`). rl-004d's 10 was also the
best of eight candidates on 4000 other fresh deals (+52.7 ± 10.8, +26.5 ± 9.8
over rl-003's 110), so about 6000 fresh deals in all back it. It gained mostly
as declarer: among RuleBots it declares in 57% of seats (RuleBot itself 25%),
61% of them Flip, mostly at level 9, and makes 49% of them.

**Exploitability.** x-003 (`results/x-003/`): a fresh learner from
`bc-explore`, 100 iterations alone in one seat against three copies of
rl-003's iteration 110, improved from −60.8 to −26.5 ± 13.5 per deal against
the policy's own result in that seat (+34.3 ± 14.8). x-004
(`results/x-004/`), the same against rl-004d's iteration 10, went from −127.2
to **−120.2 ± 21.8**: +7.0 ± 14.2 in 100 iterations. It mostly learned not to
declare. So against a learner from imitation the new policy is harder to
exploit, but that is a weak attacker. x-004b, an exploiter started from the
policy itself (`--init` and `--init-critic` from rl-004d's 10, `--critic-warmup
2`; `results/x-004b/`), is the stronger test. It went from −3.7 ± 14.6 at
iteration 10 to +21.0 ± 19.0 at 90 and **+16.6 ± 19.7** at 100: paired, +20.3
± 20.1 over its 100 iterations, at the edge of significance, and nearly all of
it as declarer (+57 to +84). So a learner that starts from the policy finds
about 20 points per deal by declaring against three copies of it: its
defence, the weakest role, is exploitable to that extent. Longer, the
exploiter might find more.

**Search** (`results/arena/rl-003-0110-search.out`): belief-sampled search
over iteration 110 (8 worlds, card play only) scored +31.4 ± 28.6 over 200
deals, where the plain policy scored +38.5 ± 28.3 on the same 200 deals (from
the recorded arena run). So search adds nothing yet, as LEARNING.md expected
until the belief head is fitted to the policy. It took 62 minutes on 6 cores.
Over rl-004d's iteration 10 (`results/arena/rl-004d-0010-search.out`) the
same: +47.5 ± 47.0 over 200 deals, where the plain policy scored +63.3 ± 48.0
on the same deals (72 minutes on 6 cores, alongside an exploiter).

### The review's measures on the workstation (27 September)

The handoff checkpoints, measured as §6 now prescribes, on the reporting
seeds 0, 41 and 42 (6000 deals; `results/workstation-2026-09/reference.txt`,
made by `reference.sh` beside it):

| Policy | Card play (`play:`) | vs RuleBot | vs rl-003's 110 | vs rl-004d's 10 | vs exploiter x-004d-0010 |
|---|---|---|---|---|---|
| bc-explore | −8.0 ± 1.0 | −11.2 ± 2.3 | −75.2 ± 5.9 | −135.6 ± 10.0 | −141.1 ± 9.5 |
| rl-003's 110 | +0.6 ± 1.9 | +23.7 ± 5.6 | 0 | −42.0 ± 8.5 | −47.1 ± 7.6 |
| rl-004d's 10 | +3.9 ± 1.9 | +56.1 ± 9.0 | +67.4 ± 9.3 | 0 | −4.4 ± 7.2 |
| rl-004d's 100 | +4.9 ± 1.9 | +51.8 ± 9.2 | +70.0 ± 9.1 | −0.2 ± 6.6 | −9.0 ± 6.9 |
| x-004b (exploiter of the 10) | +3.2 ± 1.9 | +27.2 ± 10.6 | +66.4 ± 10.0 | +9.4 ± 8.0 | −3.2 ± 6.8 |
| rl-005's 2000 (28 September; "rl-005" below) | +13.3 ± 2.1 | +108.4 ± 11.8 | +162.3 ± 11.2 | +100.0 ± 11.3 | +95.1 ± 10.4 |
| rl-005's 3600 (28 September; "rl-005, continued" below) | +19.9 ± 2.1 | +129.1 ± 13.7 | | +161.3 ± 12.5 | |
| rl-006's 2000 (29 September; "rl-006" below) | +23.4 ± 2.3 | +147.3 ± 16.3 | | | |
| rl-006's 4000 (29 September; "rl-006, continued" below) | +24.8 ± 2.2 | +181.4 ± 17.8 | | | |
| **rl-006's 5400** (29 September; "rl-006, to 6000" below) | **+26.3 ± 2.3** | **+185.9 ± 17.6** | | | |

- **Card play is a little above RuleBot's**, not level with it: +3.9 ± 1.9
  over 6000 deals, almost all as partner (+13). The 100 plays the cards
  better than the 10 (+1.0 ± 0.9 paired) but loses a little elsewhere.
- **Its weakest reference is its exploiter.** A clone trained 100
  iterations against rl-004d's 10 (`learn.exploit`, results/x-004d-0010)
  beats it by **+15.5 ± 9.7** per deal (seeds 101 and 102), nearly all as
  declarer, as x-004b did (+20.3 ± 20.1 on the laptop). Exploiters overfit:
  x-004b scores only +27 against RuleBot.
- **The belief head knows little** (`learn.beliefs`,
  `results/workstation-2026-09/beliefs-rl-004d-0010.txt`): no better than
  the prior of the room left in each place during the auction and the
  set-up, 7–12% better in early play and 19% in the last five tricks, in
  self-play. A probe fitting only the head, or a small MLP, on the frozen
  trunk's summary gains nothing (1.175 → 1.166): the trunk does not carry
  the information, so fitting the head (T2.5) needs a trunk of its own.
- **Search with the critic more than triples the edge in card play** (T3.1,
  `critic-reply:`, 100 worlds; `results/workstation-2026-09/critic-search-rl-004d-0010.txt`).
  On fixed contracts over 2000 deals (seed 0): the policy +3.0 ± 3.4,
  search **+12.3 ± 4.0, paired +9.2 ± 4.2**; as declarer +27.9 (the policy
  −0.8), as defender +7.1 (−0.1). Judged straight after the card
  (`critic:`) it loses heavily (−121 ± 49 over 30 deals): the critic never
  saw positions with another seat to act. It costs about 0.3 s per searched
  decision on one core (encoding the replies' views, and copying worlds),
  so the 2000 deals took 40 minutes on 12 workers.

### The sweep (27 September, commit 903dd24)

"Next", step 1: `results/workstation-2026-09/sweep.sh`, six runs of 200
iterations from rl-004d's 10, each with one change from the control (the
league with RuleBot, the start, rl-003's 110, rl-004d's 100, x-004b and
x-004d-0010, snapshots every 10 iterations; an exploiter every 50 iterations
for 25; magnet 0.1 with `--magnet-ema 0.01`; entropy 0.01, policy lr 1e-4,
one epoch, `--explore-bids 0.15 --explore-levels 0.1 --stake-scaling`). On
the workstation, 12 workers on the GPU, 20:09–22:08: 18.4 minutes a run
(about 2.0 s playing and 2.3 s updating per iteration), 25.9 for the
two-epoch one (4.2 s updating). Each run's files are in `results/sw-*/`.

Judged by `judge.sh` on the choosing seeds 31 and 32 (4000 deals,
`results/workstation-2026-09/judge.txt`), at iteration 200. Each line is
paired with the control, per deal; **bold** is significant (the whole
interval on one side of 0). The last column is the run's own in-run
exploiters' margins, at iterations 50, 100, 150 and 200 (each about ±10):

| Run | Card play (`play:`) | vs RuleBot | vs rl-003's 110 | vs rl-004d's 10 | vs x-004d-0010 | In-run exploiters |
|---|---|---|---|---|---|---|
| `sw-control` (itself, not paired) | +7.9 ± 2.2 | +42.1 ± 13.5 | +86.6 ± 13.3 | +15.1 ± 8.9 | +0.1 ± 8.8 | −3.1, +11.4, +8.2, +8.8 |
| the start, rl-004d's 10 | −1.3 ± 1.5 | +0.3 ± 8.2 | **−15.9 ± 9.7** | **−15.1 ± 8.9** | −4.9 ± 8.2 | |
| `sw-entropy-0.03` | −0.6 ± 1.6 | +5.5 ± 7.9 | +0.3 ± 9.5 | +4.2 ± 10.8 | +7.6 ± 9.5 | +3.5, +8.9, +16.2, +20.9 |
| `sw-entropy-0.1` | +1.0 ± 1.9 | **−10.0 ± 8.2** | +1.9 ± 9.3 | −1.3 ± 11.4 | **+10.0 ± 9.8** | +7.3, +21.0, −3.9, −3.6 |
| `sw-lr-2.5e-4-2ep` | **−3.0 ± 1.7** | **−13.5 ± 8.0** | −5.0 ± 9.8 | **−12.7 ± 11.5** | **−15.6 ± 9.7** | +10.2, +0.2, +9.5, +8.7 |
| `sw-no-explore-levels` | −0.7 ± 1.6 | **−17.1 ± 8.0** | **−10.5 ± 8.8** | −8.5 ± 10.6 | −6.6 ± 9.0 | +7.5, +0.6, +2.1, +14.7 |
| `sw-no-stake-scaling` | −1.5 ± 1.5 | −6.6 ± 7.7 | −5.2 ± 9.2 | −7.0 ± 10.4 | −5.1 ± 8.6 | +9.9, +0.7, −3.7, +1.8 |

What the runs did:

- **The control improved on its start**, which no run from a best
  checkpoint had shown on fresh deals before: +15.9 ± 9.7 against a field of
  rl-003's 110 and +15.1 ± 8.9 against rl-004d's 10 itself, level against
  RuleBot (+0.3 ± 8.2) and the exploiter, and card play +1.3 ± 1.5. Its
  in-run curve was flat (+37 to +65, one paired change significant either
  way at a time, no three falls). So the league and the EMA magnet at 0.1
  did not drift in 200 iterations as rl-004b's copied magnet at 0.1 did.
- **All six logs were healthy**: no non-finite values, no skipped steps,
  the critic explaining 72–74% of the variance on average (one iteration
  fell to 28%), the league grown to 28 members, and the learner's win rates
  against members 0.45–0.49 on average over the last 20 iterations. Entropy stayed at 0.50–0.52 except where the bonus rose:
  0.55 → 0.62 over 200 iterations at 0.03, and 0.54 → 1.03 at 0.1 (magnet KL
  0.08–0.11, made 41% → 36%).
- **The faster learning rate hurt at once.** At 2.5e-4 with two epochs
  approx_kl was 0.017 (0.005) and the clip fraction 0.11 (0.044); the in-run
  score fell from about +50 to +28 in the first 10 iterations and stayed at
  +26 to +44.
- **Exploring levels matters after all.** Without it the learner's own
  contracts in self-play were lower (level 9.24, made 49%; with exploration
  its logged contracts include the explored ones, 9.5 and 43%), and it lost
  ground against RuleBot and rl-003's 110; in-run it fell to +25 by
  iteration 200. So although exploring levels never made higher contracts
  pay (the rl-004 line), without it the policy's bidding loses.

Deciding (the rules in "Next", step 2):

1. **No change is taken** (rule 1). Entropy 0.03 is significantly better in
   nothing. Entropy 0.1 is significantly better against one field only
   (x-004d-0010, +10.0 ± 9.8) and significantly worse against RuleBot. The
   learning rate 2.5e-4 with two epochs is significantly worse in card play
   and against three of the four fields.
2. **`--explore-levels` stays**: without it, significantly worse against
   RuleBot and rl-003's 110. **`--stake-scaling` goes** (rule 2): without it,
   significantly worse in none. Unrounded, its card play is −1.46 ± 1.55
   (interval −3.01 to +0.09), so this is a near thing, and all five of its
   point estimates are negative; the rule gives a tie to the simpler
   setting, and I followed it. A one-change check with it back would settle
   it ("Next").
3. Rule 3 does not apply (no entropy qualified).
4. **Rule 4**: the sweep could not show any change better, so the review's
   recipe, except what was significantly worse than the control: **entropy
   0.03** (worse in nothing) is taken; `--policy-lr 2.5e-4 --ppo-epochs 2`
   (worse in four measures) is not, so the lr stays 1e-4 with one epoch.
5. **Rule 5** does not exclude entropy 0.03. Its in-run exploiters gained
   more as the run went on (+3.5 → +20.9 ± 11.8; the control's +8.8 ± 9.3 at
   200), but that difference, +12 ± 15, is within noise; at 0.1 the
   exploiters' gains fell to −3.6; and against the fixed exploiter
   x-004d-0010 the entropy 0.03 run scores +7.6 ± 9.5 above the control.
   Watched in the long run, whose exploiters train 50 iterations.

**rl-005's settings:** the command in "Next", step 3, with `--entropy 0.03`,
`--explore-bids 0.15 --explore-levels 0.1`, without `--stake-scaling`, and the
default policy lr (1e-4) and one PPO epoch.

### rl-005 (27–28 September, commit ba22604): the long run

"Next", step 3, with the sweep's settings, started at 22:22 on the
workstation (12 workers on the GPU, the update on the GPU):

```sh
nohup python -m learn.selfplay \
    --init runs/rl-004d/checkpoints/policy-0010.pt \
    --init-critic runs/rl-004d/checkpoints/critic-0010.pt --critic-warmup 2 \
    --league-add runs/rl-003/checkpoints/policy-0110.pt runs/rl-004d/checkpoints/policy-0100.pt \
        runs/x-004b/policy.pt runs/x-004d-0010/policy.pt \
    --exploit-every 100 --exploit-iterations 50 \
    --magnet 0.1 --magnet-ema 0.01 --explore-bids 0.15 --explore-levels 0.1 \
    --entropy 0.03 \
    --deals 1024 --iterations 2100 --eval-every 10 --eval-deals 2000 \
    --out runs/rl-005 > runs/rl-005.out 2>&1 &
```

An iteration took about 5.4–5.8 s (about 2 s playing, 2.2 s updating, and
the league's loading and the evaluation every 10), and each exploiter phase
(50 iterations of 1024 deals, then its margin on 2000 deals) about 97 s.
It finished all 2100 iterations at 01:49, in 3 hours 27 minutes: 76.3
million recorded decisions (about 36,000 per iteration).

Decisions during the run:

- Iterations 10–100: in-run +55.5 to +40.3 and back to +52.0, paired changes
  within ±12 (two single significant falls, each followed by a rise);
  entropy 0.57–0.59, approx_kl about 0.005, magnet KL about 0.04, the critic
  0.65–0.77, no skipped steps. The first exploiter (iteration 100) gained
  +6.3 ± 9.9.
- Iterations 100–500: in-run flat, +31 to +61 (paired changes within ±13,
  never two significant falls in a row). The exploiters at 200, 300, 400 and
  500 gained +9.3, +1.1, +9.9 and +16.7 (each about ±11). The league was
  full (50 members) by iteration 500. Entropy rises steadily, about 0.035
  per 100 iterations (window means 0.575, 0.612, 0.646, 0.680, 0.709), with
  approx_kl easing from 0.0055 to 0.0045 and magnet KL flat at 0.031–0.034:
  the bonus at 0.03 is slowly widening the policy, as in the sweep (0.55 →
  0.62 over 200). Not a stop rule (only a collapse is), and the evaluations
  play greedily; watched against card play.
- **Card play at iteration 500** (`results/rl-005/play-check-0500.txt`, seeds
  31 and 32, 4000 deals, alongside the run with 4 workers): **+8.6 ± 2.3,
  paired +2.1 ± 1.8 over the start** (+6.6 ± 2.3), as declarer +14.2 (+7.7).
  A small gain, significant; no stop rule.
- Iterations 500–1000: the in-run score rose to +60 to +79 from about
  iteration 580 (+73.2 at 1000, the best in-run evaluation of any run);
  paired, 10 → 1000 is +17.7 ± 14.6 and 500 → 1000 +33.6 ± 11.6. In
  self-play the learner's contracts moved from Flip to Halves (window
  means: Flip 46% → 31%, Halves 26% → 39%), level steady at 9.6, made 41%.
  Entropy's rise slowed (0.758, 0.784, 0.807, 0.824 by window: +0.02 per
  100 iterations), approx_kl 0.004, magnet KL 0.026, no skipped steps,
  about 5.9 s per iteration with the exploiters. The exploiters at 600–1000
  gained +1.0, +5.5, +3.9, +2.4 and +19.3 ± 12.2.
- **Card play at iteration 1000** (`results/rl-005/play-check-1000.txt`):
  **+13.1 ± 2.4, paired +6.5 ± 2.2 over the start**: twice the start's
  edge over RuleBot, as declarer +20.6 (+7.7) and defender +6.1 (+0.2).
  Card play had been flat through the whole rl-004 line; this is the first
  clear gain since.
- Iterations 1000–1500: the in-run score kept rising, +61 to +105 (+83.4 at
  1500; paired, 10 → 1500 +27.9 ± 16.2, 1000 → 1500 +10.2 ± 13.1), with no
  two significant falls in a row. In self-play Halves grew to 43% of the
  learner's contracts and Flip fell to 26%, the level edged up to 9.7,
  made 40–41%. Entropy's rise slowed further (0.840, 0.850, 0.864, 0.872,
  0.880 by window), approx_kl 0.004, magnet KL 0.025, the critic 0.75, no
  skipped steps. The exploiters at 1100–1500 gained +6.2, +7.6, +4.7, −6.2
  and −7.7: none of the last five significant.
- **Card play at iteration 1500** (`results/rl-005/play-check-1500.txt`):
  **+15.2 ± 2.4, paired +8.6 ± 2.4 over the start** (500: +2.1, 1000:
  +6.5): still rising, as declarer +25.5 and defender +7.7.
- Iterations 1500–2100: the in-run score rose again, to +80 to +116 (+115.9
  at 2000, +96.9 at 2100; paired, 10 → 2000 +60.4 ± 17.9). Averaged over
  windows of 300 iterations it rose steadily through the whole run: +45.5,
  +52.2, +64.5, +69.7, +80.5, +90.3, +104.8. One pair of significant falls in
  a row (1700 and 1710, −12.3 and −9.9), then +26.2 ± 10.9 at 1720; the stop
  rule never fired. Entropy 0.88 → 0.92 by the end (window means +0.01–0.02
  per 100 iterations), approx_kl 0.0036, magnet KL 0.023, the critic 0.75,
  no skipped steps and no non-finite values in the whole run. The belief
  loss crept up from 1.18 to 1.20 over the run. The exploiters at 1600–2100
  gained −2.1, +10.2, +4.4, +5.2, −9.3 and +12.4 (each ±12–18): over the
  whole run none gained significantly more than the first (+6.3 ± 9.9), and
  the mean of the last ten, +1.9, is below the first ten's, +7.5.

**Choosing** (§6, `results/rl-005/choose.sh`, `choose.txt`): every 200
iterations and the last, on seeds 31 and 32 (4000 deals), paired with the
start (rl-004d's 10, which scores +6.6 ± 2.3 in card play, +42.4 against
RuleBot, +70.7 against rl-003's 110 and −4.8 against x-004d-0010 there):

| Iteration | Card play | vs RuleBot | vs rl-003's 110 | vs rl-004d's 10 | vs x-004d-0010 | Smallest |
|---|---|---|---|---|---|---|
| 200 | −0.6 ± 1.5 | −17.7 ± 8.1 | +0.6 ± 9.5 | +14.7 ± 9.3 | +6.5 ± 8.6 | −17.7 |
| 400 | +0.3 ± 1.8 | +3.5 ± 9.0 | +31.4 ± 10.2 | +27.9 ± 9.2 | +13.6 ± 8.9 | +3.5 |
| 600 | +3.4 ± 1.8 | +18.7 ± 9.9 | +32.8 ± 11.4 | +36.4 ± 11.8 | +33.5 ± 12.2 | +18.7 |
| 800 | +3.2 ± 2.0 | +14.1 ± 9.9 | +32.7 ± 11.2 | +50.0 ± 10.9 | +34.1 ± 10.7 | +14.1 |
| 1000 | +6.5 ± 2.2 | +18.9 ± 10.4 | +39.6 ± 11.6 | +45.8 ± 10.5 | +46.1 ± 11.0 | +18.9 |
| 1200 | +8.3 ± 2.3 | +29.8 ± 10.7 | +51.8 ± 11.9 | +60.2 ± 11.1 | +47.8 ± 11.4 | +29.8 |
| 1400 | +9.0 ± 2.4 | +33.4 ± 11.5 | +71.7 ± 12.7 | +78.0 ± 12.7 | +76.0 ± 13.0 | +33.4 |
| 1600 | +8.2 ± 2.4 | +30.1 ± 11.9 | +73.3 ± 12.7 | +98.2 ± 12.4 | +80.0 ± 13.0 | +30.1 |
| 1800 | +8.3 ± 2.5 | +46.4 ± 11.6 | +75.3 ± 13.1 | +100.9 ± 12.4 | +84.2 ± 12.8 | +46.4 |
| **2000** | **+11.0 ± 2.6** | **+62.1 ± 12.3** | **+85.2 ± 13.5** | **+101.9 ± 13.9** | **+89.4 ± 15.0** | **+62.1** |
| 2100 | +11.1 ± 2.7 | +47.9 ± 11.9 | +80.4 ± 13.0 | +82.2 ± 12.6 | +76.0 ± 13.1 | +47.9 |

Every field's result rises through the run, and RuleBot's is the smallest
at every checkpoint. **Iteration 2000 is chosen**: its smallest (+62.1) is the
largest, and it is no tie: against RuleBot, paired directly, it beats 2100
by 14.2 ± 9.8 and 1800 by 15.7 ± 9.6 (`choose-2000-vs-neighbours.txt`).
RuleBot's own score, the sanity check, is +104.6 ± 14.7 there (the start
+42.4).

**Reporting** iteration 2000 on seeds 0, 41 and 42 (6000 deals,
`results/rl-005/report-2000.txt`; the row is added to "The review's
measures on the workstation" above), paired with the start:

| Measure | rl-004d's 10 | rl-005's 2000 | Paired |
|---|---|---|---|
| Card play on fixed contracts | +3.9 ± 1.9 | **+13.3 ± 2.1** | **+9.4 ± 2.1** |
| Full game against RuleBot | +56.1 ± 9.0 | **+108.4 ± 11.8** | **+52.4 ± 10.2** |
| against rl-003's 110 | +67.4 ± 9.3 | **+162.3 ± 11.2** | **+94.9 ± 10.9** |
| against rl-004d's 10 | 0 | **+100.0 ± 11.3** | **+100.0 ± 11.3** |
| against x-004d-0010 | −4.4 ± 7.2 | **+95.1 ± 10.4** | **+99.5 ± 11.6** |
| Exploiter margin (`learn.exploit`, seeds 101, 102) | +15.5 ± 9.7 | **+2.6 ± 14.3** | |

- **Card play more than tripled its edge over RuleBot** (+3.9 → +13.3),
  the review's primary measure. On fixed contracts the roles are the
  baseline's, so here the split is meaningful: as declarer +19.8 (the start
  +1.2) and defender +10.8 (+1.4), partner level (+12.2, +12.9). The gain
  is in declaring and defending, the two roles where the start was level
  with RuleBot.
- **The full game roughly doubled against RuleBot**, and against the other
  learned fields it gained about 100 points per deal: the exploiter that
  took +4.4 from the start now loses 95 to the new policy.
- **Exploiter margin**: a clone trained 100 iterations against it (x-005-2000,
  `results/x-005-2000/`) ends at +2.6 ± 14.3, not significant (the start's:
  +15.5 ± 9.7). Its in-run curve against the policy wandered between −9 and
  +24 without a trend. The interval is wider than the start's, as all of
  the new policy's full-game intervals are (±11.8 against RuleBot, the
  start ±9.0): it plays for larger stakes.
- **The belief head did not improve** (`beliefs-2000.txt`): no better than
  the prior during the auction and the set-up, 7%, 10% and 14% better in
  tricks 1–4, 5–8 and 9–13 (the start: 7%, 12%, 19%). Its weight is small
  (0.1) and nothing uses it yet; T2.5's separate belief network remains the
  way to a useful one.
- **The bidding is bolder, and still runs ahead of the play**
  (`margins-2000.txt`, 2000 deals, at each seat's first bid). Among RuleBots
  it bids in 96% of seats (the start 93%), bidding beats passing by +168 ±
  21 (+127 ± 16), and a level higher loses more: −315 ± 30 (−203 ± 23), for
  every strength of hand; with no ace or Joker, bidding costs −48 ± 49
  (−19 ± 37). Its contracts are made 47% as bid, 30% one level up and 16%
  two up (49%, 32%, 17%). In self-play it bids at its first decision in only
  51% of seats, bid − pass +211 ± 49. So the gain is not from making higher
  contracts: they still do not pay.
- **How it bids among RuleBots** (`learn.report` on 2000 recorded deals,
  seed 0, `results/rl-005/report-bidding-2000.txt`; the record stays in
  `runs/arena/`): it declares in 70% of its seats (the start 57%), 78% of
  them Flip (61%), and almost only at level 9: 5300 of its 5962 contracts
  (the start's were 1020 at 8, 3666 at 9 and 244 at 10); 575 at 10. It
  makes 47% (49%) and scores +269 per contract (+243). It bids in 89–96% of
  seats whatever its hand, and its highest bid is 9.04 with no ace or Joker
  and 9.26 with five (the start 8.53 to 8.94): it opens at 9 Flip with
  nearly any hand, the start's style taken further. Alone it loses 875 per
  contract in 5.6% of seats (805, 4.9%). This is the bidding war REVIEW.md
  §2.4 calls the equilibrium of these rules, and against every field in the
  reference set it pays; the exploiter, which starts from the policy,
  did not find a counter in 100 iterations.
- **Search on top** (`play:critic-reply:`, 100 worlds, 500 deals of seed 0,
  `results/rl-005/critic-search-2000.txt`, 20 minutes on 12 workers): the
  policy's card play +8.6 ± 6.9, with search **+20.2 ± 7.7, paired +11.6 ±
  9.4**, as declarer +32.9 and defender +25.9. The start on the same deals:
  +2.2 plain, +11.5 with search. So search adds about as much to the better
  policy as to the start (its +9.2 ± 4.2 over 2000 deals), and the two gains
  add up: the best card play so far is rl-005's 2000 with critic search.

What rl-005 shows:

- **Card play learns once the loop stops working against itself.** Over
  the rl-004 line (560 iterations with restarts and a strong magnet) card
  play did not improve; over rl-005 (2100 iterations, one run, the league,
  an EMA magnet at 0.1, entropy 0.03) it rose steadily, +2.1, +6.5, +8.6
  and +11.0 over the start at 500, 1000, 1500 and 2000 on seeds 31 and 32
  (+9.4 ± 2.1 at 2000 on the reporting seeds).
- **No drift, no cycling, no collapse in 2100 iterations**, where rl-004b
  and rl-004c slid within 40–120. Every reference field's result rises
  through the run, and the in-run curve too. The league (half the deals
  against its members drawn by priority, 15% against exploiters, the rest
  plain self-play) and the moving magnet held where restarts and a stronger
  copied magnet did not.
- **Not yet seen levelling off**: the in-run curve and card play were
  still rising at the end. 76 million decisions is below the 10^8 aimed at
  (about 36,000 per iteration, not 47,500).
- **The bidding did not learn to read the hand.** The gain in the full
  game comes with ever bolder, hand-blind level-9 Flip bids; higher
  contracts still lose, and a weak hand still bids.
- **Entropy rose throughout** (0.53 → 0.93), slowing as it went; the
  greedy policy still improved, but a lower bonus, or one that decays,
  is worth a test ("Next").

### The reference set (28 September, commit f4ac0c5)

"Done: rl-005 carried on", task 2. rl-005's 2000 beats rl-003's 110,
rl-004d's 10 and x-004d-0010 by about 95 to 160 per deal, so as fields they no longer tell
candidates apart. **The reference set from now on: RuleBot, rl-004d's 10,
rl-005's 2000 and its exploiter x-005-2000** (§6). The two policies against
it on the reporting seeds 0, 41 and 42 (6000 deals, `results/rl-005/choose.sh`
with `SEEDS="0 41 42"`, `results/rl-005/reference-2026-09-28.txt`; under 2
minutes on 12 workers):

| Policy | Card play (`play:`) | vs RuleBot | vs rl-004d's 10 | vs rl-005's 2000 | vs x-005-2000 |
|---|---|---|---|---|---|
| rl-004d's 10 | +3.9 ± 1.9 | +56.1 ± 9.0 | 0 | −93.7 ± 12.2 | −85.7 ± 10.5 |
| rl-005's 2000 | +13.3 ± 2.1 | +108.4 ± 11.8 | +100.0 ± 11.3 | 0 | −7.2 ± 10.5 |
| paired, 2000 − 10 | **+9.4 ± 2.1** | **+52.4 ± 10.2** | **+100.0 ± 11.3** | **+93.7 ± 12.2** | **+78.5 ± 12.8** |

- The first two columns repeat rl-005's report exactly (same deals, same
  deterministic agents), as they should.
- **Against a field of its own exploiter, rl-005's 2000 is level**
  (−7.2 ± 10.5): one seat of the policy among three x-005-2000s, the
  reverse of `learn.exploit`'s margin (one exploiter among three policies,
  +2.6 ± 14.3). Both say the exploiter found no significant counter in 100
  iterations. rl-004d's 10 loses 86 per deal to it.
- For a candidate from the continuation, paired with rl-005's 2000, the
  fields that can tell are RuleBot, rl-005's 2000 itself (the ancestor) and
  x-005-2000; rl-004d's 10 is kept as the older anchor.
- The set is still chosen by hand; the published systems rate each
  checkpoint against their earlier ones automatically (an Elo or TrueSkill
  ladder). Worth building later, not part of this work.

### rl-005, continued (28 September, commit f4ac0c5): 2100 → 4000

"Done: rl-005 carried on", task 1, with rl-005's own settings (a resume
changes nothing else):

```sh
nohup python -m learn.selfplay --resume runs/rl-005 --iterations 4000 >> runs/rl-005.out 2>&1 &
```

From 09:46 to 12:54 on the workstation (12 workers on the GPU, the update
on the GPU, about 11 GB of GPU memory): 1900 iterations in 3 hours 8
minutes, about 5.9 s each with the exploiters. 69.3 million recorded
decisions (about 36,500 per iteration); the whole run's 145.6 million are
past the 10^8 T4.1 aimed at. The files are in `results/rl-005/` again
(`run.json` now lists the resume).

Window means of the log (`value_ev` is the critic's share of the variance;
level, made, Flip and Halves are the learner's own contracts in self-play):

| Iterations | In-run vs RuleBot | Entropy | `value_ev` | Level | Made | Flip | Halves |
|---|---|---|---|---|---|---|---|
| 1801–2100 (before) | +104.8 | 0.915 | 0.754 | 9.73 | 41% | 25% | 42% |
| 2101–2400 | +104.9 | 0.934 | 0.743 | 9.74 | 41% | 24% | 43% |
| 2401–2700 | +109.7 | 0.950 | 0.748 | 9.76 | 41% | 25% | 42% |
| 2701–3000 | +116.9 | 0.955 | 0.750 | 9.84 | 41% | 23% | 41% |
| 3001–3300 | +117.9 | 0.959 | 0.754 | 9.92 | 40% | 24% | 41% |
| 3301–3600 | +116.5 | 0.963 | 0.752 | 9.99 | 40% | 29% | 39% |
| 3601–3900 | +124.1 | 0.956 | 0.754 | 10.10 | 40% | 35% | 36% |
| 3901–4000 | +134.4 | 0.963 | 0.748 | 10.10 | 41% | 36% | 34% |

Decisions during the run:

- **Healthy throughout**: no non-finite value and no skipped step;
  approx_kl 0.0035–0.0037, magnet KL 0.022–0.023, clip fraction 0.03; the
  critic 0.73–0.76 on window means (single iterations down to 0.19 at 2293,
  back at once); the belief loss 1.202 → 1.197. **Entropy levelled off**
  at about 0.96: by 300-iteration window it rose 0.015–0.02 a window to
  2700, then under 0.005 a window, level from about 3000.
- **The in-run curve kept rising**: +77.8 to +153.7 (the best at 3750;
  +142.2 at 4000); paired, 2100 → 4000 +45.3 ± 18.2 and 2000 → 4000 +26.4 ±
  18.4. One pair of significant falls in a row (3620 and 3630, −31.6 and
  −16.5), then +30.8 ± 14.2 at 3640; the stop rule never fired.
- In self-play the learner's contracts moved back from Halves towards Flip
  after 3300 and rose to level 10.1, made 40–41%.
- **The in-run exploiters** (50 iterations, every 100) gained +4.4, −4.6,
  +21.7, −12.1, +4.9, +9.9, +5.0, +7.6, +6.8, −3.7, +0.9, −10.2, −10.8, −1.0,
  +4.5, +21.0, +1.9, −8.4 and +16.9 (each ±12–21): mean +2.9, two of the
  nineteen significant (2400: +21.7 ± 14.0; 4000: +16.9 ± 15.9).
- **Card-play checks** on the fresh choosing seeds 33 and 34 (4000 deals,
  alongside the run with 4 workers; `play-check-2500.txt`, `-3000`,
  `-3500`), where rl-005's 2000 scores +16.5 ± 2.6: at 2500 +16.8 ± 2.6,
  paired +0.4 ± 2.3; at 3000 +19.6 ± 2.7, **paired +3.1 ± 2.4**; at 3500
  +18.8 ± 2.6, paired +2.4 ± 2.5. No stop rule (none significantly
  below the 2000), and a slower gain than in the first 2000 iterations.

**Choosing** (§6, `results/rl-005/choose-4000.txt`): every 200th
iteration from 2200 and the last, on seeds 33 and 34 (4000 deals), paired
with rl-005's 2000, which there scores +16.5 ± 2.6 in card play, +112.0 ±
14.7 against RuleBot, +94.3 ± 14.1 against rl-004d's 10 and −5.6 ± 13.9
against x-005-2000:

| Iteration | Card play | vs RuleBot | vs rl-004d's 10 | vs rl-005's 2000 | vs x-005-2000 | Smallest |
|---|---|---|---|---|---|---|
| 2200 | −0.3 ± 2.1 | −12.3 ± 10.2 | +1.8 ± 12.8 | +15.6 ± 13.8 | +5.6 ± 13.9 | −12.3 |
| 2400 | +2.1 ± 2.2 | −20.8 ± 10.7 | −4.8 ± 13.5 | +8.1 ± 13.1 | +3.1 ± 14.3 | −20.8 |
| 2600 | +1.5 ± 2.3 | −8.2 ± 11.6 | +22.8 ± 15.0 | +20.7 ± 14.8 | +19.2 ± 14.4 | −8.2 |
| 2800 | +1.6 ± 2.4 | −3.7 ± 12.4 | +33.2 ± 14.6 | +29.2 ± 15.9 | +34.8 ± 15.6 | −3.7 |
| 3000 | +3.1 ± 2.4 | +8.0 ± 11.8 | +34.0 ± 14.5 | +48.6 ± 14.6 | +41.9 ± 15.3 | +8.0 |
| 3200 | +2.6 ± 2.3 | −23.8 ± 13.5 | +29.9 ± 15.6 | +60.5 ± 15.9 | +58.3 ± 16.6 | −23.8 |
| 3400 | +2.2 ± 2.5 | +3.4 ± 13.2 | +46.7 ± 15.3 | +71.5 ± 16.1 | +63.0 ± 15.5 | +3.4 |
| **3600** | **+5.7 ± 2.5** | **+19.4 ± 13.0** | **+72.4 ± 16.1** | **+81.1 ± 16.0** | **+90.6 ± 16.8** | **+19.4** |
| 3800 | +4.7 ± 2.6 | +22.9 ± 13.1 | +72.1 ± 16.9 | +94.1 ± 15.9 | +101.2 ± 17.3 | +22.9 |
| 4000 | +3.6 ± 2.6 | +30.9 ± 13.6 | +75.0 ± 16.8 | +85.1 ± 16.4 | +100.0 ± 17.3 | +30.9 |

RuleBot is the smallest field at every checkpoint again, and it is where
the continuation gained least: level or below the 2000 up to 3400, while
against the learned fields the gain rose steadily to +70 to +100. By the
smallest, 4000 leads (+30.9), but 3600 and 3800 are within its interval;
**paired directly with 4000** (`choose-4000-vs-neighbours.txt`) they are
ties on every field (against RuleBot 3600 −11.5 ± 12.4, 3800 −8.0 ± 11.5;
none of the eight results significant). §6 breaks a tie by card play,
where 3600 is best (+5.7 over the 2000; paired with 4000, +2.1 ± 2.1), so
**iteration 3600 is chosen**. Its unpaired score against RuleBot, the
sanity check, is +131.4 ± 16.7 (the 2000: +112.0 ± 14.7).

**Reporting** iteration 3600 on seeds 0, 41 and 42 (6000 deals,
`report-3600.txt`, paired with rl-005's 2000; `report-3600-vs-rl-004d-0010.txt`,
paired with rl-004d's 10):

| Measure | rl-004d's 10 | rl-005's 2000 | rl-005's 3600 | 3600 − 2000 | 3600 − rl-004d's 10 |
|---|---|---|---|---|---|
| Card play on fixed contracts | +3.9 ± 1.9 | +13.3 ± 2.1 | **+19.9 ± 2.1** | **+6.7 ± 2.1** | **+16.0 ± 2.4** |
| Full game against RuleBot | +56.1 ± 9.0 | +108.4 ± 11.8 | **+129.1 ± 13.7** | **+20.7 ± 10.6** | **+73.1 ± 12.1** |
| against rl-004d's 10 | 0 | +100.0 ± 11.3 | **+161.3 ± 12.5** | **+61.3 ± 12.7** | **+161.3 ± 12.5** |
| against rl-005's 2000 | −93.7 ± 12.2 | 0 | **+79.9 ± 13.0** | **+79.9 ± 13.0** | **+173.6 ± 15.0** |
| against x-005-2000 | −85.7 ± 10.5 | −7.2 ± 10.5 | **+87.9 ± 11.7** | **+95.1 ± 12.9** | **+173.6 ± 14.1** |
| Exploiter margin (`learn.exploit`, seeds 101, 102) | +15.5 ± 9.7 | +2.6 ± 14.3 | **+32.0 ± 15.4** | | |

- **Card play gained again**, +6.7 ± 2.1 over the 2000 (five times the
  start's edge over RuleBot now: +19.9 against +3.9), most of it as
  declarer: +40.9 (the 2000 +19.8), partner +14.5 (+12.2), defender +12.5
  (+10.8).
- **The full game gained against every reference field**, least against
  RuleBot (+20.7 ± 10.6) and most against the learned ones (+60 to +95).
- **But the exploiter margin rose to +32.0 ± 15.4**, significant, where
  the 2000's was +2.6 ± 14.3 (`results/x-005-3600/`, `margin.txt` for the
  roles). The clone's result against the policy climbed steadily, +5.1 at
  its iteration 10, +17 to +39 from 30 to 90 and +62.1 at 100 (in-run,
  ±20–33), where x-005-2000's wandered around 0; about 70% of its margin
  comes as declarer (+57.7 in 38.7% of its seats; x-005-2000 declared in
  27.3%). So 1600 more iterations made the policy better against every
  fixed field and, on one exploiter each, more exploitable by a clone that
  declares more against it. The league's own 50-iteration exploiters found
  significant margins twice (2400, 4000) but little against the policy
  near 3600 (+4.5 ± 17.8 at 3600); whether longer exploiters would have
  caught it is a hypothesis ("Next").
- For information, not chosen: **iteration 4000** on the same deals scores
  card play +19.7 ± 2.2 (+6.5 ± 2.0 over the 2000), and +141.6 ± 14.0
  against RuleBot (+33.1 ± 11.0), +81.9, +104.7 and +104.6 over the 2000
  against the three learned fields: higher point estimates than 3600's in
  the full game, by 21 and 25 against rl-004d's 10 and the 2000. The two
  were not paired directly on these seeds; on the choosing seeds they tie
  (`choose-4000-vs-neighbours.txt`). The choice stands, as the reporting
  seeds do not choose; 4000 is also the resume state's policy.
- **The belief head is unchanged** (`beliefs-3600.txt`): no better than the
  prior before the play, 7.6%, 10.4% and 14.0% better in tricks 1–4, 5–8
  and 9–13 (the 2000: 7.2%, 10.3%, 14.0%).
- **Search adds little on top of 3600** (`critic-search-3600.txt`,
  `play:critic-reply:`, 100 worlds, 500 deals of seed 0, 48 minutes on 4
  workers alongside the entropy check): the policy's card play +14.0 ± 6.4,
  with search +17.7 ± 6.7, paired +3.7 ± 7.9. On the same deals the
  2000 goes from +8.6 to +18.7 ± 7.8 with search (+10.1 paired; +20.2 in its
  own run, the worlds being sampled). So with search the two play about
  equally well: the continuation learned much of what search added, and
  over 500 deals critic search is not shown to add anything to this policy.
- The bidding and the alone contracts are in "Bidding and alone contracts"
  below: among RuleBots 3600 declares in 76% of its seats, 81% of them
  Flip, 87% at level 9, and a level higher still loses for every strength
  of hand.

What the continuation shows:

- **Still improving, but more slowly**: card play +6.7 from 2000 to 3600
  (after +9.4 in the first 2000 iterations) and level since; the in-run
  curve +26 from 2000 to 4000 after +60 from 10 to 2000, and still rising
  at 4000, as was the full game against RuleBot. Against RuleBot, the one
  field that does not learn, the gain was small and came late (from 3600).
- **Exploitability is now the weakest point.** A 100-iteration clone finds
  +32 per deal, mostly by declaring more; the league's 50-iteration
  exploiters did not show it near 3600. Confirming it, then longer
  exploiters in the league and x-005-3600 as a league member and reference
  field, are the natural next steps ("Next").

### Bidding and alone contracts (28 September): REVIEW.md T4.3 and T4.2

"Done: rl-005 carried on", task 3. Analysis only: from 2000 recorded deals (seed 0) of rl-005's
2000 and of the chosen 3600, each in one seat among three RuleBots, with
RuleBot itself in the same seats of the same deals (the all-RuleBot lineup)
as the baseline. Everything is per played seat (7928 of 8000; 72 were
thrown in) unless it says otherwise. New:

- `learn.report` now splits each agent's alone contracts by cause (RULES.md
  §5: it called an ace it held, or the called ace was in the cat) and counts
  the own-ace calls where another legal call was an ace it did not hold
  ("with a choice"); `tests/test_evaluate.py` checks both on four deals.
- `results/rl-005/scripts/alone.py` gives the detail (`alone-2000.txt`,
  `alone-3600.txt`), and `results/rl-005/scripts/hand_reading.py` how the
  bidding depends on the hand (`hand-reading-2000.txt`, `-3600.txt`). Both
  were checked by independent recomputation from the engine's replays; it
  found one real error (Halves contracts put in the wrong row of "when it
  learnt it was alone", fixed) and some overstated claims (fixed).

**Alone contracts (T4.3)** (`alone-2000.txt`, `alone-3600.txt`):

| | rl-005's 2000 | rl-005's 3600 | RuleBot, same seats |
|---|---|---|---|
| Contracts declared | 5962 | 6058 | 1982 |
| Alone | 441 (5.6% of seats), −875 each | 450 (5.7%), −783 | 219 (2.8%), +2 |
| by calling its own ace | 7, **all forced** (it held all four), +1166 | 5, all forced, −576 | 33 (17 with a choice), −237 |
| because the ace was in the cat | 434 (5.5%), −907, made 15% | 445 (5.6%), −785, made 18% | 186 (2.3%), +44, made 40% |
| called ace not held, found in the cat | 7.3% ± 0.7% | 7.4% ± 0.7% | 9.5% ± 1.3% |
| points per seat from ace-in-cat contracts | −49.7 | −44.1 | +1.0 |
| the same seats' deals, against RuleBot there, per seat of all 8000 | −53.1 | −47.6 | |

- **The cause is the cat, not the call.** The policy never called its own
  ace when it had a choice (0 of 7, 0 of 5; RuleBot did in 17 of its 33,
  all Plain contracts where the one ace it lacked was in the suit it then
  named trumps). When it calls an ace it does not hold, the ace is in the
  cat 7.3–7.4% of the time: chance, since the other 42 cards hold it with
  probability 3/42 = 7.1%. RuleBot's calls find the cat more often (9.5%),
  probably because of which auctions it wins. Nobody can know at the call,
  which comes before the Flip and the exchange. So **extending
  `learn.margins` to the call is not needed** (task 3 made it conditional
  on own-ace calls with a choice being common).
- **It plays alone twice as often as RuleBot only because it declares
  three times as often** (5962 contracts against 1982; 1752 of them with no
  ace), and alone contracts cost it far more: 82–85% fail, 3.1 tricks short
  on average (its partnered contracts: 50% fail, 2.1 short), at triple
  settlement. By its aces the alone result is −1411, −1004, −318 and +525
  with 0, 1, 2 and 3 (the 2000); by high-card points, −1956 to −2080 per
  alone contract below 7 points and positive only from 19. By Jokers the
  rate is flat (6.5–7.5% with 0 to 2). By kind, most are Flip (331 of 441
  at the 2000, −1016 each; 350 of 450 at 3600, −839); alone in Clubs is
  rare and mixed (33 at +145 at the 2000, 14 at −326 at 3600); by level, nearly all are at 9 (388 of
  441, −1011). The same seats' deals are not bad deals: RuleBot scores +71
  on them.
- **It usually knows before the exchange**: the Flip turned the called ace
  up in 49% (the 2000) and 55% (3600) of these contracts, and in Halves it
  is asked to name trumps itself in 12–16%; only 30–40% learn it when they
  pick up the cat. Those that know early do worst (Flip −954 to −1147).
- **Price**: about 48–53 points per seat (of all 8000) against RuleBot in
  the same seats, about a third of what the rest of its game gains (+158
  per seat on the other deals, the 2000). It is part of what bidding hand-blind at level 9
  costs, and `learn.margins`' bid − pass already includes it.

**The bidding (T4.2)** (`margins-3600.txt`, `hand-reading-*.txt`,
`report-bidding-3600.txt`). `learn.margins` at each seat's first bid,
2000 deals: what bidding gained over passing, and what one level higher
would have gained, by the seat's aces and Jokers:

| Aces + Jokers | Seats | bid − pass, 2000 | bid − pass, 3600 | +1 − bid, 2000 | +1 − bid, 3600 |
|---|---|---|---|---|---|
| 0 | 1039 | −48 ± 49 | −46 ± 54 | −433 ± 76 | −382 ± 80 |
| 1 | 2706 | +110 ± 33 | +127 ± 37 | −345 ± 49 | −347 ± 53 |
| 2 | 2564 | +204 ± 36 | +210 ± 44 | −296 ± 50 | −325 ± 66 |
| 3 | 1259 | +328 ± 58 | +364 ± 73 | −202 ± 89 | −240 ± 102 |
| 4+ | 360 | +425 ± 137 | +447 ± 166 | −283 ± 199 | −247 ± 234 |
| all, among RuleBots | 7928 | +168 ± 21 | +182 ± 25 | −315 ± 30 | −323 ± 36 |
| all, in self-play | 7928 | +211 ± 49 | +304 ± 49 | −240 ± 55 | −286 ± 68 |

How much the bidding reads the hand (`hand_reading.py`; high-card points
count A 4, K 3, Q 2, J 1 and a Joker 4):

| | rl-005's 2000 | rl-005's 3600 | RuleBot, same seats |
|---|---|---|---|
| Bids at all | 95.7% | 96.5% | 55.6% |
| r(high-card points, bids at all) | 0.02 | 0.03 | 0.50 |
| r(aces + Jokers, bids at all) | −0.03 | −0.05 | 0.51 |
| seats that bid: r(high-card points, highest bid) | 0.20 | 0.25 | 0.31 |
| seats that bid: r(longest suit, highest bid) | 0.35 | 0.35 | 0.43 |
| seats that bid: highest bid per 10 high-card points | +0.15 | +0.25 | +0.60 |
| where it bid and RuleBot passed, 0–6 points: its seat's score minus RuleBot's | −76 ± 54 | −83 ± 58 | |
| the same, 10–12 points | +97 ± 50 | +151 ± 49 | |

- **A level higher still loses, for every strength of hand**: among
  RuleBots significantly in every row, at 2000 and at 3600 (−323 ± 36
  overall; −247 ± 234 even with four or more aces and Jokers); in
  self-play too, though with four or more the interval is wide (−311 ±
  430). Its contracts are made 48% as bid, 33%
  one level up and 18% two up. So **the bidding-first curriculum is not
  indicated**: LEARNING.md's condition (a level higher pays for strong
  hands and the policy does not bid it) does not hold.
- **Whether it bids still ignores the hand**: it bids in 96% of seats
  whatever it holds (correlation with strength 0.02–0.03; RuleBot 0.50).
  The level it reaches reads the hand a little, and a little more at 3600
  (+0.25 per 10 points among seats that bid, against RuleBot's +0.60), and
  the longest suit most (0.35; RuleBot 0.43). Among RuleBots it declares
  (alone included) in 76% of seats at 3600, 81% of them Flip, 87% at level
  9 (the 2000: 75%, 78%, 89%); at 3600 a few more reach 11 (286 contracts,
  5%; the 2000: 84).
- **What hand-blind bidding costs is small.** By its own counterfactual,
  passing the weakest hands (no ace or Joker, 13% of seats) would gain
  about +46 ± 54 each, not significant: about 6 per seat (at most about
  13); in self-play bidding there is +96 ± 101 (not significant; the
  2000: +150 ± 111). Against RuleBot's pass with
  the same cards it loses −83 ± 58 at 0–6 points but gains from 10 points
  up; that comparison includes its card play and RuleBot's replies, so it
  is not the bid's value alone.

**Recommendation.** No change to training for the bidding or the call.
The hand-blind 9 Flip opening is close to a best response to both fields
it meets (RuleBot and itself); a level higher does not pay, so there is
nothing for a bidding curriculum to find. The alone contracts are the
cat's 7% at triple stakes on hands it should arguably not declare, already
priced into its bids. Keep rerunning `learn.margins` and `hand_reading.py`
at each chosen checkpoint: if +1 − bid turns positive for strong hands,
or bid − pass turns significantly negative for weak ones, revisit. The
place where bidding may yet lose is against an opponent that exploits it:
x-005-3600 wins mostly by declaring more against it (see "rl-005,
continued").

### The entropy check (28 September, commit 98b046f)

"Done: rl-005 carried on", task 4: `results/rl-005/entropy-check.sh`, two runs of 200
iterations from rl-005's 2000 (policy and critic) with rl-005's other
settings, exploiters every 50 iterations for 25, and a league seeded with
rl-004d's 10, rl-005's 1000, x-004d-0010 and x-005-2000: the control at
rl-005's 0.03 and one at 0.01. On the workstation, 13:21–14:11, 12 workers
on the GPU: 25.8 and 23.8 minutes (about 7–8 s an iteration, slower than
the sweep's 5.5 s because the critic search ran alongside on 4 workers).
Each run's files are in `results/ent-*/`. Their `run.json` says
`98b046f-dirty`: the working tree then held this write-up and the
`learn.report` change, which self-play does not use.

Judged by `judge.sh` on the choosing seeds 31 and 32 (4000 deals,
`results/rl-005/entropy-judge.txt`), at iteration 200, each line paired with
the control; **bold** is significant. The last column is the run's own
in-run exploiters' margins at 50, 100, 150 and 200 (each about ±13):

| Run | Card play (`play:`) | vs RuleBot | vs rl-004d's 10 | vs rl-005's 2000 | vs x-005-2000 | In-run exploiters |
|---|---|---|---|---|---|---|
| `ent-control` (itself, not paired) | +16.3 ± 2.5 | +89.4 ± 14.5 | +102.1 ± 13.4 | +18.0 ± 12.7 | +15.5 ± 12.0 | −11.7, +0.1, +22.6, +8.1 |
| the start, rl-005's 2000 | +1.4 ± 2.0 | **+15.2 ± 9.3** | −0.2 ± 12.4 | **−18.0 ± 12.7** | **−16.1 ± 12.7** | |
| `ent-0.01` | +1.4 ± 2.1 | **+20.6 ± 9.4** | +7.6 ± 12.0 | +0.0 ± 14.0 | +1.1 ± 12.6 | −5.7, +5.7, −2.3, −5.0 |

What the runs did:

- **Both logs were healthy**: no non-finite values, no skipped steps, the
  critic 0.72–0.76 on 50-iteration means, the league grown to 28 members.
  **Entropy at 0.01 fell steadily without collapsing**: 0.89, 0.88, 0.86
  and 0.85 by 50-iteration window (the start 0.92); the control's rose
  slightly, 0.93 to 0.94. In-run both scored about +100 to +113, 0.01 a
  little higher in each window.
- **The control improved on its start against two of the three learned
  fields** (+18.0 against a field of the start itself and +16.1 against its
  exploiter; level against rl-004d's 10) but
  **lost ground against RuleBot** (−15.2 ± 9.3): the same pattern as the
  continuation, where RuleBot was the field that gained least.
- **At 0.01 the run kept its RuleBot score** (+20.6 ± 9.4 over the control,
  about +5 over the start) and matched the control against every learned
  field; card play +1.4 ± 2.1 (interval −0.7 to +3.5). Its in-run exploiters
  gained less on average (−1.8, the control's +4.8; the difference not
  significant); the control's at 150 (+22.6 ± 12.5) was the only
  significant one.

**Deciding** (task 4's rule): 0.01 is taken for the next new run only if
it is significantly better in card play or against at least two reference
fields, and significantly worse in none. It is significantly better against
one field (RuleBot) and worse in none, and its card play is not significant.
**So entropy 0.03 stays** (a tie keeps the current setting). Rule 5 would
not have excluded it: its exploiters gained no more and its entropy did not
collapse. It is a near thing in the direction of 0.01: by §6's choosing
criterion (the smallest of the reference fields, paired with the start) the
0.01 run's smallest is about +5 (RuleBot) where the control's is −15.2 (also
RuleBot), and all four of 0.01's full-game point estimates are at or above
the control's. A longer check, or a new run with 0.01, would settle it
("Next"). A resumed run keeps its own settings, so this decides only the
next new run.

### Exploitability, confirmed or not (29 September, commit 12e1d55)

"Next", step 1: a second 100-iteration exploiter against rl-005's 3600 on seed 1, and one of 200 iterations against rl-005's 2000 (seed 0), one after the other on the workstation (the GPU, default workers), 00:34–00:42, 146 s and 290 s of training plus the margins. Neither skipped a step. The files are in `results/x-005-3600-s1/` and `results/x-005-2000-200/` (`margin.txt` for the roles).

| Exploiter | Against | Iterations | Seed | Margin (seeds 101, 102, 4000 deals) | Declarer: seats, advantage | Partner | Defender |
|---|---|---|---|---|---|---|---|
| x-005-2000 | rl-005's 2000 | 100 | 0 | +2.6 ± 14.3 | 27.3%, +19.4 | +2.4 | −7.1 |
| **x-005-2000-200** | rl-005's 2000 | **200** | 0 | **+10.9 ± 16.7** | 32.2%, −5.4 | +22.7 | +16.6 |
| x-005-3600 | rl-005's 3600 | 100 | 0 | +32.0 ± 15.4 | 38.7%, +57.7 | +29.6 | +9.4 |
| **x-005-3600-s1** | rl-005's 3600 | 100 | **1** | **+18.6 ± 15.4** | 32.3%, +54.9 | −9.3 | +6.8 |

In-run `vs_target` (1000 deals of seed 12345, each about ±20–34):

| Iteration | x-005-2000 | x-005-2000-200 | x-005-3600 | x-005-3600-s1 |
|---|---|---|---|---|
| 10 | +18.4 | −5.4 | +5.1 | −5.6 |
| 30 | −3.3 | −9.1 | +20.4 | +16.6 |
| 50 | +2.8 | −19.5 | +33.9 | +29.2 |
| 70 | +23.6 | −20.4 | +36.9 | +19.4 |
| 100 | +9.2 | −14.9 | +62.1 | +33.1 |
| 130 | | +9.6 | | |
| 160 | | +12.4 | | |
| 200 | | +14.3 | | |

- **Confirmed.** x-005-3600-s1's margin, +18.6 ± 15.4, is significant (its interval, +3.2 to +34.0, is above 0), and it comes the same way as x-005-3600's: as declarer, +54.9 in 32.3% of its seats (x-005-3600: +57.7). The two 3600 exploiters pooled are about +25 per deal. Its in-run curve rose from 20 iterations on, as x-005-3600's did.
- **The 2000 stays hard to exploit with twice the training.** x-005-2000-200's +10.9 ± 16.7 is not significant; its in-run curve was below 0 for 120 iterations and about +10 after, and it wins nothing as declarer (−5.4), only as partner and defender. So 3600 is more exploitable than the 2000, by an opponent that declares more against it, and 100 iterations were not merely lucky in reaching it there.
- Both ran after the throughput changes (a compiled update with a bfloat16 trunk; PERFORMANCE.md), x-005-2000 and x-005-3600 before them; same method and budget otherwise.

So step 2, rl-006 with 100-iteration exploiters in the league, goes ahead.

### rl-006 (29 September, commit 52de0e1): longer exploiters in the league

"Next", step 2: from rl-005's 3600 (policy and critic) with rl-005's settings and one change, `--exploit-iterations 100` (rl-005: 50), and a league seeded with rl-005's 2000 and 4000, x-005-2000 and x-005-3600:

```sh
nohup python -m learn.selfplay \
    --init runs/rl-005/checkpoints/policy-3600.pt \
    --init-critic runs/rl-005/checkpoints/critic-3600.pt --critic-warmup 2 \
    --league-add runs/rl-005/checkpoints/policy-2000.pt runs/rl-005/checkpoints/policy-4000.npz \
        runs/x-005-2000/policy.pt runs/x-005-3600/policy.pt \
    --exploit-every 100 --exploit-iterations 100 \
    --magnet 0.1 --magnet-ema 0.01 --explore-bids 0.15 --explore-levels 0.1 \
    --entropy 0.03 \
    --deals 1024 --iterations 2000 --eval-every 10 --eval-deals 2000 \
    --out runs/rl-006 > runs/rl-006.out 2>&1 &
```

From 00:42 to 02:35 on the workstation (22 workers, the networks on the GPU, about 4 GB of GPU memory): 2000 iterations in 1 hour 53 minutes, about 2.1 s each (collecting 1.0 s, updating 1.1 s) and 3.4 s with the exploiter phases (an exploiter of 100 iterations and its margin took about 2 minutes every 100). 73.5 million recorded decisions (about 36,800 per iteration). The files are in `results/rl-006/`.

Window means of the log (as in "rl-005, continued"; level, made, Flip and Halves are the learner's own contracts in self-play):

| Iterations | In-run vs RuleBot | Entropy | `value_ev` | Level | Made | Flip | Halves |
|---|---|---|---|---|---|---|---|
| 1–250 | +116.2 | 0.949 | 0.752 | 10.16 | 40% | 36% | 37% |
| 251–500 | +128.4 | 0.949 | 0.763 | 10.23 | 40% | 40% | 33% |
| 501–750 | +139.8 | 0.944 | 0.759 | 10.30 | 40% | 45% | 30% |
| 751–1000 | +152.1 | 0.953 | 0.756 | 10.28 | 40% | 47% | 29% |
| 1001–1250 | +146.1 | 0.956 | 0.762 | 10.30 | 40% | 47% | 29% |
| 1251–1500 | +147.9 | 0.946 | 0.765 | 10.34 | 40% | 49% | 28% |
| 1501–1750 | +133.0 | 0.946 | 0.762 | 10.38 | 41% | 47% | 30% |
| 1751–2000 | +130.4 | 0.946 | 0.762 | 10.38 | 41% | 47% | 30% |

Decisions during the run:

- **Healthy throughout**: no non-finite value and no skipped step; approx_kl about 0.0035 (at most 0.0062), magnet KL 0.022 (at most 0.033), clip fraction 0.03; the critic 0.75–0.77 on window means (single iterations down to 0.53 at 971); the belief loss 1.199 → 1.190; entropy level at 0.94–0.96.
- **The in-run curve rose, then levelled**: +99.3 at 10, +151.1 at 2000; paired, 10 → 1000 +41.3 ± 20.3, 10 → 2000 +51.8 ± 22.5, 1000 → 2000 +10.5 ± 23.7. The window means peaked at 751–1000. One pair of significant falls in a row (1780 and 1790, −18.4 and −39.6, down from the run's high window), then +27 at 1800; the stop rule never fired.
- In self-play the learner's contracts moved further from Halves to Flip (36% → 47–49%) and up to level 10.4, made 40–41%.
- **The in-run exploiters** (100 iterations, every 100) gained −3.4, +4.5, −4.9, +1.4, +7.0, **+25.6**, +10.5, −15.7, −9.1, +3.4, +9.4, +8.7, −5.7, −2.7, +15.4, +11.3, +21.8, +1.6, +7.0 and +16.4 (each ±18–24): mean +5.1, one of the twenty significant (600: +25.6 ± 23.5; the next, at 700, +10.5). rl-005's 50-iteration ones: mean +2.9, two of nineteen significant. The last five average +11.6.
- **The league lost its seeded members early.** It filled to 50 by about iteration 450, and `--league-add`'s members join as snapshots of iteration 0, which the league thins first; exploiters it never thins. So rl-005's 2000 and 4000, x-005-2000, x-005-3600 and the start all left between iterations 409 and 441, and at 2000 the league held RuleBot, the twenty exploiters and 29 snapshots. The exploiters were drawn most (by priority). The learner met x-005-3600, the exploiter that found +32, for about a fifth of the run.
- **Card-play checks** on the fresh choosing seeds 35 and 36 (4000 deals, alongside the run with 4 workers; `play-check-0500.txt`, `-1000`, `-1500`), where rl-005's 3600 scores +24.0 ± 2.8: at 500 +21.7 ± 2.9, **paired −2.3 ± 2.1** (significantly below: one of the two in a row the stop rule needs); at 1000 +26.2 ± 3.1, paired +2.1 ± 2.3; at 1500 +25.6 ± 3.0, paired +1.6 ± 2.3. No stop rule.

**Choosing** (§6, `results/rl-006/choose.txt`): every 200th iteration and the last, on seeds 35 and 36 (4000 deals), paired with rl-005's 3600, which there scores +24.0 ± 2.8 in card play, +120.5 ± 17.0 against RuleBot, +71.9 ± 16.1 against rl-005's 2000 and −18.8 ± 15.0 against x-005-3600:

| Iteration | Card play | vs RuleBot | vs rl-005's 2000 | vs rl-005's 3600 | vs x-005-3600 | Smallest |
|---|---|---|---|---|---|---|
| 200 | +0.3 ± 1.9 | −10.7 ± 12.4 | +30.0 ± 16.5 | +23.1 ± 14.7 | +22.8 ± 16.0 | −10.7 |
| 400 | +0.7 ± 2.1 | −3.2 ± 12.1 | +30.0 ± 15.1 | +16.4 ± 14.2 | +19.4 ± 15.7 | −3.2 |
| 600 | −0.7 ± 2.2 | −5.9 ± 15.1 | +40.5 ± 18.5 | +44.1 ± 18.0 | +60.0 ± 19.0 | −5.9 |
| 800 | −0.3 ± 2.4 | +9.1 ± 14.4 | +53.1 ± 18.9 | +59.0 ± 17.6 | +68.6 ± 19.6 | +9.1 |
| 1000 | +2.1 ± 2.3 | +16.2 ± 14.9 | +50.5 ± 18.8 | +49.3 ± 16.7 | +55.0 ± 18.5 | +16.2 |
| 1200 | +1.3 ± 2.2 | +30.3 ± 15.3 | +63.2 ± 19.2 | +52.5 ± 17.9 | +61.3 ± 20.1 | +30.3 |
| 1400 | +1.0 ± 2.3 | +14.3 ± 15.1 | +71.6 ± 19.5 | +66.5 ± 17.3 | +87.1 ± 19.6 | +14.3 |
| 1600 | +1.8 ± 2.3 | +1.7 ± 16.4 | +71.5 ± 19.7 | +79.2 ± 18.2 | +82.8 ± 20.4 | +1.7 |
| 1800 | +3.9 ± 2.3 | −4.0 ± 17.2 | +70.2 ± 21.1 | +86.2 ± 18.7 | +88.2 ± 21.4 | −4.0 |
| **2000** | **+2.8 ± 2.4** | **+29.0 ± 16.6** | **+63.2 ± 19.4** | **+55.9 ± 17.8** | **+66.9 ± 20.0** | **+29.0** |

RuleBot is again the smallest field at every checkpoint; against the three learned fields every checkpoint from 800 gains +49 to +88, against RuleBot −11 to +30. By the smallest, **1200 leads** (+30.3 ± 15.3). 2000 (+29.0) and 1000 (+16.2) are within its interval, so they were **paired directly with 1200** (`choose-1200-vs-neighbours.txt`): 2000 is tied with it on every measure (card play +1.5 ± 2.2, the four fields −1.3 ± 15.1, +0.0 ± 17.9, +3.4 ± 17.5, +5.6 ± 17.7); 1000 is not (against RuleBot −14.1 ± 13.9, significant). **The exploiters decide between the tied** (seed 0, 100 iterations; `results/x-006-1200/`, `results/x-006-2000/`): x-006-1200 +4.2 ± 19.1 (declarer −8.5, partner −6.2, defender +17.7), x-006-2000 +8.9 ± 18.6 (declarer +66.2 in 29.2% of seats, partner +12.4, defender −29.0). Neither is significant, so card play decides and **iteration 2000 is chosen** (+1.5 ± 2.2 over 1200 paired directly; +2.8 against 1200's +1.3 over 3600). Its unpaired score against RuleBot, the sanity check, is +149.6 ± 19.9 (3600: +120.5 ± 17.0). It is also the resume state's policy.

**Reporting** iteration 2000 on seeds 0, 41 and 42 (6000 deals, `report-2000.txt`, paired with rl-005's 3600, rl-005's 2000 alongside):

| Measure | rl-005's 2000 | rl-005's 3600 | **rl-006's 2000** | rl-006's 2000 − rl-005's 3600 |
|---|---|---|---|---|
| Card play on fixed contracts | +13.3 ± 2.1 | +19.9 ± 2.1 | **+23.4 ± 2.3** | **+3.5 ± 2.0** |
| Full game against RuleBot | +108.4 ± 11.8 | +129.1 ± 13.7 | **+147.3 ± 16.3** | **+18.1 ± 13.7** |
| against rl-005's 2000 | 0 | +79.9 ± 13.0 | **+116.5 ± 15.4** | **+36.6 ± 16.7** |
| against rl-005's 3600 | −88.1 ± 12.8 | 0 | **+39.5 ± 14.9** | **+39.5 ± 14.9** |
| against x-005-3600 | −74.8 ± 14.2 | +1.6 ± 12.5 | **+61.9 ± 14.8** | **+60.3 ± 16.7** |
| Exploiter margin, seed 0 (`learn.exploit`, seeds 101, 102) | +2.6 ± 14.3 | +32.0 ± 15.4 | +8.9 ± 18.6 (took part in choosing) | |
| Exploiter margin, seed 1 | | +18.6 ± 15.4 | **+18.5 ± 17.7** | |

- **Card play gained again**, +3.5 ± 2.0 over 3600, mostly as declarer (+48.1; 3600 +40.9) and defender (+15.7; +12.5), partner level (+14.6).
- **The full game gained against every reference field**, least against RuleBot (+18.1 ± 13.7) and most against x-005-3600 (+60.3 ± 16.7): the exploiter that took +32 from 3600 now loses 62 a deal to rl-006's 2000. On the Summary's seed-0 deals it scores +157.9 ± 29.0 against RuleBot, +101.4 ± 26.5 over rl-004d's 10 (`summary-seed0-2000.txt`).
- **But its own exploiter margin did not fall.** Its seed-1 exploiter (`results/x-006-2000-s1/`) takes **+18.5 ± 17.7**, just significant, as 3600's seed-1 did (+18.6 ± 15.4), and in the same way: as declarer, +69.7 in 27.3% of its seats (partner +26.1, defender −13.7). Its in-run curve rose to +44.1 at 70 and ended at +7.9. Pooled over the two seeds, rl-006's 2000 gives up about +14 and rl-005's 3600 about +25, but with one or two exploiters each (±15–19 apiece) that difference is not significant.
- **The belief head is unchanged** (`beliefs-2000.txt`): no better than the prior before the play, then 7.9%, 10.9% and 14.5% better (3600: 7.6%, 10.4%, 14.0%).
- **Its bidding now reads the hand**, for the first time in these runs (`hand-reading-2000.txt`, `report-bidding-2000.txt`, `alone-2000.txt`, `margins-2000.txt`; among RuleBots, 2000 deals of seed 0). It bids at all in 76% of its seats (3600: 96%; RuleBot 56%), more with more high cards (68% at 0–6 HCP, 92% at 19+; 3600: 96% and 98%); among seats that bid, its highest bid rises +0.65 a level per 10 HCP (3600: +0.25; RuleBot: +0.60), r(HCP, highest bid) 0.49 (3600: 0.25). It declares in 63% of its seats (3600: 76%), 84% of them Flip; level 9 51%, 10 40%, 11 9% (3600: 87% at 9); made 48%; +158 per seat against RuleBot's +0 there (3600: +122). Among RuleBots a level higher still loses at every count of aces and Jokers (−386 to −667 a contract), and with none it bids in 80% of seats for +2 ± 61 over passing. The called ace is in the cat in 7.5% ± 0.7% of its contracts (chance 7.1%), as before.

What rl-006 shows:

- **Better on every fixed measure**: card play, and the full game against every reference field, the old exploiter included, with the bidding starting to follow the hand. The gain came in the first 1000 to 1200 iterations; after that the checkpoints tie.
- **Not less exploitable.** A fresh 100-iteration exploiter still finds about +10 to +20 by declaring more against it, as against 3600, while the league's own exploiters found a significant margin only once in twenty. Two things may hold the margin up: the league thinned x-005-3600 and the other seeded members out by iteration 441, and its exploiters, each trained against one learner snapshot, are weaker opponents than a fresh one trained against the final policy. With one exploiter per seed and ±15–19 each, measuring the margin better comes first ("Next").

### rl-006, continued (29 September, commit ea59c67): 2000 → 4000

"Next", step 1, agreed by the user that morning ("more training"), with rl-006's own settings:

```sh
nohup python -m learn.selfplay --resume runs/rl-006 --iterations 4000 >> runs/rl-006.out 2>&1 &
```

From 10:45 to 12:39 on the workstation (22 workers, the GPU): 2000 iterations in 1 hour 53 minutes, 3.4 s each with the exploiter phases (collecting 1.0 s, updating 1.1 s). 73.4 million recorded decisions (the run's 146.9 million). `run.json` records the resume at ebacb88, the same commit before the history rewrite of that morning (`results/history-rewrite-2026-09-29.txt`).

Window means of the log (as in "rl-006"):

| Iterations | In-run vs RuleBot | Entropy | `value_ev` | Level | Made | Flip | Halves |
|---|---|---|---|---|---|---|---|
| 1751–2000 (before) | +130.4 | 0.946 | 0.762 | 10.38 | 41% | 47% | 30% |
| 2001–2250 | +140.6 | 0.944 | 0.771 | 10.40 | 41% | 47% | 29% |
| 2251–2500 | +138.9 | 0.942 | 0.777 | 10.41 | 41% | 48% | 28% |
| 2501–2750 | +130.2 | 0.944 | 0.776 | 10.42 | 41% | 47% | 30% |
| 2751–3000 | +129.0 | 0.941 | 0.775 | 10.48 | 40% | 44% | 30% |
| 3001–3250 | +138.6 | 0.938 | 0.779 | 10.48 | 41% | 48% | 26% |
| 3251–3500 | +159.3 | 0.937 | 0.776 | 10.49 | 41% | 50% | 26% |
| 3501–3750 | +168.9 | 0.937 | 0.776 | 10.50 | 40% | 48% | 27% |
| 3751–4000 | +163.5 | 0.934 | 0.775 | 10.52 | 40% | 50% | 25% |

Decisions during the run:

- **Healthy throughout**: no non-finite value and no skipped step; approx_kl about 0.0033 (at most 0.0071), magnet KL 0.022 (at most 0.032), clip fraction 0.03; the critic 0.77–0.78 on window means (single iterations down to 0.61 at 2382); the belief loss 1.190 → 1.188; entropy 0.93–0.94.
- **The in-run curve rose late**: window means +129 to +141 up to 3250, then +159 to +169, the line's best (the single best +198.3 at 3480). Two pairs of significant falls in a row (2550 and 2560, down to +85.1; 2740 and 2750, down to +82.9), each followed at once by a significant rise; the stop rule never fired.
- In self-play the learner's contracts moved a little further from Halves (30% → 25%) and up to level 10.5, made 40–41%.
- **The in-run exploiters** gained −4.6, −25.9, +9.4, **+27.1**, +20.6, −6.2, +0.8, −6.3, −7.7, +8.5, +15.2, −34.3, +3.0, −4.1, −5.2, +4.0, +19.2, −10.3, +11.5 and +1.2 (each ±21–26): mean +0.8, one of twenty significant (2400). At 4000 the league holds RuleBot, 40 exploiters and 9 snapshots.
- **Card-play checks** on the fresh choosing seeds 37 and 38 (4000 deals, alongside the run with 4 workers; `play-check-2500.txt`, `-3000`, `-3500`), where rl-006's 2000 scores +24.4 ± 2.9: at 2500 +27.1 ± 2.7, **paired +2.7 ± 1.9**; at 3000 +26.0 ± 3.0, paired +1.6 ± 2.2; at 3500 +26.1 ± 2.8, paired +1.7 ± 2.3. No stop rule.

**Choosing** (§6, `results/rl-006/choose-4000.txt`): every 200th iteration from 2200 and the last, on seeds 37 and 38 (4000 deals), paired with rl-006's 2000, against the new reference set. rl-006's 2000 there scores +24.4 ± 2.9 in card play, +148.7 ± 20.2 against RuleBot, +66.6 ± 18.1 against rl-005's 3600 and −7.6 ± 17.5 against x-006-2000-s1:

| Iteration | Card play | vs RuleBot | vs rl-005's 3600 | vs rl-006's 2000 | vs x-006-2000-s1 | Smallest |
|---|---|---|---|---|---|---|
| 2200 | +0.3 ± 1.8 | +4.1 ± 13.9 | +3.2 ± 16.6 | +10.7 ± 15.7 | +2.4 ± 16.7 | +2.4 |
| 2400 | +2.0 ± 1.9 | −32.1 ± 14.3 | +7.2 ± 16.9 | +2.1 ± 16.0 | +11.3 ± 17.3 | −32.1 |
| 2600 | +0.7 ± 2.0 | −28.8 ± 15.2 | +8.3 ± 17.8 | +21.1 ± 17.6 | +26.1 ± 18.7 | −28.8 |
| 2800 | +1.7 ± 2.1 | −55.0 ± 15.6 | +7.0 ± 19.4 | +30.7 ± 19.5 | +20.2 ± 21.1 | −55.0 |
| 3000 | +1.6 ± 2.2 | −0.7 ± 14.7 | +20.2 ± 18.2 | +61.0 ± 18.0 | +46.3 ± 19.6 | −0.7 |
| 3200 | +3.3 ± 2.2 | +0.8 ± 15.5 | +26.3 ± 19.0 | +48.3 ± 18.7 | +42.0 ± 19.9 | +0.8 |
| 3400 | +2.5 ± 2.1 | −7.9 ± 16.5 | +34.2 ± 18.5 | +43.3 ± 18.3 | +26.6 ± 20.4 | −7.9 |
| 3600 | +3.1 ± 2.1 | −8.5 ± 17.9 | +41.5 ± 20.4 | +57.3 ± 21.0 | +59.6 ± 21.2 | −8.5 |
| 3800 | +2.0 ± 2.3 | −2.8 ± 17.9 | +44.8 ± 20.5 | +57.5 ± 20.6 | +54.2 ± 22.2 | −2.8 |
| **4000** | **+3.3 ± 2.3** | **+22.4 ± 17.6** | **+56.3 ± 20.6** | **+72.1 ± 20.5** | **+64.6 ± 22.7** | **+22.4** |

RuleBot is the smallest field at every checkpoint again, and 2400 to 2800 lost to rl-006's 2000 there (−29 to −55) while gaining against the learned fields. **Iteration 4000 leads** (+22.4 ± 17.6), and no other candidate's smallest is within its interval (+4.8 to +40.0), so it is **chosen without ties**; each of its five paired results is significant. Its unpaired score against RuleBot, the sanity check, is +171.1 ± 21.8 (the 2000: +148.7 ± 20.2).

**Reporting** iteration 4000 on seeds 0, 41 and 42 (6000 deals, `report-4000.txt`, paired with rl-006's 2000, rl-005's 3600 alongside):

| Measure | rl-005's 3600 | rl-006's 2000 | **rl-006's 4000** | rl-006's 4000 − 2000 |
|---|---|---|---|---|
| Card play on fixed contracts | +19.9 ± 2.1 | +23.4 ± 2.3 | **+24.8 ± 2.2** | +1.4 ± 1.7 |
| Full game against RuleBot | +129.1 ± 13.7 | +147.3 ± 16.3 | **+181.4 ± 17.8** | **+34.2 ± 14.3** |
| against rl-005's 3600 | 0 | +39.5 ± 14.9 | **+106.6 ± 17.9** | **+67.2 ± 17.2** |
| against rl-006's 2000 | −57.8 ± 15.1 | 0 | **+64.6 ± 17.1** | **+64.6 ± 17.1** |
| against x-006-2000-s1 | −65.7 ± 15.5 | −24.2 ± 15.0 | **+55.6 ± 17.6** | **+79.8 ± 18.9** |
| Exploiter margin, seed 1 (`learn.exploit`, seeds 101, 102) | +18.6 ± 15.4 | +18.5 ± 17.7 | +23.4 ± 20.2 | |

- **The full game gained against every field**, +34.2 ± 14.3 against RuleBot (+181.4, the line's best; +171.1 ± 31.4 on the Summary's seed-0 deals, `summary-seed0-4000.txt`) and +65 to +80 against the learned ones.
- **Card play gained a little**: +1.4 ± 1.7 on the reporting seeds, +3.3 ± 2.3 on the choosing seeds; as declarer +54.0 (the 2000: +48.1), partner +13.7, defender +15.9. The progress sweep has it rising to about +26 by 3200 and level after.
- **The exploiter margin stays at about +20**: `results/x-006-4000-s1/` takes +23.4 ± 20.2, as declarer +81.3 in 20.1% of its seats (partner +37.3, defender −4.4). Its in-run curve stayed below 0 (−23.2 at 100), which shows how rough one exploiter is.
- **The belief head improved a little** (`beliefs-4000.txt`): 8.4%, 11.6% and 16.0% better than the prior in play (the 2000: 7.9%, 10.9%, 14.5%).
- **The bidding follows the hand a little more** (`hand-reading-4000.txt`, `report-bidding-4000.txt`, `alone-4000.txt`, `margins-4000.txt`; among RuleBots): it bids in 73% of its seats (the 2000: 76%), r(HCP, bid at all) 0.27 (0.16), r(long suit, bid at all) 0.35 (0.21), +0.64 a level per 10 HCP; it declares in 64% of its seats at a mean level of 9.82 (9.58), 82% of them Flip, made 48%; +173 a seat against RuleBot's +0 there (the 2000: +159). A level higher still loses (−600 ± 58 a contract), and the called ace is in the cat in 7.6% ± 0.7% of its contracts (chance 7.1%).

What the continuation shows:

- **More training kept paying**, mostly in the full game (+34 against RuleBot, +65 to +80 against the learned fields) and in how the bidding reads the hand; card play gained little after about 3200. The gain came late, after a dip against RuleBot at 2400 to 2800.
- **Exploitability is unchanged**, at about +20 a deal by one exploiter per policy, while everything else improved.
- **The league is now mostly exploiters** (40 of 50); going further needs them capped ("Next").

### rl-006, to 6000 (29 September, commit 9fe7032): 4000 → 6000, the league's exploiters capped

"Next", steps 1 and 2, agreed by the user on 29 September.

**The cap** (commit 9fe7032, `learn/selfplay.py`): `Settings.league_exploiters = 20` and `--league-exploiters` ("exploiters kept in the league at most; the oldest go first"). `League.add` drops the oldest exploiters (by iteration) and their files beyond the cap, before the snapshots are thinned to the league's size; RuleBot and the newest snapshot always stay. The cap comes from the settings, not the saved state, so a resumed run gets it, and a resume now records in its `resumed` entry of `run.json` the settings that differ from `settings.json` (rl-006's: `{"league_exploiters": 20}`). Three tests: a league over its cap drops its oldest exploiters and their files; the cap comes from the settings after `load_state_dict`; a resume records the settings it changed.

```sh
nohup python -m learn.selfplay --resume runs/rl-006 --iterations 6000 >> runs/rl-006.out 2>&1 &
```

The resume state at 4000, with its league, was copied to `runs/rl-006-state-4000/` first (on the workstation only), since the cap deletes the dropped exploiters' files.

From 13:37 to 15:30 on the workstation (22 workers, the GPU): 2000 iterations in 1 hour 53 minutes, 3.4 s each with the exploiter phases (collecting 1.0 s, updating 1.1 s). 73.0 million recorded decisions (the run's 219.9 million). `run.json` records the resume at 9fe7032 with `"settings": {"league_exploiters": 20}`.

**The league before and after.** At 4000 it held RuleBot, 40 exploiters (100 to 4000) and 9 snapshots (2560, 3200, 3520, 3840, 3920, 3960, 3980, 3990, 4000). At the first member added after the resume (learner-4010) the 20 oldest exploiters (100 to 2000) went, and snapshots filled the room: by about 4200 the league was full again, and from then on it held RuleBot, 20 exploiters and 29 snapshots, as rl-006's did at its 2000. At 6000: the exploiters of 4100 to 6000, and snapshots from 2560 to 6000, sparse long ago (2560, 3200, 3840, 4160, 4480, 4800, 4960, ...) and every 10 iterations over the last 80. Of the learner's deals against league members, snapshots took 43% (2001–4000: 30%), exploiters 55% (68%) and RuleBot 2%.

Window means of the log (as in "rl-006"):

| Iterations | In-run vs RuleBot | Entropy | `value_ev` | Level | Made | Flip | Halves |
|---|---|---|---|---|---|---|---|
| 3751–4000 (before) | +163.5 | 0.934 | 0.775 | 10.52 | 40% | 50% | 25% |
| 4001–4250 | +164.9 | 0.928 | 0.784 | 10.59 | 39% | 49% | 23% |
| 4251–4500 | +164.6 | 0.930 | 0.780 | 10.59 | 40% | 49% | 25% |
| 4501–4750 | +154.2 | 0.940 | 0.782 | 10.53 | 41% | 49% | 27% |
| 4751–5000 | +141.4 | 0.936 | 0.784 | 10.57 | 40% | 47% | 28% |
| 5001–5250 | +156.2 | 0.940 | 0.784 | 10.58 | 40% | 48% | 27% |
| 5251–5500 | +171.4 | 0.938 | 0.783 | 10.60 | 40% | 49% | 26% |
| 5501–5750 | +164.3 | 0.945 | 0.780 | 10.58 | 40% | 47% | 26% |
| 5751–6000 | +169.2 | 0.944 | 0.783 | 10.58 | 40% | 50% | 24% |

Decisions during the run:

- **Healthy throughout**: no non-finite value and no skipped step; approx_kl about 0.0033 (at most 0.0065), magnet KL 0.022 (at most 0.031), clip fraction 0.03; the critic 0.78 on window means (single iterations down to 0.56 at 4578); the belief loss 1.184 → 1.182; entropy 0.93–0.95, a little up from 0.934.
- **The in-run curve stayed level**: window means +141 to +171, the line's best window (+171.4) at 5251–5500 and the single best +206.8 at 5500; +168.0 at 4010, +159.4 at 6000. Single significant falls, but never two in a row; the stop rule never fired.
- In self-play the learner's contracts barely moved: level 10.5–10.6, made 39–41%, Flip 47–50%, Halves 23–28%.
- **The in-run exploiters** gained −13.9, −9.1, +3.5, −8.3, +4.8, **+25.8**, +20.9, +2.2, **+27.9**, −28.1, −4.9, −1.5, +6.5, −5.2, −13.6, **+39.8**, +14.9, −21.1, −0.1 and +3.0 (each ±23–28): mean +2.2, three of twenty significant (4600, 4900, 5600; +39.8 the largest of rl-006's sixty), where 2001–4000 had a mean of +0.8 and one of twenty. With one exploiter per 100 iterations that is not a clear change.
- **Card-play checks** on the fresh choosing seeds 39 and 40 (4000 deals, alongside the run with 4 workers; `play-check-4500.txt`, `-5000`, `-5500`), where rl-006's 4000 scores +27.4 ± 2.7: at 4500 +29.6 ± 2.9, **paired +2.2 ± 2.1**; at 5000 +28.2 ± 2.8, paired +0.7 ± 2.1; at 5500 +30.2 ± 2.9, **paired +2.8 ± 2.1**. No stop rule.

**Choosing** (§6, `results/rl-006/choose-6000.txt`): every 200th iteration from 4200 and the last, on seeds 39 and 40 (4000 deals), paired with rl-006's 4000 (itself a candidate, at 0), against the new reference set. rl-006's 4000 there scores +27.4 ± 2.7 in card play, +177.3 ± 21.8 against RuleBot, +61.5 ± 22.4 against rl-006's 2000 and +29.5 ± 20.3 against x-006-4000-s1:

| Iteration | Card play | vs RuleBot | vs rl-006's 2000 | vs rl-006's 4000 | vs x-006-4000-s1 | Smallest |
|---|---|---|---|---|---|---|
| 4200 | +0.3 ± 1.8 | −7.1 ± 15.7 | −8.3 ± 19.7 | +14.2 ± 20.0 | −2.2 ± 19.9 | −8.3 |
| 4400 | +0.9 ± 1.9 | +6.7 ± 15.8 | +17.4 ± 20.6 | +27.2 ± 19.8 | +8.1 ± 19.7 | +6.7 |
| 4600 | +0.7 ± 2.1 | −38.0 ± 17.6 | +1.3 ± 22.6 | +15.0 ± 21.5 | +5.6 ± 21.5 | −38.0 |
| 4800 | +2.6 ± 2.3 | −57.3 ± 17.8 | +3.7 ± 22.2 | +30.0 ± 21.4 | +0.4 ± 22.0 | −57.3 |
| 5000 | +0.7 ± 2.1 | −23.3 ± 18.2 | +25.3 ± 21.9 | +44.4 ± 22.0 | +30.5 ± 22.2 | −23.3 |
| 5200 | +3.0 ± 2.2 | −13.8 ± 18.1 | +9.1 ± 23.7 | +9.2 ± 22.0 | +24.9 ± 23.1 | −13.8 |
| **5400** | **+1.8 ± 2.1** | **−5.6 ± 17.6** | **+27.3 ± 22.6** | **+37.4 ± 21.1** | **+26.4 ± 22.2** | **−5.6** |
| 5600 | +1.9 ± 2.1 | −15.8 ± 18.1 | +13.7 ± 22.5 | +33.8 ± 21.9 | +15.4 ± 22.0 | −15.8 |
| 5800 | +1.2 ± 2.1 | −29.8 ± 18.1 | +9.4 ± 23.3 | +25.2 ± 22.7 | +12.2 ± 23.4 | −29.8 |
| 6000 | +3.7 ± 2.2 | −15.0 ± 18.5 | +31.4 ± 22.6 | +53.0 ± 21.5 | +23.6 ± 22.4 | −15.0 |

RuleBot is the smallest field at every checkpoint but 4200, and **no checkpoint beats the 4000 against RuleBot**: from 4600 on all are below it there (−6 to −57; 4600, 4800 and 5800 significantly), while against the learned fields most gain (+9 to +53), as 2400 to 2800 did in the continuation to 4000. Card play gains a little (6000 +3.7 ± 2.2 and 5200 +3.0 ± 2.2, significant). By the smallest, **4400 leads** (+6.7 ± 15.8); within its interval are the 4000 itself (0), 5400 (−5.6) and 4200 (−8.3), so 5400 and 4200 were **paired directly with 4400** (`choose-4400-vs-neighbours.txt`, where 4400 scores +28.3 ± 2.7 in card play and +184.0 ± 20.4 against RuleBot): 5400 is tied with it on every measure (card play +0.9 ± 2.1, the four fields −12.3 ± 16.7, +10.0 ± 21.3, +10.2 ± 22.0, +18.3 ± 20.9); 4200 is not (against rl-006's 2000 −25.7 ± 18.5, significant). Between the tied, 4000, 4400 and 5400, **card play decides and iteration 5400 is chosen** (+1.8 ± 2.1 over the 4000, +0.9 ± 2.1 over 4400 paired directly); unlike 4400 (+27.2 ± 19.8 against the 4000 only), it also gains significantly against all three learned fields. 6000, with the best card play, is out by the smallest (−15.0 against RuleBot, below 4400's interval). Its unpaired score against RuleBot, the sanity check, is +171.8 ± 22.0 (the 4000: +177.3 ± 21.8; 6000: +162.3 ± 21.7).

**Reporting** iteration 5400 on seeds 0, 41 and 42 (6000 deals, `report-5400.txt`, paired with rl-006's 4000, rl-006's 2000 alongside; the batch `runs/report-rl-006-6000run.sh` on the workstation):

| Measure | rl-006's 2000 | rl-006's 4000 | **rl-006's 5400** | rl-006's 5400 − 4000 |
|---|---|---|---|---|
| Card play on fixed contracts | +23.4 ± 2.3 | +24.8 ± 2.2 | **+26.3 ± 2.3** | +1.6 ± 1.7 |
| Full game against RuleBot | +147.3 ± 16.3 | +181.4 ± 17.8 | **+185.9 ± 17.6** | +4.4 ± 13.9 |
| against rl-006's 2000 | 0 | +64.6 ± 17.1 | **+95.1 ± 17.5** | **+30.5 ± 17.6** |
| against rl-006's 4000 | −83.0 ± 17.5 | 0 | **+17.7 ± 17.1** | **+17.7 ± 17.1** |
| against x-006-4000-s1 | −53.4 ± 16.0 | +1.7 ± 15.8 | **+33.1 ± 16.3** | **+31.4 ± 17.5** |
| against x-006-5400-s1 (`vs-x-006-5400-s1.txt`) | | −18.0 ± 18.2 | **+13.5 ± 16.5** | **+31.6 ± 18.5** |
| Exploiter margin, seed 1 (`learn.exploit`, seeds 101, 102) | +18.5 ± 17.7 | +23.4 ± 20.2 | **−18.3 ± 20.3** | |

- **Level with the 4000 in card play and against RuleBot** (+1.6 ± 1.7 and +4.4 ± 13.9; +185.9, where the progress sweep has 4400 at +197.3; on the Summary's seed-0 deals +175.1 ± 30.9, +118.6 ± 31.8 over rl-004d's 10, `summary-seed0-5400.txt`). Card play by role: declarer +60.3 (the 4000: +54.0), partner +13.1 (+13.7), defender +16.2 (+15.9).
- **Better against every learned field**, +18 to +32, each just significant: rl-006's 2000, the 4000 itself, and both exploiters of the 4000 and of 5400. x-006-5400-s1 as a field was measured after choosing, to settle the next reference set: it is the learned field the 4000 does worst against (−18.0).
- **Its exploiter found nothing**: `results/x-006-5400-s1/` loses 18.3 ± 20.3 a deal against it (declarer −26.8 in 24.2% of its seats, partner +55.1, defender −48.7), where the 4000's and the 2000's took about +20, as declarer. Its in-run curve ended at +32.5 ± 42.9 at 100 (a different, noisier measure), so one exploiter per policy stays rough: this is the first margin in the line below 0, not yet proof that 5400 is safe.
- **The belief head improved a little again** (`beliefs-5400.txt`): 8.7%, 11.8% and 16.4% better than the prior in play (the 4000: 8.4%, 11.6%, 16.0%).
- **The bidding is much as the 4000's** (`hand-reading-5400.txt`, `report-bidding-5400.txt`, `alone-5400.txt`, `margins-5400.txt`; among RuleBots): it bids in 71% of its seats (the 4000: 73%), r(HCP, bid at all) 0.25 (0.27), r(long suit, bid at all) 0.32 (0.35), among seats that bid +0.71 a level per 10 HCP (+0.64) and r(HCP, highest bid) 0.48 (0.44); it declares in 62% of its seats (64%) at a mean level of 9.81 (9.82), 80% of them Flip, made 48%; +177 a seat against RuleBot's +0 there (+173). A level higher still loses (−608 ± 60 a contract), and the called ace is in the cat in 7.6% ± 0.7% of its contracts (chance 7.1%), as before.

**rl-006's iteration 5400 is the best policy**, a small step: no worse than the 4000 on any measure and better against every learned field and its own exploiter. Its weights (`runs/rl-006/checkpoints/policy-5400.{pt,npz}`, `critic-5400.pt`) and its exploiter's (`runs/x-006-5400-s1/policy.{pt,npz}`) are committed.

What the continuation to 6000 shows:

- **The capped league ran as planned**: exploiters at 20, snapshots back to 29, the learner meeting its past selves in 43% of its league deals instead of 30%. Nothing broke, and the gains against learned opponents (+18 to +32) may owe something to the past selves; with one run, the cap and 2000 more iterations cannot be told apart.
- **The gains are smaller than in 2000 → 4000**: level against RuleBot and in card play, +18 to +32 against the learned fields (2000 → 4000: +34 against RuleBot, +65 to +80 against the learned fields). Against RuleBot the checkpoints swing by ±50 between neighbours with the bidding (4600 and 4800 fell 38 and 57 below the 4000 on the choosing seeds), and no checkpoint of the continuation beats the 4000 there.
- **Card play is creeping up** (the progress sweep: +24.8 at 4000, +26.3 at 5400, +28.1 at 6000, the line's best; 6000 was out of the choice by its −15 against RuleBot).
- **The in-run exploiters found a little more** (three significant of twenty, up to +39.8), while the chosen checkpoint's own exploiter found nothing: one exploiter each, so neither is a clear change.

### Progress along the workstation line (29 September)

`results/progress/`: every 200th checkpoint of rl-005 (from rl-004d's 10) and of rl-006 (from rl-005's 3600), 31 in all at first, 41 after rl-006 carried on to 4000 and 51 after 6000, on the reporting seeds 0, 41 and 42 (6000 deals; `sweep.sh`, 50 minutes on the GPU), drawn with the in-run curve and the exploiter margins by `plot.py` into **`progress.html`**, the progress graph (open it in a browser; rl-006's iteration i is at 3600 + i on its axis).

- **Card play rose along the whole line**: +3.9 at rl-004d's 10, +11.1 at rl-005's 1000, +13.3 at 2000, +17.5 at 3000, +19.9 at 3600; in rl-006 +21.7 at 200, +21.6 at 1200, +23.4 at 2000 (paired with rl-004d's 10, +19.5 ± 2.5). About +7 per 1000 iterations over the first 1400, about +2 per 1000 after, and still rising at the end.
- **The full game against RuleBot is noisier** (±9 to ±17 a checkpoint, and bidding moves it by tens of points between neighbours): +56 at the start, between +82 and +112 over rl-005's 1800 to 3400, +129 to +142 over 3600 to 4000; in rl-006 between +108 and +149, with no clear trend after its 1000.
- **The league's exploiters** found between about −15 and +25 throughout (each ±12 to ±24), with no trend.
- **rl-006's 2000 to 4000** (added after it carried on): card play +23.2 at 2200, +26.4 at 3200 and 3600, +24.8 at 4000; the full game +99 to +163 between 2200 and 3800, then +181.4 at 4000.
- **rl-006's 4000 to 6000** (the capped league; "rl-006, to 6000"): card play +24.2 to +26.8 over 4200 to 5800, then +28.1 at 6000, the line's best (+26.3 at the chosen 5400); the full game +185.1 and +197.3 at 4200 and 4400, down to +136.4 and +134.1 at 4600 and 4800, back to +185.9 at 5400, +169.1 at 6000. The exploiter diamond at 5400 (9000 on the axis) is the line's first below 0 (−18.3).

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

### The rl-004 line (26 September): rl-004, rl-004b, rl-004c, rl-004d

§4's command, on 8 workers with the update on MPS, 45–57 s per iteration (up
to 75 s while analyses ran alongside). Each run started from the best
checkpoint of the one before (policy and critic), with one change. The user
was away for the run; every decision, and why, is in the runs' entries below.

| Run | Start | Change | Iterations | Stopped because |
|---|---|---|---|---|
| rl-004 | rl-003's 110 | rl-004's plan ("Next" of the last session) | 116 | its gains at 80 were lost by 110: hand-blind jumps to level 9 |
| rl-004b | rl-004's 80 | the exploration correction fixed (commit 89034e8) | 98 | slid from its peak at 30–50, towards Flip and declaring more |
| rl-004c | rl-004b's 40 | `--magnet 0.3` | 236 | held its level for about 120 iterations, then slid back |
| rl-004d | rl-004c's 150 | `--magnet 1.0` | 108 | held the line's best level but no longer improved on it |

In-run evaluations (2000 deals; rl-003's 110 scores +22.5 ± 10.0 on them
under the new scoring), at the iteration in brackets:

| Run | | | | | | |
|---|---|---|---|---|---|---|
| rl-004 | +14.3 (10) | −12.0 (30) | +21.2 (50) | +27.4 (80) | +5.0 (100) | −1.7 (110) |
| rl-004b | +23.5 (10) | +30.9 (30) | +30.8 (50) | +20.1 (70) | +12.7 (90) | |
| rl-004c | +35.8 (10) | +29.7 (50) | +42.2 (100) | +46.7 (150) | +26.4 (200) | +26.9 (230) |
| rl-004d | +48.5 (10) | +34.8 (50) | +46.6 (100) | | | |

On 4000 fresh deals never used for choosing (seeds 21 and 23), paired against
rl-003's 110 (+26.2 ± 6.7): rl-004's 80 +5.6 ± 6.2, rl-004b's 40 +8.7 ± 7.0,
rl-004c's 100 +12.2 ± 9.3 and 150 +20.1 ± 10.1, rl-004d's 10 +26.5 ± 9.8;
rl-004d's 100 scored +52.6, level with its 10 (0.0 ± 6.6). Each restart's
best beat the one before.

What happened, and why:

- **A bug in the exploration's correction** (fixed in commit 89034e8, details
  in rl-004's entry). PPO clipped its ratio against the explored odds, not the
  policy's own, so each update moved chance from the policy's favourite bids
  to explored ones whatever they scored. With it fixed, the auction sharpened
  (its entropy fell from 1.63 to 1.32 nats) instead of evening out.
- **Higher contracts did not start to pay.** `learn.margins` among RuleBots,
  at each seat's first bid:

  | Policy | Made one level up | "+1 − bid", all hands | "+1 − bid", 4+ aces and Jokers |
  |---|---|---|---|
  | rl-003's 110 (2000 deals) | 43% | −46 ± 12 | +95 ± 69 |
  | rl-004's 80 | 34% | −107 ± 26 | +82 ± 154 |
  | rl-004b's 50 | 34% | −124 ± 28 | +92 ± 155 |
  | rl-004c's 60 | 31% | −205 ± 32 | −73 ± 198 |
  | rl-004c's 160 | 30% | −257 ± 33 | −186 ± 179 |
  | rl-004d's 10 (2000 deals) | 32% | −203 ± 23 | −93 ± 131 |

  (1000 deals where not stated; `results/rl-004*/margins-*.txt`,
  `results/rl-004/final-margins*.txt`.) One level up, its contracts are made
  less often than rl-003's, and no strength of hand gains from a level more.
- **The bidding did not follow the play; it ran ahead of it.** It bid higher
  anyway: among RuleBots its contracts' level rose from 8.18 to 8.84, mostly
  9, and are made 49% of the time (rl-003: 56%). Its first bid beats passing
  by more (+127 ± 16; rl-003 +86 ± 10), and with no ace or Joker by −19 ±
  37 (−45 ± 21). It still bids in 91% of seats with no ace or Joker, though
  its highest bid now climbs a little with the hand (8.53 with none to 8.94
  with five; rl-003 flat at 8.02–8.18). In the duel that follows it competes
  less (a second bid in 44% of seats; rl-003 69%).
- **Where the gain comes from** (`learn.report` on the recorded arena,
  `results/arena/rl-004d-0010-report.txt`, against rl-003's, recorded under
  the old scoring): it declares in 57% of seats (rl-003 45%), 61% of them Flip
  (18%) and 14% Halves (54%), and scores +243 per contract (+198) though it
  makes fewer. It scores +464 as partner (+329) and −377 as defender (−280);
  alone it loses 805 per contract (370). By role against RuleBot:
  declarer +106, partner +121, defender −90 (rl-003's 110 under the new
  scoring: +61, +67, −62).
- **Self-play drifts towards what beats its own defence.** With the bug gone,
  the policy still moved steadily to Flip and to declaring more. In sampled
  self-play that made it a *better* declarer (rl-004b: +228 → +292 per
  contract), but RuleBot's defence punished it among RuleBots. The magnet
  slowed the drift: at 0.1 the peak held about 40 iterations, at 0.3 about
  120, and at 1.0 the policy held its level but stopped improving. Restarting
  from the best checkpoint, chosen on fresh deals, turned the drift into
  steady progress.
- **Defending is still the weakest role**, and it got worse as self-play moved
  to Flip (in-run defender −69 → −121 over rl-004c): the learner then defends
  plain and Clubs contracts, the ones RuleBot bids, less and less often.
- **Odd: the in-run curve misled twice.** A run started from a checkpoint
  chosen as the best on the evaluation deals looks worse than its start on
  those deals (the start is biased upwards), and rl-004d looked flat while
  fresh deals showed it at the line's best. Choose and judge on deals never
  used for choosing.

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

### rl-004c (26 September, commit 89034e8): stopped at iteration 236 of 450

From rl-004b's iteration 40 (policy and critic), started at 16:32 with
rl-004b's command except `--magnet 0.3` (`--init
runs/rl-004b/checkpoints/policy-0040.pt --init-critic
runs/rl-004b/checkpoints/critic-0040.pt ... --magnet 0.3 ... --out
runs/rl-004c`). Iteration 3: clip fraction 0.058, approx_kl 0.006, the critic
explained 73%.

Decisions during the run:

- Iterations 10–50: +35.8, +23.9, +29.2, +29.1, +29.7; one significant fall
  (10 → 20, −11.9 ± 6.8), otherwise flat, and level with its start throughout.
  The magnet held the policy closer: magnet KL about 0.011 per window
  (rl-004b about 0.022), clip fraction about 0.05. But in self-play the
  learner's level still climbed (9.34 → 9.58 over 40 iterations, made 43.6%
  → 41.9%, Clubs 22% → 30%), so the stronger magnet slows how far the policy
  moves but not this drift.
- Fresh deals. On `--seed 7` (`results/rl-004c/rl-004c-compare-seed7-a.txt`) its
  iterations 20–50 scored +33 to +37, 10–13 below rl-004b's iteration 40.
  But seed 7 is where iteration 40 was chosen as the best of several, so its
  +46.5 there is biased upwards. On deals not used for choosing (`--seed 11`,
  `results/rl-004c/rl-004c-compare-seed11-a.txt`): rl-004b's iteration 40
  **+45.0 ± 12.0**, and **+12.1 ± 10.4 above rl-003's iteration 110** (+32.9
  there); rl-004's iteration 80 +40.5; rl-004c's iteration 50 +38.0, level
  with rl-004b's 40 (−7.0 ± 10.3). So rl-004c holds the peak's level instead
  of sliding from it, but has not improved on it. Carried on.
- Iteration 60: +37.0 ± 14.8, up 7.3 ± 6.8 and +14.4 ± 12.9 above rl-003's
  iteration 110 on the evaluation deals. `learn.margins` on it (about
  iteration 200 of the line; `results/rl-004c/margins-0060*.txt`, 1000
  deals): among RuleBots its contracts are made 48% as bid, 31% one level up
  and 17% two up (rl-004b's iteration 50: 55%, 34%, 19%), and "+1 − bid" is
  −205 ± 32 (−124), with no strength of hand gaining from a level more (four or
  more aces and Jokers: −73 ± 198). In self-play: 53%, 33%, 16%, "+1 − bid"
  −175 ± 54. It bids a little higher and makes a little less, while its
  score holds or rises: exploring levels still has not taught it to make
  higher contracts.
- Iterations 70–100: +38.0, +36.4, +35.2, **+42.2** (the best in-run
  evaluation of any run), no significant change between any two, each
  significantly above rl-003's iteration 110 (+12.7 to +19.7). On fresh
  deals (`--seed 13`, `results/rl-004c/rl-004c-compare-seed13-a.txt`): its
  iteration 100 +36.2 ± 15.0 and 70 +31.2, rl-004b's 40 +30.8, rl-003's 110
  +21.7. The best policies now differ by 5–10 points, less than 2000 paired
  deals resolve (about ±12). Over three fresh sets rl-004b's 40 beat rl-003's
  110 by +24, +12.1 ± 10.4 and +9.1 ± 10.1.
- Iterations 110–160: +41.0, +45.3, +36.0, +37.8, **+46.7**, +37.8: swings of
  about 9 (two significant falls, each followed by a rise), around a level
  that keeps edging up. Partner rose (+91 → +155) and defender fell (−90 →
  −114). The log stayed steady from iteration 40: self-play level 9.55–9.60,
  made 41–42%, entropy 0.52, clip fraction 0.045–0.049, magnet KL 0.011,
  critic 0.71–0.73, no skipped steps; Flip grew from 33% to 46% of the
  learner's contracts in self-play.
- `learn.margins` at iteration 160 (about 300 of the line;
  `results/rl-004c/margins-0160*.txt`): among RuleBots it bids at its first
  decision in 94% of seats (86% at 60), its contracts are made 47% as bid,
  30% one level up and 15% two up, "+1 − bid" is −257 ± 33, and no strength
  of hand gains from a level more (four or more: −186 ± 179). In self-play
  48%, 33%, 17%, "+1 − bid" −185 ± 60. Over the whole line, exploring levels
  has not made higher contracts pay (one level up: 43% for rl-003, now 30%),
  yet the bidding grew bolder and the score rose with it: the bidding has
  run ahead of the play rather than following it.
- Iterations 170–210: +46.4, +39.2, +31.8, +26.4, +31.9. Three falls in a row
  too small to be significant each, but 170 → 190 lost 14.6 ± 9.2; declarer
  fell (+82 → +62) as Flip grew to about half its contracts, rl-004b's
  signature. At 200 it was level with its start (−3.6) and with iterations
  60–100. On RuleBot's auction positions
  (`results/rl-004c/rl-004c-phases-0100-0200.txt`) the auction's entropy
  *fell*, 1.45 (rl-004b's 40) → 1.35, 1.33, 1.21 at 100, 150 and 200, as it
  sharpened towards Flip (21% → 44% of its chance) and away from Halves (37%
  → 11%); its greedy bid agreed with its start's on 48%, 30% and 23%. So
  TRAINING.md's cue for less exploration (a rising entropy) was absent: the
  drift is self-play's own preference. Decision: if 210 fell below its start,
  restart from rl-004c's best with a stronger magnet, the one lever left; if
  it recovered, as after the dip at 130, carry on. It recovered (+31.9, +5.5 ±
  8.1). Carried on.
- Iterations 220 and 230: +32.7, +26.9. Four evaluations (200–230) at +26 to
  +33, the level it started from; the +36 to +47 of iterations 60–170 was
  gone, and defender reached −121 (−69 at the start). A likely reason for
  the defence: as self-play moved to Flip, the learner defended plain and
  Clubs contracts played as RuleBot plays them (it never bids Flip or Halves)
  less and less often. Fresh deals (`--seed 17`,
  `results/rl-004c/rl-004c-compare-seed17-a.txt`): rl-004b's 40 +33.5;
  rl-004c's 100 +36.2, 120 +37.1, **150 +41.5 ± 17.3** (+8.0 ± 14.4), 170
  +25.6, and 230 **+13.7, −19.7 ± 14.2 below rl-004b's 40**.
- 20:03, decision: stopped rl-004c at iteration 236 (cycling that wipes out
  the gains: its peak band was lost, and on fresh deals 230 was significantly
  below the run's start) and started rl-004d from its iteration 150 (the best
  on seed 17 and in the run) with one change, `--magnet 1.0`. The magnet is
  the one lever left: at 0.3 the peak held about three times as long as at
  0.1, and less exploration is not indicated (the auction's entropy fell).

### rl-004d (26 September, commit 89034e8): stopped at iteration 108 of 450

From rl-004c's iteration 150 (policy and critic), started at 20:03 with
rl-004c's command except `--magnet 1.0`. At iteration 3 the clip fraction
was 0.030 and approx_kl 0.003, at TRAINING.md's line for "no learning"
(about 0.03): if it stays there with paired changes near 0 for 50
iterations, the magnet is too strong.

Decisions during the run:

- Iterations 10–60: +48.5, +38.6, +35.8, +37.9, +34.8, +33.6; one
  significant fall (10 → 20), then paired changes within ±3. Clip fraction
  0.032–0.035, magnet KL about 0.005 per window (0.3: 0.011), self-play level
  9.43–9.57, Flip 49% → 60% of the contracts it declared in the evaluation.
  On the evaluation deals it sat 8–12 below its start, but that start was
  chosen as the best of rl-004c's evaluations on those very deals, so it is
  biased upwards.
- A screen on 4000 fresh deals never used for choosing (seeds 21 and 23,
  pooled; `results/rl-004d/screen-seeds21-23.txt`), paired against rl-003's
  iteration 110 (+26.2 ± 6.7): rl-004's 80 +31.8 (+5.6 ± 6.2); rl-004b's 40
  +34.9 (+8.7 ± 7.0); rl-004c's 100 +38.4 (+12.2 ± 9.3), 120 +34.3 (+8.1 ±
  9.8), 150 +46.3 (**+20.1 ± 10.1**); rl-004d's 10 **+52.7 ± 10.8 (+26.5 ±
  9.8)** and 50 +47.4 (+21.2 ± 9.7). So the line kept improving from restart
  to restart even where the in-run curve looked flat, and rl-004d holds the
  best level rather than stagnating below it.
- Iterations 70–100: +39.8, +43.3, +44.0, +46.6. On the same fresh deals
  (`results/rl-004d/screen-seeds21-23-rl-004d-100.txt`) its iterations 70, 80,
  90 and 100 scored +48.1, +49.7, +49.9 and +52.6, all level with its
  iteration 10 (paired −4.6 to 0.0, ±6). It held the line's best level but no
  longer improved on it.
- 21:40, decision (agreed with the user, who was back by then): stopped
  rl-004d at iteration 108 and ended training. Over about 560 iterations
  (116 + 98 + 236 + 108) the line had plateaued at its best level, and the
  remedy for "no learning" (a weaker magnet) is what drifted in rl-004b and
  rl-004c. On to §6 with the finalists rl-004d's 10 and 100 and rl-004c's 150.

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
- When moves are played from other odds than the policy's (exploration), PPO
  must clip against the policy that collected them and weigh by the odds.
  Clipping against the played odds silently biased every update towards the
  explored moves (rl-003's rising auction entropy, rl-004's jumps). Test the
  update's gradient on explored moves, not only that the odds are recorded.
- Choose and judge on deals never used for choosing. The in-run maximum is
  biased upwards, so a run started from it looks worse than its start on the
  same deals, and small differences between good checkpoints need 4000 or
  more paired deals to show.
- Self-play drifts towards whatever beats its own weaknesses (here, declaring
  Flip against its own weak defence), which a fixed opponent like RuleBot can
  punish. A stronger magnet slows the drift; restarting from the best
  checkpoint, chosen on fresh deals, turned it into progress.
- Exploring higher levels did not teach the policy to make them: over about
  560 iterations its contracts one level up were made less often, not more.
- One long run with a league and a moving magnet beat restarts from the
  best checkpoint (rl-005): 2100 iterations without drift, and card play,
  flat over the whole rl-004 line, rose steadily. Judge by card play on
  fixed contracts and paired reference fields on held-out seeds; the in-run
  curve agreed this time, but the choosing seeds decided.

## Next: RuleBot in a tenth of the deals, then rl-006 on to 8000 (proposed, 29 September; not agreed)

Written 29 September 2026, evening, for the user and the next agent. **Nothing here is agreed**: anything that trains needs the user's agreement first. Read "rl-006, to 6000" and "Progress along the workstation line" in Results, then §5 and §6, and the "Done" records below for how the last steps were run ("Done: rl-006, longer exploiters in the league" has the shell, the shared machine and waiting on PIDs). The progress graph is `results/progress/progress.html`.

**Where it stands.** The best policy is **rl-006's 5400** (`runs/rl-006/checkpoints/policy-5400.{pt,npz}`, `critic-5400.pt`, committed). On the reporting seeds it is level with rl-006's 4000 in card play (+1.6 ± 1.7; +26.3 ± 2.3 against RuleBot) and against RuleBot (+4.4 ± 13.9; +185.9), and better against every learned field (+18 to +32), and its exploiter found nothing (−18.3 ± 20.3). rl-006's resume state is at 6000 (card play +28.1 on the progress sweep, the line's best, but −15 against RuleBot on the choosing seeds). The league keeps at most 20 exploiters (`--league-exploiters`). **The web game's bot (`web/bot.npz`) is still rl-005's 3600**: replacing it with rl-006's 5400 is the user's call (`cp runs/rl-006/checkpoints/policy-5400.npz web/bot.npz`); on every measure here 5400 is better, and it is the first best policy whose exploiter found nothing.

**Why.** RuleBot decides every choice: it is the smallest of the reference fields at nearly every checkpoint of the last three choices, so a checkpoint is chosen by how it does against RuleBot. Yet the learner meets RuleBot in about 1% of its deals (one league member of fifty, drawn like the others; 1.1% of the deals in 4001–6000). Its result against RuleBot swings by up to 50 a deal between neighbouring checkpoints as the bidding drifts, and the last 2000 iterations gained nothing there while gaining +18 to +32 against the learned fields. RuleBot is also the only opponent in training that bids cautiously (when both bid, its highest level is 8.1 where 5400's is 10.0). So the one change proposed is to let the learner practise against it more: a tenth of the deals, about nine times as many as now. More training stays the default otherwise; the gains are slowing (2000 → 4000: +34 against RuleBot, +65 to +80 against the learned fields; 4000 → 6000: level, +18 to +32), but card play is still creeping up.

### Step 1 (proposed): RuleBot in a tenth of the deals (code, with tests)

- A setting `rule_share: float = 0.0` in `Settings`, with `--rule-share` ("deals against RuleBot, besides its draws as a league member"). In `_league_lineups`, that share of the deals goes to RuleBot in the same lineups as league deals (one to three seats); 0 keeps every earlier run as it was.
- A test beside the league's: with `rule_share` 0.5, about half the deals of an iteration are against RuleBot, and none with 0.
- A resume keeps a setting missing from `run.json` at the command line's value, so `--resume runs/rl-006 --rule-share 0.1` works without editing `run.json`, and the resume records it in its `resumed` entry.

### Step 2 (proposed): carry rl-006 on to 8000 with it (about 2 hours)

```sh
nohup python -m learn.selfplay --resume runs/rl-006 --iterations 8000 --rule-share 0.1 >> runs/rl-006.out 2>&1 &
```

Watch it as §5 says, with card-play checks at 6500, 7000 and 7500 on the choosing seeds 43 and 44, paired with rl-006's 5400. Then choose as §6 says, among every 200th iteration from 6200 and the last, paired with rl-006's 5400 (a candidate at 0), against the new reference set:

```sh
REF="rule runs/rl-006/checkpoints/policy-4000.npz runs/rl-006/checkpoints/policy-5400.npz runs/x-006-5400-s1/policy.pt"
SEEDS="43 44" START=runs/rl-006/checkpoints/policy-5400.npz RUN=runs/rl-006 FROM=6200 FIELDS="$REF" \
    sh results/rl-005/choose.sh > results/rl-006/choose-8000.txt
```

Report the chosen one on seeds 0, 41 and 42 paired with rl-006's 5400, with one exploiter (`--seed 1`) and its bidding: `runs/report-rl-006-6000run.sh` on the workstation is the batch used for 5400 (change its `REF`, `START` and `C` to the new set). Extend the graph (`CHOSEN` in `plot.py`). The question the run answers: does the full game against RuleBot rise again, without losing the learned fields? If the user prefers no change, the same run without `--rule-share` is the plain alternative.

### At the end

`pytest -q` and `ruff check . && ruff format --check .`; the Results entries, and the Summary's and "The review's measures" rows for a new best; "Where things stand" updated; this section turned into a "Done" record with a new "Next"; commit and push `training`, then merge it into `main` by pull request with the tests green (CLAUDE.md) and bring `training` level with `main`. Force-add only standout weights ("What is committed under `runs/`").

### Later (not agreed)

- **The exploiter margin from several exploiters** (seeds 2 to 5 of `learn.exploit` against rl-005's 3600 and rl-006's 2000, about 30 minutes) and **a league that keeps its `--league-add` members** (they join as snapshots of iteration 0 and are thinned first): proposed on 29 September and set aside for more training; worth it only if the margin grows or a decision hangs on it.
- **Entropy 0.01**: a one-change twin of rl-006 at 0.01, or a longer check than 200 iterations.
- An **Elo or TrueSkill ladder** that rates every 200th checkpoint against the earlier ones automatically.
- The **`--stake-scaling`** one-change check.
- T2.5's **separate belief network**: the belief head has barely improved in 10,000 iterations (rl-006's 5400: 8.7%, 11.8% and 16.4% better than the prior in play; rl-004d's 10: 7–12% and 19% in the last five tricks).
- **Search**: critic search added only +3.7 ± 7.9 to rl-005's 3600; not measured since.
- **Bidding**: rl-006's 5400 bids 72% of the hands without an ace or a Joker, for −16 ± 76 over passing among RuleBots (rl-006's 2000: 80%, +2 ± 61); a level higher still loses at every strength.
- To play against the best policy: `python -m web.server --bot runs/rl-006/checkpoints/policy-5400.npz` (the default is still rl-005's 3600).
- **Collect while updating** (the throughput review's step 4): an iteration would take about the longer of the two halves, about 1.3 s instead of 2.45, and the GPU would stop pulsing. The deals then come from a policy one update behind, so PPO needs the decoupled objective (Hilton et al. 2021): log-chances recomputed under the update's starting weights, the ratio clipped against those, each step weighed by them over the collecting odds. `ppo_objective` already has that form for explored bids. It changes the algorithm, so it needs the user's agreement and a paired run against a synchronous twin. PERFORMANCE.md, "Training throughput", "Next".

### What is committed under `runs/`

**Runs live on the workstation only** (the user, 29 September): resume states, leagues, every checkpoint, exploiters and recorded deals stay in the ignored `runs/` of `~/danish-wist-training`. Only standout weights are force-added (`git add -f`), as the agent judges: each policy that has been the best (its `policy-NNNN.pt`, `.npz` and `critic-NNNN.pt`) and the exploiters used as reference fields. On 29 September the earlier chosen sets (resume states, leagues, every 200th checkpoint, most exploiters, recorded deals; 163 files) were taken out of the index, still on disk, and out of the history of `training`, `main` and `throughput` on GitHub (the refs of pull requests #14 and #15 still reach the old commits; only GitHub Support can drop them): those branches were rewritten from the first commit that added run files (65589f1, now ea70e92) on, which changed 28 commits' hashes; `results/history-rewrite-2026-09-29.txt` maps old to new, and the hashes cited in this file are the new ones (records such as `results/*/run.json` keep the old). What remains, 21 files then, 26 after rl-006's 4000 and 31 after its 5400:

- `runs/bc-explore.{pt,npz}`: the imitation start.
- The policies that have been the best, each with its critic: rl-003's 110, rl-004d's 10, rl-005's 2000, rl-005's 3600, rl-006's 2000, rl-006's 4000 and **rl-006's 5400** (`runs/<run>/checkpoints/policy-NNNN.{pt,npz}`, `critic-NNNN.pt`), added as each became the best.
- The reference exploiters x-005-3600, x-006-2000-s1, x-006-4000-s1 and x-006-5400-s1 (`policy.{pt,npz}`).


## Done: room for snapshots in the league, then rl-006 on to 6000 (29 September)

Agreed by the user on 29 September (the handover prompt below) and carried out that day: Results, "rl-006, to 6000". The plan as it was written follows; its "Later" list and the record of what is committed under `runs/` moved to "Next" above.

**Where it stood.** The best policy was **rl-006's 4000** (`runs/rl-006/checkpoints/policy-4000.{pt,npz}`, `critic-4000.pt`, committed; also the resume state's policy). On the reporting seeds it beats rl-006's 2000 in the full game against every reference field (+34.2 ± 14.3 against RuleBot, +181.4; +65 to +80 against the learned fields) and a little in card play (+1.4 ± 1.7; +24.8 ± 2.2 against RuleBot). Its exploiter margin stays at about +20 (+23.4 ± 20.2). **The web game's bot (`web/bot.npz`) is still rl-005's 3600**, as the user chose on 29 September.

**Why.** More training is still paying: 2000 more iterations gained +34 to +80 in the full game and made the bidding read the hand more, and the full game was still rising at 4000 (card play less so). But the league never thins exploiters, and at 4000 it holds 40 of them and 9 past selves: past about 4500 iterations exploiters would fill it and the learner would stop meeting its past selves. Exploiters stay as they are otherwise: one `learn.exploit` per chosen checkpoint as the alarm.

### Step 1: cap the exploiters in the league (code, with tests)

`League` (`learn/selfplay.py`) never thins exploiters (`_thin` drops snapshots only). Cap them:

- A setting `league_exploiters: int = 20` in `Settings`, with `--league-exploiters` ("exploiters kept in the league at most; the oldest go first"), passed to `League`.
- In `League.add`, once there are more exploiters than the cap, drop the oldest (by iteration) and its file, before the snapshots are thinned to `size`; RuleBot and the newest snapshot always stay, as now.
- `League.load_state_dict` restores `size` from the saved state. The cap must come from the settings, not the state, so that a resumed run gets it (rl-006's state has none).
- A resume takes every argument from `run.json` except `--resume`, `--out`, `--workers` and `--device`, and one missing there keeps its default, so `--resume runs/rl-006` runs with the default cap without editing `run.json`. `settings.json` is written only when a run starts: record the cap the resumed run used (for example in its `resumed` entry of `run.json`).
- Tests in `tests/test_selfplay.py`, beside `test_a_full_league_thins_its_oldest_snapshots_most`: a league over its cap keeps the newest exploiters and drops the oldest and their files; RuleBot and the newest snapshot stay; the cap holds after `load_state_dict`.

With 20 exploiters of 50 the league keeps about 29 past selves, as rl-006's did at its 2000. rl-006's 40 exploiters drop to the newest 20 at the first member added after the resume. Changing `learn/` needs no rules change.

### Step 2: carry rl-006 on to 6000 (about 2 hours)

```sh
nohup python -m learn.selfplay --resume runs/rl-006 --iterations 6000 >> runs/rl-006.out 2>&1 &
```

Watch it as §5 says, with card-play checks at 4500, 5000 and 5500 on the choosing seeds 39 and 40, paired with rl-006's 4000. Then choose as §6 says, among every 200th iteration from 4200 and the last, paired with rl-006's 4000 (itself a candidate, at 0), against the new reference set:

```sh
REF="rule runs/rl-006/checkpoints/policy-2000.npz runs/rl-006/checkpoints/policy-4000.npz runs/x-006-4000-s1/policy.pt"
SEEDS="39 40" START=runs/rl-006/checkpoints/policy-4000.npz RUN=runs/rl-006 FROM=4200 FIELDS="$REF" \
    sh results/rl-005/choose.sh > results/rl-006/choose-6000.txt
```

Report the chosen one on seeds 0, 41 and 42 the same way, paired with rl-006's 4000, with one exploiter (`learn.exploit ... --seed 1`) as its margin and its bidding as for the 4000 (`runs/report-rl-006-4000.sh` on the workstation is the batch that was used). Extend the graph (`sh results/progress/sweep.sh && python results/progress/plot.py`; add the chosen iteration to `CHOSEN` in `plot.py` first). Write it up as "rl-006, to 6000" in Results, with the league change (what it held before and after the cap). Force-add only the chosen checkpoint's weights and its exploiter, and only if it becomes the best.

### At the end

`pytest -q` and `ruff check . && ruff format --check .`; the Results entries, and the Summary's and "The review's measures" rows for a new best; "Where things stand" updated; this section turned into a "Done" record with a new "Next" (proposed, not agreed); commit and push `training`, then merge it into `main` by pull request with the tests green (CLAUDE.md) and bring `training` level with `main`. Force-add only standout weights ("What is committed under `runs/`").

### The handover prompt (29 September, afternoon)

Given to the next agent, with a fresh context, with the user's agreement to steps 1 and 2:

> You are continuing the Danish Wist learned-bot project on the RTX 5090 workstation. Work in the git worktree ~/danish-wist-training on the branch `training`, with the Python in its .venv (`git pull` first; `export PATH=$PWD/.venv/bin:$PATH`).
>
> Read, in this order: CLAUDE.md; TRAINING.md from the top ("Where things stand"), then its "Next" section in full, then §5 ("Watching it") and §6 ("After the run: measuring and choosing"); then the "rl-006, continued" and "Progress along the workstation line" entries in Results. The progress graph is results/progress/progress.html.
>
> Your task is TRAINING.md "Next", steps 1 and 2, which I have agreed to: cap the exploiters in the league (code in learn/selfplay.py, with tests), then carry rl-006 on to 6000 iterations with `--resume`, watch it as §5 says with the card-play checks at 4500, 5000 and 5500, then choose, report and measure the chosen checkpoint as "Next" says (one `learn.exploit` exploiter as its margin, its bidding), and extend the progress graph. You don't need to ask me before the code change, the run, the checks or the exploiter. Stop and report to me if a stop rule fires, or if something unexpected makes the plan unsound; anything else that trains needs my agreement.
>
> Record decisions and results in TRAINING.md's Results as you go, at the level of detail of "rl-006, continued", and copy the run's and exploiter's small files to results/. Runs stay on this machine: force-add only the weights of a checkpoint that becomes the best (and its exploiter). Don't replace `web/bot.npz`. Run `pytest -q` and `ruff check . && ruff format --check .` before committing. At the end, push `training`, merge it into `main` by pull request with the tests green, and bring `training` level with `main`.
>
> The machine is shared with other sessions; leave their processes alone. Anything longer than a few minutes runs detached; stop your own runs by killing their main process by its PID, never `pkill -f` a pattern, and wait on a PID or a file, never on `pgrep -f "<pattern>"` (it matches the waiting shell itself).
>
> When you finish, tell me in plain terms how the capped league went, how the chosen checkpoint compares with rl-006's 4000 on card play, the full game against the reference fields and the exploiter margin, what the graph shows, and what you would do next.

## Done: rl-006 carried on to 4000 (29 September)

Agreed by the user on 29 September ("Resume rl-006 to 4000") and carried out that day: Results, "rl-006, continued". The plan as it was written follows; its "Later" list, the record of what is committed under `runs/` and the handover prompt moved to "Next" above.

**Where it stood.** The best policy was **rl-006's 2000** (`runs/rl-006/checkpoints/policy-2000.{pt,npz}`, `critic-2000.pt`, committed; also the resume state's policy). On the reporting seeds it beats rl-005's 3600 in card play (+3.5 ± 2.0; +23.4 ± 2.3 against RuleBot) and in the full game against every reference field (+18.1 against RuleBot, +36.6, +39.5 and +60.3 against rl-005's 2000, rl-005's 3600 and x-005-3600), and its bidding has begun to follow the hand. **But a fresh exploiter still takes +18.5 ± 17.7 from it**, as from 3600 (+18.6 ± 15.4 on the same seed), by declaring more. **The web game's bot (`web/bot.npz`) is still rl-005's 3600**: replacing it with rl-006's 2000 is the user's call (`cp runs/rl-006/checkpoints/policy-2000.npz web/bot.npz`); on these measures it is better everywhere and no more exploitable.

**Why more training, and why little exploiting** (the user's question, 29 September). An exploiter stands in for an opponent who adapts to the bot, which none of self-play's opponents does (RuleBot and the bot's own past selves). The holes they find are +10 to +25 per deal, small beside the gains (about +150 over RuleBot, +40 over the day before's best), and they did not grow while the rest improved. So training is the lever, and the exploiter stays as an alarm: one `learn.exploit` per chosen checkpoint, acted on only if the margin clearly grows. The five-exploiter measurement and the league change proposed earlier that day are set aside ("Later").

### Step 1: carry rl-006 on to 4000 (about 2 hours)

With rl-006's own settings (a resume changes nothing else):

```sh
nohup python -m learn.selfplay --resume runs/rl-006 --iterations 4000 >> runs/rl-006.out 2>&1 &
```

At 4000 its league holds about 40 exploiters and 9 snapshots, as rl-005's did at its 4000. Watch it as §5 says, with card-play checks at 2500, 3000 and 3500 on the fresh choosing seeds 37 and 38, paired with rl-006's 2000. Then choose as §6 says, among every 200th iteration from 2200 and the last, paired with rl-006's 2000 (itself a candidate, at 0), against the new reference set, breaking ties by card play:

```sh
REF="rule runs/rl-005/checkpoints/policy-3600.npz runs/rl-006/checkpoints/policy-2000.npz runs/x-006-2000-s1/policy.pt"
SEEDS="37 38" START=runs/rl-006/checkpoints/policy-2000.npz RUN=runs/rl-006 FROM=2200 FIELDS="$REF" \
    sh results/rl-005/choose.sh > results/rl-006/choose-4000.txt
```

Report the chosen one (NNNN) on seeds 0, 41 and 42 the same way, paired with rl-006's 2000, with one exploiter (`learn.exploit ... --seed 1`) as its margin, and its bidding (`learn.report`, `hand_reading.py`) as for rl-006's 2000. Extend the graph: `sh results/progress/sweep.sh && python results/progress/plot.py` (about 45 minutes; add NNNN to `CHOSEN` in `plot.py` first). Write it up as "rl-006, continued" in Results. Force-add only the chosen checkpoint's weights, and only if it becomes the best.

### Step 2: past 4000, a league with room for snapshots

The league never thins exploiters, so beyond about 4500 iterations they fill it (50 members) and the learner stops meeting its past selves. If step 1 is still improving at 4000, cap the exploiters in `League` (`learn/selfplay.py`: thin the oldest beyond, say, 20; with tests) before carrying on. Needs the user's agreement.

### At the end

`pytest -q` and `ruff check . && ruff format --check .`; the Results entries; "Where things stand" updated; this section turned into a "Done" record with a new "Next"; commit and push `training` (no pull request to `main` unless the user asks). Force-add only standout weights ("What is committed under `runs/`").

## Done: rl-006, longer exploiters in the league (29 September)

Agreed by the user on 29 September ("Kick off the training that has been prepared and is ready to continue") and carried out that night: step 1 confirmed the exploitability of rl-005's 3600 (Results, "Exploitability, confirmed or not"), and step 2 ran rl-006 and chose its 2000 (Results, "rl-006"). The plan as it was written, on 28 September, follows; its "Later" list and the record of what is committed under `runs/` moved to "Next" above.

**Where it stands.** The best policy is **rl-005's 3600**
(`runs/rl-005/checkpoints/policy-3600.{pt,npz}`, `critic-3600.pt`,
committed), and since 28 September the web game's default bot
(`web/bot.npz`, a copy of its `.npz`; replace it only when the user asks).
It plays the cards at +19.9 ± 2.1 against RuleBot (rl-004d's 10: +3.9) and
beats every reference field, but **a 100-iteration exploiter takes +32.0 ±
15.4 from it** (rl-005's 2000: +2.6 ± 14.3; one exploiter each), mostly by
declaring more against it; the league's own 50-iteration exploiters did not
show this near 3600. The in-run curve and the full game against RuleBot
were still rising at 4000; card play gained +6.7 from 2000 to 3600 and was
level after. The bidding needs no curriculum (T4.2), the call no change
(T4.3), and entropy 0.03 stays (0.01 was a near thing).

### Before starting

Start the Claude Code session in `~/danish-wist-training`, so that this
branch's `CLAUDE.md` is the one loaded (the main checkout's still describes
the MacBook). Each shell command runs in a fresh shell: begin every one with
`cd ~/danish-wist-training && export PATH=$PWD/.venv/bin:$PATH &&`, and
repeat variables such as `REF` in every command that uses them (`choose.sh`
silently falls back to its old defaults when `FIELDS` or `SEEDS` is empty).

```sh
cd ~/danish-wist-training && git pull            # the branch `training`
export PATH=$PWD/.venv/bin:$PATH
python -m pytest -q                              # about 3 minutes; all should pass
nvidia-smi                                       # the GPU should be nearly empty
pgrep -af '^python -m learn\.'                   # no other runs of ours
```

- **The machine is shared** with another Claude session on the CPU only;
  leave its processes alone. A training run holds about 6 GB of GPU memory and starts a worker on all cores but two (mostly idle, waiting for the GPU); anything alongside it on the GPU uses `--workers 4`. **The time estimates below were made before the throughput work of 28 September: runs now take about half as long.**
- **Anything longer than a few minutes** runs detached (`nohup ... &`, as
  below); wait for it from a background shell (the Bash tool's
  `run_in_background`), not in the foreground.
- **Stop a process by its exact PID** (`pgrep -f '^python -m
  learn.selfplay'`, anchored, to find it). **Wait on a PID or a file, never
  on `pgrep -f "<pattern>"`**: the harness runs each command inside `bash
  -c "... <command> ..."`, so an unanchored pattern matches the waiting
  shell itself and the waiter never ends (three such waiters looped for
  hours on 27–28 September). For example, in a background shell: `while
  kill -0 $PID; do sleep 15; done`, or wait until `log.jsonl` reaches the
  last iteration.
- `runs/` is gitignored, but chosen checkpoints are force-added (`git add
  -f`) with the user's agreement of 28 September; §7's older advice on
  checkpoints and pull requests is superseded by this section.

### Step 1: confirm the exploitability (about 15 minutes)

The two margins come from one exploiter each. Repeat 3600's with another
seed, and give the 2000 an exploiter twice as long, to see whether the
policy became more exploitable or 100 iterations happened to reach it there
and not before (GPU, 12 workers; one after the other, detached):

```sh
nohup sh -c '
python -m learn.exploit runs/rl-005/checkpoints/policy-3600.pt \
    --critic runs/rl-005/checkpoints/critic-3600.pt --seed 1 --out runs/x-005-3600-s1 \
    > runs/x-005-3600-s1.out 2>&1
python -m learn.exploit runs/rl-005/checkpoints/policy-2000.pt \
    --critic runs/rl-005/checkpoints/critic-2000.pt --iterations 200 --out runs/x-005-2000-200 \
    > runs/x-005-2000-200.out 2>&1
' > runs/step1.out 2>&1 &
```

Each writes `margin.json`. For each, copy `run.json`, `settings.json`,
`log.jsonl`, `evals.jsonl` and `margin.json` to `results/<exploiter>/`, and
the role split at the end of its `.out` (from "exploiter margin against")
to `results/<exploiter>/margin.txt`, as for x-005-3600. **Confirmed** if
x-005-3600-s1's margin is significant (its interval above 0). Record both in
Results ("Exploitability, confirmed or not"), with each exploiter's in-run
`vs_target` curve beside x-005-2000's and x-005-3600's. Force-add both
exploiters' `policy.pt` and `policy.npz` (as x-005-2000's and x-005-3600's
are), commit and push. **If it is not confirmed, stop and report** before
training anything else.

### Step 2: rl-006, exploiters as long as `learn.exploit`'s (about 4 hours)

From rl-005's 3600 with rl-005's settings and one change, `--exploit-iterations
100` (rl-005: 50): the league's exploiters then train as long as the one that
found +32. The league starts with exactly these four: rl-005's 2000 and 4000,
x-005-2000 and x-005-3600 (step 1's exploiters are measurements and do not
join):

```sh
nohup python -m learn.selfplay \
    --init runs/rl-005/checkpoints/policy-3600.pt \
    --init-critic runs/rl-005/checkpoints/critic-3600.pt --critic-warmup 2 \
    --league-add runs/rl-005/checkpoints/policy-2000.pt runs/rl-005/checkpoints/policy-4000.npz \
        runs/x-005-2000/policy.pt runs/x-005-3600/policy.pt \
    --exploit-every 100 --exploit-iterations 100 \
    --magnet 0.1 --magnet-ema 0.01 --explore-bids 0.15 --explore-levels 0.1 \
    --entropy 0.03 \
    --deals 1024 --iterations 2000 --eval-every 10 --eval-deals 2000 \
    --out runs/rl-006 > runs/rl-006.out 2>&1 &
```

At about 5.5 s an iteration plus about 190 s for each exploiter phase, 2000
iterations take about 4 hours. Check the first few iterations' log (the
league's members, `exploiter_margin` at 100) before leaving it.

**Watch** as §5 says, every 30–60 minutes: `python -m learn.curve
runs/rl-006`, the log's health, that it is still growing, and
`exploiter_margin` every 100 iterations: with longer exploiters it should
first be larger (they find more), then fall if the learner closes the gap.
**Card-play checks** at iterations 500, 1000 and 1500, alongside the run,
on the fresh choosing seeds 35 and 36, paired with rl-005's 3600 (for 1000
and 1500, list the earlier checkpoints too, as rl-005's checks did):

```sh
mkdir -p results/rl-006
python -m learn.arena --workers 4 --field rule --seeds 35 36 --deals 2000 \
    --candidate play:runs/rl-005/checkpoints/policy-3600.npz play:runs/rl-006/checkpoints/policy-0500.npz \
    > results/rl-006/play-check-0500.txt
```

**Stop and report** if a value is NaN, entropy collapses towards 0, steps
are skipped, card play is significantly below rl-005's 3600 at two checks in
a row, or `vs_rulebot` falls significantly three evaluations in a row.
Otherwise let it run; don't change code during it except to fix a real bug,
and say so.

**After it: choose.** Among every 200th iteration and the last, paired with
rl-005's 3600 (itself a candidate, at 0), on seeds 35 and 36, against the
reference set RuleBot, rl-005's 2000, rl-005's 3600 and x-005-3600 (§6):

```sh
REF="rule runs/rl-005/checkpoints/policy-2000.npz runs/rl-005/checkpoints/policy-3600.npz runs/x-005-3600/policy.pt"
SEEDS="35 36" START=runs/rl-005/checkpoints/policy-3600.npz RUN=runs/rl-006 FROM=200 FIELDS="$REF" \
    sh results/rl-005/choose.sh > results/rl-006/choose.txt
```

1. **Rank** by the smallest of the reference fields' paired results
   (rl-005's 3600 counts as 0). If rl-005's 3600 leads, rl-006 did not beat
   it: say so, report the best rl-006 candidate for information only, and
   skip the rest.
2. **Ties.** Pair the leader (iteration LLLL) directly with every candidate
   whose smallest is within the leader's interval; a candidate is **tied**
   with the leader if none of its paired results in that direct pairing is
   significant (as in "rl-005, continued"):

   ```sh
   REF="rule runs/rl-005/checkpoints/policy-2000.npz runs/rl-005/checkpoints/policy-3600.npz runs/x-005-3600/policy.pt"
   SEEDS="35 36" START=runs/rl-006/checkpoints/policy-LLLL.npz FIELDS="$REF" \
       C="runs/rl-006/checkpoints/policy-AAAA.npz runs/rl-006/checkpoints/policy-BBBB.npz" \
       sh results/rl-005/choose.sh > results/rl-006/choose-LLLL-vs-neighbours.txt
   ```

3. **Exploiters decide among tied ones** (new for rl-006; §6 otherwise
   applies): run `learn.exploit` (seed 0, 100 iterations, `--out
   runs/x-006-NNNN`) on the leader and on each candidate tied with it.
   Prefer a candidate whose margin is not significant (its interval
   includes 0) over one whose margin is significant; among those left,
   card play decides, as §6 says. Without ties the leader is chosen and its
   seed-0 exploiter is simply its margin.

**Report** the chosen iteration NNNN on seeds 0, 41 and 42, paired with
rl-005's 3600, with rl-005's 2000 alongside, and measure it. Its seed-0
exploiter took part in choosing, so its margin there is biased low: report
a second exploiter on seed 1 as its margin, with the seed-0 one beside it.

```sh
REF="rule runs/rl-005/checkpoints/policy-2000.npz runs/rl-005/checkpoints/policy-3600.npz runs/x-005-3600/policy.pt"
SEEDS="0 41 42" START=runs/rl-005/checkpoints/policy-3600.npz FIELDS="$REF" \
    C="runs/rl-006/checkpoints/policy-NNNN.npz runs/rl-005/checkpoints/policy-2000.npz" \
    sh results/rl-005/choose.sh > results/rl-006/report-NNNN.txt
python -m learn.exploit runs/rl-006/checkpoints/policy-NNNN.pt \
    --critic runs/rl-006/checkpoints/critic-NNNN.pt --seed 1 --out runs/x-006-NNNN-s1 \
    > runs/x-006-NNNN-s1.out 2>&1
python -m learn.beliefs runs/rl-006/checkpoints/policy-NNNN.pt --deals 2000 > results/rl-006/beliefs-NNNN.txt
{ python -m learn.margins runs/rl-006/checkpoints/policy-NNNN.npz --field rule --deals 2000
  python -m learn.margins runs/rl-006/checkpoints/policy-NNNN.npz --deals 2000; } > results/rl-006/margins-NNNN.txt
python -m learn.arena --candidate runs/rl-006/checkpoints/policy-NNNN.npz --field rule --deals 2000 \
    --record runs/arena/rl-006-NNNN-vs-rule.jsonl > runs/arena/rl-006-NNNN-vs-rule.out
python -m learn.report runs/arena/rl-006-NNNN-vs-rule.jsonl > results/rl-006/report-bidding-NNNN.txt
python results/rl-005/scripts/alone.py runs/arena/rl-006-NNNN-vs-rule.jsonl > results/rl-006/alone-NNNN.txt
python results/rl-005/scripts/hand_reading.py runs/arena/rl-006-NNNN-vs-rule.jsonl > results/rl-006/hand-reading-NNNN.txt
```

(Record the arena to a fresh file: `--record` appends.) Each exploiter's
small files go to `results/x-006-*/` as in step 1.

**Write it up** as "rl-006" in Results, at the level of "rl-005, continued"
(its window table, the checks, the choosing and reporting tables, the
exploiters, the bidding), and copy `run.json`, `settings.json`, `log.jsonl`
and `evals.jsonl` to `results/rl-006/`. **Commit** the same kind of
checkpoint set as rl-005's, force-added: the resume set (`state.pt`,
`run.json`, `settings.json`, `policy.pt`, `policy.npz`, `critic.pt`, and
`git add -f -A runs/rl-006/league`), the chosen checkpoint's `.pt`, `.npz`
and critic, every 200th `.npz`, the policies (`policy.pt`, `policy.npz`) of
every exploiter made by `learn.exploit` in steps 1 and 2 (`runs/x-005-*`,
`runs/x-006-*`), and the recorded deals. If the committed files under
`runs/` pass about 200, raise it with the user. **Do not replace
`web/bot.npz`**: say in the report whether the chosen one beats rl-005's
3600 on every measure, and let the user decide.

### At the end

`pytest -q` and `ruff check . && ruff format --check .`; the Results
entries, and the Summary table's and "The review's measures" rows for the
chosen checkpoint; "Where things stand" updated; this section turned into a
"Done" record (as "Done: rl-005 carried on" was) with a new "Next"; commit
and push `training` (no pull request to `main` unless the user asks).

### The handover prompt (28 September)

The prompt written for the user to give the next agent (with a fresh
context) to carry out steps 1 and 2; giving it is the agreement to them:

> You are continuing the Danish Wist learned-bot project on the RTX 5090
> workstation. Work in the git worktree ~/danish-wist-training on the branch
> `training`, with the Python in its .venv (git pull first).
>
> Read, in this order: CLAUDE.md; TRAINING.md from the top ("Where things
> stand"), then its "Next" section in full, then §5 ("Watching it") and §6
> ("After the run: measuring and choosing"); then the "rl-005, continued",
> "Bidding and alone contracts" and "The entropy check" entries in Results.
>
> Your task is TRAINING.md "Next", steps 1 and 2, which I (the user) have
> already agreed to: confirm the exploitability of rl-005's 3600 (step 1, two
> `learn.exploit` runs, about 15 minutes); if step 1 confirms it as defined
> there, run rl-006 (step 2, about 4 hours) with the command given there,
> watching it as §5 says, doing the card-play checks at 500, 1000 and 1500,
> then choosing (including the exploiter margins of the tied candidates),
> reporting and measuring as step 2 describes. You don't need to ask me
> before starting those runs and exploiters; anything else that trains needs
> my agreement. Stop and report to me if step 1 does not confirm the
> exploitability, if one of the stop rules fires, or if something unexpected
> makes the plan unsound.
>
> Record decisions and results in TRAINING.md's Results as you go, at the
> level of detail of the "rl-005, continued" entry, and copy each run's and
> exploiter's small files to results/<name>/. Commit checkpoints as step 2
> describes (a chosen set, not every checkpoint). Do not replace
> `web/bot.npz` (the web game's default bot, rl-005's 3600) unless I ask.
> Run `pytest -q` and `ruff check . && ruff format --check .` before
> committing. Commit to `training` and push it after step 1 and at the end.
> Don't open a pull request to main unless I ask.
>
> The machine is shared with another Claude session working on another
> project (CPU only); leave its processes alone. Stop your own runs by
> killing their main Python process by its PID; never `pkill -f` a pattern,
> and never wait on `pgrep -f "<pattern>"` (it matches the waiting shell
> itself); wait on a PID or a file instead, as "Next" explains.
>
> When you finish, tell me in plain terms whether the exploitability was
> confirmed, how rl-006 went and how its chosen checkpoint compares with
> rl-005's 3600 and 2000 on the review's measures (card play on fixed
> contracts, the reference fields, the exploiter margin), whether you think
> it should replace 3600 as the web game's bot, and what you would do next.

## Done: rl-005 carried on, and three pieces of work around it

Written 28 September 2026 for the next agent, and **done on 28 September**
("The reference set", "rl-005, continued", "Bidding and alone contracts"
and "The entropy check" in Results). Kept as the record of how they were
run. **The user had agreed to the
four tasks below, including the runs in tasks 1 and 4.** Anything else
that trains needs the user's agreement first. Read this section, §5 and §6,
then "rl-005" in Results. Order: task 2 first (short), then start task 1
and do task 3 while it runs, then task 4 once the GPU is free.

### Before starting

```sh
cd ~/danish-wist-training && git pull            # the branch `training`
export PATH=$PWD/.venv/bin:$PATH
python -m pytest -q                              # about 3 minutes; all should pass
nvidia-smi                                       # the GPU should be nearly empty
pgrep -af "learn\."                              # no other runs of ours
```

- **The machine is shared** with another Claude session on the CPU only
  (`~/hard-poc`); leave its processes alone. **GPU memory** limits doing two
  things at once: a training run holds about 13 GB, so anything alongside it
  uses `--workers 4`. **Stop a run by killing its main Python process**
  (`pgrep -f '^python -m learn.selfplay'`); never `pkill -f` a pattern that
  also matches your own shell.
- **What is committed under `runs/`** (force-added, 28 September):
  rl-005's resume set (`state.pt`, `run.json`, `settings.json`, the latest
  `policy.pt`, `policy.npz`, `critic.pt`, and `league/`), rl-005's 2000
  (`policy-2000.pt`, `.npz`, `critic-2000.pt`), its 1000 (`policy-1000.pt`,
  a league member for task 4), every 200th checkpoint's `.npz` (the choosing
  candidates), the exploiter x-005-2000's policy, and 2000 recorded deals of
  rl-005's 2000 among RuleBots (`runs/arena/rl-005-2000-vs-rule.jsonl`). On
  the workstation the whole run directory is there anyway. From a fresh
  clone, copy `results/rl-005/log.jsonl` and `evals.jsonl` into
  `runs/rl-005/` before resuming (a resume trims the log to the saved
  iteration; checked on 28 September from exactly this set).
- **Within a run the league fills itself**: a snapshot joins every 10
  iterations and an exploiter every 100, up to 50 members (older ones thin
  out), drawn by priority, as in AlphaStar's league and OpenAI Five's past
  versions. Nobody adds checkpoints by hand. `--league-add` only seeds a new
  run with other runs' policies; a resumed run keeps its own league.

### Task 2: the reference set (about 15 minutes; do it first)

**Done on 28 September** ("The reference set" in Results; §6 updated).

rl-005's 2000 beats rl-003's 110, rl-004d's 10 and x-004d-0010 by about 100
per deal, so they no longer discriminate, and RuleBot was the smallest field
at every checkpoint. **The reference set from now on: RuleBot, rl-004d's 10,
rl-005's 2000 and the latest exploiter, x-005-2000.** Measure the two
policies against it on the reporting seeds, then write it into §6 (the
text, and the example commands):

```sh
REF="rule runs/rl-004d/checkpoints/policy-0010.npz runs/rl-005/checkpoints/policy-2000.npz runs/x-005-2000/policy.pt"
SEEDS="0 41 42" START=runs/rl-004d/checkpoints/policy-0010.npz \
    C=runs/rl-005/checkpoints/policy-2000.npz FIELDS="$REF" \
    sh results/rl-005/choose.sh > results/rl-005/reference-2026-09-28.txt
```

Add a short table of it to Results. The evaluation set is the one manual
step left: the published systems rate each checkpoint automatically against
their earlier ones (an Elo or TrueSkill ladder). Worth building later (each
checkpoint every 200 iterations played, paired, against the ones before);
not part of these four tasks.

### Task 1: carry rl-005 on to 4000 iterations (about 3.2 hours)

```sh
nohup python -m learn.selfplay --resume runs/rl-005 --iterations 4000 >> runs/rl-005.out 2>&1 &
```

Only `--iterations`, `--workers` and `--device` can change on a resume; the
league, the optimisers, the magnet and the random state carry on. At rl-005's
pace (about 5.9 s per iteration with the exploiters) 1900 iterations take
about 3.2 hours. Watch as §5 says, every 30–60 minutes: `python -m
learn.curve runs/rl-005`, the log's health, `exploiter_margin` every 100
iterations, and that the log is still growing. Entropy was 0.93 at 2100 and
rising by about 0.01 per 100 iterations: expected, not a stop rule.

**Card-play checks** at iterations 2500, 3000 and 3500, alongside the run,
on the fresh choosing seeds 33 and 34 (§6's table):

```sh
python -m learn.arena --workers 4 --field rule --seeds 33 34 --deals 2000 \
    --candidate play:runs/rl-005/checkpoints/policy-2000.npz play:runs/rl-005/checkpoints/policy-2500.npz
```

**Stop and report** if a value is NaN, entropy collapses towards 0, steps
are skipped, card play is significantly below rl-005's 2000 at two checks in
a row, or `vs_rulebot` falls significantly three evaluations in a row (§5).
Otherwise let it run to 4000; don't change code during it except to fix a
real bug, and say so.

**After it: choose, report, measure** (§6's protocol, with task 2's set).
Choose among every 200th iteration from 2200 and the last, paired with
rl-005's 2000 (itself a candidate, at 0), on seeds 33 and 34:

```sh
SEEDS="33 34" START=runs/rl-005/checkpoints/policy-2000.npz FROM=2200 FIELDS="$REF" \
    sh results/rl-005/choose.sh > results/rl-005/choose-4000.txt
```

Choose by the smallest of the reference fields' paired results (rl-005's
2000 counts as 0), ties within the paired interval broken by card play.
Then report the chosen one on seeds 0, 41 and 42, paired with rl-005's
2000, with rl-004d's 10 alongside:

```sh
SEEDS="0 41 42" START=runs/rl-005/checkpoints/policy-2000.npz FIELDS="$REF" \
    C="runs/rl-005/checkpoints/policy-NNNN.npz runs/rl-004d/checkpoints/policy-0010.npz" \
    sh results/rl-005/choose.sh > results/rl-005/report-NNNN.txt
python -m learn.exploit runs/rl-005/checkpoints/policy-NNNN.pt \
    --critic runs/rl-005/checkpoints/critic-NNNN.pt --out runs/x-005-NNNN   # 4 minutes
python -m learn.beliefs runs/rl-005/checkpoints/policy-NNNN.pt --deals 2000
python -m learn.arena --field rule --seeds 0 --deals 500 --worlds 100 \
    --candidate play:runs/rl-005/checkpoints/policy-NNNN.npz \
    play:critic-reply:runs/rl-005/checkpoints/policy-NNNN.pt  # about 10 minutes
```

Write it up as "rl-005, continued" in Results, at the level of rl-005's
entry; copy `run.json`, `settings.json`, `log.jsonl` and `evals.jsonl` to
`results/rl-005/` again. **Commit the checkpoints** (the user agreed on 28
September to commit them while they are small): the new resume set, with
`git add -f -A runs/rl-005/league` so that thinned-out members are removed
too; the chosen checkpoint's `policy-NNNN.pt`, `.npz` and `critic-NNNN.pt`;
every 200th `.npz` from 2200; and the new exploiter's policy. Not every
checkpoint: rl-005 has 630 files (1.7 GB). If the committed set grows to
hundreds of files, raise it with the user.

### Task 3: the bidding and alone contracts (REVIEW.md T4.2, T4.3), while task 1 runs

Analysis, not training: use `--workers 4` for anything on the GPU while task
1 runs. Deliverable: a Results entry "Bidding and alone contracts" with the
tables and a recommendation. No change to training settings, the rules or
the engine without the user.

- **Alone contracts (T4.3).** Among RuleBots rl-005's 2000 plays alone in
  5.6% of its seats at −875 per contract (RuleBot: 1.3%, −53), about 49
  points per deal (`results/rl-005/report-bidding-2000.txt`). By RULES.md §5
  a declarer is alone when it calls an ace it holds or the called ace is in
  the cat. From the record (`runs/arena/rl-005-2000-vs-rule.jsonl`) split
  its alone contracts by cause, by the aces and Jokers it held, by kind and
  level, and by result, and check how often it called its own ace when
  another call was allowed. `learn.report` has the roles but not the cause:
  add it (a flag or a short section, with a test), or a script under
  `results/rl-005/scripts/`. If calling its own ace with a choice is
  common, measure what another call would have scored: `learn.margins`
  varies only auction decisions, so that means extending it to the call
  (code with tests).
- **The bidding (T4.2).** The bidding-first curriculum (LEARNING.md) is
  indicated only if a level higher pays for strong hands and the policy
  does not bid it. So far a level higher loses for every strength of hand
  (−315 ± 30 at 2000, `results/rl-005/margins-2000.txt`). Rerun `learn.margins`
  (`--field rule`, and in self-play) on task 1's chosen checkpoint, record
  2000 deals of it among RuleBots and run `learn.report` on them, and say
  whether that changed and how much the bidding now reads the hand.

### Task 4: entropy, a one-change check (about 50 minutes), after task 1

Entropy rose from 0.53 to 0.93 over rl-005 at a bonus of 0.03. Two runs of
200 iterations from rl-005's 2000 (policy and critic) with rl-005's other
settings, exploiters every 50 iterations for 25, and a league seeded with
rl-004d's 10, rl-005's 1000, x-004d-0010 and x-005-2000: the control at 0.03
and one at 0.01. Both scripts were checked on 28 September (10 iterations,
100 deals).

```sh
nohup sh results/rl-005/entropy-check.sh > runs/entropy-check.out 2>&1 &
# when both runs are done:
PREFIX=ent START=runs/rl-005/checkpoints/policy-2000.npz FIELDS="$REF" \
    sh results/workstation-2026-09/judge.sh > results/rl-005/entropy-judge.txt
```

Deciding, from `entropy-judge.txt` (each line paired with `ent-control`):
**0.01 is taken for the next new run** if it is significantly better in
card play or against at least two reference fields, and significantly
worse in none; otherwise 0.03 stays (a tie keeps the current setting). As
in the sweep, a setting whose exploiters gain much more, or whose entropy
collapses, is not taken. A resumed run keeps its own settings, so this
decides the next new run, not task 1. Record the table and the decision in
Results ("The entropy check") and copy each run's small files to
`results/ent-*/`.

### At the end

`pytest -q` and `ruff check . && ruff format --check .`; the Results entries;
REVIEW.md's T4.2 and T4.3 marked with what was found; "Where things stand"
and "Next" updated; commit and push `training` (no pull request to `main`
unless the user asks). Then tell the user in plain terms how the
continuation went and how its chosen checkpoint compares with rl-005's 2000
and rl-004d's 10 on the review's measures, what the bidding and alone
analysis found, what the entropy check decided, and what you would do next.

### Later (as planned then; superseded by "Next")

- **Search** (T3.2, T3.3): critic search still adds about +11 in card play
  on top of the better policy; expert iteration needs a much cheaper search
  first (about 0.3 s per decision on one core now).
- **`--stake-scaling`** was dropped on a near tie (card play −1.5, interval
  −3.0 to +0.1, all five point estimates negative); a one-change check with
  it back would settle it.
- **A new run seeded with the new reference policies** (`--league-add`
  rl-005's best and its exploiters) once the continuation levels off.
- To play against the policy: `python -m web.server --bot
  runs/rl-005/checkpoints/policy-2000.npz`.

## Done: the sweep, then the long run

Written 27 September 2026 for the next agent, and **done on 27–28
September** ("The sweep" and "rl-005" in Results). The user had agreed to
both runs. They are REVIEW.md's T2.3 and T2.6 (the sweep) and T4.1 (the
long run). Kept as the record of how they were run; the commands serve for
the next ones.

### Before starting

```sh
cd ~/danish-wist-training && git pull            # the branch `training`
.venv/bin/python -m pytest -q                    # about 4 minutes; all should pass
nvidia-smi                                       # the GPU should be nearly empty
pgrep -af "learn\."                              # no other runs of ours
```

Everything the runs read is committed under `runs/` (force-added):
rl-004d's 10 (`policy-0010.pt`, `critic-0010.pt`, the start), rl-003's 110,
rl-004d's 100, the exploiters x-004b and x-004d-0010. Put `.venv/bin` first
on `PATH` for the scripts: `export PATH=$PWD/.venv/bin:$PATH`.

Things to know about the machine:

- **It is shared.** Another Claude session may be working on another project
  (`~/hard-poc`), on the CPU only. It makes timings vary; leave it alone.
- **GPU memory is the limit on doing two things at once.** A training run
  holds about 13 GB (12 workers of about 1 GB, and the update). An arena
  alongside it should use `--workers 4` (about 1 GB each); critic search
  more per worker. Two runs at once would slow each other (the GPU switches
  between processes): run things one after another.
- **Stop a run by killing its main Python process** (`kill <pid>`); its
  workers stop by themselves. Find it with `pgrep -f '^python -m learn.selfplay'`
  or `ps -eo pid,args | grep learn.selfplay`. Never `pkill -f` a pattern that
  also matches your own shell's command line.
- `--resume runs/<run>` carries a stopped run on exactly (§4).

### Step 1: the sweep (about 2 hours)

Six runs of 200 iterations from rl-004d's 10, one change each from a
control: the league on (RuleBot, the start, rl-003's 110, rl-004d's 100,
x-004b and x-004d-0010, and snapshots as they come), an exploiter trained
every 50 iterations for 25, magnet 0.1 with `--magnet-ema 0.01`, 1024 deals
per iteration, evaluation every 10 on 2000 deals. `results/workstation-2026-09/sweep.sh`
runs them one after another, skipping any already run:

| Run | Change from the control |
|---|---|
| `sw-control` | none: entropy 0.01, policy lr 1e-4, one epoch, `--explore-bids 0.15 --explore-levels 0.1 --stake-scaling` |
| `sw-entropy-0.03`, `sw-entropy-0.1` | `--entropy` (T2.3) |
| `sw-lr-2.5e-4-2ep` | `--policy-lr 2.5e-4 --ppo-epochs 2` (T2.3) |
| `sw-no-explore-levels` | without `--explore-levels` (T2.6) |
| `sw-no-stake-scaling` | without `--stake-scaling` (T2.6) |

```sh
nohup sh results/workstation-2026-09/sweep.sh > runs/sweep.out 2>&1 &
```

Each takes about 20 minutes (the two-epoch one about 30): an iteration is
about 2 s playing and 3 s updating (6 s with two epochs), and each exploiter
phase about a minute. Every run was checked for one iteration on 27
September. Watch as §5 says: `tail runs/sweep.out`, `tail runs/sw-*.out`,
`python -m learn.curve runs/sw-control`. Healthy, from a 30-iteration trial:
`approx_kl` about 0.005, `magnet_kl` rising to 0.02–0.03 and flat, entropy
steady, `skipped_steps` 0, and in `league.against` win rates of 0.45–0.5
(a single deal's result is mostly luck, so they differ little and the
draws are near uniform). `exploiter_margin` appears at iterations 50, 100,
150 and 200. Stop and report only for §5's reasons (NaN, collapsing
entropy); 200 iterations are too few for the `vs_rulebot` stop rule. If a
run stops part way, `--resume` it by hand to 200 iterations
(`python -m learn.selfplay --resume runs/sw-... --iterations 200`) before
running the script again: the script skips any run that has a log.

### Step 2: judge the sweep (about 10 minutes)

```sh
sh results/workstation-2026-09/judge.sh > results/workstation-2026-09/judge.txt
```

It plays every run's iteration 200, and the start, on the choosing seeds 31
and 32 (4000 deals): card play on fixed contracts, and the full game against
each reference field (RuleBot, rl-003's 110, rl-004d's 10, x-004d-0010). All
are paired with `sw-control`, so each "paired" line is one change's effect.
It ends with each run's in-run exploiter margins (lower is harder to exploit).

Deciding the long run's settings, from judge.txt:

1. **A change is taken** if, paired with the control, it is significantly
   better (the whole interval above 0) in card play or against at least two
   reference fields, and significantly worse in none.
2. **A removal is taken** (`sw-no-explore-levels`, `sw-no-stake-scaling`) if
   it is significantly worse in none: the simpler setting wins a tie.
3. If both entropies qualify, take the one better in card play. Settings
   that qualify are combined.
4. **If the sweep cannot tell them apart**, take the review's recipe (T4.1):
   entropy 0.03, `--policy-lr 2.5e-4 --ppo-epochs 2`, except any setting that
   was significantly worse than the control.
5. Also look at the exploiter margins and the logs' health: a setting whose
   exploiters gain much more, or whose entropy collapses, is not taken
   whatever else it scores.

Write the table and the decision, with its reasons, into Results
("The sweep") before starting the long run, and commit it with each run's
`run.json`, `settings.json`, `log.jsonl` and `evals.jsonl` copied to
`results/sw-*/` (§7).

### Step 3: the long run (about 4–6 hours)

From rl-004d's 10 again, not from a sweep run: one clean run, no restarts
(REVIEW.md §2.4). About 10^8 recorded decisions: 2100 iterations of 1024
deals (about 47,500 each). With the settings from step 2 in place of the
marked ones:

```sh
nohup python -m learn.selfplay \
    --init runs/rl-004d/checkpoints/policy-0010.pt \
    --init-critic runs/rl-004d/checkpoints/critic-0010.pt --critic-warmup 2 \
    --league-add runs/rl-003/checkpoints/policy-0110.pt runs/rl-004d/checkpoints/policy-0100.pt \
        runs/x-004b/policy.pt runs/x-004d-0010/policy.pt \
    --exploit-every 100 --exploit-iterations 50 \
    --magnet 0.1 --magnet-ema 0.01 --explore-bids 0.15 \
    --explore-levels 0.1 --stake-scaling \
    --entropy 0.03 --policy-lr 2.5e-4 --ppo-epochs 2 \
    --deals 1024 --iterations 2100 --eval-every 10 --eval-deals 2000 \
    --out runs/rl-005 > runs/rl-005.out 2>&1 &
# step 2 decides --explore-levels, --stake-scaling, --entropy, --policy-lr and --ppo-epochs
```

Watching (§5), every 30–60 minutes: `python -m learn.curve runs/rl-005`, the
log's health, `exploiter_margin` every 100 iterations (falling is good: the
policy is getting harder to exploit), `league.members` (at most 50; older
snapshots thin out), and that the log is still growing. At iterations 500,
1000 and 1500, the card-play test alongside the run, with few workers:

```sh
python -m learn.arena --workers 4 --field rule --seeds 31 32 --deals 2000 \
    --candidate play:runs/rl-004d/checkpoints/policy-0010.npz play:runs/rl-005/checkpoints/policy-0500.npz
```

**Stop and report** if a value is NaN, entropy collapses, steps are skipped,
or card play is significantly below the start at two checks in a row.
Otherwise let it run to the end; don't change code during it except to fix
a real bug, and say so (the rl-004 line's lesson). Update Results as it goes
(the rl-004 entries show the level of detail), and commit now and then.

### Step 4: after the long run

§6's protocol. Choose among the checkpoints every 200 iterations and the
last, with the start, on seeds 31 and 32 (`judge.sh`'s commands with those
candidates); report the chosen one on seeds 0, 41 and 42 in the table of
"The review's measures on the workstation"; its exploiter margin with
`learn.exploit` (4 minutes); `learn.beliefs`; `learn.margins --field rule`;
and card play with critic search, `play:critic-reply:<policy.pt>` against
its plain play on 500 deals (about 10 minutes; 2000 took 40). Then the
Results entry, `results/rl-005/`, REVIEW.md's T2.3, T2.6 and T4.1 ticked,
commit and push `training`, and report to the user. Leave a pull request to
`main` for the user to ask for.

### After that (REVIEW.md §4, still open; see "Next" above)

T2.4 (expected-SARSA advantages); T2.5 as a belief network with its own
trunk (a head fit alone cannot help: see the probe in Results); T3.2
(weighting worlds by the likelihood of the others' actions); T3.3 (expert
iteration: distilling critic search into the policy, which needs a much
cheaper search, since it costs about 0.3 s per decision on one core now);
T3.4 (an endgame solver). (T4.2 and T4.3 were done on 28 September:
"Bidding and alone contracts".)

### Earlier plans (superseded)

The review's todo list (27 September) superseded the list below.

Previous (26 September), in rough order:

1. **Defending.** It is the weakest role in every run and got worse as
   self-play moved to Flip (in-run −69 → −121 over rl-004c). *The review
   showed this is the auction's doing, not defensive play: see REVIEW.md §2.3.*
   A broader field would help: more deals with RuleBot and older snapshots in
   some seats (`Settings.opponent_share` is 0.25 and not a command-line flag
   yet) and a real league (LEARNING.md step 7). Both are code changes.
2. **An averaged policy**, the remedy for cycling named in the last session's
   plan and not tried, since it needs code: keep an exponential moving average
   of the policy's weights and evaluate and export that. The magnet slowed
   the drift (0.3 held about 120 iterations, 1.0 held but stopped improving)
   without stopping it. *The review recommends a magnet of about 0.1 with an
   EMA reference instead of a stronger one.*
3. **Drop `--explore-levels`?** It did not make higher contracts pay, and
   moves 10% of each bid's chance up one or two levels in every deal. A run
   from rl-004d's 10 without it (one change) would show whether it helped at
   all.
4. **The hand-blind bidding and the defence.** It still bids in 91% of seats
   with no ace or Joker. x-004b, an exploiter started from the policy itself,
   found about +20 per deal in 100 iterations, nearly all as declarer against
   the policy's defence (Summary). A longer exploiter, and training against
   exploiters (a league), are the tests and the cure.
5. **Tools.** The analysis scripts of this session (`results/rl-004/scripts/`:
   several policies on the same fresh deals from several seeds, pooled and
   paired) would make a small `learn.arena` extension: `--seeds` and paired
   comparison of several candidates. `learn.report` and `learn.contracts` overlap and could be
   merged.

Other open work, from LEARNING.md and the review: fit the belief head to the
final policy before search relies on it; and an engine question, that the
FUCDIC phase is visible to every seat (a declined fucdic tells the others the
declarer holds at most one card of the called suit; training is not affected,
since only the declarer acts in it).

## Handoff

- **Work moved to the 5090 workstation** on 27 September 2026 (REVIEW.md §5
  and todo T0, done that day): Intel Core Ultra 9 285K (8 performance + 16
  efficiency cores), 60 GB, RTX 5090 (32 GB), Linux. The git worktree
  `~/danish-wist-training` holds `training` (the main checkout
  `~/danish-wist` is on the default branch); Python
  `~/danish-wist-training/.venv/bin/python` is uv's CPython 3.13.14 with
  PyTorch 2.14.0+cu130 and NumPy 2.5.3 (NVIDIA driver 595.91.07). All tests
  pass there. The baselines reproduce exactly: `learn.arena` gives rl-004d's
  10 **+56.5 ± 15.7** against RuleBot and the fixed-contract card-play test
  (`hybrid_arena.py ... net-play`) **+3.0 ± 3.4**. The throughput work (T0.3)
  and its numbers are in PERFORMANCE.md, "The RTX 5090 workstation": about
  4.5 s per 1024-deal iteration instead of 56 s, and 7.6 s for a 2000-deal
  arena. The checkpoints the next run needs (the imitation start, rl-003's
  110, rl-004d's 10 and 100 with critics and resumable state, the x-004b
  exploiter, one recorded arena) are committed on this branch, force-added
  under the ignored `runs/`.
- **On the workstation, `runs/`** holds the committed handoff set, the
  exploiter x-004d-0010 (its `policy.pt` and `.npz` committed: it is a
  reference field and a league member) and the arena record. Everything
  else there is made by the runs above: the sweep's `sw-*/` (not
  committed) and **`rl-005/`** (`state.pt` resumes it, `checkpoints/` holds
  policy, critic and `.npz` every 10 iterations, `league/` the league's
  members; carried on to 4000 on 28 September), with **rl-005's best
  policy `rl-005/checkpoints/policy-3600.*` and `critic-3600.pt`** (before
  it the 2000), the exploiters `x-005-2000/` and `x-005-3600/`, the
  entropy check's `ent-control/` and `ent-0.01/`, and
  `arena/rl-005-2000-vs-rule.jsonl` and `-3600-vs-rule.jsonl` (2000
  recorded deals each among RuleBots).
- **Checkpoints in git.** On 28 September the user agreed to commit
  checkpoints while they are small, and to review that if hundreds pile up.
  So a chosen set of rl-005 is force-added (about 250 MB after the
  continuation; listed in "Next", "Committed under `runs/`"): what resumes
  it, the best policies, the choosing candidates, the exploiters and the
  recorded deals, not all of its 1200 checkpoint files (2.9 GB). Commit the
  same kind of set after each run.
- **The laptop keeps the rest of `runs/`** (gitignored), under its
  `~/danish-wist-training/runs/`:
  - `bc.pt`, `bc-explore.pt` and their `.npz`: the imitation starts.
  - `rl-001/` to `rl-003/`, `rl-004/`, `rl-004b/`, `rl-004c/`, `rl-004d/`:
    each with `state.pt` (resumable with `--resume`), `checkpoints/` (policy,
    critic and `.npz` every 10 iterations), logs and evaluations.
    rl-004d's `checkpoints/policy-0010.*` and `critic-0010.pt` were the
    best policy until 28 September (then rl-005's 2000, now its 3600).
  - `x-003/`, `x-004/`, `x-004b/`: the exploiters against rl-003's 110 and
    rl-004d's 10.
  - `arena/`, `analysis/`: the evaluations above (copied to `results/`).
- **Review scripts** (`results/review-2026-09/scripts/`): `hybrid_arena.py`
  (bidding and card play measured apart; `net-play` is the fixed-contract
  card-play test, ±3.5 over 2000 deals), `role_split.py` and
  `defender_split.py` (from a recorded arena: each role's advantage with the
  contract unchanged against changed).
- **Tools:** `learn.curve` (a run's curve and paired changes, `--follow` to
  watch), `learn.arena` (duplicate evaluation; several `--candidate`s and
  `--seeds`, paired with the first; `play:<policy>` for card play on fixed
  contracts; `critic-reply:<policy.pt>` for critic search; `--record` to keep
  every deal), `learn.exploit` (the exploiter margin), `learn.beliefs` (the
  belief head by phase), `learn.report` (from recorded deals: roles, contracts,
  alone contracts by cause, and bidding by hand strength), `learn.contracts` (plays policies itself:
  contracts by kind, level and result, `--phases`, `--sample`, `--field
  rule`), and `learn.margins` (each bid against a pass and one or two levels
  higher, same cards: whether the bidding fits the play). `learn.report` and
  `learn.contracts` overlap and could be merged.
- **Recorded arena deals** of rl-003's iteration 110, rl-002's last policy,
  and rl-004c's 150 and rl-004d's 10 and 100 are in
  `runs/arena/*-vs-rule.jsonl` (15 MB each; those of rl-004d's 10 and
  rl-005's 2000 and 3600 are committed, the rest stay on the laptop).
- **Before a long run,** agree it with the user. The user expects Flip to prove
  strong and bidding to grow high and aggressive, since a made contract pays
  for every trick; both have held so far.
