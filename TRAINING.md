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

**Where things stand** (28 September 2026, early morning, on the
workstation): the sweep and the long run agreed with the user are done.
The sweep ("The sweep" in Results) chose entropy 0.03 and dropped
`--stake-scaling`; the long run, **rl-005** (2100 iterations from rl-004d's
10 with the league, 3.5 hours), ran without drift or a stop rule. Its
iteration 2000 is **the best policy**: on the reporting seeds (6000 deals)
its **card play on fixed contracts is +13.3 ± 2.1** against RuleBot (rl-004d's
10: +3.9 ± 1.9, paired **+9.4 ± 2.1**), the full game +108.4 ± 11.8 against
RuleBot (**+52.4 ± 10.2** over rl-004d's 10) and about +100 against every
learned reference field, and its exploiter margin is +2.6 ± 14.3 (rl-004d's
10: +15.5 ± 9.7). With critic search its card play reaches +20.2 ± 7.7 (500
deals). Its bidding is bolder and still ignores the hand. See "rl-005" in
Results, then "Next" near the end.

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
`collect_s` and `update_s` per iteration. The update runs on `--device` and
the workers' networks on `--worker-device`; both default to the GPU, and
`--workers` to 12 when the workers use it (PERFORMANCE.md, "The RTX 5090
workstation").

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
   RuleBot, rl-003's 110, rl-004d's 10 and the latest exploiter, with
   several candidates on the same deals, paired (`--candidate A B ...`).
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
| **31, 32** | **Choosing** checkpoints from now on |
| **0, 41, 42** | **Reporting** (0 is where every earlier headline number was measured) |
| 101, 102 | `learn.exploit`'s margin |

**Choosing a checkpoint.** On seeds 31 and 32, 2000 deals each: the
candidates' card play, and their full game against each reference field,
paired with rl-004d's 10. Choose by the smallest of the reference fields'
paired results, and break ties within the paired interval by card play.
RuleBot's full-arena score is a sanity check (drop a run that falls below its
start there), not the objective. Then report the chosen one on seeds 0, 41
and 42 with the same commands, and its exploiter margin.

```sh
C="runs/rl-005/checkpoints/policy-0100.npz runs/rl-005/checkpoints/policy-0200.npz"
BEST=runs/rl-004d/checkpoints/policy-0010.npz
python -m learn.arena --field rule --seeds 31 32 --deals 2000 \
    --candidate play:$BEST $(for c in $C; do echo play:$c; done)
for field in rule runs/rl-003/checkpoints/policy-0110.npz $BEST runs/x-004d-0010/policy.pt; do
    python -m learn.arena --field $field --seeds 31 32 --deals 2000 --candidate $BEST $C
done
python -m learn.exploit runs/rl-005/checkpoints/policy-0200.pt \
    --critic runs/rl-005/checkpoints/critic-0200.pt --out runs/x-005-0200
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
| **rl-004d at iteration 10** | new | **+56.5 ± 15.7** (declarer +106, partner +121, defender −90; **+38.2 ± 14.2 over rl-003's 110**) |
| rl-004d at iteration 100 | new | +49.0 ± 16.3 (+30.8 ± 15.1 over rl-003's 110) |
| **rl-005 at iteration 2000** (workstation, 28 September) | new | **+105.1 ± 20.3** (**+48.6 ± 17.9 over rl-004d's 10**; `results/rl-005/summary-seed0.txt`) |

**rl-005's iteration 2000 is now the best policy**
(`runs/rl-005/checkpoints/policy-2000.npz` on the workstation, not
committed; chosen and reported as §6 says, in "rl-005" below). Until then
rl-004d's iteration 10 was the best
(`runs/rl-004d/checkpoints/policy-0010.npz`). It was also the
best of eight candidates on 4000 other fresh deals (+52.7 ± 10.8, +26.5 ± 9.8
over rl-003's 110), so about 6000 fresh deals in all back it. It gains mostly
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
| **rl-004d's 10** | **+3.9 ± 1.9** | **+56.1 ± 9.0** | **+67.4 ± 9.3** | 0 | **−4.4 ± 7.2** |
| rl-004d's 100 | +4.9 ± 1.9 | +51.8 ± 9.2 | +70.0 ± 9.1 | −0.2 ± 6.6 | −9.0 ± 6.9 |
| x-004b (exploiter of the 10) | +3.2 ± 1.9 | +27.2 ± 10.6 | +66.4 ± 10.0 | +9.4 ± 8.0 | −3.2 ± 6.8 |
| **rl-005's 2000** (28 September; "rl-005" below) | **+13.3 ± 2.1** | **+108.4 ± 11.8** | **+162.3 ± 11.2** | **+100.0 ± 11.3** | **+95.1 ± 10.4** |

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

### The sweep (27 September, commit 4199555)

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

### rl-005 (27–28 September, commit c6a20a4): the long run

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

### Imitation

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

## Next

Written 28 September 2026 after rl-005. Suggestions, in order; **agree any
run with the user first.**

1. **Carry rl-005 on** (`python -m learn.selfplay --resume runs/rl-005
   --iterations 4000`, about 3.5 hours more; it keeps the league, the
   optimisers and the magnet). It was still improving at the end, in card
   play and against every field, and had 76 million decisions, not the 10^8
   aimed at. One run, no restart (REVIEW.md §2.4); the card-play check every
   500 iterations and §6's choosing at the end, as for rl-005.
2. **Strengthen the reference set.** rl-005's 2000 beats rl-003's 110,
   rl-004d's 10 and x-004d-0010 by about 100 per deal, so they no longer
   discriminate, and RuleBot was the smallest field at every checkpoint.
   Add rl-005's 2000 and its exploiter x-005-2000 (`runs/x-005-2000/`) as
   fields (and to the league with `--league-add` in a new run), and choose
   by them.
3. **The bidding** (T4.2, T4.3). Card play is now clearly above RuleBot's,
   the review's condition for revisiting the bidding, yet the bids ignore
   the hand more than ever (level 9 Flip in about 95% of seats) and a level
   higher still loses (`learn.margins`, −315 ± 30). Alone contracts cost
   about 49 points per deal (5.6% of seats at −875): check the call and the
   level with aces in hand. The bidding-first curriculum in LEARNING.md is
   the tool if margins show a level up paying for strong hands.
4. **Entropy.** It rose from 0.53 to 0.93 over the run at 0.03, slowing.
   The greedy policy improved throughout, so this is not urgent; a one-change
   check (0.01 again, or a bonus decaying to 0.01) from rl-005's 2000 would
   show whether the widening costs anything now.
5. **Search** (T3.2, T3.3): critic search still adds about +11 in card play
   on top of the better policy. Expert iteration needs a much cheaper search
   first (about 0.3 s per decision on one core now).
6. `--stake-scaling` was dropped on a near tie (card play −1.5, interval
   −3.0 to +0.1, all five measures' point estimates negative); a one-change
   check from rl-005's 2000 with it back would settle it.

To play against the new policy: `python -m web.server --bot
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
T3.4 (an endgame solver); T4.2 and T4.3.

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
  else there is made by the runs above: the sweep's `sw-*/` and
  **`rl-005/`** (not committed; `state.pt` resumes it, `checkpoints/` holds
  policy, critic and `.npz` every 10 iterations, `league/` the league's
  members), **rl-005's best policy `rl-005/checkpoints/policy-2000.*` and
  `critic-2000.pt`**, its exploiter `x-005-2000/`, and
  `arena/rl-005-2000-vs-rule.jsonl` (2000 recorded deals among RuleBots).
- **The laptop keeps the rest of `runs/`** (gitignored), under its
  `~/danish-wist-training/runs/`:
  - `bc.pt`, `bc-explore.pt` and their `.npz`: the imitation starts.
  - `rl-001/` to `rl-003/`, `rl-004/`, `rl-004b/`, `rl-004c/`, `rl-004d/`:
    each with `state.pt` (resumable with `--resume`), `checkpoints/` (policy,
    critic and `.npz` every 10 iterations), logs and evaluations.
    **rl-004d's `checkpoints/policy-0010.*` and `critic-0010.pt` are the best
    policy** and the natural start of the next run.
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
  belief head by phase), `learn.report` (from recorded deals: roles, contracts, and
  bidding by hand strength), `learn.contracts` (plays policies itself:
  contracts by kind, level and result, `--phases`, `--sample`, `--field
  rule`), and `learn.margins` (each bid against a pass and one or two levels
  higher, same cards: whether the bidding fits the play). `learn.report` and
  `learn.contracts` overlap and could be merged.
- **Recorded arena deals** of rl-003's iteration 110, rl-002's last policy,
  and rl-004c's 150 and rl-004d's 10 and 100 are in
  `runs/arena/*-vs-rule.jsonl` (15 MB each, too big for git).
- **Before a long run,** agree it with the user. The user expects Flip to prove
  strong and bidding to grow high and aggressive, since a made contract pays
  for every trick; both have held so far.
