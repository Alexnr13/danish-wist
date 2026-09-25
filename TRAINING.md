# Training — brief for the `training` branch

**Goal:** run and look after the first real self-play training of the learned
bot on the MacBook Pro (M1 Pro, 16 GB), and report what happened. Read
`LEARNING.md` for the method and `CLAUDE.md` for how the project works. You
run experiments; you do not change `danish_wist/` or `learn/`. If something
in them blocks you, stop and report it (command, traceback, log lines) so the
`learning` agent can fix it.

**Sharing the laptop.** The `performance` agent also works on this machine.
Heavy runs spoil each other's timings, so agree with the user before starting
a long run, and pause its benchmarking while training runs.

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
1024 deals takes about 55 s on MPS (5 s playing, 48 s updating, 6 s for a
1000-deal evaluation every 10 iterations), so 450 iterations take about 7 h:

```sh
caffeinate -i nohup python -m learn.selfplay --init runs/bc-explore.pt \
    --iterations 450 --deals 1024 --ppo-epochs 1 --critic-warmup 5 \
    --eval-every 10 --eval-deals 1000 --out runs/rl-002 > runs/rl-002.out 2>&1 &
```

To stop it, `kill` the Python process (not `caffeinate`, which is its child):
it stops the workers too, and `--resume` carries on later.

`caffeinate -i` stops the Mac sleeping while it is open and on power; closing
the lid still sleeps it. The first 5 iterations train only the critic
(`warmup` in the log), since imitation never trains a value head.

If the run stops, run `python -m learn.selfplay --resume runs/rl-002`. It
carries on from the last finished iteration with everything the run had (the
networks, optimisers, magnet, snapshot pool and random state, from
`state.pt`) and its own settings from `run.json`; only `--iterations` (the
total), `--workers` and `--device` can change. (With several workers, games
finish in a varying order, so no two runs are bit-identical anyway.) A new
run refuses a directory that already holds one.

## 5. Watching it

Every 30–60 minutes read `runs/rl-002/log.jsonl`, and check that it changed
in the last few minutes (a hang shows no error). One line per iteration:

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
collapses. Every evaluation uses the same 1000 deals, so compare evaluations
as paired differences (`evals.jsonl` keeps each one's per-deal results). Don't
change settings in code to "fix" it. Report the evidence.

## 6. After the run

`runs/rl-002/checkpoints/` keeps the policy and critic from every 10th
iteration. If the in-run curve peaked before the end, compare the best few
checkpoints on fresh deals before choosing one (the in-run maximum over the
same 1000 deals is biased upwards):

```sh
python -m learn.arena --candidate runs/rl-002/policy.npz --field rule --deals 2000
python -m learn.arena --candidate runs/bc-explore.npz --field rule --deals 2000
python -m learn.selfplay --init runs/bc-explore.pt --exploit runs/rl-002/policy.pt \
    --iterations 100 --deals 1024 --critic-warmup 5 --eval-every 10 --out runs/x-002
```

To see how the bidding changed over the run, compare the checkpoints' contracts in
self-play and among RuleBots, and their choices by phase (including how likely
each is to bid Flip or Halves, which only sampling can discover):

```sh
python -m learn.contracts --phases runs/bc-explore.npz runs/rl-002/checkpoints/policy-*0.npz
python -m learn.contracts --field rule rule runs/bc-explore.npz runs/rl-002/policy.npz
```

The exploiter is the exploitability test: a fresh learner in one seat against the
policy in the other three. Belief-sampled search costs about 90 s per deal in
one process, so run it as 8 processes on disjoint deals and pool them (the
mean weighted by deals; the CI as sqrt(sum(n_i² ci_i²)) / N):

```sh
for s in 0 1 2 3 4 5 6 7; do
  python -m learn.arena --candidate search:runs/rl-002/policy.npz --field rule \
      --deals 25 --seed $s > runs/search-$s.out &
done; wait
```

Then play a few deals yourself against it for the user to try:
`python -m web.server --bot runs/rl-002/policy.npz`.

## 7. Reporting

Add a "Results" section below, one entry per run: date, commit, command,
workers and device, time per iteration, the `vs_rulebot` curve (a short
table), the final arena numbers with confidence intervals, the exploiter's
`vs_target`, and anything odd. Copy each run's `run.json`, `settings.json`,
`log.jsonl` and `evals.jsonl` to `results/<run>/` (they are small;
checkpoints stay in the ignored `runs/`). Commit to `training`, merge `main`
in, and open a pull request to `main`.

## Results

(none yet)
