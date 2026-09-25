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
per iteration. Try `--workers 6` and `8`, and keep the faster one.

## 3. Imitation start (about half an hour)

```sh
python -m learn.imitate --deals 20000 --epochs 3 --out runs/bc.pt
```

Expect about 85–90% agreement with RuleBot, and a duplicate score against a
RuleBot field near zero. This writes `runs/bc.pt` and `runs/bc.npz`.

## 4. The main run

Size it from the smoke test's timings to take roughly 4–8 hours. For
example, 300 iterations of 512 deals:

```sh
caffeinate -i nohup python -m learn.selfplay --init runs/bc.pt \
    --iterations 300 --deals 512 --eval-every 10 --eval-deals 200 \
    --out runs/rl-001 > runs/rl-001.out 2>&1 &
```

`caffeinate -i` stops the Mac sleeping. If the run stops, carry on with
`--resume runs/rl-001` (the iteration count restarts at 1; the log keeps
appending).

## 5. Watching it

Every 30–60 minutes read `runs/rl-001/log.jsonl`. One line per iteration:

| Field | Healthy |
|---|---|
| `vs_rulebot` ± `vs_rulebot_ci95` (every 10 iterations) | Rising above 0 and staying there. **The number that matters.** |
| `value_loss` | Falling, then flat |
| `belief_loss` | Falling, then flat |
| `entropy` | Falling slowly; not collapsing towards 0 in a few iterations |
| `magnet_kl` | Small (under about 0.2) |
| `clip_fraction` | About 0.05–0.3 |
| `collect_s`, `update_s` | Steady |

**Stop and report** if: any value is NaN; `vs_rulebot` falls for three
evaluations in a row, well outside its confidence interval; or entropy
collapses. Don't change settings in code to "fix" it. Report the evidence.

## 6. After the run

```sh
python -m learn.arena --candidate runs/rl-001/policy.npz --field rule --deals 2000
python -m learn.arena --candidate runs/bc.npz --field rule --deals 2000
python -m learn.arena --candidate search:runs/rl-001/policy.npz --field rule --deals 200
python -m learn.selfplay --init runs/bc.pt --exploit runs/rl-001/policy.pt \
    --iterations 50 --eval-every 10 --out runs/x-001        # exploitability
```

Then play a few deals yourself against it for the user to try:
`python -m web.server --bot runs/rl-001/policy.npz`.

## 7. Reporting

Add a "Results" section below, one entry per run: date, commit, command,
workers and device, time per iteration, the `vs_rulebot` curve (a short
table), the final arena numbers with confidence intervals, the exploiter's
`vs_target`, and anything odd. Copy each run's `settings.json` and
`log.jsonl` to `results/<run>/` (they are small; checkpoints stay in the
ignored `runs/`). Commit to `training`, merge `main` in, and open a pull
request to `main`.

## Results

(none yet)
