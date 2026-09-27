# Review of the learned bot, September 2026: findings, lessons and the todo list

Written 27 September 2026 after a review of the rl-004 line (`TRAINING.md`,
`results/rl-004/FINDINGS.md`), the training code in `learn/`, two diagnostic
experiments run on the laptop, and two literature surveys
(`literature-trick-taking.md`, `literature-self-play.md`). It is the brief for
the next agent, who works on the 5090 workstation. Read this, then `CLAUDE.md`,
`LEARNING.md` and `TRAINING.md`; the todo list is §4 and the migration steps §5.

## 1. Headline

The recipe (PPO self-play, a critic that sees the hidden cards, a magnet, a
belief head, belief-sampled search) is the right one and the engineering is
careful. But the headline metric hides the main fact: **after 1.17 million
self-play deals, the network's card play is no better than RuleBot's.** Every
point of rl-004d's +56.5 against RuleBot comes from the auction. The training
loop is tuned to slow itself down (magnet 1.0) and to select for exploiting one
passive bot (restarts from the best-vs-RuleBot checkpoint), and card play,
which is where trick-taking strength has always come from, has no direct
learning signal and no working search behind it yet.

## 2. Findings from the diagnostics

Scripts in `scripts/`, outputs in this directory. All on the arena's 2000 deals
(seed 0), against a RuleBot field, rl-004d's iteration 10 unless stated.

**2.1 Card play is flat since imitation** (`card-play-by-checkpoint.txt`). A
hybrid agent lets RuleBot run the auction and contract set-up while the network
plays the cards (`hybrid_arena.py ... net-play`). The stake never changes, so
this test resolves to ±3.5 instead of the arena's ±15.

| Policy | Network plays cards in RuleBot's contracts |
|---|---|
| bc-explore (imitation) | −7.1 ± 1.5 |
| rl-003 it.110 | −0.1 ± 3.4 |
| rl-004c it.150 | +2.7 ± 3.5 |
| rl-004d it.10 (best) | +3.0 ± 3.4 |
| rl-004d it.100 | +4.1 ± 3.3 |

Card play changed a lot over training (agreement with RuleBot fell from 97% to
61%) but did not get stronger in ordinary contracts. What it did learn is to
play its own high Flip contracts better than RuleBot would (next table).

**2.2 Decomposition of the +56.5** (`hybrid-rl-004d-0010.txt`):

| Agent | Advantage per deal |
|---|---|
| network throughout | +56.5 ± 15.7 |
| network bids only; RuleBot sets up the contract and plays | +33.0 ± 13.2 |
| network bids and sets up the contract; RuleBot plays the cards | +26.3 ± 15.6 |
| RuleBot bids and sets up; network plays the cards | +3.0 ± 3.4 |

So about +30 is the auction and about +30 is playing its own chosen (Flip,
level 9) contracts better than RuleBot's heuristics would.

**2.3 The by-role split is misleading** (`role-split.txt`). With the contract
identical to the all-RuleBot baseline, the network's defence is −4 ± 12 and its
declaring +42 ± 73. The whole "defender −90" comes from the 1172 seats where it
pushed RuleBot up a level (mean +1.2 levels) and then passed. That is a cost of
bidding, already netted in `learn.margins`' "bid − pass". The user's view,
agreed: against a field that bids differently, declarer/partner/defender
splits are meaningless; the counter to a conservative bidder is bidding higher,
not better defence. **Do not read by-role arena numbers as play quality.**

**2.4 Other observations.**
- The belief head has barely learned: loss 1.17 against a prior entropy of about
  1.28 at the auction (13/42 per hand, 3/42 out). Search's "belief-weighted"
  worlds are near uniform, and 8 worlds is far too few anyway (Kermit saturates
  at 160). Rollouts cost about 90 s per deal, so the search cannot be measured
  on enough deals to matter.
- The magnet at 1.0 pins the policy: approx KL 0.003 per iteration, clip
  fraction 0.03. That is why rl-004d "held but stopped improving".
- The opponent pool is the last 8 snapshots, used in a quarter of deals, with
  RuleBot in about 5% of deals: nearly pure latest-self self-play, the
  configuration the literature finds worst for drift and forgetting.
- Restarting from the checkpoint that scores best against RuleBot is
  max-of-noisy-estimates selection on the reporting opponent, and each restart
  discards the optimiser state and the pool. The "drift" it fights is self-play
  doing its job: under these scoring rules declaring pays even at 35–40% made
  (bid 9 Flip: made 9 tricks +720, failed by two −320), so a bidding war is the
  equilibrium, not a pathology.
- Alone contracts cost about 40 points per deal (4.9% of seats at −805).
  Mostly the luck of the cat holding the called ace; worth a check of the call
  and level with aces in hand.
- Throughput: 48 s update against 7 s play per 1024 deals, about 1000 decisions
  per second on MPS. Total self-play so far is 1–2 orders of magnitude below
  where published systems first beat strong baselines.
- The decoupled-PPO fix is correct; the critic is the unbiased history-plus-
  state form; exact resume and the golden replay test are good practice.
  Tests and lint pass.

## 3. Lessons from the literature (details and citations in the two surveys)

1. **Strength in trick-taking play comes from determinised search plus a good
   inference model.** Skat's Kermit (160 worlds, exact open-card solver) still
   beats the best learned Skat policies by 2.6 to 9.8 tournament points per
   game, including the 2024 transformer-planning system. Every competitive
   bridge engine is PIMC plus a double-dummy solver. The biggest lever in Skat
   was the belief model: conditioning inference on the full history gave +3–4
   TP/G, and weighting worlds by the likelihood of every observed action under
   a learned policy a further +1.6–2.3, mostly on defence.
2. **Pure self-play RL beat strong baselines only at 10^9 steps or more**
   (DouZero 30 GPU-days, PerfectDou 2.5×10^9 samples, Suphx 44 GPUs × 2 days).
   PerfectDou's ablation: the perfect-information critic alone added little;
   oracle features and reward shaping mattered more. Fan & Farina (2026) show
   GAE's sampled backups add avoidable variance in imperfect-information
   self-play and replace them with expected-SARSA advantages from a
   full-information critic, which we already have.
3. **Regularisation:** in a 7,000-run study (Rudolph et al., ICLR 2026) PPO,
   PPG and MMD were on par; the entropy coefficient was the most important
   hyperparameter (0.05–0.2 on normalised rewards), the KL magnet weight far
   down the list. Systems that use a magnet keep its weight moderate (0.05–0.2)
   and put the effort into how the reference moves (EMA or periodic reset).
4. **Populations, not magnets, stop drift:** AlphaStar's league (50%
   prioritised fictitious self-play against the whole pool, exploiters, never
   restarting), OpenAI Five's 20% against the full history, Diplomacy's best
   response to the time-average of checkpoints with one checkpoint for all
   opponents. Exploiter margin is the standard exploitability proxy.
   Checkpoints are selected against a held-out reference set; the scripted bot
   is used only to discard runs.
5. **Bidding:** every state-of-the-art bridge bidder pretrains on a strong
   bidder then runs PPO with fictitious play; contract-value estimation beat
   imitation in Skat. `learn.margins` already shows our bids are near a best
   response to our play, so bidding is not the bottleneck now.
6. No published work on call-ace whist, and no principled treatment of
   hidden-partner inference beyond consistent deal sampling and
   policy-likelihood weighting of worlds.

## 4. Todo list, in order

Each item names the files it touches and what "done" looks like. Every code
change needs tests (`CLAUDE.md`), and `pytest`, `ruff check .` and `ruff format
--check .` must pass before a commit. Agree any long run with the user first.

### Phase 0: the workstation (before anything else)

- [ ] **T0.1 Set up.** Clone, `git checkout training`, `uv venv -p 3.13
  --managed-python`, install a CUDA build of PyTorch that supports the RTX 5090
  (Blackwell, compute capability 12.0), then `uv pip install -e ".[learn,dev]"`
  and `pytest -q`. Record the Python and torch versions in `TRAINING.md`
  "Handoff".
- [ ] **T0.2 Restore `runs/`.** Unpack `handoff-2026-09.tar.gz` (see §5) into
  the repository root. Check the baseline reproduces: `python -m learn.arena
  --candidate runs/rl-004d/checkpoints/policy-0010.npz --field rule --deals
  2000` should give +56.5 ± 15.7, and `python
  results/review-2026-09/scripts/hybrid_arena.py
  runs/rl-004d/checkpoints/policy-0010.npz 2000 8 net-play` +3.0 ± 3.4 (the
  exact numbers depend only on the seed, not the machine).
- [ ] **T0.3 CUDA path and throughput.** `learn.selfplay` picks `mps` when
  available, else `cpu`: pass `--device cuda`. `CHUNK = 256` in
  `learn/selfplay.py` was tuned for MPS; make it a setting and find the fastest
  value on the 5090 (start at 2048). Run the smoke test (`TRAINING.md` §2) and
  a timing run at 1024 deals with `--workers` at the core count minus two;
  record `collect_s` and `update_s`. Target: the update well under the
  collection time, so that deals per hour is set by the CPU workers. If
  collection then dominates, raise `games_in_flight` and consider batched GPU
  inference in the workers (`PERFORMANCE.md`). Run `python -m learn.bench` and
  add the numbers to `PERFORMANCE.md`.

### Phase 1: measurement, before any training

- [ ] **T1.1 Fixed-contract card-play test as a tool.** Turn
  `scripts/hybrid_arena.py` into `learn/hybrid.py` (or an arena candidate
  syntax such as `play:<policy.npz>`), with a test that a hybrid of RuleBot and
  RuleBot equals RuleBot, and document it in `TRAINING.md` §6. This is the
  primary progress metric for card play: ±3.5 over 2000 deals.
- [ ] **T1.2 Paired multi-seed screening in `learn.arena`.** Fold
  `results/rl-004/scripts/screen.py` in: several candidates, several seeds,
  pooled, each paired against the first. Used for checkpoint selection (T1.4).
- [ ] **T1.3 Exploiter margin as a standard metric.** A short recipe (and
  ideally one command) that clones the policy, trains it with `--exploit`
  against the frozen policy in the other three seats for a fixed budget (say
  100 iterations of 1024 deals, as x-004b), and reports the paired gain. This is
  the exploitability proxy; log it for every policy that is compared.
- [ ] **T1.4 Checkpoint selection protocol.** Write into `TRAINING.md` §6:
  select on seeds never used for reporting, against a reference set (RuleBot,
  rl-003 it.110, rl-004d it.10, the latest exploiter), by the fixed-contract
  test plus the min over the reference set; report on other seeds. RuleBot's
  full-arena score is a sanity check, not the objective. Retire the by-role
  interpretation (§2.3).
- [ ] **T1.5 Belief head diagnostic.** Report the belief loss by phase (auction,
  early play, late play) against the prior entropy of the same positions, so
  that "learned" means well below the prior late in play. Small script or a
  flag on `learn.contracts`.

### Phase 2: the training loop (code, each with tests)

- [ ] **T2.1 A league instead of the last eight snapshots.** In
  `learn/selfplay.py`: an unbounded (or capped at 30–50, oldest thinned)
  snapshot pool saved to disk and restored by `--resume`; RuleBot as a member;
  prioritised fictitious self-play with f_hard(p) = (1 − p)^p using tracked
  pairwise results of the learner against each member; per deal, **one** pool
  member fills all the non-learner seats (hidden partners need the correlation);
  flags `--league-share` (opponent deals; start at 0.5) and `--exploiter-share`.
  Expose `opponent_share` on the command line meanwhile.
- [ ] **T2.2 Exploiters in the loop.** Every N iterations clone the learner,
  train it K iterations against the frozen learner, add it to the pool, log its
  margin (`exploiter_margin` in `log.jsonl`). AlphaStar's main exploiters are
  the model.
- [ ] **T2.3 Magnet and entropy.** Replace the every-10-iterations copy with a
  parameter EMA reference (τ about 0.01 per update; keep the periodic-reset
  option), default weight 0.1; make the entropy coefficient a flag (now a
  constant 0.01) and sweep 0.01 / 0.03 / 0.1 on normalised advantages; make the
  policy learning rate a flag and try 2.5e-4 with 2–4 PPO epochs. Judge each
  by T1.1 and T1.3, not by the RuleBot arena. Stop restarting runs; carry on
  with a better opponent mix.
- [ ] **T2.4 Expected-SARSA(λ) advantages** from the oracle critic instead of
  GAE (Fan & Farina 2026, arXiv 2605.19235): the critic already sees every
  hidden card, so the expectation over the acting player's policy at each step
  is computable. Optional; after T2.1–T2.3 have run once.
- [ ] **T2.5 Fit the belief head.** A supervised fit on recorded deals of the
  current policy (cheap: the targets are in every record) before search relies
  on it; then consider replacing the 52-card head with, or adding, a head for
  "who holds the called ace", the quantity that decides the team.
- [ ] **T2.6 One-change checks.** A run from rl-004d it.10 without
  `--explore-levels` (it never made higher contracts pay), and one without
  `--stake-scaling` (never tested alone). Cheap, and they settle two open
  questions in `FINDINGS.md`.

### Phase 3: card play through search and distillation

- [ ] **T3.1 One-ply search with the oracle critic.** For each legal play and
  each of N ≥ 100 worlds sampled by `learn.worlds`, apply the play and evaluate
  the resulting position for our seat with the critic (`encode_oracle`), on the
  GPU in one batch; choose the best mean. Milliseconds per decision instead of
  90 s per deal. Measure with T1.1 (this agent is play-only, so the test is
  exact). Target: clearly above the policy's +3. Then compare rollouts at depth
  1–2 with critic leaves.
- [ ] **T3.2 Policy-likelihood world weighting.** Weight each sampled world by
  the product of the policy's probabilities of the other players' observed
  actions in that world (Rebstock et al. 2019, arXiv 1905.10911), with the
  partner's identity as part of the world. Measure again with T1.1; the Skat
  gains were mostly on defence.
- [ ] **T3.3 Expert iteration.** Record the search's choices during self-play
  (search must run inside the Runner workers or as a batched GPU service),
  distil them into the policy with a cross-entropy term, repeat. Measure with
  T1.1 after each round.
- [ ] **T3.4 Endgame solver (later, optional).** A perfect-information solver
  for the last six or seven tricks as an optional compiled path (Rust or C),
  cross-checked against a pure-Python one and against `record.py` replays, as
  `CLAUDE.md` allows. Skat's evidence: PIMC's edge is "near-perfect play in the
  later half of the game".

### Phase 4: the long run

- [ ] **T4.1 The run.** Agree it with the user. Start from rl-004d it.10
  (policy and critic), league on (T2.1, T2.2), magnet 0.1 with the EMA
  reference, entropy about 0.03–0.05, learning rate 2.5e-4, two epochs, 1024
  to 4096 deals per iteration, `--stake-scaling`, `--explore-bids 0.15`,
  `--explore-levels` per T2.6. Judge by T1.1, T1.3 and the reference set, on
  held-out seeds. No restarts. Aim for 10^8 decisions.
- [ ] **T4.2 Bidding, revisited.** Once T1.1 shows card play clearly above
  RuleBot, rerun `learn.margins`; the bidding-first curriculum in `LEARNING.md`
  pays only when a level higher pays for strong hands and the policy does not
  bid it.
- [ ] **T4.3 Alone contracts.** Check the call and level with aces in hand
  (`learn.report` on the arena records): 4.9% of seats alone at −805 is about
  40 points per deal.

## 5. Migration to the workstation

`runs/` is not in git and lived only on the laptop. `scripts/pack_runs.sh`
packed what the workstation needs into `runs/handoff-2026-09.tar.gz` (62 MB) on
the laptop, in `~/danish-wist-training/`:

- `runs/bc-explore.pt`, `.npz`: the imitation start.
- `runs/rl-003/checkpoints/policy-0110.{pt,npz}`, `critic-0110.pt`: the previous best, a reference.
- `runs/rl-004d/checkpoints/policy-0010.{pt,npz}`, `critic-0010.pt`: **the best policy, the start of the next run.**
- `runs/rl-004d/checkpoints/policy-0100.{pt,npz}`, `critic-0100.pt`; `runs/rl-004d/state.pt`, `run.json`, `settings.json`: to `--resume` rl-004d if wanted.
- `runs/x-004b/policy.{pt,npz}`: the exploiter that found +20 against rl-004d it.10.
- `runs/arena/rl-004d-0010-vs-rule.jsonl`: the recorded arena the role-split scripts read.

Copy the tarball to the workstation (scp, or any drive) and unpack it in the
repository root: `tar xzf handoff-2026-09.tar.gz`. Everything else in `runs/`
(rl-001 to rl-004c, the other arenas) is reproducible from `results/` and not
needed. The laptop's memory notes for the coding agent do not transfer; this
document and `TRAINING.md` are the record.

Workstation differences to remember: `--device cuda` (the default is `mps` or
`cpu`); `CHUNK` (T0.3); `--workers` defaults to the core count minus two; each
worker keeps a one-thread CPU copy of the policy, so RAM per worker is small.

## 6. Numbers to carry forward

| Quantity | Value |
|---|---|
| Best policy | rl-004d it.10: +56.5 ± 15.7 vs RuleBot (2000 deals, seed 0); about +50 after selection bias |
| Its card play in RuleBot's contracts | +3.0 ± 3.4 (imitation start −7.1 ± 1.5) |
| Its exploitability proxy | +20.3 ± 20.1 per deal for an exploiter started from itself, 100 iterations |
| Belief loss | 1.17 (prior about 1.28) |
| Self-play so far | about 1.17M deals, 558 iterations in the rl-004 line |
| Laptop throughput | 1024 deals per 55 s: 7 s play, 48 s update (MPS) |
| Per-deal SD of the duplicate score | about 360 points, so ±15 needs 2000 deals |
