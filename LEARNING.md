# Learned Bots — Research and Plan

Where we are heading with machine-learned bots, and why. This is a plan, not a
rule book: `RULES.md` still decides what the game is.

## 1. What kind of game this is

The shape of the game decides the method more than anything else.

| Property | Danish Wist | Consequence |
|---|---|---|
| Players | 4, zero-sum overall | **Not two-player zero-sum.** |
| Teams | Set by the called ace, hidden from the declarer and defenders | Coalitions are themselves hidden information, as in Schafkopf and Doppelkopf. |
| Hidden state | 3 unseen hands, the cat, the fucdic's identity | A huge belief space: roughly 10^20 possible deals from one seat. |
| Length | About 64 decisions per deal (auction, call, trumps, flip, exchange, fucdic, 13 tricks each) | Short episodes, so credit assignment is easy. |
| Actions | At most 55 cards or about 28 bids, always a short legal list | Small action heads with a legal-action mask. |
| Stakes | Trick value ranges from 10 to 1280, and alone is ×3 | Returns vary over three orders of magnitude, so value scaling matters. |
| Simulator | Pure Python, about 550 random deals/s per core | Cheap enough, but network inference will cost more than the engine. |

**"Optimal" needs a definition.** In two-player zero-sum games a Nash
equilibrium is safe: it cannot lose in expectation. With four players and
shifting teams, equilibria are neither unique nor safe, and no algorithm comes
with a guarantee. The practical targets are:

1. a high average score against a varied field (RuleBot, past versions, humans);
2. low *exploitability*: a bot trained only to beat ours, in one seat, should
   gain little.

## 2. What the landmark systems teach us

| System | Core idea | Fit here |
|---|---|---|
| **DeepStack** (2017), **ReBeL** (2020) | Re-solve a subgame during play, using a value network over *public belief states* | Poor. The re-solving guarantees need two-player zero-sum, and a public belief state here is a distribution over all hidden deals for four seats, which is far too large. |
| **Student/Player of Games** (2021/23) | ReBeL-style search (GT-CFR) plus AlphaZero-style self-play | Same limits as ReBeL. |
| **DeepNash** (Stratego, 2022) | Model-free self-play with R-NaD, a KL "magnet" regulariser that stops the policy from cycling, and no search | **The regulariser carries over.** The two-player theory does not, but the training recipe is sound. |
| **Pluribus** (6-player poker, 2019) | Blueprint self-play plus limited search, with no guarantees | Shows that multiplayer self-play works in practice. |
| **DouZero / PerfectDou** (Dou Dizhu, 2021/22) | 3 players with a hidden-hand 2-vs-1 team. Self-play RL, with a critic that sees all hands in training but not in play | **The closest precedent that actually works.** |
| **Suphx** (Mahjong, 2020) | "Oracle guiding": train with hidden information, then remove it | Same idea as PerfectDou's critic. |
| **Skat, Bridge bots** (Kermit, Ben, NooK; transformer planning 2024; outer-learning 2025) | A learned policy plus a *belief/inference model* that samples plausible deals, with search run in the sampled worlds | **Belief-sampled search is the state of the art for trick-taking play.** |
| **Hanabi** (SPARTA, learned belief search) | Search on top of a fixed policy, using a learned belief model to sample worlds | The same test-time search pattern. |
| **Policy gradient re-evaluated** (ICLR 2026) | Well-tuned PPO / magnetic mirror descent match or beat CFR-, fictitious-play- and PSRO-based deep RL in imperfect-information games | **Use plain regularised policy gradient, not deep CFR.** |

What "modern" adds in 2026 is chiefly architecture: a **transformer** that
reads the deal as a sequence of events handles variable-length histories,
card sets and attention to "who played what" naturally. It is still a small
model (a few million parameters) trained from scratch on self-play. Large
language models are not good players here: they are slow and poor at card
counting.

## 3. Recommended recipe

> **Regularised self-play policy gradient (PPO with an R-NaD/MMD-style magnet)
> on a small transformer over the observation history, with a
> perfect-information critic and an auxiliary belief head in training, and
> belief-sampled search at play time.**

In more detail:

1. **Observation → tokens.** Use only `Deal.view(seat)`, with seats relative
   to the viewer. Tokens are: the hand as a set of card tokens; one token per
   public event (bid, pass, called suit, trumps, turned cat card, fucdic,
   each card played with its seat and trick); and phase and contract tokens.
   Encode with a transformer (about 4–6 layers, width 128–256).
2. **Heads.**
   - *Policy:* a pointer over the hand's card tokens for plays and discards,
     plus small fixed heads for bids, suits and yes/no choices, masked to
     `legal_actions`.
   - *Value (critic):* in training only, it also sees every hidden card.
     This greatly reduces variance, and the actor never sees what it should
     not.
   - *Belief (auxiliary):* predicts where every unseen card is (each
     opponent, the cat, or the fucdic), and who the partner is. It is
     supervised from the true deal, it is cheap, and it improves the
     representation. It is also what the search in step 5 samples from.
3. **Scale the returns.** Once the contract is fixed, predict the score in
   units of trick value (bounded, about ±39). In the auction, where returns
   span 10–15,000, use a symlog two-hot value head. Never reshape the reward
   itself, because bidding must weigh the true stakes.
4. **Self-play loop.** One shared network plays all four seats. Train with
   PPO plus a KL penalty towards a slowly updated magnet policy, as DeepNash
   and MMD do. Draw opponents from a pool of past snapshots plus RuleBot, so
   the bot does not overfit to itself.
5. **Test-time search.** At a decision, sample N deals from the belief head
   that are consistent with the public record: shown voids, the fucdic, and
   the partner's known position. Roll each action out with the policy network
   for all seats, and pick the best average. Rolling out with the policy,
   rather than solving each sampled deal with perfect information, avoids
   PIMC's "strategy fusion" error. The strength is adjustable through N
   (0 means no search).
6. **Optional: expert iteration.** Distil the search's choices back into the
   policy and repeat.

Varying strength falls out of this at no extra cost: use earlier snapshots,
less search, or a temperature on the policy.

## 4. Measuring progress

- **Duplicate scoring.** Play each deal four times, rotating seats, so luck
  largely cancels out. Report the mean score per deal with a confidence
  interval.
- **A fixed field:** RandomBot, RuleBot and frozen snapshots.
- **Exploiters.** Freeze the main bot in three seats and train a fresh bot in
  the fourth. The exploiter's gain is our exploitability proxy.
- Break results down by phase (auction outcomes, contracts made or failed,
  and declarer, partner and defender results), so we can see where the
  strength comes from.

## 5. Implementation order

Each step is testable and useful on its own.

1. **Evaluation harness:** an `Agent` protocol, duplicate tournaments and
   statistics. It needs no new dependencies.
2. **Observation encoder:** view → tokens, and action ↔ head index, with
   tests that it uses only what the view shows.
3. **Network plus behaviour cloning of RuleBot:** proves the pipeline end to
   end. This is where PyTorch arrives, as an optional `learn` extra that
   stays out of the engine.
4. **Self-play PPO** with the oracle critic, belief head and magnet. Target:
   clearly beat RuleBot in duplicate play.
5. **Belief-sampled search.**
6. **Exploiters and a snapshot league.** Then iterate.

Progress: steps 1–3 are done: `learn/arena.py`, `learn/encoding.py`, and
`learn/model.py` with `learn/imitate.py` and `learn/inference.py`. Step 4
(`learn/selfplay.py`) is built: PPO with a hidden-card critic, magnet and
belief head, updating on MPS when available, with `--resume`. Collection runs
through the performance branch's `Runner` (`--workers`, default: all cores
but two). Each worker holds a one-thread PyTorch copy of the policy, which gets
new weights and a seed by `broadcast` every iteration. The learner records
its decisions and training targets in the worker, and compact trajectories
come back through `finish`. Snapshots and the exploit target are named greedy
copies, and evaluation runs through the same pool. It is ready to run on the
MacBook.
Step 6's exploiter is built too (`--exploit`).

Step 5 is built: `learn/worlds.py` samples deals consistent with one
player's view. It works out the unseen cards and their places, respects shown
voids, the fucdic and a revealed partner, then replays the public history
through the engine and keeps only worlds that show the player exactly the view
it had. That succeeded at every decision tested, in about 1.4 ms for three
worlds. `learn/search.py`'s `SearchAgent` plays every legal action out in N
worlds with a rollout agent and picks the best average. By default it searches
card play only. Searching bids with few worlds overbids (winner's curse):
−815 ± 591 per deal against RuleBot over 30 deals with 4 worlds, and declarer
in 77% of seats. With RuleBot rollouts and 8 worlds over card play it is level
with RuleBot (+0 ± 24, 40 deals), too few deals to separate them. Real
measurements, with network rollouts and many more deals, are for the MacBook
(`python -m learn.arena --candidate search --field rule`). Worlds can be
weighted by a network's belief head (`SearchAgent(..., belief=agent)`, where
`agent` is a `NetAgent` or a NumPy `NumpyAgent`). This only matters once
self-play has trained the head: an untrained head's beliefs are near uniform.

**Open engine question.** A player's view does not say whether the declarer
exchanged with the cat, although everyone at a real table sees it. Adding a
public `took_cat` field to `PlayerView` changes the engine's public interface
(and the performance branch's golden digests), so it waits for agreement
through `main`.
The encoder gives about 42 tokens per decision on average (at most 160), in
about 31 µs: roughly the same cost as the engine's own work per decision. Its
action space is one flat index of 207 actions with a legal mask, which is
simpler than a pointer head and loses nothing at this size.

**Step 3 result** (container, 4 CPU cores, about 25 minutes): the default
network has 590k parameters (width 128, 4 layers). It was trained on 261k
RuleBot decisions from 5,000 deals (forced moves skipped), for 3 epochs.

- It agrees with RuleBot on 89% of held-out decisions.
- In duplicate play over 300 deals it is **−49 ± 16** per deal against a
  RuleBot field (close, but not yet equal).
- Against a random field it scores **+11,080 ± 810**, against RuleBot's
  +12,080 ± 840.
- The exported NumPy copy chose the same move as PyTorch in 1,929 of 1,929
  decisions, at about 8 ms per single decision.

More data and epochs would close the gap, but that isn't the point:
self-play (step 4) starts from this policy.

**Step 4** is in `learn/selfplay.py`: PPO self-play with a critic that sees the
hidden cards (`encode_oracle`, training only) and a magnet KL term; mostly pure
self-play, with a quarter of deals seating RuleBot or past snapshots in some
seats. It uses its own simple single-process collector for now.

**What step 4 needs from the performance runner.** When `learn/runner.py`
lands, the learner will plug in as a batch agent. It must be able to record,
for each decision it makes, which deal and seat it belongs to, so it can
attach that seat's final score. Either of these would do:

- `choose_batch(views, keys)`, where each key identifies (game, seat); or
- `play_many` yields each finished deal together with the agent's per-deal
  records.

The learner also needs, per decision and in the worker, the oracle tokens
(`encode_oracle(deal, seat)`), which need the `Deal` itself, not just the
view. The simplest form: the runner lets a learner agent see
`(deal, seat, view)` for its own decisions.

**Agreed with the performance side** (pull request #3; its proposal is in
`PERFORMANCE.md` → "Next"): a `Runner` pool that stays up between PPO
iterations; agents named per game (snapshots as fixed named slots filled by
`broadcast(name, method, ...)`); `Decision(game, seat, view, deal)` for agents
that define `choose_decisions`, with `deal` read only for `encode_oracle`;
`finish(game, deal, agents)` returning compact trajectories; per-iteration
reseeding; and duplicate evaluation through the same pool. The policy runs as
a one-thread PyTorch copy in each worker, while the main process keeps MPS
for the update. Once the `Runner` is on `main`, `collect()` moves onto it.

**Where work runs.** The container builds and tests the method at toy scale;
real training and benchmarking happen on the MacBook. A short container run
(6 PPO iterations of 256 deals from the imitation policy) showed stable
training: critic loss 25.6 → 15.4, policy within 0.07 KL of its magnet. In
that environment one iteration spent about 11 s collecting and 120 s
updating, so the update is what the GPU should take.

## 6. Packaging and compute

**Three separate layers.** Playing against trained bots must not require the
training machinery.

- `danish_wist/` (engine and baseline bots) and `web/` keep no dependencies.
- `learn/` holds evaluation (standard library only) and, from step 3,
  training with PyTorch as an optional extra (`pip install -e ".[learn]"`).
  Torch is imported only by the modules that train.
- **Runtime inference** loads saved weights without PyTorch. The planned route
  is to export the small network's weights to a NumPy file and run the forward
  pass in NumPy, tested against the PyTorch output. That keeps the playing
  install to one light dependency. **Done in step 3:** `learn.model.export`
  writes a `.npz`; `learn.inference.NumpyAgent` plays from it and matches
  PyTorch to within 1e-4. `python -m web.server --bot model.npz` seats it in
  the browser game.

**Compute.** Measured engine speed is about 400–570 deals/s per CPU core
(about 64 decisions per deal). On the MacBook Pro M1 Pro (8 performance cores,
16 GB):

- Steps 1–3 are small, and the laptop is ample.
- For step 4, self-play runs as parallel CPU workers, each feeding batched
  network inference (MPS or CPU). Allowing for inference cost, expect a few
  hundred deals per second: roughly 1 million deals an hour. That should be
  enough to clearly beat RuleBot and to tune the recipe.
- Longer runs, belief-sampled search (which multiplies the cost of each
  decision) and exploiter training are where the 5090 workstation pays off.
  Its main gains are more CPU cores for self-play and a faster GPU for large
  batches. Move there once the laptop's runs stop improving.

The engine stays exactly as it is: pure, with no I/O, and records stay
replayable. If profiling later shows that the engine limits training, a
faster mirror can be written and cross-checked against `record.py` replays.

## References

- Moravčík et al., *DeepStack*, Science 2017.
- Brown & Sandholm, *Superhuman AI for multiplayer poker* (Pluribus), Science 2019.
- Brown et al., *ReBeL*, NeurIPS 2020.
- Schmid et al., *Player of Games* 2021 / *Student of Games*, Science Advances 2023.
- Perolat et al., *Mastering Stratego with model-free multiagent RL* (DeepNash), Science 2022.
- Zha et al., *DouZero*, ICML 2021; Yang et al., *PerfectDou*, NeurIPS 2022.
- Li et al., *Suphx: Mastering Mahjong with Deep RL*, 2020.
- Lerer et al., *SPARTA*, AAAI 2020; Hu et al., *Learned Belief Search*, 2021.
- Sokota et al., *Magnetic Mirror Descent*, ICLR 2023.
- Rudolph et al., *Reevaluating Policy Gradient Methods for Imperfect-Information Games*, ICLR 2026 (arXiv 2502.08938).
- *Transformer Based Planning in the Observation Space with Applications to Trick Taking Card Games*, arXiv 2404.13150.
- *Outer-Learning Framework for Multi-Player Trick-Taking Card Games: Skat*, arXiv 2512.15435.
