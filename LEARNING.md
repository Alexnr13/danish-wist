# Learned Bots — Research and Plan

Where we are heading with machine-learned bots, and why. This is a plan, not a
rule book: `RULES.md` still decides what the game is.

## Status and next steps

| Step | Status | Where |
|---|---|---|
| 1. Evaluation harness | Done; runs on all cores and records deals; contract, curve and bidding-margin reports | `learn/arena.py`, `learn/evaluate.py`, `learn/report.py`, `learn/contracts.py`, `learn/curve.py`, `learn/margins.py` |
| 2. Observation encoder | Done | `learn/encoding.py` |
| 3. Network and imitation of RuleBot | Done; NumPy inference for play | `learn/model.py`, `learn/imitate.py`, `learn/inference.py` |
| 4. Self-play PPO | Beats RuleBot: +56.5 ± 15.7 per deal (rl-004d, iteration 10), +38.2 ± 14.2 over rl-003's best on the same deals; exploration's PPO correction fixed. **But all of it is bidding: on fixed contracts its card play is level with RuleBot, +3.0 ± 3.4** (review, September 2026) | `learn/selfplay.py`, `TRAINING.md`, `results/review-2026-09/REVIEW.md` |
| 5. Belief-sampled search | Built; no gain yet over rl-003's policy (+31 ± 29 vs +39 ± 28, 200 deals) or rl-004d's (+48 ± 47 vs +63 ± 48): fit the belief head first | `learn/worlds.py`, `learn/search.py` |
| 6. Exploiters and a league | Exploiter built: 100 iterations from imitation found no way to beat rl-003 (−26.5 ± 13.5) or rl-004d (−120.2 ± 21.8), but one started from rl-004d itself found +20 ± 20 per deal, as declarer; the league is still just recent snapshots | `learn/selfplay.py --exploit` |

**Review, 27 September 2026** (`results/review-2026-09/REVIEW.md`, with two
literature surveys beside it): the recipe below stands, but the next work is
measurement (a fixed-contract card-play test, exploiter margin, held-out
selection), a real league with exploiters in place of a stronger magnet, and
search over sampled worlds evaluated by the oracle critic, distilled back into
the policy. Its todo list supersedes the "Next" list below, kept for the record.

Next (in rough order, as of 26 September):

1. **Defending**, the weakest role in every run, got worse as self-play moved
   to Flip: the learner rarely defends the plain and Clubs contracts RuleBot
   bids. Train against a broader field (more RuleBot and older snapshots in
   some seats; a real league, step 7).
2. **Stop self-play drifting.** With exploration's correction fixed (it had
   biased every update towards explored bids), the policy still drifts
   towards what beats its own defence, and against RuleBot that loses. A
   stronger magnet slows it (TRAINING.md, the rl-004 line); an averaged
   policy is the next thing to try. Restarting from the best checkpoint,
   chosen on fresh deals, made steady progress meanwhile.
   Higher contracts did not start to pay with `--explore-levels` (made one
   level up 43% → 32%), yet the bidding grew bolder and the score rose.
3. **A bidding-first phase, if bidding lags play** (below, "A bidding-first
   curriculum"): when `learn.margins` shows a level higher paying for strong
   hands while the policy still does not bid it.
4. **Strength levels to play against.** Let the web game offer RuleBot, the
   imitation network, the trained network, and the trained network with
   search at a chosen number of worlds.
5. **Search.**
   - Faster: copying and replaying deals is most of its cost.
   - Safe bidding: search only the policy's few most likely bids, with many
     more worlds, to avoid the winner's curse.
   - Measured properly against RuleBot and the plain network.
6. **Expert iteration.** Train the policy to copy search's choices, then
   repeat: the usual way search lifts a policy beyond itself.
7. **A real league.** Keep a spread of older snapshots, not just the last
   eight, and train against exploiters' weaknesses.
8. **Tidy this document** once the above settles: history of results to an
   appendix, the plan kept short. Merge `learn.contracts` (which plays
   policies itself, greedily or sampling, and compares choices by phase) into
   `learn.report` (which reads recorded deals): they overlap.

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

The exchange is public (RULES.md §10): `PlayerView.took_cat` is encoded for
every seat as an `EXCHANGED` token.
The encoder gives about 42 tokens per decision on average (at most 160), in
about 31 µs: roughly the same cost as the engine's own work per decision. Its
action space is one flat index of 207 actions with a legal mask, which is
simpler than a pointer head and loses nothing at this size.

**Step 3 on the MacBook** (20,000 deals, 1.04M decisions, 3 epochs, 28
minutes on CPU): 96.9% agreement with RuleBot on held-out decisions, and
**−7.2 ± 14.7** per deal against a RuleBot field over 200 deals: level with
its teacher. This is `runs/bc.pt`, the start of self-play.

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
seats. Games are played through the performance branch's `Runner`.

**Before the first long run** (September 2026) a review of the training code
led to these changes:

- **The critic predicts the expected score.** It used to regress
  symlog(score) with a squared error, which learns symexp(E[symlog score]):
  near 0 whenever a deal can go either way (+200 or −80 at even odds gives
  +0.6, not +60). It now predicts odds over 255 bins spaced evenly on a
  symlog scale, trained by cross-entropy on two-hot targets that interpolate
  in points, so its expectation is the mean score (§3.3's two-hot head).
- **Forced moves are not recorded** (about 19% of decisions): they give the
  policy no gradient.
- **The update is about twice as fast on MPS**, with the same maths: each
  512-row minibatch runs in passes of 256 rows (MPS slows sharply beyond about
  32k token rows per pass), and tokens are embedded by one multi-hot product
  (MPS's embedding backward is about 10x slower when a batch reuses few rows).
- **One PPO epoch over twice the deals**: playing costs under a tenth of an
  iteration, so fresh deals beat a second pass over old ones for the same
  number of gradient steps.
- **Runs survive crashes**: `state.pt` after every iteration, an exact
  `--resume`, numbered checkpoints, a guard against non-finite gradients and
  weights, and more diagnostics (`approx_kl`, gradient norms, the critic's
  explained variance, results by role).

Considered and left for later: length-bucketed minibatches (faster, but they
made the critic fit much worse), and a stronger entropy or magnet term (watch
entropy first). Scaling card play by the contract's trick value (§3.3) was
left too, then built as `--stake-scaling` once exploring higher contracts made
it matter. It differs from §3.3 read literally. Advantages from the exchange
on (when the stake is fixed: trumps are decided, and so is whether the
declarer is alone) are weighted by the batch's mean stake over the deal's,
and the batch is still normalised as one. Normalising bids and card play
separately would have cut the bids' share of the update from 29% to 18%
(1024 deals of rl-003, `results/analysis/update-shares-rl-003-0110.txt`):
their advantages vary more than card play's, so today they get about twice
their share of decisions, and they should keep it. The auction, the call and
trumps stay in points, because they set the stake: naming clubs in a Plain
contract doubles it, and calling one's own ace triples the declarer's. The
critic still predicts points. With `--explore-levels 0.1`, play at stakes over
160 per trick was 6% of play decisions but 39% of play's share of the update;
with scaling, 12%.

The belief head should get a short supervised fit to the final policy before
search relies on it.

**Step 4 results** (MacBook, 26 September 2026; details in `TRAINING.md`):
self-play PPO **beats RuleBot**. The best policy so far, rl-003 at iteration
110, scores **+23.5 ± 9.1** per deal against a RuleBot field over 2000 deals
(the imitation start: −8.6 ± 3.8), almost all of it as declarer (+61).
Three lessons:

- **Exploration of contracts decides the bidding.** RuleBot never bids Flip
  or Halves, so neither does its copy, and self-play cannot learn from
  choices it never samples. `imitate --explore` gives the start a few percent
  on each kind of contract; `selfplay --explore-bids` keeps them in play while
  their follow-up decisions are learned (the first Flip contracts are badly
  played and lose, so PPO drops them otherwise). With both, the policy chose
  Flip more and more by itself and made its biggest gains.
- **The bidding grows high and aggressive**, as the scoring invites (a made
  contract pays for every trick): half a level to a level above RuleBot, declaring
  in about half its seats, with Flip, Halves and Clubs replacing plain bids.
- **Self-play cycles.** The policy swings between kinds of contract from one
  checkpoint to the next, and its strength swings with it (rl-003 fell back
  from +35 to +1 in-run between iterations 110 and 180). The magnet at 0.02
  is too weak to stop it; the next run tries 0.1, then an averaged policy.

Defending is the weakest role throughout (−19 to −30 per deal at best).

**A bidding-first curriculum** (considered in September 2026, not built). The
policy learned little about judging hands in the auction, so the user asked
whether, as FACTR (Liu et al. 2025) corrupts a robot's vision early in
training so that its policy learns to use touch, we could hide or corrupt
card play so that early gains must come from bidding.

- Corrupting what players see during play would not reach the auction. Bids
  are made before any card is played, so it would only make play, and with
  it each bid's result, noisier.
- The form that transfers takes play out of the learning problem, as bridge
  programs learn to bid against double-dummy results (Yeh & Lin 2016; Gong,
  Jiang & Tian 2019). Every seat plays its cards with a fixed player,
  greedily, so a deal's result depends only on the cards and the contract,
  and only the auction and contract decisions learn. Card play is then handed
  back to the learner in a rising share of deals, as FACTR weakens its
  corruption over training.
- `learn.margins` then showed that rl-003's bidding already fits its own play.
  Bidding beats passing with all but the weakest hands, and a level higher
  loses because those contracts are made too rarely. Fitting bids to a frozen copy of that play
  would change little: the limit is card play in higher contracts, so rl-004
  explores them instead.
- The curriculum pays once play has improved and bidding has not followed:
  when `learn.margins` shows a level higher paying for strong hands while the
  policy still does not bid it.
- It needs separate networks (trunks) for bidding and card play. With
  today's shared trunk, training bids while play is frozen would drift the
  play as well.

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
- Liu et al., *FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning*, RSS 2025 (arXiv 2502.17432).
- Yeh & Lin, *Automatic Bridge Bidding Using Deep Reinforcement Learning*, 2016 (arXiv 1607.03290).
- Gong, Jiang & Tian, *Simple is Better: Training an End-to-end Contract Bridge Bidding Agent without Human Knowledge*, 2019 (OpenReview SklViCEFPH).
