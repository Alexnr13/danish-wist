# Literature survey: AI for trick-taking card games (2019–2026)

Compiled 27 September 2026 by a research agent for the review in `REVIEW.md`.
Sources were verified by fetching arXiv abstracts/HTML and extracting the PDFs of
the key papers (Alberta Skat ×3, Gong 2019, PerfectDou, DanZero+, AlphaDou, Suphx,
Long 2010, Sievers 2015). Items marked **[unverified]** could not be confirmed
from a primary source.

---

## 1. Skat

### 1.1 Kermit and the Alberta line (Buro, Furtak, Rebstock, Solinas, Sturtevant)

**Kermit** (Buro, Long, Furtak, Sturtevant, "Improving state evaluation, inference,
and search in trick-based card games", IJCAI 2009): PIMC card play with an
open-card αβ solver; bidding/declaration from tables learned from human games;
table-based "Kermit Inference" (KI) that weights worlds by opponents' bids.
Judged "comparable to expert human players". Furtak & Buro (CIG 2013) found
Kermit's PIMC **saturates at about 160 sampled worlds per move**; 160 worlds is
the standard setting in all later Alberta papers.

**Solinas, Rebstock, Buro, "Improving Search with Supervised Learning in
Trick-Based Card Games", AAAI-19** — https://arxiv.org/abs/1903.09604
- A card-location network (P(card c is with player p) for all 32 cards) trained
  on **20 million human games**; input includes the full card-play history,
  void suits, bids. World sampling for PIMC draws from the product of these
  marginals, renormalised.
- Results (2,500 matches × 2 games, 160 worlds): bidding+declaration+cardplay
  inference vs KI: **+3.4 TP/G suit, +1.3 grand, +4.1 null** (1σ ≈ 1.0–1.4).
  Without card-history input the gain disappears. Cost 0.286 s/move vs 0.093 s.
- Key finding: *inference quality is where PIMC strength came from*; the
  history-conditioned belief model is the single biggest lever.

**Rebstock, Solinas, Buro, "Learning Policies from Human Data for Skat", CoG
2019** — https://arxiv.org/abs/1905.10907
- Imitation networks (5 FC layers; largest ≈2.4M weights) for bid, declaration,
  discard and cardplay; cardplay datasets 146M–289M positions; test accuracy
  77–82%; ~16 h on one GTX 1080 Ti.
- Cardplay: imitation policy alone (Kermit bidding + net cardplay) **−2.61 TP/G
  vs Kermit's PIMC**; best full-network player **+1.05 TP/G** because the
  value-based declaration (+1.17) outweighs the cardplay loss. Speed 2.5 ms/move
  vs Kermit ~650 ms. Authors attribute PIMC's edge to "near-perfect play in the
  later half of the game".

**Rebstock, Solinas, Buro, Sturtevant, "Policy Based Inference in Trick-Taking
Card Games", CoG 2019** — https://arxiv.org/abs/1905.10911
- "Policy Inference" (PI): weight each sampled world w by ∏ₜ π_imitation(a_t |
  w, h_t) over all observed opponent actions, sampling 20,000 card configurations.
- PI vs card-location inference: **+2.32 / +0.64 (n.s.) / +1.57 TP/G**
  (suit/grand/null); vs KI +5.13/+3.12/+2.37. Most of the gain is on
  **defence**. Cost ~5× CLI.
- Control: a **cheating** Kermit (all mass on the true world) *loses* to
  non-cheating inference in suit (−3.25) and grand (−8.49): PIMC's value of a
  single world is not a good action value; averaging over plausible worlds does
  real work.

**Rebstock, Solinas, Sturtevant, Buro, "Transformer Based Planning in the
Observation Space with Applications to Trick Taking Card Games"**, arXiv
2404.13150 (Apr 2024; verified)
- Games: **Hearts, Skat, The Crew**. GO-MCTS: a GPT-2-style transformer (8
  layers, d=256, ≈6–7M params) trained to predict the next observation token and
  the outcome; MCTS entirely in observation space; leaf value V(O)=Σ p(o|O)·v(o).
- Training: population-based neural fictitious self-play; Hearts 4M random games
  then 10 iters × 500k games; Skat 4M mixed random/XSkat games then 20 iters
  (training unstable; needed a scripted bootstrap); Crew 10M then 10 × 2M.
- Results (3,000 matches): Hearts — raw policy beats xinxin (PIMC+UCT) by 0.95
  pts/game, GO-MCTS by 1.74; Skat — raw policy **−16.31 TP/G vs Kermit**,
  GO-MCTS **−9.84 TP/G** at ~42 s/turn vs 72 ms for the raw policy.

**Solinas et al., "History Filtering in Imperfect Information Games"**, arXiv
2311.14651: constructing consistent histories is intractable in general; an
MCMC sampler for consistent deals in trick-taking games (tested on Oh Hell).

### 1.2 Edelkamp's Skat programs (no neural nets)
- "Challenging Human Supremacy in Skat", SoCS 2019: bidding/skat putting from
  tables over "several million" human games; trick play = expert rules +
  open-card search; belief-space search for Ouvert.
- "Knowledge-Based Paranoia Search" (arXiv 2104.05423); "On the Power of
  Refined Skat Selection" (arXiv 2104.02997); "Improving Computer Play in Skat
  with Hope Cards", CG 2022 (doi:10.1007/978-3-031-34017-8_12): "for the first
  time" won a 20-series online tournament against a top German player.
- "A Framework for General Trick-Taking Card Games", KI 2024
  (doi:10.1007/978-3-031-70893-0_6): one engine for Belote, Tarot, Doppelkopf,
  Spades, Hearts, Euchre, Schafkopf with "bidding, team building, game
  selection". Paywalled; **no numbers on hidden-partner games verified.**
- **"Outer-Learning Framework for Playing Multi-Player Trick-Taking Card
  Games: A Case Study in Skat", arXiv 2512.15435 (Dec 2025) — exists, single
  author Edelkamp.** Pre-play decisions are lookup tables keyed by feature
  hashes, seeded from >200M human games then augmented with **30M AI self-play
  games** (>1 month on 16 cores). Card play is the unchanged open-card solver +
  paranoia search. Agreement with the open-card solver on 83,844 human games
  84.5% → 84.78%. Claims deep learning has had "only marginal impact" in
  Skat/Bridge; uses no neural network. Modest, incremental.

---

## 2. Bridge

### 2.1 Bidding with RL (scored by double-dummy)

| System | Method | Data / compute | vs WBridge5 | Source |
|---|---|---|---|---|
| Yeh & Lin 2016 | DQN-style, non-competitive | self-generated deals | — | arXiv 1607.03290 |
| Rong, Qin, An 2019 | partner-hand estimator + policy net, SL then RL, competitive | human deals | **+0.25 IMP/b, 64 boards** | arXiv 1903.00900 |
| Gong, Jiang, Tian 2019 "Simple is Better" | A3C self-play from scratch, 200-unit MLP, no belief loss | 2.5M DDS-precomputed hands; **4–5 h on one GPU** | **+0.41 IMP/b, 64 boards**; auxiliary partner-hand belief loss *hurt* | ICML-W 2019 |
| Tian et al. 2020 Joint Policy Search | JPS on the self-play baseline | — | **+0.63 IMP/b, 1k boards** | arXiv 2008.06495 |
| Lockhart et al. (DeepMind) 2020 | imitation of 1M WBridge5 deals + particle-sampling search + policy iteration | 200k Adam steps | network-only +0.57±0.05; **+search +0.85±0.05, 10k boards** | arXiv 2011.14124 |
| Qiu, Wang, You, Zhou 2024 | SL → PG self-play + value net → **Belief Monte Carlo Search** at test time | one machine | **+0.98 IMP/deal, 10k deals** | IEEE JAS, doi:10.1109/JAS.2024.124488 |
| Kita et al. 2024 "Simple, Solid, Reproducible Baseline" | SL pretrain (1M boards) → **PPO + fictitious self-play**, action masking | 12.5M boards, 8,192 envs, 10k PPO updates | **+1.24±0.19 IMP/b, 1k boards** (SOTA); **RL from scratch could not beat WB5** | arXiv 2406.10306, CoG 2024 |

Take-aways: (a) every SOTA bidding result since 2020 uses **SL pretraining on a
strong bot's bids + self-play RL**; (b) fictitious/opponent-pool play stabilises
PPO; (c) **test-time belief search over sampled hands** adds ≈+0.3 IMP/b on top
of the network; (d) an *auxiliary partner-hand loss did not help* in Gong 2019,
but explicit belief nets used *at search time* did.

### 2.2 Card play and full engines
- **GIB / Jack / WBridge5 / Q-Plus / Micro Bridge**: sample deals consistent with
  the auction and play, solve each with a double-dummy solver, pick the best
  average: PIMC with an exact world evaluator. Still the competition-winning
  architecture.
- **αμ** (Cazenave & Ventos, arXiv 1911.07960; 2101.12639): search over vectors
  of worlds with DDS at leaves; partly corrects strategy fusion in declarer play.
- **NooK (NukkAI), 2022**: probabilistic inductive logic programming +
  Monte-Carlo; declarer play only at 3NT; **won 67 of 80 sets** against 8
  champions with WBridge5 defending. Not a full-game system.
- **BEN** (github.com/lorserker/ben): NN bidding trained on millions of
  auto-bid deals; card play = NN-sampled deals + DDS. Lost to WBridge5 5.2 by 12
  IMPs over 160 boards (2024).
- **Deep-RL bridge card play 2023–26**: **no** peer-reviewed result where a
  learned policy beats DDS-sampling engines at card play. Neural double-dummy
  estimators exist (Kowalik & Mańdziuk 2021, doi:10.1007/978-3-030-92273-3_2).

---

## 3. Other games

| Game | System | Method | Scale | Strength | Source |
|---|---|---|---|---|---|
| Dou Dizhu (3p, 2-vs-1) | **DouZero** (ICML 2021) | Deep Monte-Carlo, LSTM+MLP, 45 actors | 4×1080Ti, **30 days**; beat SL in 2 d, DeltaDou in 10 d | Botzone #1 of 344 | arXiv 2106.06135 |
| Dou Dizhu | **PerfectDou** (NeurIPS 2022) | **PPO+GAE, perfect-information critic, imperfect actor**, oracle reward | 2.5e9 samples; vanilla PPO without oracle critic/reward: WP 0.509 | SOTA | arXiv 2203.16406 |
| Dou Dizhu | AlphaDou 2024 | DMC with win-prob/value factorisation, bidding end-to-end | one RTX 4090 server | learned bidding > SL bidding | arXiv 2407.10279 |
| Dou Dizhu / HUNL | Fan & Farina 2026 "GAE Falls Short…" | Expected-SARSA(λ) advantage replacing GAE in imperfect-info self-play | — | consistent gains over PPO/GAE | arXiv 2605.19235 |
| Guandan (4p, known partners) | DanZero 2022 / DanZero+ 2023 | DMC (160 CPUs + 1 GPU, 30 d) then PPO restricted to the DMC's top-2 actions | PPO stage < 1 day | beats DMC model 55%, rule bots 86–93% | arXiv 2210.17087, 2312.02561 |
| Mahjong | **Suphx** (2020) | SL on Tenhou logs → PG with global reward predictor + **oracle guiding** (perfect-info features dropped out 1→0) + run-time policy adaptation | 44 GPUs × 2 days per agent | stable rank 8.74 (top humans 7.46) | arXiv 2003.13590 |
| Mahjong | Tjong 2024 | 15M-param transformer, hierarchical action | 7 d SL, 2 GPUs | top 1% Botzone | doi:10.1049/cit2.12298 |
| Big 2 (4p, no partners) | Charlesworth 2018; Patwa 2026 | PPO self-play, no search; entropy reg. important | small | beats amateur humans | arXiv 1808.10442, 2605.28863 |
| Gongzhu (4p Hearts-like) | ScrofaZero 2021 | tabula-rasa DRL + importance sampling + Bayesian inference | — | "human expert level" | arXiv 2102.07495 |
| Wizard | Schumacher & Pleines CoG 2022 | DQN; LSTM and tree search did **not** beat plain DQN | — | 66–87% bid accuracy | arXiv 2205.13834 |
| Spades | "Bidding in Spades" 2019 | expected-utility bidding + ML correction | — | > rule bots | arXiv 1912.11323 |
| Hearts | xinxin (Sturtevant) | PIMC+UCT, 50 worlds × 2000 runs | — | beaten by GO-MCTS | 2404.13150 |
| Doppelkopf (**hidden partner**) | Sievers & Helmert, KI 2015 | UCT with random *consistent* card assignment; ensemble-UCT (10 assignments × 1000 rollouts) | 10,000 rollouts/move | beats simple baselines; level with a human in 48 games; over-plays solos | ai.dmi.unibas.ch/papers/sievers-helmert-ki2015.pdf |
| Doppelkopf | Obenaus, FU Berlin BA 2017 | UCT + LSTM next-card prior | — | LSTM "did not improve UCT significantly" | thesis |
| Schafkopf (**hidden partner, Rufspiel**) | no peer-reviewed work; GitHub PPO/Q-learning hobby projects | — | none reported | — |
| Tarot (**called king**) | only Edelkamp KI 2024 framework (rule/search) | — | none | — |

**Hidden-partner inference in the literature is thin.** The only treatments are
(i) constraint-consistent deal sampling in Doppelkopf UCT (team identity emerges
from where the ♣Q are assigned; no learned belief), (ii) Edelkamp's rule-based
"team building" (paywalled), (iii) Rebstock's policy-based inference in Skat,
the closest principled template: weight worlds by the likelihood of every
observed action under a learned policy; the partner's identity is just one more
latent variable of the world. Nobody has published a learned belief head + PIMC
for a called-ace game.

---

## 4. PIMC, its critiques, and alternatives

- **Long, Sturtevant, Buro, Furtak, AAAI 2010** ("Understanding the Success of
  PIMC Sampling"): errors are *strategy fusion* and *non-locality*. Three
  synthetic properties (leaf correlation, bias, disambiguation factor) predict
  PIMC's loss vs Nash. Measured on human Skat and Hearts games: correlation
  0.8–1.0, disambiguation ≈0.6, so PIMC loses only ≈0.1 (on ±1) to equilibrium.
  Trick-taking games are the *favourable* regime for PIMC.
- **ISMCTS** (Cowling, Powley, Whitehouse, IEEE TCIAIG 2012,
  doi:10.1109/TCIAIG.2012.2200894): information-set trees with a fresh
  determinisation per iteration; has rarely beaten PIMC+exact solver in
  trick-taking games.
- **Extended PIMC** (Arjonilla, Saffidine, Cazenave, CoG 2024, arXiv 2408.02380):
  delay the perfect-information resolution to reduce strategy fusion.
- **Learned Belief Search** (Hu, Lerer, Brown, Foerster 2021, arXiv 2106.09086):
  learned belief, sample worlds, evaluate each candidate action by *rolling out
  the blueprint policy* (no fusion); in Hanabi 55–91% of exact search's
  (SPARTA, arXiv 1912.02318) benefit at 4.6–35.8× less compute. The direct
  ancestor of the project's belief-sampled rollout search.
- **Learned value networks over determinised worlds**: Bridge's DDS is an exact
  evaluator; neural DD estimators are cheap approximate ones; PerfectDou's
  perfect-information critic is exactly a learned value of a determinised world
  but is not used at test time. **No 2023–26 paper evaluates sampled worlds with
  a learned perfect-information value net inside PIMC in a trick-taking game.**
- Public-state search (ReBeL, Student of Games arXiv 2112.03178, LAMIR arXiv
  2510.05048): principled but two-player zero-sum only.
- Regularised self-play: DeepNash / R-NaD (arXiv 2206.15378) and MMD (arXiv
  2206.05825). Fan & Farina 2026 (arXiv 2605.19235): Expected-SARSA(λ)
  advantages instead of GAE in imperfect-information self-play.

---

## 5. Call-Ace / Esmakker / Danish whist
No academic or preprint work found (English and Danish terms). Only commercial
apps and hobby repos. **The project has no published baseline to compare against.**

---

## 6. Concrete lessons for the Danish-whist bot

1. **Search with a good belief is what carried card-play strength in every
   trick-taking game with a strong system.** Kermit (160 worlds, exact solver)
   still beats the best learned Skat policies by 2.6 TP/G (imitation) and 9.8
   TP/G (GO-MCTS with search). Bridge engines are PIMC+DDS. Belief-sampled
   search at test time is the right architecture; the evidence suggests
   **100–320 worlds** (Kermit saturates at 160; xinxin uses 50).
2. **The belief model is the highest-leverage component.** Conditioning
   inference on the full action history gave +3–4 TP/G, and weighting worlds
   by the *likelihood of observed actions under a learned policy* a further
   +1.6–2.3 TP/G, mostly on defence. For hidden partners, treat partner identity
   as part of the world and score worlds by ∏π(a_t|w,h_t) using the policy
   itself. Product-of-marginals sampling from a belief head was the *weaker*
   method.
3. **Do not evaluate worlds with one-world-optimal play alone.** The
   cheating-Kermit result and Long 2010 show PIMC's per-world values are
   biased; rolling out with the *policy* (LBS style) avoids strategy fusion. A
   learned perfect-information value (the project's oracle critic) as the leaf
   evaluator is unexplored in the literature and cheap.
4. **Training scale of comparable systems**: Skat imitation nets 20M human
   games; GO-MCTS 4M bootstrap + 5–20M self-play games; DouZero 30 GPU-days;
   DanZero 30 days × 160 CPUs; PerfectDou 1e9–2.5e9 samples; Suphx 88 GPU-days
   per agent; bridge bidding 12.5M boards (Kita), 4–5 GPU-hours (Gong). 1.2M
   deals is 1–2 orders of magnitude below the self-play systems that reached
   top strength, but on par with the bridge-bidding regime.
5. **Bidding**: every SOTA bridge bidder pretrains on a strong bot's bids and
   then runs PPO/self-play with fictitious play; pure from-scratch RL failed to
   beat WBridge5 in the careful 2024 replication. Value-based declaration
   (+1.17 TP/G) beat imitation in Skat: estimate contract EVs rather than
   imitate.
6. **Oracle critic**: validated by PerfectDou, Suphx, DouZero+/DanZero variants.
   Keep it.
7. **PPO specifics**: entropy regularisation and latest-model self-play helped
   (Gong 2019, Patwa 2026); opponent pools/fictitious play stabilised the
   cooperative bridge case (Kita 2024); Expected-SARSA(λ) advantages if variance
   is a problem (Fan & Farina 2026).
8. **Auxiliary belief loss**: hurt Gong 2019's bidder as a training auxiliary;
   helped when the belief net was *used at decision time*. Measure whether the
   belief head is actually consumed by the search.
9. **Evaluation protocol** used by the field: mirrored/duplicate deals with all
   seat permutations, 2,500–5,000 matches, paired tests; stake-weighted score,
   not win rate.
10. **Hidden-partner-specific**: no published solution; constraint-consistent
    sampling alone gives weak partner inference (Sievers' UCT misjudged solos);
    an LSTM prior on top of UCT didn't help (Obenaus). Policy-likelihood
    weighting of worlds is the best-supported method to import.

## Verification flags
- **Verified from primary text**: all Alberta Skat numbers, Gong 2019, Kita
  2024, Lockhart 2020, Qiu 2024, DouZero, PerfectDou, DanZero/DanZero+, Suphx,
  Long 2010, Sievers 2015, 2512.15435, 2404.13150, LBS, EPIMC, Wizard, Big 2,
  Bidding in Spades.
- **Abstract/secondary only**: Edelkamp KI 2024 framework, Hope Cards result,
  SoCS 2019, Rong 2019 data size, NooK method, Tjong numbers, ScrofaZero.
- **Unverified / suspect**: World Computer Bridge Championship 2024 and 2025
  winners.
