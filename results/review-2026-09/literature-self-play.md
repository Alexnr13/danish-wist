# Literature survey: self-play RL for a 4-player hidden-partnership card game

Compiled 27 September 2026 by a research agent for the review in `REVIEW.md`.
Numbers read from a paper's own text are stated as fact; the rest is marked
*(unverified)*. Numbers re-checked against extracted text: DeepNash η and Δ_m
table, MMD moving-magnet appendix H.5, Rudolph et al. appendix E/G and venue.

---

## 1. Self-play stability via regularised policy gradient

### R-NaD / DeepNash — Perolat et al., Science 378:990 (2022), arXiv:2206.15378
- **Reward transformation**: at turn t the acting player receives r − η·log(π/π_reg)
  and the others r + η·log(π/π_reg), so the transformed game stays zero-sum.
  **η = 0.2, constant throughout training.**
- **How often π_reg is updated:** the regularised game is solved for a fixed
  number of learner steps Δ_m, then π_reg := the target/EMA network's policy
  (EMA rate 0.001). **Δ_m = 10k learner steps for m ≤ 100; 100k for
  100 < m ≤ 165; 35k after**; 7.21M learner steps in all; batch 768
  trajectories per step. The switch is smoothed over the next iteration.
  Optimizer state carries over.
- Learner: v-trace with the η log-ratio terms folded into the Q estimator,
  NeuRD update, lr 5e-5, Adam(β1 = 0, β2 = 0.999).
- Theory: two-player zero-sum only. Larger η → faster convergence to a fixed
  point more biased toward π_reg, so more outer iterations needed.
- Only multiplayer R-NaD study found: Dewey et al., "Mastering Liar's Poker",
  arXiv:2511.03724 (3-player): ran R-NaD unchanged; observed convergence
  without guarantees.

### Magnetic Mirror Descent — Sokota et al., ICLR 2023, arXiv:2206.05825
- π_{t+1} = argmax_π E q_t − α·KL(π, ρ) − (1/η)·KL(π, π_t). Default magnet ρ is
  **fixed uniform** (then the α term is an entropy bonus); fixed point is a QRE.
- Two ways to reach Nash: anneal α_t = η_t = 1/√t, or a **moving magnet**
  (App. H.5): ρ_{t+1} ∝ ρ_t^{1−η̃} · π_{t+1}^{η̃} with **α = 1, η = 0.1,
  η̃ = 0.05** per iteration, "at a much faster rate than annealing". Tabular.
- Deep variant (App. J): PPO with a constant reverse KL to the *previous*
  policy plus entropy, α_t = η_t = 0.05·√(10M/t). Two-player only.

### Rudolph et al., "Reevaluating Policy Gradient Methods for Imperfect-Information Games", arXiv:2502.08938 — **ICLR 2026 (poster)**, verified
- 7 algorithms (PPO, PPG, MMD; NFSP; PSRO; ESCHER; R-NaD) × 5 two-player games;
  50 configs × 3 seeds × 10M steps per game–algorithm, then top-5 × 10 seeds:
  **7,000 runs, >345,000 CPU-hours.** Metric: exact exploitability. All
  algorithms: 3×512 MLP, **one shared model for both players**.
- Findings: R-NaD, NFSP, ESCHER and PSRO failed to outperform generic PG
  methods; **PPO, PPG and MMD were roughly on par**, so the magnet term bought
  nothing measurable over PPO with a high entropy bonus.
- What mattered: **the entropy coefficient is the top hyperparameter in nearly
  every game**; then learning rate, clip, vf_coef; MMD's kl_coef ranks 2nd–7th.
  **Best exploitability at entropy coefficient 0.05–0.2**, far above library
  defaults (0–0.01).
- Hyperparameters (App. G): PPO lr 2.5e-4 annealed, batch 1024, 4 minibatches,
  4 epochs, clip 0.1, **ent_coef 0.05**, vf_coef 0.5, max_grad_norm 0.5, GAE λ
  0.95, advantage normalisation on, target_kl disabled. MMD = same + reverse KL
  to the previous policy with **constant kl_coef 0.05**.

### 2024–2026 follow-ups
- **EMAgnet** (Maidment et al., arXiv:2606.23995): PPO self-play with KL toward a
  *parameter-space EMA* of the policy, θ_mag ← (1−τ)θ_mag + τθ after each PPO
  epoch; τ in [1e-5, 0.1], λ_KL in [0.01, 32], constant. Beats uniform-magnet PPO
  in 8/9 small games. The closest match to "PPO + KL to a lagging copy".
- **GARIP** (arXiv:2606.22688): collapse tracks the *peak* lag of the reference;
  a running-average reference (ρ = 0.01 per update) minimises it.
- **NashPG** (arXiv:2510.18183): KL to a reference reset every 10,000 updates,
  **α = 0.2 with a U-shaped sensitivity**.
- **APMD** (Abe et al., ICML 2024, arXiv:2305.16610): reference reset every 200
  iterations, μ = 0.1; last-iterate rates for N-player monotone games (the only
  >2-player theory found).
- Practical PPO self-play, 2026: **Generals.io** (arXiv:2606.23348): plain PPO,
  current-policy self-play, *no* KL term, entropy 0.05·(t+1)^−0.2, parameter EMA
  τ = 0.999 for deployment; 4 days on 4×H200; #1 on the ladder. **Big 2**
  (Patwa, arXiv:2605.28863; 4-player): one shared PPO policy in all seats, no KL,
  ent_coef 0.05 best, lr 3e-5, current-policy self-play > checkpoint self-play >
  fixed-opponent training; 320k games, 7–13 h on a laptop.

---

## 2. Population / league methods against drift and cycling

### AlphaStar — Vinyals et al., Nature 2019, doi:10.1038/s41586-019-1724-z
- Main agents: **35% self-play, 50% PFSP against all past league players, 15%
  PFSP vs forgotten main players and past main exploiters.** Snapshot every
  2×10^9 steps; main agents never reset.
- PFSP: opponent B sampled ∝ f(P[A beats B]); **f_hard(x) = (1−x)^p** (default;
  "helps with integrating information from exploits, as these are strong but
  rare counter strategies, and a uniform mixture would be able to just ignore
  them"); **f_var(x) = x(1−x)** for main exploiters and struggling agents.
- Main exploiters: train only against current main agents; reset to supervised
  parameters. League exploiters: PFSP over the whole league.
- Ablations (Elo / min-win-rate vs all past versions): **pFSP+SP 1540 / 71%; SP
  1519 / 46%; pFSP 1273 / 70%; FSP 1143 / 69%**. League composition: main only
  1540 / 6% relative population performance; + main exploiters 1693 / 35%;
  + league exploiters 1824 / 62%.

### PSRO — Lanctot et al., NeurIPS 2017, arXiv:1711.00832
- **Joint Policy Correlation**: independent RL loses **34–72%** of reward when
  paired with a different run. A cheap, direct test for "self-play policy
  overfit to its co-trained selves". Plain PSRO can *increase* exploitability
  between iterations (Anytime PSRO, arXiv:2201.07700).

### OpenAI Five — Berner et al. 2019, arXiv:1912.06680
- **80% vs latest parameters, 20% vs past versions**, "to avoid strategy
  collapse in which the agent forgets how to play against a wide variety of
  opponents". Pool = **entire history**; sampled by a quality score updated on
  each win.

### Balduzzi et al., ICML 2019, arXiv:1901.08106
- "**Self-play assumes transitivity**: that local improvements imply global
  improvements. The assumption fails in nontransitive games." Rectified Nash
  population growth beat self-play; training against a much weaker opponent
  yields no learning signal.

### Other evidence on the past-self fraction
- NFSP (arXiv:1603.01121): act from the average policy 90% of the time.
- Bansal et al., ICLR 2018, arXiv:1710.03748: "training against the latest
  opponent leads to worst performance"; sampling uniformly from the last half or
  the whole history was best.
- Hernandez et al., CoG 2019: naive self-play "clearly exhibits cyclic
  catastrophic forgetting"; full-history uniform does not.
- 7-player Diplomacy: Anthony et al., NeurIPS 2020, arXiv:2006.04635: iterated
  best response cycles; best-responding to the *time-average* of checkpoints,
  sampling **one** historical checkpoint for *all* opponents to preserve
  correlations, was best. DORA (arXiv:2110.02924): pure self-play from scratch
  reached incompatible equilibria in 7-player. Diplodocus / Cicero: KL toward a
  human anchor with a *distribution* of λ.

---

## 3. Asymmetric / oracle critics
- **Baisero & Amato, AAMAS 2022, arXiv:2105.11674**: a state-only critic V(s)
  is a *biased* target for a history-conditioned policy; the **history-state
  critic V(h,s) is unbiased**. Lyu et al., JAIR 2023, arXiv:2408.14597:
  state-based critics "may incur unbounded bias"; history+state critics are
  unbiased and "usually a favorable trade-off". The project's critic (view plus
  hidden cards) is in the unbiased regime.
- **Suphx**: oracle information goes into the *actor*, then dropped out with
  Bernoulli(γ_t) decaying 1 → 0; plain distillation "does not work well". SL on
  44M state-action pairs; each RL agent 1.5M self-play games on 44 GPUs for 2
  days.
- **PerfectDou** (arXiv:2203.16406): PPO+GAE, value net sees all hands plus an
  *oracle reward*. 880 CPU cores + 8 GPUs, 2.5×10^9 steps. **Ablation at 10^9
  steps: vanilla PPO WP 0.509; critic seeing only public info 0.717; full
  0.732 / ADP 1.27** — the perfect-information critic alone added little;
  oracle features + oracle reward mattered.
- **DouZero** (arXiv:2106.06135): 48-core Xeon + 4×1080 Ti, **30 days**; beats
  SL in 1–2 days, DeltaDou in 3–10 days; ~10^9 steps in ~50 h.
- **Fan & Farina, "GAE Falls Short in Imperfect-Information Self-Play RL",
  arXiv:2605.19235 (May 2026)**: GAE's sampled multi-step backups add variance
  from sampled opponent/own actions even with an exact critic; VRPO = full-
  information Q-critic + Expected-SARSA(λ) traces inside PPO; ≥15% lower
  exploitability on small games, +33 mBB/hand vs Slumbot; Dou Dizhu 2.65×10^9
  steps in ~55 h on 4×RTX 5090.

---

## 4. Credit assignment with hidden teams
- **DeepRole** (Serrino et al., NeurIPS 2019, arXiv:1906.02330; Avalon): value
  net outputs P(win | role assignment) for every assignment; a generic head
  generalised worse. 60% wins with 4 humans vs human 48%.
- **Werewolf** (Xu et al., ICML 2024, arXiv:2310.18940): explicit hidden-role
  deduction + MAPPO with population training. Removing deduction: 0.30 → 0.16;
  **self-play instead of population training: 0.30 → 0.26 / 0.70 → 0.66**.
- **Hanabi** (parameter-shared across seats): auxiliary hand-prediction head
  helped 2p, **hurt 3–5p self-play**, but helped cross-play robustness (Other-
  Play, arXiv:2003.02979); with better infrastructure "no longer helpful" (OBL).
- **Trick-taking with secret partners**: Doppelkopf UCT with consistent card
  assignment (Sievers & Helmert 2015); Schafkopf/Sheepshead: GitHub only
  (tobiasemrich/SchafkopfRL: PPO, shared net, LSTM; "team concept not well
  understood: agent sometimes plays higher trump than teammate").
- **One shared network across seats is standard** (Rudolph et al., Hanabi, Big
  2, DeepNash). No controlled ablation of role-conditioned heads exists.

---

## 5. Pluribus — Brown & Sandholm, Science 2019
- Blueprint: MCCFR, **8 days on a 64-core server, 12,400 CPU core-hours**. Play:
  depth-limited search; at leaves each remaining player chooses among **k = 4
  continuation strategies**. "Do not have known strong theoretical guarantees
  outside two-player zero-sum" but work empirically. No 2020–2025 multiplayer
  reuse of biased continuation strategies found beyond Pluribus.

---

## 6. Training-scale reference points

| System (game) | Method | Games / steps | Hardware | Wall-clock | Baseline beaten |
|---|---|---|---|---|---|
| DouZero 2021 (Dou Dizhu) | DMC self-play | ~10^9 steps in ~50 h; released ≈10^10 | 48 cores + 4×1080 Ti | 30 d (DeltaDou at 3–10 d) | DeltaDou, SL |
| PerfectDou 2022 | PPO, perfect-info critic + oracle reward | 2.5×10^9 steps | 880 CPU cores + 8 GPUs | — | DouZero |
| Fan & Farina 2026 (Dou Dizhu) | PPO + Q-critic | 2.65×10^9 steps | 4×RTX 5090 | ~55 h | PerfectDou-budget |
| DanZero 2022 (Guandan) | DMC | — | 160 CPUs + RTX 3070 | 30 d | 8 rule bots |
| Suphx 2020 (Mahjong) | SL + PG w/ oracle | SL 44M pairs; RL 1.5M games/agent | 44 GPUs | 2 d per agent | 10 dan Tenhou |
| DeepNash 2022 (Stratego) | R-NaD | 7.21M learner steps × 768 traj | 768 + 256 TPU nodes | *(months, unverified)* | 84% vs humans |
| AlphaHoldem 2022 (HUNL) | Trinal-clip PPO, K-best self-play | 2.7×10^9 hands | 8×TITAN V + 64 cores | 3 d | Slumbot +111.6 mbb/h |
| Lockhart et al. 2020 (bridge bidding) | imitation + search + policy iteration | 50–170M learner obs | — | — | WBridge5 +0.85 IMPs/deal |
| Rebstock et al. 2019 (Skat) | imitation, human data | 23M–289M samples | 1×1080 Ti | — | Kermit +1.17 TP/game |
| Charlesworth 2018 (Big 2) | PPO self-play | 1.5×10^8 steps ≈ 3M games | 4 cores + 1 GPU | ~2 d | amateur humans |
| Patwa 2026 (Big 2) | PPO shared policy | 320k games | 6-core laptop | 7–13 h | rule bots |

The project's ~1.2M deals sits 1–2 orders of magnitude below every crossover
point above except the laptop-scale Big 2 runs.

---

## 7. Checkpoint selection and evaluation
- **Balduzzi et al., "Re-evaluating evaluation", NeurIPS 2018, arXiv:1806.02643**:
  Elo assumes transitivity; Nash averaging is invariant to clones.
- **Timbers et al., IJCAI 2022, arXiv:2004.09677**: approximate exploitability
  via a learned best response; "head-to-head performance and exploitability
  are complementary".
- **Agarwal et al., NeurIPS 2021, arXiv:2108.13264**: reporting max-during-
  training is "generally incomparable with end-performance"; use IQM +
  stratified bootstrap. Winner's curse: select on one set, report on another
  (Bastani et al., arXiv:2510.18161). Robust test-opponent sets: Morrill et
  al., UAI 2023, arXiv:2306.07372.
- Practice: **AlphaStar Unplugged** (arXiv:2308.03526) uses the built-in bot
  only "as a validation metric, to tune hyper-parameters and discard" runs, and
  *reports* the minimum win rate over held-out reference agents.
- **No paper analyses "restart from the checkpoint that scores best against one
  fixed scripted bot"**; the closest evidence is JPC and the Werewolf/AlphaStar
  population ablations.

---

## What this implies for the project's setup

1. **Raising the magnet weight (0.02 → 1.0) is not the lever the literature
   supports.** Everywhere the magnet is a *stability* device with moderate
   weight (R-NaD η = 0.2 with the reference re-anchored every 10k–100k steps;
   MMD's moving magnet at η̃ = 0.05; NashPG α = 0.2, U-shaped). A large weight
   on a slowly refreshed copy pins the policy near it, consistent with approx
   KL 0.003–0.007 per iteration. Rudolph et al. found the KL weight far less
   important than the entropy coefficient. The drift toward beating one's own
   weak defence is the self-play-transitivity failure of Balduzzi/JPC, which a
   regulariser toward *yourself* cannot address. **Recommendation:** magnet
   0.05–0.2 with a continuously moving reference (EMA τ ≈ 0.01 per update, or a
   reset every few hundred updates); entropy coefficient around 0.05 on
   normalised advantages.
2. **Fix the opponent population, not the regulariser.** A pool of the last 8
   snapshots used occasionally is the configuration every study found worst.
   Evidence-backed mixes: ~40–50% current self-play, ~35–50% PFSP with
   f_hard = (1−p)^p over an unbounded (or ≥30–50) snapshot pool **with the
   scripted bot inside the pool** (its weight then rises whenever it beats
   you), ~10–15% vs exploiters. Because the game has hidden partners, sample
   *one* historical checkpoint for all three other seats in a deal.
3. **Add exploiters**: every N iterations clone the current network, train it
   against the frozen current policy in the other seats, record its margin
   (the exploitability proxy, a better progress metric than the scripted bot)
   and add it to the PFSP pool. AlphaStar: main exploiters 6% → 35% relative
   population performance, league exploiters → 62%.
4. **Restarting from the checkpoint that scores best vs the scripted bot is
   unsound as practised**: max-of-noisy-estimates selection on the reporting
   opponent, selecting for that bot's blind spots (JPC 34–72% reward loss), and
   each restart discards the population memory that prevents cycling. Use the
   bot to *discard* runs, select on a held-out seed set against a reference set
   (bot + several past snapshots + latest exploiter), report on another seed set.
5. **Oracle critic: keep it; it is the unbiased V(h,s) form.** Expect modest
   gains from the critic alone (PerfectDou 0.717 → 0.732); try Expected-
   SARSA(λ) advantages from the full-information critic instead of GAE.
6. **Auxiliary hidden-card head: mixed evidence.** Keep it cheap; a head for
   "who holds the called ace" is better motivated than all 52 cards; the belief
   is worth most when *used at decision time*.
7. **A shared network for all seats is standard.**
8. **Ranges from the PPO-for-imperfect-information literature:** lr 2.5e-4
   annealed (swept 3e-5…2e-3), clip 0.1–0.2, 4 epochs × 4 minibatches on
   1,024-sample batches, entropy 0.05–0.2, reverse-KL 0.05 constant. None
   report per-update KL as a progress variable; the comparison points are
   exploitability / best-response margin and min-win-rate vs the pool.
9. **Scale:** 1.2M deals is 1–2 orders of magnitude below where DouZero,
   PerfectDou, AlphaHoldem or Suphx first beat strong baselines. Slow learning
   is partly expected; adding regularisation that further slows it is the wrong
   direction. Throughput and the opponent mix are the levers with evidence.
   Every convergence result surveyed is two-player zero-sum or N-player
   monotone; for a 4-player hidden-team game the precedents are empirical and
   all relied on population/averaging rather than a single regularised line.

### Unverified / not found
- AlphaStar's numeric p in f_hard; DeepNash wall-clock; Liar's Poker η.
- No academic Schafkopf/Sheepshead/Tarock partner-inference paper; no controlled
  ablation of role-conditioned heads; no paper on restart-from-best-vs-bot.
