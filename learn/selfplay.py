"""Self-play reinforcement learning (LEARNING.md §3, step 4).

One network (the *learner*) plays many deals at once, mostly against itself
and sometimes against RuleBot or frozen past versions of itself. Its
decisions are then improved with PPO:

- **Critic with hidden cards.** A separate value network reads
  `encode_oracle` (the player's view plus every hidden card), in training
  only. It predicts the final score as odds over bins spaced evenly on a
  symmetric-log scale (a "two-hot" head), so scores from 10 to 100,000 are
  all resolved and its expected score is unbiased, without reshaping the
  reward.
- **Advantages** come from GAE over each seat's own decisions in a deal. The
  only reward is that seat's score at the end. Forced moves (one legal
  action) are played but not recorded: there is nothing to learn from them.
- **Magnet.** A KL penalty keeps the policy near a slowly refreshed copy of
  itself (as in DeepNash and magnetic mirror descent), so self-play does not
  cycle or chase its own tail.

    python -m learn.selfplay --init runs/bc.pt --iterations 200 --out runs/rl
    python -m learn.selfplay --resume runs/rl --iterations 200   # carry on to 200
    python -m learn.selfplay --init runs/bc.pt --exploit runs/rl/policy.pt --out runs/x
    python -m learn.selfplay --init runs/rl --explore-bids 0.15 --out runs/rl2  # from rl's networks

A run's directory holds `log.jsonl` (one line per iteration), `evals.jsonl`
(each evaluation's per-deal results, for paired comparisons), `run.json` (the
command line and commit) and `settings.json`; the latest `policy.pt`,
`critic.pt` and `policy.npz`, with numbered copies in `checkpoints/` at every
snapshot and evaluation; and `state.pt`, from which `--resume` carries on
exactly where the run stopped: the same iteration count, optimisers, magnet,
snapshot pool and random state.

`--exploit` trains an *exploiter*: a fresh learner in one seat against a frozen
policy in the other three. How much it gains over that policy (`vs_target`
in the log) measures how exploitable the policy is (LEARNING.md §4).

Games are played by `--workers` processes through `learn.runner.Runner`, each
with a copy of the policy that gets new weights every iteration and runs on
`--worker-device` (an NVIDIA GPU when there is one, else a CPU thread); the
update runs on `--device` (an NVIDIA GPU, else Apple's, else the CPU).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import signal
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from danish_wist.actions import Action
from danish_wist.bidding import NUM_PLAYERS, Bid
from danish_wist.bots import RuleBot
from danish_wist.game import Deal, Phase
from danish_wist.scoring import trick_value

from .arena import default_workers, random_positions
from .contracts import KINDS, contract_kind
from .encoding import (
    ACTIONS,
    NOT_HIDDEN,
    PHASES,
    Observation,
    belief_targets,
    encode_oracle,
    observe,
)
from .evaluate import evaluate
from .model import Net, NetAgent, NetConfig, collate, device_of, export, inputs, load, save
from .runner import Decision, Runner


@dataclass(frozen=True)
class Settings:
    deals_per_iteration: int = 256
    league_share: float = 0.5  # deals against a league member drawn by priority (`League`)
    exploiter_share: float = 0.15  # deals against an exploiter, once there are any
    league_size: int = 50  # members at most (RuleBot, snapshots, exploiters)
    snapshot_every: int = 10  # iterations between snapshots joining the league
    exploit_every: int = 0  # iterations between training exploiters (0: none)
    exploit_iterations: int = 50  # each exploiter's training
    games_in_flight: int = 256  # deals each worker keeps going at once
    gae_lambda: float = 0.95
    ppo_epochs: int = 1  # deals are cheap to play, so fresh ones beat a second pass
    batch_size: int = 512
    chunk: int = 256  # rows per forward and backward pass (see `_chunks`)
    clip: float = 0.2
    policy_lr: float = 1e-4
    critic_lr: float = 3e-4
    entropy: float = 0.01
    magnet: float = 0.02  # weight of KL(policy || magnet)
    magnet_ema: float = 0.0  # its step towards the policy per iteration (0: a copy, refreshed)
    belief: float = 0.1  # weight of the auxiliary loss for predicting unseen cards
    critic_warmup: int = 0  # first iterations that train only the critic
    explore_bids: float = 0.0  # share of each bid's chance that play moves to other kinds
    explore_levels: float = 0.0  # share that play moves to the same kind one and two levels up
    stake_scaling: bool = False  # weigh decisions from the exchange on by the deal's stake


# --- Collecting experience ---------------------------------------------------


@dataclass
class Step:
    """One learner decision, stored compactly."""

    observation: Observation  # NumPy arrays
    oracle: np.ndarray  # tokens including hidden cards, for the critic
    belief: np.ndarray  # where each unseen card really is (`encoding.belief_targets`)
    action: int
    log_prob: float  # the chance it was played with (`explored` odds in the auction)
    policy_log_prob: float  # the policy's own chance of it


@dataclass
class Trajectory:
    """Everything one learner seat decided in one deal, and what it scored."""

    steps: list[Step]
    reward: float
    contract: tuple[str, int, bool] | None = None  # (kind, level, made) if this seat declared
    stake: float = 0.0  # this seat's points per trick (see `stake`)


def stake(deal: Deal, seat: int) -> float:
    """What each trick is worth to `seat` in a finished deal: the trick value, and three
    times it for a declarer alone (0 if the deal was thrown in).

    Once trumps are decided the stake cannot change, so from then on every
    decision's result is a whole number of these units.
    """
    if deal.redeal:
        return 0.0
    value = trick_value(deal.bid, deal.trumps)
    return float(3 * value if deal.alone and seat == deal.declarer else value)


_EXCHANGE = PHASES.index(Phase.EXCHANGE)


def stake_is_fixed(observation: Observation) -> bool:
    """Whether a decision comes after trumps are decided (its first token is the phase)."""
    return int(observation.tokens[0][3]) >= _EXCHANGE


# Agent names in each runner worker (see `make_agents`).
LEARNER, EXPLOITER, RULE, EVAL, TARGET = "learner", "exploiter", "rule", "eval", "target"
SLOTS = 8  # league members loaded in each worker at once (see `League.draw`)


def slot_name(slot: int) -> str:
    return f"opponent-{slot}"


def _compact(observation: Observation) -> Observation:
    return Observation(
        np.asarray(observation.tokens, dtype=np.int16),
        np.asarray(observation.legal, dtype=np.int16),
    )


_BIDS = torch.tensor([i for i, a in enumerate(ACTIONS) if isinstance(a, Bid)])
_LEVELS = torch.tensor([ACTIONS[i].level for i in _BIDS])
SAME_LEVEL = (_LEVELS[:, None] == _LEVELS[None, :]).float() - torch.eye(len(_BIDS))
_SAME_KIND = torch.tensor(
    [[ACTIONS[i].attachment is ACTIONS[j].attachment for j in _BIDS] for i in _BIDS]
)
_UP = _LEVELS[None, :] - _LEVELS[:, None]  # (from, to): how many levels higher `to` is
HIGHER = (_SAME_KIND & (_UP >= 1) & (_UP <= 2)).float()  # the same kind, one or two levels up


def explored(
    probs: torch.Tensor, legal: torch.Tensor, share: float, levels: float = 0.0
) -> torch.Tensor:
    """The odds the learner bids from while it explores (`Settings.explore_bids` and
    `explore_levels`).

    `share` of each bid's chance moves, evenly, to the other legal kinds of
    contract at its level (plain, Flip, Clubs, Halves), so kinds the policy
    has almost given up on stay in play while it learns to play them.
    `levels` of it moves, evenly, to the same kind one and two levels up, so
    that higher contracts are played, and their play learned, before the
    policy would choose them. Other actions keep their chances. probs and
    legal are (B, NUM_ACTIONS).
    """
    bids = probs[:, _BIDS]
    kept = bids.clone()
    arriving = torch.zeros_like(bids)
    for matrix, part in ((SAME_LEVEL, share), (HIGHER, levels)):
        targets = matrix * legal[:, None, _BIDS]  # (B, from, to): where each bid's share may go
        counts = targets.sum(-1)
        moved = part * bids * (counts > 0)
        kept -= moved
        arriving += ((moved / counts.clamp(min=1))[:, :, None] * targets).sum(1)
    result = probs.clone()
    result[:, _BIDS] = kept + arriving
    return result


class Learner:
    """The policy being trained, as it runs in a worker.

    It samples its actions and records each real decision (not forced moves)
    with its training targets (critic tokens and where the unseen cards are,
    both read from the deal) under (game, seat), for `trajectories` to collect
    when the game ends. Only the view goes into the policy. With `explore` or
    `explore_levels`, it bids from `explored` odds, and records each action's
    chance under them as well as the policy's own, for `ppo_objective`.
    """

    def __init__(
        self,
        config: NetConfig,
        worker: int,
        explore: float = 0.0,
        explore_levels: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.net = Net(config).to(device).eval()
        self.worker = worker
        self.explore = explore
        self.explore_levels = explore_levels
        self.generator = torch.Generator()
        self.steps: dict[tuple[int, int], list[Step]] = defaultdict(list)

    def load(self, state: dict) -> None:
        self.net.load_state_dict(state)

    def seed(self, seed: int) -> None:
        self.generator.manual_seed(seed * 1009 + self.worker)  # a different stream per worker

    @torch.no_grad()
    def choose_decisions(self, decisions: list[Decision]) -> list[Action]:
        observations = [observe(decision.view) for decision in decisions]
        tokens, padding, legal = collate(observations)
        device = device_of(self.net)
        logits = self.net(tokens.to(device), padding.to(device), legal.to(device))[0]
        own = torch.log_softmax(logits, dim=-1).cpu()  # sampled on the CPU, on any device
        log_probs = own
        auction = torch.tensor([d.view.phase is Phase.AUCTION for d in decisions])
        if (self.explore or self.explore_levels) and auction.any():
            probs = own.exp()
            probs[auction] = explored(
                probs[auction], legal[auction], self.explore, self.explore_levels
            )
            log_probs = probs.log()
        actions = torch.multinomial(log_probs.exp(), 1, generator=self.generator).squeeze(-1)
        chosen = log_probs.gather(1, actions[:, None]).squeeze(-1)
        chosen_own = own.gather(1, actions[:, None]).squeeze(-1)
        for decision, observation, action, log_prob, own_log_prob in zip(
            decisions,
            observations,
            actions.tolist(),
            chosen.tolist(),
            chosen_own.tolist(),
            strict=True,
        ):
            if len(observation.legal) == 1:
                continue  # a forced move: it gives the policy no gradient
            oracle = encode_oracle(decision.deal, decision.seat, observation.tokens)
            oracle = np.asarray(oracle, dtype=np.int16)
            belief = np.asarray(belief_targets(decision.deal, decision.seat), dtype=np.int8)
            step = Step(_compact(observation), oracle, belief, action, log_prob, own_log_prob)
            self.steps[decision.game, decision.seat].append(step)
        return [ACTIONS[action] for action in actions.tolist()]


class Frozen(NetAgent):
    """A network that only plays, greedily: past snapshots, exploit targets, evaluation."""

    def __init__(self, config: NetConfig, device: str = "cpu") -> None:
        super().__init__(Net(config).to(device))

    def load(self, state: dict) -> None:
        self.net.load_state_dict(state)
        self.net.eval()


def make_agents(
    worker: int,
    config: dict,
    slots: int,
    one_thread: bool,
    explore: float = 0.0,
    explore_levels: float = 0.0,
    device: str = "cpu",
) -> dict:
    """The agents in each runner worker, by name. Their weights arrive by `broadcast`.

    Their networks run on `device`: on a GPU, each worker's forward passes cost
    little, and the worker's time goes to the engine and the encoding.
    """
    if one_thread:
        torch.set_num_threads(1)  # one process per core already
    net_config = NetConfig(**config)
    agents = {
        LEARNER: Learner(net_config, worker, explore, explore_levels, device),
        EXPLOITER: Learner(net_config, worker, explore, explore_levels, device),
        RULE: RuleBot(),
        EVAL: Frozen(net_config, device),
        TARGET: Frozen(net_config, device),
    }
    return agents | {slot_name(slot): Frozen(net_config, device) for slot in range(slots)}


def trajectories(
    game: int, deal: Deal, agents: dict, learner: str = LEARNER
) -> tuple[int, list[Trajectory], list[int]]:
    """Runs in the worker when a game ends: the learner's decisions in it with their scores,
    and the deal's scores (the learner's results against its opponents)."""
    steps = agents[learner].steps
    contract = None
    if not deal.redeal:
        contract = (contract_kind(deal.bid), deal.bid.level, deal.scores[deal.declarer] > 0)
    found = [
        Trajectory(
            steps.pop((game, seat)),
            float(deal.scores[seat]),
            contract if seat == deal.declarer else None,
            stake(deal, seat),
        )
        for seat in range(NUM_PLAYERS)
        if (game, seat) in steps
    ]
    return game, found, list(deal.scores)


def contract_stats(found: list[Trajectory]) -> dict:
    """The contracts the learner declared: the share of each kind, the mean level, how
    often they were made."""
    contracts = [t.contract for t in found if t.contract is not None]
    if not contracts:
        return {}
    return {
        "declared": {
            k: round(sum(c[0] == k for c in contracts) / len(contracts), 4) for k in KINDS
        },
        "level": round(float(np.mean([c[1] for c in contracts])), 2),
        "made": round(float(np.mean([c[2] for c in contracts])), 3),
    }


def collect(
    runner: Runner,
    lineups: list[list[str]],
    rng: random.Random,
    learner: str = LEARNER,
    scores: dict[int, list[int]] | None = None,
) -> list[Trajectory]:
    """Play one random deal per lineup (an agent name per seat), recording `learner`.

    With `scores`, each deal's final scores are put in it, by the lineup's index.
    """
    runner.broadcast(learner, "seed", rng.getrandbits(32))
    games = list(zip(random_positions(len(lineups), rng), lineups, strict=True))
    found = []
    for game, trajectories_found, final in runner.play(
        games, finish=partial(trajectories, learner=learner)
    ):
        found += trajectories_found
        if scores is not None:
            scores[game] = final
    return found


def exploit_lineups(count: int, rng: random.Random, learner: str = LEARNER) -> list[list[str]]:
    """The learner in one random seat, the frozen target in the other three."""
    lineups = []
    for _ in range(count):
        lineup = [TARGET] * NUM_PLAYERS
        lineup[rng.randrange(NUM_PLAYERS)] = learner
        lineups.append(lineup)
    return lineups


def lineups_against(opponents: list[str | None], rng: random.Random) -> list[list[str]]:
    """One lineup per deal: the learner in every seat, or with the deal's opponent (an agent
    name) in 1–3 random seats. One opponent fills all of them: a hidden partner plays
    realistically only beside its own kind."""
    lineups = []
    for opponent in opponents:
        lineup = [LEARNER] * NUM_PLAYERS
        if opponent is not None:
            for seat in rng.sample(range(NUM_PLAYERS), rng.randint(1, NUM_PLAYERS - 1)):
                lineup[seat] = opponent
        lineups.append(lineup)
    return lineups


def outcome(lineup: list[str], scores: list[int]) -> float:
    """The learner's result in a deal against another agent: 1 if its seats together scored
    more than nothing, 0 if less, 1/2 if nothing."""
    total = sum(score for name, score in zip(lineup, scores, strict=True) if name == LEARNER)
    return 1.0 if total > 0 else 0.0 if total < 0 else 0.5


@dataclass
class Member:
    """A league member: a past learner, an exploiter or RuleBot (no weights)."""

    weights: dict | None
    iteration: int  # when it joined
    exploiter: bool = False
    wins: float = 0.0  # the learner's results against it, lately (see `League.record`)
    games: float = 0.0


class League:
    """The opponents the learner meets besides itself (REVIEW.md T2.1; AlphaStar's league).

    Members are snapshots of the learner, exploiters trained against it, and
    RuleBot, which is always in. Opponents are drawn by prioritised fictitious
    self-play: in proportion to f_hard(x) = (1 - x)^2, where x is the
    learner's recent win rate against the member (`outcome`), so members it
    struggles against come up more. Beyond `size` members, past snapshots are
    thinned where they are closest together for their age, so older ones grow
    sparse and the pool spans the whole run.
    """

    DECAY = 0.9  # each iteration's weight in a member's win rate, against the ones before

    def __init__(self, size: int = 50, folder: Path | None = None) -> None:
        self.size = size
        self.folder = folder  # where members' weights are kept, one file each, if anywhere
        self.members: dict[str, Member] = {RULE: Member(None, 0)}

    def add(self, name: str, weights: dict, iteration: int, exploiter: bool = False) -> None:
        self.members[name] = Member(weights, iteration, exploiter)
        if self.folder is not None:
            self.folder.mkdir(parents=True, exist_ok=True)
            _replace(self._file(name), partial(torch.save, weights))
        while len(self.members) > self.size and self._thin():
            pass

    def _file(self, name: str) -> Path:
        return self.folder / f"{name.replace('/', '_')}.pt"

    def _thin(self) -> bool:
        snapshots = sorted(
            (m.iteration, name)
            for name, m in self.members.items()
            if m.weights is not None and not m.exploiter
        )
        if len(snapshots) < 2:
            return False  # nothing to thin: the newest snapshot always stays
        newest = snapshots[-1][0]
        # Leave out the one whose neighbours are closest together, for their age.
        crowded = min(
            range(len(snapshots) - 1),
            key=lambda i: (
                (snapshots[i + 1][0] - (snapshots[i - 1][0] if i else 0))
                / (newest - snapshots[i][0] + 1)
            ),
        )
        name = snapshots[crowded][1]
        del self.members[name]
        if self.folder is not None:
            self._file(name).unlink(missing_ok=True)
        return True

    def win_rate(self, name: str) -> float:
        member = self.members[name]
        return (member.wins + 1) / (member.games + 2)  # 1/2 before any games

    def draw(self, count: int, rng: random.Random, exploiters: bool = False) -> list[str]:
        """`count` members, with repeats, by priority (exploiters only, if asked)."""
        names = [n for n, m in self.members.items() if m.exploiter or not exploiters]
        if not names:
            return []
        weights = [(1 - self.win_rate(name)) ** 2 for name in names]
        return rng.choices(names, weights, k=count)

    def record(self, name: str, outcomes: list[float]) -> None:
        member = self.members[name]
        member.wins = self.DECAY * member.wins + sum(outcomes)
        member.games = self.DECAY * member.games + len(outcomes)

    def state_dict(self) -> dict:
        """Everything but the weights, which are in the folder."""
        return {
            "size": self.size,
            "members": {n: vars(m) | {"weights": None} for n, m in self.members.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.size = state["size"]
        self.members = {n: Member(**m) for n, m in state["members"].items()}
        for name, member in self.members.items():
            if name != RULE:
                member.weights = torch.load(self._file(name), weights_only=True)


# --- Learning ----------------------------------------------------------------


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(x.abs())


VALUE_BINS = 255  # the critic's value head: odds over this many bins (one of them 0)
VALUE_RANGE = 12.0  # the outermost bins, in symlog units: about ±160,000 points


def value_bins(device: torch.device | str = "cpu") -> torch.Tensor:
    """The critic's bins, in points: evenly spaced on a symmetric-log scale."""
    return symexp(torch.linspace(-VALUE_RANGE, VALUE_RANGE, VALUE_BINS, device=device))


def two_hot(values: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Each value as weights on the two bins around it, whose expectation is the value.

    The weights interpolate in points, not in symlog units, so a critic that
    learns the average of these targets predicts the expected score.
    """
    values = values.clamp(bins[0], bins[-1])
    upper = torch.searchsorted(bins, values).clamp(1, len(bins) - 1)
    lower = upper - 1
    weight = (values - bins[lower]) / (bins[upper] - bins[lower])
    result = torch.zeros(len(values), len(bins), device=values.device)
    result.scatter_(1, lower[:, None], (1 - weight)[:, None])
    return result.scatter_add_(1, upper[:, None], weight[:, None])


def expected_value(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """The critic's prediction in points, from its logits over the bins."""
    return (torch.softmax(logits, dim=-1) * bins).sum(-1)


def as_critic(net: Net) -> Net:
    """A critic made from `net`: the same trunk, with a fresh value head over `VALUE_BINS`."""
    critic = Net(replace(net.config, value_bins=VALUE_BINS))
    trunk = {name: t for name, t in net.state_dict().items() if not name.startswith("value.")}
    critic.load_state_dict(trunk, strict=False)
    return critic


def advantages(values: list[float], reward: float, lam: float) -> list[float]:
    """GAE for one trajectory whose only reward arrives after its last step."""
    result, running = [0.0] * len(values), 0.0
    for t in reversed(range(len(values))):
        following = values[t + 1] if t + 1 < len(values) else 0.0
        delta = (reward if t + 1 == len(values) else 0.0) + following - values[t]
        running = delta + lam * running
        result[t] = running
    return result


@dataclass
class Batch:
    observations: list[Observation]
    oracles: list[np.ndarray]
    beliefs: torch.Tensor  # (N, 52) true places of unseen cards, NOT_HIDDEN elsewhere
    actions: torch.Tensor
    old_log_probs: torch.Tensor  # each move's log-chance under the policy that collected it
    played_log_probs: torch.Tensor  # and under the odds it was played from (see `ppo_objective`)
    advantages: torch.Tensor
    returns: torch.Tensor  # λ-returns in points: the critic's targets
    value_ev: float  # the share of the final scores' variance that the critic explains


def _chunks(index: list[int], lengths: np.ndarray, size: int) -> list[list[int]]:
    """Minibatch `index` in passes of `size` rows, by length (`lengths[i]`: row i's tokens).

    A pass is padded to its longest row, and the average row is a third of the
    longest, so rows of similar length go together. The minibatch itself stays
    as drawn (length-bucketed minibatches made the critic fit much worse), and
    the passes' gradients add up to the same whole.
    """
    ordered = sorted(index, key=lengths.__getitem__)
    return [ordered[start : start + size] for start in range(0, len(ordered), size)]


def _oracle_inputs(oracles: list[np.ndarray], device: torch.device) -> list[torch.Tensor]:
    return inputs([Observation(o, np.zeros(0, dtype=np.int64)) for o in oracles], device)


@torch.no_grad()
def _critic_values(critic: Net, oracles: list[np.ndarray], chunk: int = 256) -> torch.Tensor:
    """The critic's expected score for each decision, in points (on the CPU)."""
    critic.eval()
    device = device_of(critic)
    bins = value_bins(device)
    lengths = np.array([len(o) for o in oracles])
    values = torch.empty(len(oracles))
    for part in _chunks(list(range(len(oracles))), lengths, chunk):
        logits = critic(*_oracle_inputs([oracles[i] for i in part], device))[1]
        values[part] = expected_value(logits, bins).cpu()
    return values


def explained_variance(predicted: torch.Tensor, actual: torch.Tensor) -> float:
    return float(1 - (actual - predicted).var() / actual.var().clamp(min=1e-8))


def stake_weights(trajectories: list[Trajectory]) -> torch.Tensor:
    """Each step's weight under `Settings.stake_scaling`.

    Before the stake is fixed (the auction, the call, trumps) a decision can
    change what the deal is worth, so it keeps its weight of 1 and is judged in
    points. From the exchange on, the weight is the batch's mean stake over this
    deal's: every deal then teaches card play equally, instead of the few with
    the highest stakes drowning out the rest, while those decisions together keep
    roughly the share of the update they had (their advantages grow with the stake).
    """
    fixed = torch.tensor([stake_is_fixed(s.observation) for t in trajectories for s in t.steps])
    stakes = torch.tensor([t.stake for t in trajectories for _ in t.steps], dtype=torch.float32)
    weights = torch.ones(len(fixed))
    if fixed.any():
        weights[fixed] = stakes[fixed].mean() / stakes[fixed]
    return weights


def prepare(trajectories: list[Trajectory], critic: Net, settings: Settings) -> Batch:
    """Advantages and value targets for the collected steps, on the critic's device."""
    steps = [step for trajectory in trajectories for step in trajectory.steps]
    values = _critic_values(critic, [s.oracle for s in steps], settings.chunk)
    all_advantages, start = [], 0
    for trajectory in trajectories:
        end = start + len(trajectory.steps)
        found = advantages(values[start:end].tolist(), trajectory.reward, settings.gae_lambda)
        all_advantages += found
        start = end
    rewards = torch.tensor([t.reward for t in trajectories for _ in t.steps])
    device = device_of(critic)
    adv = torch.tensor(all_advantages, dtype=torch.float32)
    returns = adv + values  # in points: the critic's targets, whatever the policy's weights
    if settings.stake_scaling:
        adv = adv * stake_weights(trajectories)
    return Batch(
        [s.observation for s in steps],
        [s.oracle for s in steps],
        torch.as_tensor(np.stack([s.belief for s in steps]), dtype=torch.long).to(device),
        torch.tensor([s.action for s in steps]).to(device),
        torch.tensor([s.policy_log_prob for s in steps]).to(device),
        torch.tensor([s.log_prob for s in steps]).to(device),
        ((adv - adv.mean()) / (adv.std() + 1e-8)).to(device),
        returns.to(device),
        explained_variance(values, rewards),
    )


ChunkLoss = Callable[[list[int]], tuple[torch.Tensor, dict[str, torch.Tensor]]]


def ppo_objective(
    new: torch.Tensor, old: torch.Tensor, played: torch.Tensor, adv: torch.Tensor, clip: float
) -> torch.Tensor:
    """PPO's clipped objective per step, for moves played from other odds than the policy's.

    `new`, `old` and `played` are the log-chances of each move under the policy
    being updated, the policy that collected it, and the odds it was really
    played from (`explored` in the auction, the policy's own elsewhere). The
    ratio is clipped against the old policy, not the played odds, and the step
    is weighed by old / played (decoupled PPO, Hilton et al. 2021). Clipping
    against the played odds would start an explored bid's ratio far below
    1 - clip and the policy's favourite's above 1 + clip, so that only good
    results of explored bids and bad ones of favourites would count: an update
    that drifts towards whatever is explored, whatever it scores.
    """
    ratio = torch.exp(new - old)
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip)
    return torch.exp(old - played) * torch.min(ratio * adv, clipped * adv)


def _policy_loss(
    policy: Net, magnet: Net, batch: Batch, settings: Settings, index: list[int]
) -> ChunkLoss:
    """The PPO loss of minibatch `index`, one chunk at a time (see `_step`)."""
    size = len(index)
    hidden = (batch.beliefs[index] != NOT_HIDDEN).sum().clamp(min=1)

    def chunk_loss(chunk: list[int]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tokens, padding, legal = inputs([batch.observations[i] for i in chunk], device_of(policy))
        summary = policy.summarise(tokens, padding)
        logits, _ = policy.heads(summary, legal)
        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            magnet_log_probs = torch.log_softmax(magnet(tokens, padding, legal)[0], dim=-1)
        # Illegal actions have log-probability -inf; zero them before multiplying,
        # since 0 * -inf is NaN and would poison the gradients.
        probs = log_probs.exp()
        safe = log_probs.masked_fill(~legal, 0.0)
        magnet_safe = magnet_log_probs.masked_fill(~legal, 0.0)

        new = log_probs.gather(1, batch.actions[chunk, None]).squeeze(-1)
        old = batch.old_log_probs[chunk]
        log_ratio = new - old
        ratio = torch.exp(log_ratio)
        objective = ppo_objective(
            new, old, batch.played_log_probs[chunk], batch.advantages[chunk], settings.clip
        )
        beliefs = policy.beliefs(summary).flatten(0, 1)
        terms = {
            "policy_loss": -objective.sum() / size,
            "entropy": -(probs * safe).sum() / size,
            "magnet_kl": (probs * (safe - magnet_safe)).sum() / size,
            "belief_loss": F.cross_entropy(
                beliefs, batch.beliefs[chunk].flatten(), ignore_index=NOT_HIDDEN, reduction="sum"
            )
            / hidden,
        }
        loss = (
            terms["policy_loss"]
            - settings.entropy * terms["entropy"]
            + settings.magnet * terms["magnet_kl"]
            + settings.belief * terms["belief_loss"]
        )
        terms["clip_fraction"] = ((ratio - 1).abs() > settings.clip).float().sum() / size
        terms["approx_kl"] = ((ratio - 1) - log_ratio).sum() / size  # KL(old || new)
        return loss, terms

    return chunk_loss


def _critic_loss(critic: Net, batch: Batch, bins: torch.Tensor, index: list[int]) -> ChunkLoss:
    """The critic's loss on minibatch `index`: cross-entropy with the two-hot returns."""

    def chunk_loss(chunk: list[int]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        oracle = _oracle_inputs([batch.oracles[i] for i in chunk], device_of(critic))
        targets = two_hot(batch.returns[chunk], bins)
        loss = F.cross_entropy(critic(*oracle)[1], targets, reduction="sum") / len(index)
        return loss, {"value_loss": loss}

    return chunk_loss


def _step(
    net: Net,
    optimiser: torch.optim.Optimizer,
    chunk_loss: ChunkLoss,
    chunks: list[list[int]],
    norm: str,
) -> dict[str, torch.Tensor] | None:
    """One optimiser step on a minibatch, with its forward and backward in `chunks`.

    Each chunk's loss is its share of the minibatch's, so the gradients add up
    to the whole minibatch's. Returns the diagnostics and the gradient norm
    (before clipping), as tensors on the device, or None if the gradient was
    not finite: then the step is skipped. Only that check waits for the GPU.
    """
    optimiser.zero_grad()
    totals: dict[str, torch.Tensor] = {}
    for chunk in chunks:
        loss, terms = chunk_loss(chunk)
        loss.backward()
        for name, value in terms.items():
            totals[name] = totals.get(name, 0.0) + value.detach()
    grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    if not torch.isfinite(grad_norm):
        return None
    optimiser.step()
    return totals | {norm: grad_norm}


def _attention(device: torch.device):
    """Plain attention on CUDA: for sequences this short (about 40 tokens) it beats the fused
    kernels PyTorch picks there, by about 10% of the update on the 5090, with the same results."""
    return sdpa_kernel(SDPBackend.MATH) if device.type == "cuda" else nullcontext()


def update(
    policy: Net,
    critic: Net,
    magnet: Net,
    optimisers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    batch: Batch,
    settings: Settings,
    rng: random.Random,
    train_policy: bool = True,
) -> dict[str, float]:
    """PPO epochs over the batch (only the critic's, without `train_policy`).

    Returns mean losses and diagnostics, and `skipped_steps`: optimiser steps
    left out because their gradient was not finite.
    """
    policy_optimiser, critic_optimiser = optimisers
    policy.train()
    critic.train()
    magnet.eval()
    bins = value_bins(device_of(critic))
    lengths = np.array([len(o.tokens) for o in batch.observations])
    oracle_lengths = np.array([len(o) for o in batch.oracles])
    stats: dict[str, list[torch.Tensor]] = defaultdict(list)
    steps = skipped = 0
    order = list(range(len(batch.actions)))
    with _attention(device_of(policy)):
        for _ in range(settings.ppo_epochs):
            rng.shuffle(order)
            for start in range(0, len(order), settings.batch_size):
                index = order[start : start + settings.batch_size]
                parts = [
                    (
                        critic,
                        critic_optimiser,
                        _critic_loss(critic, batch, bins, index),
                        _chunks(index, oracle_lengths, settings.chunk),
                        "critic",
                    )
                ]
                if train_policy:
                    loss = _policy_loss(policy, magnet, batch, settings, index)
                    chunks = _chunks(index, lengths, settings.chunk)
                    parts.insert(0, (policy, policy_optimiser, loss, chunks, "policy"))
                for net, optimiser, chunk_loss, chunks, name in parts:
                    steps += 1
                    found = _step(net, optimiser, chunk_loss, chunks, f"{name}_grad_norm")
                    if found is None:
                        skipped += 1
                        continue
                    for key, value in found.items():
                        stats[key].append(value)
    if skipped == steps:
        raise FloatingPointError("every update step had a non-finite gradient")
    means = {key: torch.stack(values).mean() for key, values in stats.items()}
    return dict(zip(means, torch.stack(list(means.values())).tolist(), strict=True)) | {
        "skipped_steps": skipped
    }


# --- The training loop -------------------------------------------------------

STATE = "state.pt"  # everything `--resume` needs, rewritten after every iteration


def _cpu_state(net: Net) -> dict:
    return {name: t.detach().to("cpu", copy=True) for name, t in net.state_dict().items()}


def _check_finite(*nets: Net) -> None:
    """Stop before non-finite weights reach the magnet, the snapshot pool or a checkpoint."""
    for net in nets:
        if not torch.stack([p.isfinite().all() for p in net.parameters()]).all():
            raise FloatingPointError("the weights are no longer finite; nothing was saved")


def _replace(path: Path, write: Callable[[str], object]) -> None:
    """Write `path` through a temporary file, so that a crash never leaves half a file."""
    temporary = path.with_name(f".tmp-{path.name}")  # keeps the suffix, so np.savez adds none
    write(str(temporary))
    os.replace(temporary, path)


def _append(path: Path, entry: dict) -> None:
    with path.open("a") as file:
        file.write(json.dumps(entry) + "\n")


def _keep_until(path: Path, iteration: int) -> None:
    """Drop lines logged after `iteration` (the state a resume carries on from), or cut short."""
    if path.exists():
        kept = []
        for line in path.read_text().splitlines():
            try:
                if json.loads(line)["iteration"] <= iteration:
                    kept.append(line)
            except json.JSONDecodeError:
                pass  # a line cut short by a crash
        path.write_text("".join(line + "\n" for line in kept))


def _save_networks(out: Path, iteration: int, policy: Net, critic: Net, keep: bool) -> None:
    """The latest networks, and with `keep` a numbered copy of each in checkpoints/."""
    files = {
        "policy.pt": partial(save, policy),
        "critic.pt": partial(save, critic),
        "policy.npz": partial(export, policy),
    }
    for name, write in files.items():
        _replace(out / name, write)
        if keep:
            (out / "checkpoints").mkdir(exist_ok=True)
            stem, suffix = name.split(".")
            _replace(out / "checkpoints" / f"{stem}-{iteration:04d}.{suffix}", write)


@dataclass
class Learning:
    """A network being trained: the policy, its critic and magnet, and their optimisers."""

    policy: Net
    critic: Net
    magnet: Net
    optimisers: tuple[torch.optim.Optimizer, torch.optim.Optimizer]

    @classmethod
    def start(cls, policy: Net, critic: Net, settings: Settings, device: str) -> Learning:
        policy, critic = policy.to(device), critic.to(device)
        optimisers = (
            torch.optim.AdamW(policy.parameters(), lr=settings.policy_lr),
            torch.optim.AdamW(critic.parameters(), lr=settings.critic_lr),
        )
        return cls(policy, critic, copy.deepcopy(policy), optimisers)

    def move_magnet(self, settings: Settings, iteration: int) -> None:
        """Towards the policy by `magnet_ema` of the way each iteration, or (without it) onto
        the policy every `snapshot_every` iterations."""
        if settings.magnet_ema:
            with torch.no_grad():
                magnet, policy = list(self.magnet.parameters()), list(self.policy.parameters())
                torch._foreach_lerp_(magnet, policy, settings.magnet_ema)
        elif iteration % settings.snapshot_every == 0:
            self.magnet.load_state_dict(self.policy.state_dict())


def _learn(
    runner: Runner,
    learning: Learning,
    lineups: list[list[str]],
    settings: Settings,
    rng: random.Random,
    learner: str = LEARNER,
    train_policy: bool = True,
    scores: dict[int, list[int]] | None = None,
) -> dict:
    """One PPO iteration: play `lineups`, recording `learner`, and update its networks.

    Returns the log entry's statistics; each deal's scores go into `scores`.
    """
    started = time.perf_counter()
    runner.broadcast(learner, "load", _cpu_state(learning.policy))
    found = collect(runner, lineups, rng, learner, scores)
    collected = time.perf_counter()
    batch = prepare(found, learning.critic, settings)
    entry: dict = {"decisions": len(batch.actions)}
    if not train_policy:
        entry["warmup"] = True  # only the critic learns
    policy, critic, magnet, optimisers = (
        learning.policy,
        learning.critic,
        learning.magnet,
        learning.optimisers,
    )
    entry |= update(policy, critic, magnet, optimisers, batch, settings, rng, train_policy)
    entry["value_ev"] = batch.value_ev
    entry["mean_reward"] = float(np.mean([t.reward for t in found]))
    entry |= contract_stats(found)
    entry["collect_s"] = round(collected - started, 1)
    entry["update_s"] = round(time.perf_counter() - collected, 1)
    _check_finite(policy, critic)
    return entry


LEAGUE_DRAWS, EXPLOITER_DRAWS = 6, 2  # members drawn each iteration (together at most SLOTS)


def _load(runner: Runner, league: League, members: list[str], loaded: dict[int, str]) -> dict:
    """The agent name of each member: RuleBot's own, or a worker slot loaded with its weights.

    `loaded` (slot -> member) is kept between iterations, so a member drawn
    again is not sent to the workers again.
    """
    wanted = [m for m in dict.fromkeys(members) if m != RULE]
    held = {member: slot for slot, member in loaded.items() if member in wanted}
    free = [slot for slot in range(SLOTS) if loaded.get(slot) not in held]
    names = {RULE: RULE}
    for member in wanted:
        if member not in held:
            held[member] = slot = free.pop(0)
            runner.broadcast(slot_name(slot), "load", league.members[member].weights)
            loaded[slot] = member
        names[member] = slot_name(held[member])
    return names


def _league_lineups(
    runner: Runner,
    league: League,
    loaded: dict[int, str],
    settings: Settings,
    rng: random.Random,
) -> tuple[list[list[str]], dict[int, str]]:
    """This iteration's lineups, and the league member each league deal is against.

    `league_share` of the deals go to members drawn from the whole league,
    `exploiter_share` to exploiters (self-play while there are none), and the
    rest are self-play.
    """
    drawn = league.draw(LEAGUE_DRAWS, rng)
    exploiters = league.draw(EXPLOITER_DRAWS, rng, exploiters=True)
    names = _load(runner, league, drawn + exploiters, loaded)
    members: list[str | None] = []
    for _ in range(settings.deals_per_iteration):
        u = rng.random()
        if u < settings.exploiter_share:
            members.append(rng.choice(exploiters) if exploiters else None)
        elif u < settings.exploiter_share + settings.league_share:
            members.append(rng.choice(drawn))
        else:
            members.append(None)
    lineups = lineups_against([None if m is None else names[m] for m in members], rng)
    return lineups, {i: m for i, m in enumerate(members) if m is not None}


def _record(
    league: League,
    lineups: list[list[str]],
    against: dict[int, str],
    scores: dict[int, list[int]],
) -> dict:
    """Update each member's win rate from this iteration's deals; the log's summary."""
    results: dict[str, list[float]] = defaultdict(list)
    for game, member in against.items():
        results[member].append(outcome(lineups[game], scores[game]))
    for member, found in results.items():
        league.record(member, found)
    return {
        "members": len(league.members),
        "against": {m: [len(r), round(league.win_rate(m), 3)] for m, r in results.items()},
    }


EXPLOITER_SEED = 777  # the deals an exploiter's margin is measured on


def _exploiter(
    runner: Runner,
    learning: Learning,
    league: League,
    settings: Settings,
    iteration: int,
    rng: random.Random,
    device: str,
) -> dict:
    """Train a copy of the learner against it, frozen, and add the copy to the league.

    REVIEW.md T2.2: every `exploit_every` iterations, for `exploit_iterations`.
    The copy plays one seat against the learner in the other three; its
    margin, measured on fixed deals at the end, is the learner's
    exploitability proxy (`learn.exploit` measures the same on its own).
    """
    started = time.perf_counter()
    runner.broadcast(TARGET, "load", _cpu_state(learning.policy))
    copied = Learning.start(
        copy.deepcopy(learning.policy), copy.deepcopy(learning.critic), settings, device
    )
    for step in range(1, settings.exploit_iterations + 1):
        lineups = exploit_lineups(settings.deals_per_iteration, rng, EXPLOITER)
        _learn(runner, copied, lineups, settings, rng, EXPLOITER)
        copied.move_magnet(settings, step)
    runner.broadcast(EVAL, "load", _cpu_state(copied.policy))
    positions = random_positions(2000, random.Random(EXPLOITER_SEED))
    result = evaluate(runner, EVAL, TARGET, positions)
    league.add(f"exploiter-{iteration:04d}", _cpu_state(copied.policy), iteration, exploiter=True)
    return {
        "exploiter_margin": result.mean,
        "exploiter_margin_ci95": result.ci95,
        "exploiter_s": round(time.perf_counter() - started, 1),
    }


def train(
    policy: Net,
    critic: Net,
    iterations: int,
    settings: Settings,
    rng: random.Random,
    out: Path | None = None,
    eval_every: int = 0,
    eval_deals: int = 100,
    device: str = "cpu",
    target: Net | None = None,
    workers: int = 1,
    resume: bool = False,
    worker_device: str = "cpu",
    league_start: dict[str, dict] | None = None,
) -> list[dict]:
    """Run self-play training up to iteration `iterations`; returns one log entry per iteration.

    The critic comes from `as_critic`. The update runs on `device`. Games are
    played by `workers` processes (1: in this one), whose networks run on
    `worker_device`. The league starts with RuleBot, the policy as it begins
    and `league_start` (name -> weights). With a `target`, train an exploiter
    against it instead (see module docstring). With `resume`, carry on from
    `out`'s saved state.
    """
    assert critic.config.value_bins == VALUE_BINS, "make the critic with as_critic()"
    learning = Learning.start(policy, critic, settings, device)
    policy, critic = learning.policy, learning.critic
    league = League(settings.league_size, None if out is None else out / "league")
    first = 1
    if resume:
        state = torch.load(out / STATE, map_location="cpu", weights_only=True)
        for net, name in [(policy, "policy"), (critic, "critic"), (learning.magnet, "magnet")]:
            net.load_state_dict(state[name])
        for optimiser, saved in zip(learning.optimisers, state["optimisers"], strict=True):
            optimiser.load_state_dict(saved)
        if "league" in state:
            league.load_state_dict(state["league"])
        else:  # a run from before the league: its snapshots join, and RuleBot
            for name, weights in state["pool"].items():
                league.add(name, weights, 0)
        first = state["iteration"] + 1
        rng.setstate(state["rng"])
        torch.set_rng_state(state["torch_rng"])
        for name in ("log.jsonl", "evals.jsonl"):
            _keep_until(out / name, state["iteration"])
    elif out is not None and (out / "log.jsonl").exists():
        raise FileExistsError(f"{out} already holds a run: resume it, or choose another")
    elif target is None:
        league.add("start", _cpu_state(policy), 0)
        for name, weights in (league_start or {}).items():
            league.add(name, weights, 0)
    eval_positions = random_positions(eval_deals, random.Random(12345)) if eval_every else []
    make = partial(
        make_agents,
        config=asdict(policy.config),
        slots=SLOTS,
        one_thread=workers > 1,
        explore=settings.explore_bids,
        explore_levels=settings.explore_levels,
        device=worker_device,
    )
    loaded: dict[int, str] = {}  # worker slot -> the league member in it
    history = []
    with Runner(make, workers=workers, games_in_flight=settings.games_in_flight) as runner:
        if target is not None:
            runner.broadcast(TARGET, "load", _cpu_state(target))
        for iteration in range(first, iterations + 1):
            scores: dict[int, list[int]] = {}
            if target is None:
                lineups, against = _league_lineups(runner, league, loaded, settings, rng)
            else:
                lineups, against = exploit_lineups(settings.deals_per_iteration, rng), {}
            warmup = iteration <= settings.critic_warmup
            entry: dict = {"iteration": iteration}
            entry |= _learn(runner, learning, lineups, settings, rng, LEARNER, not warmup, scores)
            if against:
                entry["league"] = _record(league, lineups, against, scores)
            learning.move_magnet(settings, iteration)
            if target is None and iteration % settings.snapshot_every == 0:
                league.add(f"learner-{iteration:04d}", _cpu_state(policy), iteration)
            every = settings.exploit_every
            if target is None and every and iteration % every == 0:
                entry |= _exploiter(runner, learning, league, settings, iteration, rng, device)
            result = None
            if eval_every and iteration % eval_every == 0:
                evaluating = time.perf_counter()
                runner.broadcast(EVAL, "load", _cpu_state(policy))
                name, field = ("rulebot", RULE) if target is None else ("target", TARGET)
                result = evaluate(runner, EVAL, field, eval_positions)
                entry[f"vs_{name}"], entry[f"vs_{name}_ci95"] = result.mean, result.ci95
                entry[f"vs_{name}_roles"] = {
                    r: round(float(np.mean(v)), 1) for r, v in result.by_role.items() if v
                }
                entry["eval_s"] = round(time.perf_counter() - evaluating, 1)
            entry["time"] = round(time.time())
            if out is not None:
                out.mkdir(parents=True, exist_ok=True)
                _append(out / "log.jsonl", entry)
                if result is not None:
                    _append(out / "evals.jsonl", {"iteration": iteration, **asdict(result)})
                keep = (
                    iteration % settings.snapshot_every == 0
                    or result is not None
                    or iteration == iterations
                )
                _save_networks(out, iteration, policy, critic, keep)
                state = {
                    "iteration": iteration,
                    "policy": policy.state_dict(),
                    "critic": critic.state_dict(),
                    "magnet": learning.magnet.state_dict(),
                    "optimisers": [optimiser.state_dict() for optimiser in learning.optimisers],
                    "league": league.state_dict(),
                    "rng": rng.getstate(),
                    "torch_rng": torch.get_rng_state(),
                }
                _replace(out / STATE, partial(torch.save, state))
            print(
                " ".join(
                    f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in entry.items()
                ),
                flush=True,
            )
            history.append(entry)
    return history


def _commit() -> str:
    try:
        found = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return found.stdout.strip()


RESUMABLE = {"resume", "out", "workers", "device", "worker_device"}  # may differ when resuming
CUDA = torch.cuda.is_available()


def _interrupt(signum, frame) -> None:
    raise KeyboardInterrupt  # so that `kill` stops the workers too, via the Runner's exit


def main() -> None:
    signal.signal(signal.SIGTERM, _interrupt)
    parser = argparse.ArgumentParser(description="Self-play training with PPO.")
    parser.add_argument(
        "--init",
        type=Path,
        help="start from a checkpoint (the critic from its trunk), or a run's latest networks",
    )
    parser.add_argument("--init-critic", type=Path, help="with --init, start the critic from this")
    parser.add_argument(
        "--resume", type=Path, help="carry on the run in this directory, with its own settings"
    )
    parser.add_argument(
        "--device",
        default="cuda" if CUDA else "mps" if torch.backends.mps.is_available() else "cpu",
        help="where the update runs",
    )
    parser.add_argument(
        "--worker-device",
        default="cuda" if CUDA else "cpu",
        help="where each worker's networks run while playing",
    )
    parser.add_argument(
        "--workers", type=int, help="processes to play on (default: arena.default_workers)"
    )
    parser.add_argument("--exploit", type=Path, help="train an exploiter against this policy")
    parser.add_argument(
        "--iterations", type=int, help="in all (default: 100, or the run's own when resuming)"
    )
    parser.add_argument("--deals", type=int, default=Settings.deals_per_iteration)
    parser.add_argument("--ppo-epochs", type=int, default=Settings.ppo_epochs)
    parser.add_argument(
        "--magnet", type=float, default=Settings.magnet, help="weight of KL(policy || magnet)"
    )
    parser.add_argument(
        "--critic-warmup",
        type=int,
        default=Settings.critic_warmup,
        help="first iterations that train only the critic",
    )
    parser.add_argument(
        "--explore-bids",
        type=float,
        default=Settings.explore_bids,
        help="share of each bid's chance moved to other kinds of contract while playing",
    )
    parser.add_argument(
        "--explore-levels",
        type=float,
        default=Settings.explore_levels,
        help="share of each bid's chance moved to the same kind one and two levels up",
    )
    parser.add_argument(
        "--stake-scaling",
        action="store_true",
        help="weigh decisions from the exchange on by the batch's mean stake over the deal's",
    )
    parser.add_argument("--entropy", type=float, default=Settings.entropy, help="bonus weight")
    parser.add_argument("--policy-lr", type=float, default=Settings.policy_lr)
    parser.add_argument("--critic-lr", type=float, default=Settings.critic_lr)
    parser.add_argument(
        "--magnet-ema",
        type=float,
        default=Settings.magnet_ema,
        help="the magnet's step towards the policy per iteration (0: a copy refreshed every 10)",
    )
    parser.add_argument(
        "--league-share",
        type=float,
        default=Settings.league_share,
        help="deals against league members, drawn by priority",
    )
    parser.add_argument(
        "--exploiter-share",
        type=float,
        default=Settings.exploiter_share,
        help="deals against exploiters, once there are any",
    )
    parser.add_argument("--league-size", type=int, default=Settings.league_size)
    parser.add_argument(
        "--exploit-every",
        type=int,
        default=Settings.exploit_every,
        help="iterations between training exploiters against the learner (0: none)",
    )
    parser.add_argument("--exploit-iterations", type=int, default=Settings.exploit_iterations)
    parser.add_argument(
        "--league-add",
        nargs="*",
        default=[],
        metavar="POLICY",
        help="past policies (.pt or .npz) that join the league at the start",
    )
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-deals", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/rl"))
    args = parser.parse_args()
    args.workers = args.workers or default_workers(args.worker_device)

    if args.resume:
        if not (args.resume / STATE).exists():
            parser.error(f"{args.resume} has no {STATE} to resume from")
        run = json.loads((args.resume / "run.json").read_text())
        iterations = args.iterations
        for key, value in run["args"].items():
            if key not in RESUMABLE:
                setattr(args, key, value)
        args.out, args.iterations = args.resume, iterations or args.iterations
        run["args"]["iterations"] = args.iterations
    elif (args.out / "log.jsonl").exists():
        parser.error(f"{args.out} already holds a run: --resume it, or choose another --out")
    else:
        args.iterations = args.iterations or 100
    settings = Settings(
        deals_per_iteration=args.deals,
        ppo_epochs=args.ppo_epochs,
        critic_warmup=args.critic_warmup,
        explore_bids=args.explore_bids,
        explore_levels=args.explore_levels,
        stake_scaling=args.stake_scaling,
        magnet=args.magnet,
        magnet_ema=args.magnet_ema,
        entropy=args.entropy,
        policy_lr=args.policy_lr,
        critic_lr=args.critic_lr,
        league_share=args.league_share,
        exploiter_share=args.exploiter_share,
        league_size=args.league_size,
        exploit_every=args.exploit_every,
        exploit_iterations=args.exploit_iterations,
    )
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    launch = {"commit": _commit(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if args.resume:  # the networks' shapes; train() restores their weights and the rest
        policy, critic = load(str(args.out / "policy.pt")), load(str(args.out / "critic.pt"))
        run["resumed"] = [*run.get("resumed", []), launch | {"iterations": args.iterations}]
    else:
        if args.init and args.init.is_dir():  # another run's latest policy and critic
            policy, critic = load(str(args.init / "policy.pt")), load(str(args.init / "critic.pt"))
        else:
            policy = load(str(args.init)) if args.init else Net(NetConfig())
            critic = load(str(args.init_critic)) if args.init_critic else as_critic(policy)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "settings.json").write_text(json.dumps(asdict(settings), indent=2))
        saved = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        run = {"args": saved, **launch}
    (args.out / "run.json").write_text(json.dumps(run, indent=2))
    train(
        policy,
        critic,
        args.iterations,
        settings,
        rng,
        args.out,
        args.eval_every,
        args.eval_deals,
        args.device,
        load(str(args.exploit)) if args.exploit else None,
        args.workers,
        resume=bool(args.resume),
        worker_device=args.worker_device,
        league_start={path: _cpu_state(load(path)) for path in args.league_add},
    )


if __name__ == "__main__":
    main()
