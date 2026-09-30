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
    python -m learn.selfplay --resume runs/rl-cool --iterations 300 --cooldown 100  # a copy of rl

A resume takes the run's settings from its `run.json`; a setting given at a
resume that the run lacks (one newer than the run, such as `--cooldown`) is
written there too, so later resumes keep it. Only `RESUMABLE` settings (the
workers, the device, `--graphed-update`) may differ from one resume to the next.
With `--cooldown N` both learning rates fall linearly to a tenth over the last
N iterations up to `--iterations` (`cooldown_share`).

A run's directory holds `log.jsonl` (one line per iteration), `evals.jsonl`
(each evaluation's per-deal results, for paired comparisons), `run.json` (the
command line and commit, and each resume's, with any settings it changed) and
`settings.json` (as the run started); the latest `policy.pt`,
`critic.pt`, `policy.npz` and the magnet's `magnet.pt` and `magnet.npz`, with
numbered copies in `checkpoints/` at every snapshot and evaluation; and
`state.pt`, from which `--resume` carries on
exactly where the run stopped: the same iteration count, optimisers, magnet,
snapshot pool and random state.

`--exploit` trains an *exploiter*: a fresh learner in one seat against a frozen
policy in the other three. How much it gains over that policy (`vs_target`
in the log) measures how exploitable the policy is (LEARNING.md §4).

Games are played by `--workers` processes through `learn.runner.Runner`; the
networks they play with run in this process for all of them, on `--device`
(an NVIDIA GPU, else Apple's, else the CPU), where the update runs too.
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
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
from functools import partial
from operator import attrgetter
from pathlib import Path
from typing import NamedTuple

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
    NUM_ACTIONS,
    PHASES,
    Observation,
    belief_targets,
    hidden_tokens,
    observe,
)
from .evaluate import evaluate
from .model import Net, NetConfig, Networks, Question, device_of, export, load, save
from .runner import Decision, Runner


@dataclass(frozen=True)
class Settings:
    deals_per_iteration: int = 256
    league_share: float = 0.5  # deals against a league member drawn by priority (`League`)
    exploiter_share: float = 0.15  # deals against an exploiter, once there are any
    league_size: int = 50  # members at most (RuleBot, snapshots, exploiters)
    league_exploiters: int = 20  # exploiters kept in the league at most; the oldest go first
    snapshot_every: int = 10  # iterations between snapshots joining the league
    exploit_every: int = 0  # iterations between training exploiters (0: none)
    exploit_iterations: int = 50  # each exploiter's training
    games_in_flight: int = 256  # deals each worker keeps going at once
    gae_lambda: float = 0.95
    ppo_epochs: int = 1  # deals are cheap to play, so fresh ones beat a second pass
    batch_size: int = 512
    chunk: int = 512  # rows per forward and backward pass at most (see `_passes`)
    clip: float = 0.2
    policy_lr: float = 1e-4
    critic_lr: float = 3e-4
    cooldown: int = 0  # the run's last iterations, over which both rates fall to a tenth
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
    """The policy being trained, as it plays in a worker; its network runs in the runner's
    process (`networks`), for every worker at once.

    It samples its actions from the network's answer and records each real
    decision (not forced moves) with its training targets (critic tokens and
    where the unseen cards are, both read from the deal) under (game, seat),
    for `trajectories` to collect when the game ends. Only the view goes into
    the policy. With `explore` or `explore_levels`, it bids from `explored`
    odds, and records each action's chance under them as well as the policy's
    own, for `ppo_objective`.
    """

    def __init__(
        self, network: str, worker: int, explore: float = 0.0, explore_levels: float = 0.0
    ) -> None:
        self.network = network
        self.worker = worker
        self.explore = explore
        self.explore_levels = explore_levels
        self.generator = torch.Generator()
        self.steps: dict[tuple[int, int], list[Step]] = defaultdict(list)

    def seed(self, seed: int) -> None:
        self.generator.manual_seed(seed * 1009 + self.worker)  # a different stream per worker

    def ask(self, decisions: list[Decision]) -> Question:
        return Question.of([observe(decision.view) for decision in decisions])

    def act(self, decisions: list[Decision], question: Question, answer) -> list[Action]:
        own = torch.log_softmax(question.logits(answer), dim=-1)  # sampled here, on the CPU
        log_probs = own
        auction = torch.tensor([d.view.phase is Phase.AUCTION for d in decisions])
        if (self.explore or self.explore_levels) and auction.any():
            legal = own > -torch.inf
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
            question.observations(),
            actions.tolist(),
            chosen.tolist(),
            chosen_own.tolist(),
            strict=True,
        ):
            if len(observation.legal) == 1:
                continue  # a forced move: it gives the policy no gradient
            hidden = np.array(hidden_tokens(decision.deal, decision.seat), dtype=np.int16)
            oracle = np.concatenate([observation.tokens, hidden.reshape(-1, 5)])
            belief = np.asarray(belief_targets(decision.deal, decision.seat), dtype=np.int8)
            step = Step(observation, oracle, belief, action, log_prob, own_log_prob)
            self.steps[decision.game, decision.seat].append(step)
        return [ACTIONS[action] for action in actions.tolist()]


class Frozen:
    """A network that only plays, greedily: past snapshots, exploit targets, evaluation. Its
    network runs in the runner's process (`networks`)."""

    def __init__(self, network: str) -> None:
        self.network = network

    def ask(self, decisions: list[Decision]) -> Question:
        return Question.of([observe(decision.view) for decision in decisions])

    def act(self, decisions: list[Decision], question: Question, answer) -> list[Action]:
        return [ACTIONS[i] for i in question.logits(answer).argmax(dim=-1).tolist()]


def make_agents(worker: int, slots: int, explore: float = 0.0, explore_levels: float = 0.0) -> dict:
    """The agents in each runner worker, by name. Their networks are `networks`."""
    agents = {
        LEARNER: Learner(LEARNER, worker, explore, explore_levels),
        EXPLOITER: Learner(EXPLOITER, worker, explore, explore_levels),
        RULE: RuleBot(),
        EVAL: Frozen(EVAL),
        TARGET: Frozen(TARGET),
    }
    return agents | {slot_name(slot): Frozen(slot_name(slot)) for slot in range(slots)}


def networks(config: NetConfig, slots: int, device: str) -> Networks:
    """The networks of `make_agents`' agents, on `device`, for the runner's process. Their
    weights are loaded (`Networks.load`) before each play."""
    names = [LEARNER, EXPLOITER, EVAL, TARGET, *map(slot_name, range(slots))]
    return Networks({name: Net(config) for name in names}, device)


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
    struggles against come up more. Beyond `exploiters` exploiters, the oldest
    go. Beyond `size` members, past snapshots are thinned where they are
    closest together for their age, so older ones grow sparse and the pool
    spans the whole run.
    """

    DECAY = 0.9  # each iteration's weight in a member's win rate, against the ones before

    def __init__(self, size: int = 50, folder: Path | None = None, exploiters: int = 20) -> None:
        self.size = size
        self.folder = folder  # where members' weights are kept, one file each, if anywhere
        self.exploiters = exploiters  # at most; from the settings, not the saved state
        self.members: dict[str, Member] = {RULE: Member(None, 0)}

    def add(self, name: str, weights: dict, iteration: int, exploiter: bool = False) -> None:
        self.members[name] = Member(weights, iteration, exploiter)
        if self.folder is not None:
            self.folder.mkdir(parents=True, exist_ok=True)
            _replace(self._file(name), partial(torch.save, weights))
        exploiters = sorted((m.iteration, n) for n, m in self.members.items() if m.exploiter)
        for _, oldest in exploiters[: max(0, len(exploiters) - self.exploiters)]:
            self._drop(oldest)
        while len(self.members) > self.size and self._thin():
            pass

    def _file(self, name: str) -> Path:
        return self.folder / f"{name.replace('/', '_')}.pt"

    def _drop(self, name: str) -> None:
        del self.members[name]
        if self.folder is not None:
            self._file(name).unlink(missing_ok=True)

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
        self._drop(snapshots[crowded][1])
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


def advantages(
    values: np.ndarray, rewards: np.ndarray, lengths: np.ndarray, lam: float
) -> np.ndarray:
    """GAE for trajectories laid end to end, `lengths[i]` steps each, whose only reward
    (`rewards[i]`) arrives after their last step.

    The trajectories are set right-aligned in a table, so that the recursion runs over its
    columns for all of them at once: in float64, operation for operation as one trajectory
    at a time in Python's floats would do it, so the numbers are the same.
    """
    width = int(lengths.max())
    placed = np.arange(width) >= width - lengths[:, None]  # each row's steps, at its right end
    table = np.zeros(placed.shape)
    table[placed] = values
    following, reward = np.zeros(placed.shape), np.zeros(placed.shape)
    following[:, :-1] = table[:, 1:]
    reward[:, -1] = rewards
    delta = reward + following - table
    found, running = np.empty(placed.shape), np.zeros(len(lengths))
    for t in reversed(range(width)):
        running = delta[:, t] + lam * running
        found[:, t] = running
    return found[placed]


LENGTH_STEP = 16  # passes are padded to a multiple of this many tokens, so they come in few shapes


def _rounded(length: int) -> int:
    return -(-int(length) // LENGTH_STEP) * LENGTH_STEP


@dataclass
class Padded:
    """Token sequences padded into one tensor on the update's device; passes take rows of it."""

    tokens: torch.Tensor  # (N, T, 5), T the longest row's length rounded up to LENGTH_STEP
    lengths: torch.Tensor  # (N,) each row's tokens

    @classmethod
    def of(cls, sequences: list, device: torch.device) -> Padded:
        """The sequences sent end to end in one copy, then spread into their rows on `device`
        (`_spread`): padded by NumPy, a batch's took about 30 ms."""
        lengths = np.fromiter(map(len, sequences), np.int64, len(sequences))
        flat = _pinned((int(lengths.sum()), 5), torch.int16, device)
        np.concatenate(sequences, out=flat.numpy())
        counts = _to(lengths, device)
        tokens = torch.zeros(
            len(lengths), _rounded(lengths.max()), 5, dtype=torch.int32, device=device
        )
        tokens[_spread(counts, len(flat))] = flat.to(device, non_blocking=True).int()
        return cls(tokens, counts)

    def take(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The tokens of `rows`, and their padding mask."""
        padding = torch.arange(self.tokens.shape[1], device=rows.device) >= self.lengths[rows, None]
        return self.tokens[rows], padding


def _to(array: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    """A NumPy array, or a tensor on the CPU, on `device`, copied without waiting for the GPU
    to finish its work."""
    tensor = torch.as_tensor(array)
    if tensor.device.type == device.type:
        return tensor
    pinned = _pinned(tensor.shape, tensor.dtype, device)
    pinned.numpy()[...] = tensor.numpy()
    return pinned.to(device, non_blocking=True)


def _pinned(shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """An empty tensor to fill on the CPU (through `.numpy()`) and send to `device`: in pinned
    memory for an NVIDIA GPU, so that it goes without waiting for the GPU.

    Filled by NumPy, not copied by PyTorch (`pin_memory`), whose copies of more than 32,768
    numbers wake its threads: about a millisecond each after a pause, and more while the
    workers keep the cores busy.
    """
    return torch.empty(shape, dtype=dtype, pin_memory=device.type == "cuda")


def _spread(counts: torch.Tensor, total: int) -> tuple[torch.Tensor, torch.Tensor]:
    """For `total` values laid end to end, `counts[i]` of them row i's: the row of each, and
    its place in that row; worked out on the counts' device, without waiting for it."""
    device = counts.device
    rows = torch.arange(len(counts), device=device).repeat_interleave(counts, output_size=total)
    return rows, torch.arange(total, device=device) - (counts.cumsum(0) - counts)[rows]


def _legal(legal: list, device: torch.device) -> torch.Tensor:
    """(N, NUM_ACTIONS), true at each row's legal actions (`inference.legal_mask`): their
    indices sent end to end, then set on `device`."""
    counts = np.fromiter(map(len, legal), np.int64, len(legal))
    choices = _to(np.concatenate(legal), device).long()
    mask = torch.zeros(len(legal), NUM_ACTIONS, dtype=torch.bool, device=device)
    rows, _ = _spread(_to(counts, device), len(choices))
    mask[rows, choices] = torch.ones_like(choices, dtype=torch.bool)  # True would wait for the GPU
    return mask


@dataclass
class Batch:
    inputs: Padded  # what the policy saw
    legal: torch.Tensor  # (N, NUM_ACTIONS) its legal actions
    oracle: Padded  # what the critic sees (`encode_oracle`)
    beliefs: torch.Tensor  # (N, 52) true places of unseen cards, NOT_HIDDEN elsewhere
    actions: torch.Tensor
    old_log_probs: torch.Tensor  # each move's log-chance under the policy that collected it
    played_log_probs: torch.Tensor  # and under the odds it was played from (see `ppo_objective`)
    advantages: torch.Tensor
    returns: torch.Tensor  # λ-returns in points: the critic's targets
    value_ev: float  # the share of the final scores' variance that the critic explains


Pass = tuple[list[int], int]  # rows, and how many of them count


def _passes(index: list[int], size: int) -> list[Pass]:
    """Minibatch `index` in passes of `size` rows, whose gradients add up to the whole's.

    Every pass has the same shape, so that a compiled pass never meets a new
    one: `size` rows, a short pass filled up with copies of its first row that
    do not count (their weight is 0), and as many tokens as the longest row
    in the batch (`Padded`). A random minibatch nearly always holds a row
    about that long, so little is lost to the padding.
    """
    parts = (index[start : start + size] for start in range(0, len(index), size))
    return [(part + part[:1] * (size - len(part)), len(part)) for part in parts]


def _counted(rows: int, counted: int, device: torch.device) -> torch.Tensor:
    """The weight of each of a pass's rows: 1 for the first `counted`, 0 for its filling."""
    return (torch.arange(rows, device=device) < counted).float()


class _Rows:
    """Row indices of many passes, sent to the device in one copy; each pass is a slice of it."""

    def __init__(self) -> None:
        self._flat: list[int] = []
        self._tensor: torch.Tensor | None = None

    def add(self, rows: list[int]) -> slice:
        start = len(self._flat)
        self._flat += rows
        return slice(start, len(self._flat))

    def to(self, device: torch.device) -> _Rows:
        self._tensor = _to(np.array(self._flat, dtype=np.int64), device)
        return self

    def __getitem__(self, part: slice) -> torch.Tensor:
        return self._tensor[part]


def _summarise(net: Net, tokens: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    """`net.summarise`, in bfloat16 on an NVIDIA GPU and in float32 after it.

    There the trunk's matrix products run on tensor cores, several times
    faster than in float32 (a 5090's float32 and TF32 are the same speed).
    The heads, softmaxes, log-chances and losses stay in float32: rounding the
    logits to bfloat16 is what spoiled an earlier try (value loss 1.7 to 4.7,
    approximate KL four times), while this keeps both where float32 has them.
    """
    if tokens.device.type != "cuda":
        return net.summarise(tokens, padding)
    with torch.autocast("cuda", torch.bfloat16):
        return net.summarise(tokens, padding).float()


def _values(critic: Net, tokens: torch.Tensor, padding: torch.Tensor, bins: torch.Tensor):
    return expected_value(critic.value(_summarise(critic, tokens, padding)), bins)


def _compiled(function: Callable, device: torch.device) -> Callable:
    """`function` compiled when it runs on an NVIDIA GPU (fused kernels, far fewer of them:
    PERFORMANCE.md); as it is elsewhere, where compiling costs more than it saves.

    Compiled for each shape it meets, which `_passes` keeps to one per batch:
    a batch whose longest row is in a new multiple of `LENGTH_STEP` costs one
    compilation (seconds; cached on disk for the next run). Compiling for any
    shape instead left guards on the first shape's size that later shapes
    failed, recompiling mid-run until the compiler gave up.
    """
    if device.type != "cuda":
        return function
    if function not in _COMPILED:
        _COMPILED[function] = torch.compile(function, dynamic=False)
    return _COMPILED[function]


_COMPILED: dict[Callable, Callable] = {}


@torch.no_grad()
def _critic_values(critic: Net, oracles: list, chunk: int = 256) -> torch.Tensor:
    """The critic's expected score for each decision, in points (on the CPU)."""
    return _padded_values(critic, Padded.of(oracles, device_of(critic)), chunk, False).cpu()


@torch.no_grad()
def _padded_values(
    critic: Net,
    oracle: Padded,
    chunk: int,
    compiled: bool = True,
    graphs: UpdateGraphs | None = None,
) -> torch.Tensor:
    """The critic's expected scores, in passes of `chunk` rows; compiled (see `_compiled`) for
    training's many rows, and replayed from `graphs` if given (`_forward`), as it is for a
    search's calls of any size."""
    critic.eval()
    bins = value_bins(device_of(critic))
    count = len(oracle.lengths)
    size = chunk if compiled else min(chunk, count)

    def inputs(rows: torch.Tensor) -> tuple:
        return (*oracle.take(rows), bins)

    return _forward(partial(_values, critic), inputs, count, size, compiled, graphs)


@torch.no_grad()
def _forward(
    function: partial,
    inputs: Callable[[torch.Tensor], tuple],
    count: int,
    size: int,
    compiled: bool = True,
    graphs: UpdateGraphs | None = None,
) -> torch.Tensor:
    """`function(*inputs(rows))`, a number for each of `count` rows, in passes of `size` rows
    (`_passes`), without gradients: as it is, compiled (`_compiled`), or replayed from
    `graphs` (`UpdateGraphs.forward`: the same numbers, with little Python). `function` is a
    network's pass with the networks as its fixed arguments, as `Loss.terms` is."""
    device = device_of(function.args[0])
    rows = _Rows()
    passes = [(rows.add(part), counted) for part, counted in _passes(list(range(count)), size)]
    rows.to(device)
    found = torch.empty(count, device=device)
    compute = _compiled(function.func, device) if compiled else function.func
    if graphs is not None:
        graphs.check()
    with _attention(device):
        for part, counted in passes:
            index = rows[part]
            if graphs is None:
                result = compute(*function.args, *inputs(index))
            else:
                result = graphs.forward(function, inputs(index))
            found[index[:counted]] = result[:counted]
    return found


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


def _gathered(items: list, name: str, dtype: type) -> np.ndarray:
    """Each item's `name` (an attribute, or an attribute's with dots), as an array."""
    return np.fromiter(map(attrgetter(name), items), dtype, len(items))


def prepare(
    trajectories: list[Trajectory],
    critic: Net,
    settings: Settings,
    graphs: UpdateGraphs | None = None,
) -> Batch:
    """Advantages and value targets for the collected steps, all on the critic's device.

    The steps' fields are gathered by NumPy, a few whole arrays each, while the GPU works
    out the critic's values, replayed from `graphs` if given (`_forward`): little Python,
    so that another thread (collecting while updating) has Python's lock most of the time.
    """
    steps = [step for trajectory in trajectories for step in trajectory.steps]
    device = device_of(critic)
    oracle = Padded.of(list(map(attrgetter("oracle"), steps)), device)
    values = _padded_values(critic, oracle, settings.chunk, graphs=graphs)
    inputs = Padded.of(list(map(attrgetter("observation.tokens"), steps)), device)
    legal = _legal(list(map(attrgetter("observation.legal"), steps)), device)
    beliefs = np.concatenate(list(map(attrgetter("belief"), steps))).reshape(len(steps), -1)
    actions = _gathered(steps, "action", np.int64)
    old, played = (_gathered(steps, name, np.float32) for name in ("policy_log_prob", "log_prob"))
    lengths = np.fromiter(map(len, map(attrgetter("steps"), trajectories)), np.int64)
    rewards = _gathered(trajectories, "reward", np.float64)
    on_cpu = values.cpu()
    found = advantages(on_cpu.numpy(), rewards, lengths, settings.gae_lambda)
    # The last sums are the CPU's, as before; or with graphs the GPU's, the same to rounding,
    # without the threads PyTorch wakes for the CPU's (tens of milliseconds while the
    # workers keep the cores busy, as they do when collecting while updating).
    where = device if graphs is not None else torch.device("cpu")
    values = values if graphs is not None else on_cpu
    adv = _to(found.astype(np.float32), where)
    returns = adv + values  # in points: the critic's targets, whatever the policy's weights
    if settings.stake_scaling:
        adv = adv * _to(stake_weights(trajectories), where)
    rewards = _to(np.repeat(rewards, lengths).astype(np.float32), where)
    return Batch(
        inputs,
        legal,
        oracle,
        _to(beliefs, device).long(),
        _to(actions, device),
        _to(old, device),
        _to(played, device),
        _to((adv - adv.mean()) / (adv.std() + 1e-8), device),
        _to(returns, device),
        explained_variance(values, rewards),
    )


ChunkResult = tuple[torch.Tensor, dict[str, torch.Tensor]]


class Loss(NamedTuple):
    """A minibatch's loss, one pass at a time (see `_step`): `terms(*inputs(rows, counted))` is
    a pass's share of the loss and its terms. `inputs` takes the pass's rows from the batch;
    `terms` reads nothing else but the networks in its fixed arguments, so that a CUDA graph
    of it can be replayed on other rows (`_GraphedStep`)."""

    inputs: Callable[[torch.Tensor, int], tuple]  # (rows, how many count) -> the pass's inputs
    terms: partial  # (*inputs) -> the loss and its terms


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


def _policy_sums(
    policy: Net,
    magnet: Net,
    tokens: torch.Tensor,
    padding: torch.Tensor,
    legal: torch.Tensor,
    actions: torch.Tensor,
    old: torch.Tensor,
    played: torch.Tensor,
    adv: torch.Tensor,
    beliefs: torch.Tensor,
    weight: torch.Tensor,
    clip: float,
) -> dict[str, torch.Tensor]:
    """The policy's loss terms and diagnostics over one pass, summed over its rows as they
    are weighed (see `_passes`)."""
    summary = _summarise(policy, tokens, padding)
    log_probs = torch.log_softmax(policy.heads(summary, legal)[0], dim=-1)
    with torch.no_grad():
        magnet_logits = magnet.heads(_summarise(magnet, tokens, padding), legal)[0]
        magnet_log_probs = torch.log_softmax(magnet_logits, dim=-1)
    # Illegal actions have log-probability -inf; zero them before multiplying,
    # since 0 * -inf is NaN and would poison the gradients.
    probs = log_probs.exp()
    safe = log_probs.masked_fill(~legal, 0.0)
    magnet_safe = magnet_log_probs.masked_fill(~legal, 0.0)
    new = log_probs.gather(1, actions[:, None]).squeeze(-1)
    log_ratio = new - old
    ratio = torch.exp(log_ratio)
    places = policy.beliefs(summary).flatten(0, 1)
    beliefs = beliefs.masked_fill(weight[:, None] == 0, NOT_HIDDEN)
    return {
        "policy_loss": -(weight * ppo_objective(new, old, played, adv, clip)).sum(),
        "entropy": -(weight * (probs * safe).sum(-1)).sum(),
        "magnet_kl": (weight * (probs * (safe - magnet_safe)).sum(-1)).sum(),
        "belief_loss": F.cross_entropy(
            places, beliefs.flatten(), ignore_index=NOT_HIDDEN, reduction="sum"
        ),
        "clip_fraction": (weight * ((ratio - 1).abs() > clip).float()).sum(),
        "approx_kl": (weight * ((ratio - 1) - log_ratio)).sum(),  # KL(old || new)
    }


def _policy_terms(
    policy: Net,
    magnet: Net,
    settings: Settings,
    tokens: torch.Tensor,
    padding: torch.Tensor,
    legal: torch.Tensor,
    actions: torch.Tensor,
    old: torch.Tensor,
    played: torch.Tensor,
    adv: torch.Tensor,
    beliefs: torch.Tensor,
    weight: torch.Tensor,
    size: int | torch.Tensor,
    hidden: torch.Tensor,
) -> ChunkResult:
    """A pass's share of the PPO loss of a minibatch of `size` rows, whose beliefs have
    `hidden` unseen cards, and its terms."""
    sums_of = _compiled(_policy_sums, device_of(policy))
    inputs = (tokens, padding, legal, actions, old, played, adv, beliefs, weight)
    sums = sums_of(policy, magnet, *inputs, settings.clip)
    terms = {name: total / size for name, total in sums.items()}
    terms["belief_loss"] = sums["belief_loss"] / hidden
    loss = (
        terms["policy_loss"]
        - settings.entropy * terms["entropy"]
        + settings.magnet * terms["magnet_kl"]
        + settings.belief * terms["belief_loss"]
    )
    return loss, terms


def _policy_loss(
    policy: Net, magnet: Net, batch: Batch, settings: Settings, minibatch: torch.Tensor
) -> Loss:
    """The PPO loss of `minibatch` (its rows), one pass at a time (see `_step`)."""
    size = len(minibatch)
    hidden = (batch.beliefs[minibatch] != NOT_HIDDEN).sum().clamp(min=1)

    def inputs(rows: torch.Tensor, counted: int) -> tuple:
        return (
            *batch.inputs.take(rows),
            batch.legal[rows],
            batch.actions[rows],
            batch.old_log_probs[rows],
            batch.played_log_probs[rows],
            batch.advantages[rows],
            batch.beliefs[rows],
            _counted(len(rows), counted, rows.device),
            size,
            hidden,
        )

    return Loss(inputs, partial(_policy_terms, policy, magnet, settings))


def _critic_sum(
    critic: Net,
    tokens: torch.Tensor,
    padding: torch.Tensor,
    returns: torch.Tensor,
    bins: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """The critic's cross-entropy with the two-hot returns, summed over one pass as its rows
    are weighed."""
    logits = critic.value(_summarise(critic, tokens, padding))
    return (weight * F.cross_entropy(logits, two_hot(returns, bins), reduction="none")).sum()


def _critic_terms(
    critic: Net,
    tokens: torch.Tensor,
    padding: torch.Tensor,
    returns: torch.Tensor,
    bins: torch.Tensor,
    weight: torch.Tensor,
    size: int | torch.Tensor,
) -> ChunkResult:
    """A pass's share of the critic's loss on a minibatch of `size` rows."""
    sum_of = _compiled(_critic_sum, device_of(critic))
    loss = sum_of(critic, tokens, padding, returns, bins, weight) / size
    return loss, {"value_loss": loss}


def _critic_loss(critic: Net, batch: Batch, bins: torch.Tensor, size: int) -> Loss:
    """The critic's loss on a minibatch of `size` rows, one pass at a time."""

    def inputs(rows: torch.Tensor, counted: int) -> tuple:
        weight = _counted(len(rows), counted, rows.device)
        return (*batch.oracle.take(rows), batch.returns[rows], bins, weight, size)

    return Loss(inputs, partial(_critic_terms, critic))


Passes = list[tuple[torch.Tensor, int]]  # each pass's rows on the device, and how many count


def _step(
    net: Net, optimiser: torch.optim.Optimizer, loss: Loss, passes: Passes, norm: str
) -> dict[str, torch.Tensor] | None:
    """One optimiser step on a minibatch, with its forward and backward in `passes`.

    Each pass's loss is its share of the minibatch's, so the gradients add up
    to the whole minibatch's. Returns the diagnostics and the gradient norm
    (before clipping), as tensors on the device, or None if the gradient was
    not finite: then the step is skipped. Only that check waits for the GPU.
    """
    optimiser.zero_grad()
    totals: dict[str, torch.Tensor] = {}
    for rows, counted in passes:
        value, terms = loss.terms(*loss.inputs(rows, counted))
        value.backward()
        for name, term in terms.items():
            totals[name] = totals.get(name, 0.0) + term.detach()
    grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    if not torch.isfinite(grad_norm):
        return None
    optimiser.step()
    return totals | {norm: grad_norm}


def _set_capturable(optimiser: torch.optim.Optimizer, on: bool) -> None:
    """Keep AdamW's step counts on the device, so that a CUDA graph can replay its steps
    (`capturable`), or on the CPU as by default: a state saved either way resumes either way."""
    for group in optimiser.param_groups:
        group["capturable"] = on
        for param in group["params"]:
            state = optimiser.state.get(param, {})
            if "step" in state:
                state["step"] = state["step"].to(param.device if on else "cpu")


_STREAMS: dict[torch.device, torch.cuda.Stream] = {}


def _stream(device: torch.device) -> torch.cuda.Stream:
    """The one stream on `device` that the update's graphs are compiled and captured on: of
    their own, since graphs captured in two threads at once must not share a stream (as they
    would share `torch.cuda.graph`'s default one with the served networks'). It has priority
    -1, a pool of streams that nothing else here draws from (`torch.cuda.Stream()` hands out
    the streams of a pool in turn)."""
    if device not in _STREAMS:
        _STREAMS[device] = torch.cuda.Stream(device, priority=-1)
    return _STREAMS[device]


@contextmanager
def _capture(graph: torch.cuda.CUDAGraph, stream: torch.cuda.Stream, pool=None):
    """Capturing into `graph` on `stream`, with the memory of `pool` (by default, a pool of
    its own), so that other threads' CUDA calls do not fail meanwhile, as they do with the
    default mode (TRAINING.md, "Collecting while updating").

    cuBLAS keeps a workspace for each thread and stream, from their first
    matrix product until any CUDA graph is destroyed, which frees them all, and
    a graph keeps the one it was captured with. So each capture starts and
    ends with none kept, as Inductor's graphs do (`clear_cublass_cache` in
    `torch/_inductor/cudagraph_trees.py`): its matrix products get a workspace
    in its own pool, which lives as long as it does. Otherwise graphs used
    workspaces that other graphs' destruction freed (the GPU faulted), or
    shared one with the served networks' graphs, which deadlocked the GPU when
    both were replayed at once (collecting while updating).
    """
    torch._C._cuda_clearCublasWorkspaces()
    with torch.cuda.graph(graph, pool=pool, stream=stream, capture_error_mode="thread_local"):
        yield
    torch._C._cuda_clearCublasWorkspaces()


class _GraphedStep:
    """`_step` for one network and its optimiser, replayed from CUDA graphs.

    A step is two graphs. The first is a pass's forward and backward: it adds
    the pass's gradient to the network's and its terms to the step's, then
    works out the gradient's norm so far (one graph for each shape of pass,
    which `_passes` keeps to about one a batch). The second is the clip and
    the optimiser's step (captured again when the optimiser's settings, such as
    its learning rate, change). Between them the norm is read, the step's one
    wait for the GPU as in `_step`, and a step whose gradient is not finite is
    skipped. Each pass's inputs are copied into the tensors its graph reads.
    The numbers are `_step`'s; but Python launches a few dozen kernels a step
    instead of hundreds, leaving its lock to other threads (PERFORMANCE.md,
    "Graphing the update").
    """

    def __init__(self, net: Net, optimiser: torch.optim.Optimizer) -> None:
        self.net, self.optimiser = net, optimiser
        self.stream = _stream(device_of(net))
        self.pool = torch.cuda.graph_pool_handle()  # the passes' graphs share their memory
        self.used: list[torch.nn.Parameter] = []  # the weights a pass gives a gradient
        self.grads: list[torch.Tensor] = []  # their gradients, where every graph finds them
        self.norm = torch.zeros((), device=device_of(net))  # the gradient's norm
        self.names: list[str] = []  # the terms a pass adds up in `totals`
        self.totals = torch.zeros(0, device=device_of(net))
        self.passes: dict[tuple, tuple[torch.cuda.CUDAGraph, list[torch.Tensor]]] = {}
        self.stepping: tuple[list[dict], torch.cuda.CUDAGraph] | None = None
        self.signature = ()

    def __call__(self, loss: Loss, passes: Passes, norm: str) -> dict[str, torch.Tensor] | None:
        if not self.optimiser.state:  # a new optimiser makes its state in its first step
            return _step(self.net, self.optimiser, loss, passes, norm)
        inputs = [loss.inputs(rows, counted) for rows, counted in passes]
        graphs = [self._pass(loss.terms, found) for found in inputs]
        if self.used[0].grad is not self.grads[0]:  # `_step` left gradients of its own
            for param, grad in zip(self.used, self.grads, strict=True):
                param.grad = grad
        torch._foreach_zero_([*self.grads, self.totals])
        for (graph, statics), found in zip(graphs, inputs, strict=True):
            for static, value in zip(statics, found, strict=True):
                if isinstance(value, torch.Tensor):
                    static.copy_(value)
                else:
                    static.fill_(value)
            graph.replay()
        if not torch.isfinite(self.norm):
            return None
        self._stepping().replay()
        found = torch.cat([self.totals, self.norm.view(1)])
        return dict(zip([*self.names, norm], found.unbind(), strict=True))

    def _pass(self, terms: partial, inputs: tuple) -> tuple[torch.cuda.CUDAGraph, list]:
        shapes = [(x.shape, x.dtype) if isinstance(x, torch.Tensor) else type(x) for x in inputs]
        key = (terms.func, terms.args, *shapes)
        if key not in self.passes:
            self.passes[key] = self._capture_pass(terms, inputs)
        return self.passes[key]

    def _capture_pass(self, terms: partial, inputs: tuple) -> tuple[torch.cuda.CUDAGraph, list]:
        """The graph of a pass of `terms` on inputs shaped as `inputs`, and the tensors it reads
        them from; compiled and warmed up first, on the update's stream, leaving `.grad` alone."""
        device = self.norm.device
        statics = [
            x.clone() if isinstance(x, torch.Tensor) else torch.tensor(float(x), device=device)
            for x in inputs
        ]
        weights = list(self.net.parameters())
        stream = self.stream
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):  # compile, and let libraries set up, before capturing
            for _ in range(2):
                value, found = terms(*statics)
                grads = torch.autograd.grad(value, weights, allow_unused=True)  # .grad untouched
        torch.cuda.current_stream(device).wait_stream(stream)
        used = [param for param, grad in zip(weights, grads, strict=True) if grad is not None]
        if not self.used:
            self.used, self.grads = used, [torch.zeros_like(param) for param in used]
            self.names, self.totals = list(found), torch.zeros(len(found), device=device)
        assert [id(p) for p in used] == [id(p) for p in self.used] and list(found) == self.names
        for param, grad in zip(self.used, self.grads, strict=True):
            param.grad = grad
        del value, found, grads
        graph = torch.cuda.CUDAGraph()
        with _capture(graph, stream, self.pool):
            value, found = terms(*statics)
            value.backward()
            self.totals.add_(torch.stack(list(found.values())).detach())
            self.norm.copy_(torch.nn.utils.get_total_norm(self.grads))
        self.signature = self._signature()
        return graph, statics

    def _settings(self) -> list[dict]:
        return [{k: v for k, v in g.items() if k != "params"} for g in self.optimiser.param_groups]

    def _stepping(self) -> torch.cuda.CUDAGraph:
        """The graph of the clip and the optimiser's step, for the optimiser's settings now."""
        if self.stepping is None or self.stepping[0] != self._settings():
            assert all(self.optimiser.state.get(param) for param in self.used)
            _set_capturable(self.optimiser, True)
            graph = torch.cuda.CUDAGraph()
            with _capture(graph, self.stream):
                torch.nn.utils.clip_grads_with_norm_(self.used, 1.0, self.norm)
                self.optimiser.step()
            self.stepping = (self._settings(), graph)
            self.signature = self._signature()
        return self.stepping[1]

    def _signature(self) -> tuple[int, ...]:
        """Where the tensors are that the graphs read and do not own: the weights of the
        networks their passes read, and the optimiser's state. Graphs replayed after one of
        them moved would read memory that is no longer theirs."""
        nets = {id(m): m for key in self.passes for m in key[1] if isinstance(m, torch.nn.Module)}
        found = [param for net in nets.values() for param in net.parameters()]
        states = [self.optimiser.state.get(param, {}) for param in self.used]
        found += [t for state in states for t in state.values() if isinstance(t, torch.Tensor)]
        return tuple(t.data_ptr() for t in found)


class UpdateGraphs:
    """The update's passes replayed from CUDA graphs: its optimiser steps (`_GraphedStep`, one
    for each network), and passes without gradients such as `prepare`'s critic pass
    (`forward`). An NVIDIA GPU's alternative to `_step` and to running `_compiled`'s
    functions from Python."""

    def __init__(self) -> None:
        self.steps: dict[Net, _GraphedStep] = {}
        self.passes: dict[tuple, tuple] = {}  # `forward`'s graphs, by function and shapes
        self.pool = None  # the memory they share

    def check(self) -> None:
        """Forget graphs if what they read has moved (`_GraphedStep._signature`, `_weights`),
        as loading an optimiser's state moves it: they are captured again when next used."""
        for net, step in list(self.steps.items()):
            if step.signature != step._signature():
                del self.steps[net]
        for key, (*_, weights) in list(self.passes.items()):
            if weights != _weights(key[1]):
                del self.passes[key]

    def __call__(
        self, net: Net, optimiser: torch.optim.Optimizer, loss: Loss, passes: Passes, norm: str
    ) -> dict[str, torch.Tensor] | None:
        step = self.steps.get(net)
        if step is None or step.optimiser is not optimiser:
            step = self.steps[net] = _GraphedStep(net, optimiser)
        return step(loss, passes, norm)

    @torch.no_grad()
    def forward(self, function: partial, inputs: tuple) -> torch.Tensor:
        """`function(*inputs)`, compiled (`_compiled`) and without gradients, replayed from a
        CUDA graph captured for its networks (`function.args`) and the shapes of `inputs`.
        What it returns is the graph's own, which the next call may overwrite: use it first."""
        key = (function.func, function.args, *[(x.shape, x.dtype) for x in inputs])
        if key not in self.passes:
            self.passes[key] = self._capture_forward(function, inputs)
        graph, statics, found, _ = self.passes[key]
        for static, value in zip(statics, inputs, strict=True):
            static.copy_(value)
        graph.replay()
        return found

    def _capture_forward(self, function: partial, inputs: tuple) -> tuple:
        """The graph of `function` on inputs shaped as `inputs`, the tensors it reads them from
        and what it returns, and where the weights are that it reads; compiled and warmed up
        first, on the update's stream (`_capture_pass`)."""
        device = inputs[0].device
        compute = partial(_compiled(function.func, device), *function.args)
        statics = [x.clone() for x in inputs]
        stream = _stream(device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):  # compile, and let libraries set up, before capturing
            for _ in range(2):
                compute(*statics)
        torch.cuda.current_stream(device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with _capture(graph, stream, self.pool):
            found = compute(*statics)
        self.pool = graph.pool()
        return graph, statics, found, _weights(function.args)


def _weights(nets: tuple) -> tuple[int, ...]:
    """Where the weights are of those of `nets` that are networks: a graph reads them there."""
    modules = [net for net in nets if isinstance(net, torch.nn.Module)]
    return tuple(param.data_ptr() for net in modules for param in net.parameters())


def _attention(device: torch.device):
    """Plain attention on CUDA: for sequences this short (about 40 tokens) it beats the fused
    kernels PyTorch picks there, with the same results: by about 10% of the update on the 5090
    before it was compiled, and by about 5% since (1.37 against 1.44 s, 28 September)."""
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
    graphs: UpdateGraphs | None = None,
) -> dict[str, float]:
    """PPO epochs over the batch (only the critic's, without `train_policy`), each step
    replayed from `graphs` if given (on an NVIDIA GPU; the same numbers, faster).

    Returns mean losses and diagnostics, and `skipped_steps`: optimiser steps
    left out because their gradient was not finite.
    """
    step: Callable = _step
    if graphs is not None:
        graphs.check()
        step = graphs
    policy_optimiser, critic_optimiser = optimisers
    policy.train()
    critic.train()
    magnet.eval()
    device = device_of(critic)
    bins = value_bins(device)
    stats: dict[str, list[torch.Tensor]] = defaultdict(list)
    steps = skipped = 0
    order = list(range(len(batch.actions)))
    for _ in range(settings.ppo_epochs):
        rng.shuffle(order)
        rows = _Rows()  # every pass's rows go to the device at once
        minibatches = []
        pass_rows = min(settings.chunk, settings.batch_size)
        for start in range(0, len(order), settings.batch_size):
            index = order[start : start + settings.batch_size]
            passes = [(rows.add(part), counted) for part, counted in _passes(index, pass_rows)]
            minibatches.append((rows.add(index), passes))
        rows.to(device)
        with _attention(device):
            for minibatch, passes in minibatches:
                size = minibatch.stop - minibatch.start
                parts = [
                    (critic, critic_optimiser, _critic_loss(critic, batch, bins, size), "critic")
                ]
                if train_policy:
                    loss = _policy_loss(policy, magnet, batch, settings, rows[minibatch])
                    parts.insert(0, (policy, policy_optimiser, loss, "policy"))
                on_device = [(rows[part], counted) for part, counted in passes]
                for net, optimiser, loss, name in parts:
                    steps += 1
                    found = step(net, optimiser, loss, on_device, f"{name}_grad_norm")
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


def _save_networks(out: Path, iteration: int, learning: Learning, keep: bool) -> None:
    """The latest networks, and with `keep` a numbered copy of each in checkpoints/.

    The magnet is kept as a policy too: with `magnet_ema` it is the policy's
    weights averaged over about the last 1 / magnet_ema iterations, which
    plays better than the policy it averages (TRAINING.md, "The recipe for a
    long run").
    """
    files = {
        "policy.pt": partial(save, learning.policy),
        "critic.pt": partial(save, learning.critic),
        "policy.npz": partial(export, learning.policy),
        "magnet.pt": partial(save, learning.magnet),
        "magnet.npz": partial(export, learning.magnet),
    }
    for name, write in files.items():
        _replace(out / name, write)
        if keep:
            (out / "checkpoints").mkdir(exist_ok=True)
            stem, suffix = name.split(".")
            _replace(out / "checkpoints" / f"{stem}-{iteration:04d}.{suffix}", write)


def cooldown_share(iteration: int, iterations: int, cooldown: int) -> float:
    """The share of the settings' learning rates that `iteration` of a run to `iterations`
    learns at: all of them until its last `cooldown` iterations, then falling linearly to a
    tenth at the last. From the iteration's number alone, so a resume carries on exactly.

    A cooldown settles the policy as the magnet's average does (TRAINING.md, "The cooldown
    check"; Hägele et al. 2024, arXiv 2405.18392)."""
    left = iterations - iteration  # iterations still to come after this one
    if left >= cooldown:
        return 1.0
    return 0.1 + 0.9 * left / cooldown


@dataclass
class Learning:
    """A network being trained: the policy, its critic and magnet, and their optimisers; and,
    `graphed` on an NVIDIA GPU, the graphs its update's steps are replayed from."""

    policy: Net
    critic: Net
    magnet: Net
    optimisers: tuple[torch.optim.Optimizer, torch.optim.Optimizer]
    graphs: UpdateGraphs | None = None

    @classmethod
    def start(
        cls, policy: Net, critic: Net, settings: Settings, device: str, graphed: bool = False
    ) -> Learning:
        policy, critic = policy.to(device), critic.to(device)
        optimisers = (
            torch.optim.AdamW(policy.parameters(), lr=settings.policy_lr),
            torch.optim.AdamW(critic.parameters(), lr=settings.critic_lr),
        )
        graphs = UpdateGraphs() if graphed and torch.device(device).type == "cuda" else None
        return cls(policy, critic, copy.deepcopy(policy), optimisers, graphs)

    def load(self, state: dict) -> None:
        """The networks and optimisers from a run's saved state (`STATE`)."""
        for name in ("policy", "critic", "magnet"):
            getattr(self, name).load_state_dict(state[name])
        for optimiser, saved in zip(self.optimisers, state["optimisers"], strict=True):
            optimiser.load_state_dict(saved)
            _set_capturable(optimiser, self.graphs is not None)

    def set_rates(self, settings: Settings, share: float = 1.0) -> None:
        """The policy's and the critic's learning rates: `share` of the settings' (a graphed
        step captures its optimiser's step again when they change: `_GraphedStep._stepping`)."""
        rates = (settings.policy_lr, settings.critic_lr)
        for optimiser, rate in zip(self.optimisers, rates, strict=True):
            for group in optimiser.param_groups:
                group["lr"] = rate * share

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
    runner.networks.load(learner, learning.policy.state_dict())
    found = collect(runner, lineups, rng, learner, scores)
    collected = time.perf_counter()
    batch = prepare(found, learning.critic, settings, learning.graphs)
    entry: dict = {"decisions": len(batch.actions)}
    if not train_policy:
        entry["warmup"] = True  # only the critic learns
    policy, critic, magnet, optimisers = (
        learning.policy,
        learning.critic,
        learning.magnet,
        learning.optimisers,
    )
    graphs = learning.graphs
    entry |= update(policy, critic, magnet, optimisers, batch, settings, rng, train_policy, graphs)
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
            runner.networks.load(slot_name(slot), league.members[member].weights)
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
    runner.networks.load(TARGET, learning.policy.state_dict())
    policy, critic, graphed = learning.policy, learning.critic, learning.graphs is not None
    copied = Learning.start(copy.deepcopy(policy), copy.deepcopy(critic), settings, device, graphed)
    for step in range(1, settings.exploit_iterations + 1):
        lineups = exploit_lineups(settings.deals_per_iteration, rng, EXPLOITER)
        _learn(runner, copied, lineups, settings, rng, EXPLOITER)
        copied.move_magnet(settings, step)
    runner.networks.load(EVAL, copied.policy.state_dict())
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
    league_start: dict[str, dict] | None = None,
    graphed: bool = False,
) -> list[dict]:
    """Run self-play training up to iteration `iterations`; returns one log entry per iteration.

    The critic comes from `as_critic`. The update runs on `device`, and so do
    the networks that play: in this process, for all of the `workers`
    processes that play the games (1: this one). The league starts with
    RuleBot, the policy as it begins and `league_start` (name -> weights).
    With a `target`, train an exploiter against it instead (see module
    docstring). With `resume`, carry on from `out`'s saved state. With `graphed`, on an NVIDIA
    GPU, the update's steps are replayed from CUDA graphs (`UpdateGraphs`).
    """
    assert critic.config.value_bins == VALUE_BINS, "make the critic with as_critic()"
    learning = Learning.start(policy, critic, settings, device, graphed)
    policy, critic = learning.policy, learning.critic
    league = League(
        settings.league_size, None if out is None else out / "league", settings.league_exploiters
    )
    first = 1
    if resume:
        state = torch.load(out / STATE, map_location="cpu", weights_only=True)
        learning.load(state)
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
        slots=SLOTS,
        explore=settings.explore_bids,
        explore_levels=settings.explore_levels,
    )
    served = networks(policy.config, SLOTS, device)
    loaded: dict[int, str] = {}  # network slot -> the league member in it
    history = []
    with Runner(
        make, workers=workers, games_in_flight=settings.games_in_flight, networks=served
    ) as runner:
        if target is not None:
            runner.networks.load(TARGET, target.state_dict())
        for iteration in range(first, iterations + 1):
            scores: dict[int, list[int]] = {}
            if target is None:
                lineups, against = _league_lineups(runner, league, loaded, settings, rng)
            else:
                lineups, against = exploit_lineups(settings.deals_per_iteration, rng), {}
            warmup = iteration <= settings.critic_warmup
            entry: dict = {"iteration": iteration}
            share = cooldown_share(iteration, iterations, settings.cooldown)
            learning.set_rates(settings, share)
            if settings.cooldown:
                entry["lr_share"] = round(share, 6)
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
                runner.networks.load(EVAL, policy.state_dict())
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
                _save_networks(out, iteration, learning, keep)
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


RESUMABLE = {"resume", "out", "workers", "device", "graphed_update"}  # may differ when resuming
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
        help="where the networks learn and play",
    )
    parser.add_argument(
        "--workers", type=int, help="processes to play on (default: all cores but two)"
    )
    parser.add_argument(
        "--graphed-update",
        action="store_true",
        help="replay the update's steps from CUDA graphs, not launch them from Python (the same "
        "learning; on an NVIDIA GPU)",
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
        "--cooldown",
        type=int,
        default=Settings.cooldown,
        help="the last iterations up to --iterations, over which both learning rates fall "
        "linearly to a tenth",
    )
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
        "--league-exploiters",
        type=int,
        default=Settings.league_exploiters,
        help="exploiters kept in the league at most; the oldest go first",
    )
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
    args.workers = args.workers or default_workers(None)

    if args.resume:
        if not (args.resume / STATE).exists():
            parser.error(f"{args.resume} has no {STATE} to resume from")
        run = json.loads((args.resume / "run.json").read_text())
        iterations = args.iterations
        for key, value in vars(args).items():  # given, and new to the run: kept from now on
            new = key not in run["args"] and key not in RESUMABLE
            if new and value != parser.get_default(key):
                run["args"][key] = str(value) if isinstance(value, Path) else value
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
        cooldown=args.cooldown,
        league_share=args.league_share,
        exploiter_share=args.exploiter_share,
        league_size=args.league_size,
        league_exploiters=args.league_exploiters,
        exploit_every=args.exploit_every,
        exploit_iterations=args.exploit_iterations,
    )
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    launch = {"commit": _commit(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if args.resume:  # the networks' shapes; train() restores their weights and the rest
        policy, critic = load(str(args.out / "policy.pt")), load(str(args.out / "critic.pt"))
        started = json.loads((args.out / "settings.json").read_text())
        changed = {k: v for k, v in asdict(settings).items() if started.get(k) != v}
        resumed = launch | {"iterations": args.iterations, "settings": changed}
        run["resumed"] = [
            *run.get("resumed", []),
            resumed | ({"graphed_update": True} if args.graphed_update else {}),
        ]
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
        league_start={path: _cpu_state(load(path)) for path in args.league_add},
        graphed=args.graphed_update,
    )


if __name__ == "__main__":
    main()
