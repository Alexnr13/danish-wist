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
with a one-thread CPU copy of the policy that gets new weights every
iteration; the update runs on `--device` (default: Apple's GPU, "mps", when
available).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from danish_wist.actions import Action
from danish_wist.bidding import NUM_PLAYERS
from danish_wist.bots import RuleBot
from danish_wist.game import Deal

from .arena import Position, Result, random_positions, role
from .encoding import ACTIONS, NOT_HIDDEN, Observation, belief_targets, encode_oracle, observe
from .model import Net, NetAgent, NetConfig, collate, export, load, save
from .runner import Decision, Runner


@dataclass(frozen=True)
class Settings:
    deals_per_iteration: int = 256
    opponent_share: float = 0.25  # deals where some seats are RuleBot or past versions
    snapshot_every: int = 10  # iterations between refreshing the magnet and saving a snapshot
    max_snapshots: int = 8
    games_in_flight: int = 256  # deals each worker keeps going at once
    gae_lambda: float = 0.95
    ppo_epochs: int = 1  # deals are cheap to play, so fresh ones beat a second pass
    batch_size: int = 512
    clip: float = 0.2
    policy_lr: float = 1e-4
    critic_lr: float = 3e-4
    entropy: float = 0.01
    magnet: float = 0.02  # weight of KL(policy || magnet)
    belief: float = 0.1  # weight of the auxiliary loss for predicting unseen cards
    critic_warmup: int = 0  # first iterations that train only the critic


# --- Collecting experience ---------------------------------------------------


@dataclass
class Step:
    """One learner decision, stored compactly."""

    observation: Observation  # NumPy arrays
    oracle: np.ndarray  # tokens including hidden cards, for the critic
    belief: np.ndarray  # where each unseen card really is (`encoding.belief_targets`)
    action: int
    log_prob: float


@dataclass
class Trajectory:
    """Everything one learner seat decided in one deal, and what it scored."""

    steps: list[Step]
    reward: float


# Agent names in each runner worker (see `make_agents`).
LEARNER, RULE, EVAL, TARGET = "learner", "rule", "eval", "target"


def snapshot_name(slot: int) -> str:
    return f"snapshot-{slot}"


def _compact(observation: Observation) -> Observation:
    return Observation(
        np.asarray(observation.tokens, dtype=np.int16),
        np.asarray(observation.legal, dtype=np.int16),
    )


class Learner:
    """The policy being trained, as it runs in a worker.

    It samples its actions and records each real decision (not forced moves)
    with its training targets (critic tokens and where the unseen cards are,
    both read from the deal) under (game, seat), for `trajectories` to collect
    when the game ends. Only the view goes into the policy.
    """

    def __init__(self, config: NetConfig, worker: int) -> None:
        self.net = Net(config).eval()
        self.worker = worker
        self.generator = torch.Generator()
        self.steps: dict[tuple[int, int], list[Step]] = defaultdict(list)

    def load(self, state: dict) -> None:
        self.net.load_state_dict(state)

    def seed(self, seed: int) -> None:
        self.generator.manual_seed(seed * 1009 + self.worker)  # a different stream per worker

    @torch.no_grad()
    def choose_decisions(self, decisions: list[Decision]) -> list[Action]:
        observations = [observe(decision.view) for decision in decisions]
        logits, _ = self.net(*collate(observations))
        log_probs = torch.log_softmax(logits, dim=-1)
        actions = torch.multinomial(log_probs.exp(), 1, generator=self.generator).squeeze(-1)
        chosen = log_probs.gather(1, actions[:, None]).squeeze(-1)
        for decision, observation, action, log_prob in zip(
            decisions, observations, actions.tolist(), chosen.tolist(), strict=True
        ):
            if len(observation.legal) == 1:
                continue  # a forced move: it gives the policy no gradient
            oracle = np.asarray(encode_oracle(decision.deal, decision.seat), dtype=np.int16)
            belief = np.asarray(belief_targets(decision.deal, decision.seat), dtype=np.int8)
            step = Step(_compact(observation), oracle, belief, action, log_prob)
            self.steps[decision.game, decision.seat].append(step)
        return [ACTIONS[action] for action in actions.tolist()]


class Frozen(NetAgent):
    """A network that only plays, greedily: past snapshots, exploit targets, evaluation."""

    def __init__(self, config: NetConfig) -> None:
        super().__init__(Net(config))

    def load(self, state: dict) -> None:
        self.net.load_state_dict(state)
        self.net.eval()


def make_agents(worker: int, config: dict, snapshots: int, one_thread: bool) -> dict:
    """The agents in each runner worker, by name. Their weights arrive by `broadcast`."""
    if one_thread:
        torch.set_num_threads(1)  # one process per core already
    net_config = NetConfig(**config)
    agents = {
        LEARNER: Learner(net_config, worker),
        RULE: RuleBot(),
        EVAL: Frozen(net_config),
        TARGET: Frozen(net_config),
    }
    return agents | {snapshot_name(slot): Frozen(net_config) for slot in range(snapshots)}


def trajectories(game: int, deal: Deal, agents: dict) -> list[Trajectory]:
    """Runs in the worker when a game ends: the learner's decisions in it, and their scores."""
    steps = agents[LEARNER].steps
    return [
        Trajectory(steps.pop((game, seat)), float(deal.scores[seat]))
        for seat in range(NUM_PLAYERS)
        if (game, seat) in steps
    ]


def collect(runner: Runner, lineups: list[list[str]], rng: random.Random) -> list[Trajectory]:
    """Play one random deal per lineup (an agent name per seat), recording the learner."""
    runner.broadcast(LEARNER, "seed", rng.getrandbits(32))
    games = list(zip(random_positions(len(lineups), rng), lineups, strict=True))
    return [t for found in runner.play(games, finish=trajectories) for t in found]


def exploit_lineups(count: int, rng: random.Random) -> list[list[str]]:
    """The learner in one random seat, the frozen target in the other three."""
    lineups = []
    for _ in range(count):
        lineup = [TARGET] * NUM_PLAYERS
        lineup[rng.randrange(NUM_PLAYERS)] = LEARNER
        lineups.append(lineup)
    return lineups


def choose_lineups(
    count: int, opponents: list[str], share: float, rng: random.Random
) -> list[list[str]]:
    """Mostly pure self-play; in `share` of deals, 1–3 seats go to random opponents."""
    lineups = []
    for _ in range(count):
        lineup = [LEARNER] * NUM_PLAYERS
        if opponents and rng.random() < share:
            for seat in rng.sample(range(NUM_PLAYERS), rng.randint(1, NUM_PLAYERS - 1)):
                lineup[seat] = rng.choice(opponents)
        lineups.append(lineup)
    return lineups


def _scores_and_roles(game: int, deal: Deal, agents: dict) -> tuple:
    return game, deal.scores, [role(deal, seat) for seat in range(NUM_PLAYERS)]


def evaluate(runner: Runner, candidate: str, field: str, positions: list[Position]) -> Result:
    """Duplicate play (as `learn.arena.duplicate`) between two named agents, through the runner."""
    games = []
    for position in positions:
        games.append((position, [field] * NUM_PLAYERS))
        for seat in range(NUM_PLAYERS):
            lineup = [field] * NUM_PLAYERS
            lineup[seat] = candidate
            games.append((position, lineup))
    played = {
        game: (scores, roles) for game, scores, roles in runner.play(games, _scores_and_roles)
    }
    result = Result()
    for i in range(len(positions)):
        baseline = played[5 * i][0]
        advantages = []
        for seat in range(NUM_PLAYERS):
            scores, roles = played[5 * i + 1 + seat]
            advantages.append(scores[seat] - baseline[seat])
            result.by_role[roles[seat]].append(advantages[-1])
        result.per_deal.append(sum(advantages) / NUM_PLAYERS)
    return result


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
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor  # λ-returns in points: the critic's targets
    value_ev: float  # the share of the final scores' variance that the critic explains


CHUNK = 256  # rows per forward and backward pass: on Apple's GPU, bigger ones run much slower


def _chunks(index: list[int]) -> list[list[int]]:
    return [index[start : start + CHUNK] for start in range(0, len(index), CHUNK)]


def _device(net: Net) -> torch.device:
    return next(net.parameters()).device


def _inputs(observations: list[Observation], device: torch.device) -> list[torch.Tensor]:
    return [t.to(device) for t in collate(observations)]


def _oracle_inputs(oracles: list[np.ndarray], device: torch.device) -> list[torch.Tensor]:
    return _inputs([Observation(o, np.zeros(0, dtype=np.int64)) for o in oracles], device)


@torch.no_grad()
def _critic_values(critic: Net, oracles: list[np.ndarray]) -> torch.Tensor:
    """The critic's expected score for each decision, in points (on the CPU)."""
    critic.eval()
    device = _device(critic)
    bins = value_bins(device)
    values = [
        expected_value(critic(*_oracle_inputs(oracles[i : i + CHUNK], device))[1], bins).cpu()
        for i in range(0, len(oracles), CHUNK)
    ]
    return torch.cat(values)


def explained_variance(predicted: torch.Tensor, actual: torch.Tensor) -> float:
    return float(1 - (actual - predicted).var() / actual.var().clamp(min=1e-8))


def prepare(trajectories: list[Trajectory], critic: Net, settings: Settings) -> Batch:
    """Advantages and value targets for the collected steps, on the critic's device."""
    steps = [step for trajectory in trajectories for step in trajectory.steps]
    values = _critic_values(critic, [s.oracle for s in steps])
    all_advantages, start = [], 0
    for trajectory in trajectories:
        end = start + len(trajectory.steps)
        found = advantages(values[start:end].tolist(), trajectory.reward, settings.gae_lambda)
        all_advantages += found
        start = end
    rewards = torch.tensor([t.reward for t in trajectories for _ in t.steps])
    device = _device(critic)
    adv = torch.tensor(all_advantages, dtype=torch.float32)
    return Batch(
        [s.observation for s in steps],
        [s.oracle for s in steps],
        torch.as_tensor(np.stack([s.belief for s in steps]), dtype=torch.long).to(device),
        torch.tensor([s.action for s in steps]).to(device),
        torch.tensor([s.log_prob for s in steps]).to(device),
        ((adv - adv.mean()) / (adv.std() + 1e-8)).to(device),
        (adv + values).to(device),
        explained_variance(values, rewards),
    )


ChunkLoss = Callable[[list[int]], tuple[torch.Tensor, dict[str, torch.Tensor]]]


def _policy_loss(
    policy: Net, magnet: Net, batch: Batch, settings: Settings, index: list[int]
) -> ChunkLoss:
    """The PPO loss of minibatch `index`, one chunk at a time (see `_step`)."""
    size = len(index)
    hidden = (batch.beliefs[index] != NOT_HIDDEN).sum().clamp(min=1)

    def chunk_loss(chunk: list[int]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tokens, padding, legal = _inputs([batch.observations[i] for i in chunk], _device(policy))
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
        log_ratio = new - batch.old_log_probs[chunk]
        ratio = torch.exp(log_ratio)
        adv = batch.advantages[chunk]
        clipped = torch.clamp(ratio, 1 - settings.clip, 1 + settings.clip)
        beliefs = policy.beliefs(summary).flatten(0, 1)
        terms = {
            "policy_loss": -torch.min(ratio * adv, clipped * adv).sum() / size,
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
        oracle = _oracle_inputs([batch.oracles[i] for i in chunk], _device(critic))
        targets = two_hot(batch.returns[chunk], bins)
        loss = F.cross_entropy(critic(*oracle)[1], targets, reduction="sum") / len(index)
        return loss, {"value_loss": loss}

    return chunk_loss


def _step(
    net: Net, optimiser: torch.optim.Optimizer, chunk_loss: ChunkLoss, index: list[int], norm: str
) -> dict[str, float] | None:
    """One optimiser step on minibatch `index`, with its forward and backward in chunks.

    Each chunk's loss is its share of the minibatch's, so the gradients add up
    to the whole minibatch's. Returns the diagnostics and the gradient norm
    (before clipping), or None if the gradient was not finite: then the step
    is skipped.
    """
    optimiser.zero_grad()
    totals: dict[str, torch.Tensor] = {}
    for chunk in _chunks(index):
        loss, terms = chunk_loss(chunk)
        loss.backward()
        for name, value in terms.items():
            totals[name] = totals.get(name, 0.0) + value.detach()
    grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    if not torch.isfinite(grad_norm):
        return None
    optimiser.step()
    return {name: value.item() for name, value in totals.items()} | {norm: grad_norm.item()}


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
    bins = value_bins(_device(critic))
    stats: dict[str, list[float]] = defaultdict(list)
    steps = skipped = 0
    order = list(range(len(batch.actions)))
    for _ in range(settings.ppo_epochs):
        rng.shuffle(order)
        for start in range(0, len(order), settings.batch_size):
            index = order[start : start + settings.batch_size]
            parts = [(critic, critic_optimiser, _critic_loss(critic, batch, bins, index), "critic")]
            if train_policy:
                loss = _policy_loss(policy, magnet, batch, settings, index)
                parts.insert(0, (policy, policy_optimiser, loss, "policy"))
            for net, optimiser, chunk_loss, name in parts:
                steps += 1
                found = _step(net, optimiser, chunk_loss, index, f"{name}_grad_norm")
                if found is None:
                    skipped += 1
                    continue
                for key, value in found.items():
                    stats[key].append(value)
    if skipped == steps:
        raise FloatingPointError("every update step had a non-finite gradient")
    return {key: sum(values) / len(values) for key, values in stats.items()} | {
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
) -> list[dict]:
    """Run self-play training up to iteration `iterations`; returns one log entry per iteration.

    The critic comes from `as_critic`. Games are played by `workers` processes
    (1: in this one). With a `target`, train an exploiter against it instead
    (see module docstring). With `resume`, carry on from `out`'s saved state.
    """
    assert critic.config.value_bins == VALUE_BINS, "make the critic with as_critic()"
    policy, critic = policy.to(device), critic.to(device)
    magnet = copy.deepcopy(policy)
    optimisers = (
        torch.optim.AdamW(policy.parameters(), lr=settings.policy_lr),
        torch.optim.AdamW(critic.parameters(), lr=settings.critic_lr),
    )
    pool: dict[str, dict] = {}  # snapshot name -> its weights
    first = 1
    if resume:
        state = torch.load(out / STATE, map_location="cpu", weights_only=True)
        for net, name in [(policy, "policy"), (critic, "critic"), (magnet, "magnet")]:
            net.load_state_dict(state[name])
        for optimiser, saved in zip(optimisers, state["optimisers"], strict=True):
            optimiser.load_state_dict(saved)
        pool, first = state["pool"], state["iteration"] + 1
        rng.setstate(state["rng"])
        torch.set_rng_state(state["torch_rng"])
        for name in ("log.jsonl", "evals.jsonl"):
            _keep_until(out / name, state["iteration"])
    elif out is not None and (out / "log.jsonl").exists():
        raise FileExistsError(f"{out} already holds a run: resume it, or choose another")
    eval_positions = random_positions(eval_deals, random.Random(12345)) if eval_every else []
    make = partial(
        make_agents,
        config=asdict(policy.config),
        snapshots=settings.max_snapshots,
        one_thread=workers > 1,
    )
    history = []
    with Runner(make, workers=workers, games_in_flight=settings.games_in_flight) as runner:
        if target is not None:
            runner.broadcast(TARGET, "load", _cpu_state(target))
        for name, weights in pool.items():
            runner.broadcast(name, "load", weights)
        for iteration in range(first, iterations + 1):
            started = time.perf_counter()
            runner.broadcast(LEARNER, "load", _cpu_state(policy))
            if target is None:
                lineups = choose_lineups(
                    settings.deals_per_iteration, [RULE, *pool], settings.opponent_share, rng
                )
            else:
                lineups = exploit_lineups(settings.deals_per_iteration, rng)
            found = collect(runner, lineups, rng)
            collected = time.perf_counter()
            batch = prepare(found, critic, settings)
            warmup = iteration <= settings.critic_warmup
            entry: dict = {"iteration": iteration, "decisions": len(batch.actions)}
            if warmup:
                entry["warmup"] = True  # only the critic learns
            entry |= update(
                policy, critic, magnet, optimisers, batch, settings, rng, train_policy=not warmup
            )
            entry["value_ev"] = batch.value_ev
            entry["mean_reward"] = float(np.mean([t.reward for t in found]))
            entry["collect_s"] = round(collected - started, 1)
            entry["update_s"] = round(time.perf_counter() - collected, 1)
            _check_finite(policy, critic)

            if iteration % settings.snapshot_every == 0:
                magnet.load_state_dict(policy.state_dict())
                slot = (iteration // settings.snapshot_every - 1) % settings.max_snapshots
                pool[snapshot_name(slot)] = _cpu_state(policy)
                runner.broadcast(snapshot_name(slot), "load", pool[snapshot_name(slot)])
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
                    "magnet": magnet.state_dict(),
                    "optimisers": [optimiser.state_dict() for optimiser in optimisers],
                    "pool": pool,
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


RESUMABLE = {"resume", "out", "workers", "device"}  # may differ when resuming


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play training with PPO.")
    parser.add_argument("--init", type=Path, help="start the policy (and critic) from a checkpoint")
    parser.add_argument(
        "--resume", type=Path, help="carry on the run in this directory, with its own settings"
    )
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--exploit", type=Path, help="train an exploiter against this policy")
    parser.add_argument(
        "--iterations", type=int, help="in all (default: 100, or the run's own when resuming)"
    )
    parser.add_argument("--deals", type=int, default=Settings.deals_per_iteration)
    parser.add_argument("--ppo-epochs", type=int, default=Settings.ppo_epochs)
    parser.add_argument(
        "--critic-warmup",
        type=int,
        default=Settings.critic_warmup,
        help="first iterations that train only the critic",
    )
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-deals", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/rl"))
    args = parser.parse_args()

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
    )
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    launch = {"commit": _commit(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if args.resume:  # the networks' shapes; train() restores their weights and the rest
        policy, critic = load(str(args.out / "policy.pt")), load(str(args.out / "critic.pt"))
        run["resumed"] = [*run.get("resumed", []), launch | {"iterations": args.iterations}]
    else:
        policy = load(str(args.init)) if args.init else Net(NetConfig())
        critic = as_critic(policy)
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
    )


if __name__ == "__main__":
    main()
