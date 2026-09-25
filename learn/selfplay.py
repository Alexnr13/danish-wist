"""Self-play reinforcement learning (LEARNING.md §3, step 4).

One network (the *learner*) plays many deals at once, mostly against itself
and sometimes against RuleBot or frozen past versions of itself. Its
decisions are then improved with PPO:

- **Critic with hidden cards.** A separate value network reads
  `encode_oracle` (the player's view plus every hidden card), in training
  only. It predicts the final score on a symmetric-log scale, which keeps
  scores from 10 to 50,000 manageable without reshaping the reward.
- **Advantages** come from GAE over each seat's own decisions in a deal. The
  only reward is that seat's score at the end.
- **Magnet.** A KL penalty keeps the policy near a slowly refreshed copy of
  itself (as in DeepNash and magnetic mirror descent), so self-play does not
  cycle or chase its own tail.

    python -m learn.selfplay --init runs/bc.pt --iterations 200 --out runs/rl
    python -m learn.selfplay --resume runs/rl --iterations 200   # carry on
    python -m learn.selfplay --init runs/bc.pt --exploit runs/rl/policy.pt --out runs/x

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
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from danish_wist.actions import Action
from danish_wist.bidding import NUM_PLAYERS
from danish_wist.bots import RuleBot
from danish_wist.game import Deal

from .arena import random_positions
from .encoding import ACTIONS, NOT_HIDDEN, Observation, belief_targets, encode_oracle, observe
from .evaluate import evaluate
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
    ppo_epochs: int = 2
    batch_size: int = 512
    clip: float = 0.2
    policy_lr: float = 1e-4
    critic_lr: float = 3e-4
    entropy: float = 0.01
    magnet: float = 0.02  # weight of KL(policy || magnet)
    belief: float = 0.1  # weight of the auxiliary loss for predicting unseen cards


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

    It samples its actions and records each decision with its training targets
    (critic tokens and where the unseen cards are, both read from the deal)
    under (game, seat), for `trajectories` to collect when the game ends. Only
    the view goes into the policy.
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


# --- Learning ----------------------------------------------------------------


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(x.abs())


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
    targets: torch.Tensor  # value targets, in symlog units


def _device(net: Net) -> torch.device:
    return next(net.parameters()).device


def _oracle_inputs(oracles: list[np.ndarray], device: torch.device) -> list[torch.Tensor]:
    inputs = collate([Observation(o, np.zeros(0, dtype=np.int64)) for o in oracles])
    return [t.to(device) for t in inputs]


@torch.no_grad()
def _critic_values(critic: Net, oracles: list[np.ndarray], batch_size: int) -> list[float]:
    critic.eval()
    values = []
    for start in range(0, len(oracles), batch_size):
        inputs = _oracle_inputs(oracles[start : start + batch_size], _device(critic))
        values += symexp(critic(*inputs)[1]).tolist()
    return values


def prepare(trajectories: list[Trajectory], critic: Net, settings: Settings) -> Batch:
    """Advantages and value targets for the collected steps, on the critic's device."""
    steps = [step for trajectory in trajectories for step in trajectory.steps]
    values = _critic_values(critic, [s.oracle for s in steps], settings.batch_size)
    all_advantages, start = [], 0
    for trajectory in trajectories:
        end = start + len(trajectory.steps)
        all_advantages += advantages(values[start:end], trajectory.reward, settings.gae_lambda)
        start = end
    device = _device(critic)
    adv = torch.tensor(all_advantages, dtype=torch.float32)
    returns = adv + torch.tensor(values, dtype=torch.float32)
    return Batch(
        [s.observation for s in steps],
        [s.oracle for s in steps],
        torch.as_tensor(np.stack([s.belief for s in steps]), dtype=torch.long).to(device),
        torch.tensor([s.action for s in steps]).to(device),
        torch.tensor([s.log_prob for s in steps]).to(device),
        ((adv - adv.mean()) / (adv.std() + 1e-8)).to(device),
        symlog(returns).to(device),
    )


def update(
    policy: Net,
    critic: Net,
    magnet: Net,
    optimisers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    batch: Batch,
    settings: Settings,
    rng: random.Random,
) -> dict[str, float]:
    """PPO epochs over the batch. Returns mean losses and diagnostics."""
    policy_optimiser, critic_optimiser = optimisers
    policy.train()
    critic.train()
    magnet.eval()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    order = list(range(len(batch.actions)))
    for _ in range(settings.ppo_epochs):
        rng.shuffle(order)
        for start in range(0, len(order), settings.batch_size):
            index = order[start : start + settings.batch_size]
            inputs = collate([batch.observations[i] for i in index])
            tokens, padding, legal = (t.to(_device(policy)) for t in inputs)
            summary = policy.summarise(tokens, padding)
            logits, _ = policy.heads(summary, legal)
            log_probs = torch.log_softmax(logits, dim=-1)
            with torch.no_grad():
                magnet_log_probs = torch.log_softmax(magnet(tokens, padding, legal)[0], dim=-1)
            # Illegal actions have log-probability -inf; zero them before multiplying,
            # since 0 * -inf is NaN and would poison the gradients.
            probs = log_probs.exp()
            safe = log_probs.masked_fill(~legal, 0.0)
            entropy = -(probs * safe).sum(-1).mean()
            kl = (probs * (safe - magnet_log_probs.masked_fill(~legal, 0.0))).sum(-1).mean()

            new = log_probs.gather(1, batch.actions[index, None]).squeeze(-1)
            ratio = torch.exp(new - batch.old_log_probs[index])
            adv = batch.advantages[index]
            clipped = torch.clamp(ratio, 1 - settings.clip, 1 + settings.clip)
            policy_loss = -torch.min(ratio * adv, clipped * adv).mean()
            beliefs = batch.beliefs[index]
            belief_loss = (
                F.cross_entropy(
                    policy.beliefs(summary).flatten(0, 1),
                    beliefs.flatten(),
                    ignore_index=NOT_HIDDEN,
                )
                if (beliefs != NOT_HIDDEN).any()
                else summary.new_zeros(())
            )
            loss = (
                policy_loss
                - settings.entropy * entropy
                + settings.magnet * kl
                + settings.belief * belief_loss
            )
            policy_optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            policy_optimiser.step()

            oracle = _oracle_inputs([batch.oracles[i] for i in index], _device(critic))
            value_loss = F.mse_loss(critic(*oracle)[1], batch.targets[index])
            critic_optimiser.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_optimiser.step()

            for name, value in [
                ("policy_loss", policy_loss),
                ("value_loss", value_loss),
                ("belief_loss", belief_loss),
                ("entropy", entropy),
                ("magnet_kl", kl),
                ("clip_fraction", ((ratio - 1).abs() > settings.clip).float().mean()),
            ]:
                totals[name] += value.item()
            count += 1
    return {name: total / count for name, total in totals.items()}


# --- The training loop -------------------------------------------------------


def _cpu_state(net: Net) -> dict:
    return {name: t.detach().cpu() for name, t in net.state_dict().items()}


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
) -> list[dict]:
    """Run self-play training; returns one log entry per iteration.

    Games are played by `workers` processes (1: in this one). With a `target`,
    train an exploiter against it instead (see module docstring).
    """
    policy, critic = policy.to(device), critic.to(device)
    magnet = copy.deepcopy(policy)
    optimisers = (
        torch.optim.AdamW(policy.parameters(), lr=settings.policy_lr),
        torch.optim.AdamW(critic.parameters(), lr=settings.critic_lr),
    )
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
        snapshots: list[str] = []
        for iteration in range(1, iterations + 1):
            started = time.perf_counter()
            runner.broadcast(LEARNER, "load", _cpu_state(policy))
            if target is None:
                lineups = choose_lineups(
                    settings.deals_per_iteration,
                    [RULE, *snapshots],
                    settings.opponent_share,
                    rng,
                )
            else:
                lineups = exploit_lineups(settings.deals_per_iteration, rng)
            found = collect(runner, lineups, rng)
            collected = time.perf_counter()
            batch = prepare(found, critic, settings)
            entry = {"iteration": iteration, "decisions": len(batch.actions)}
            entry |= update(policy, critic, magnet, optimisers, batch, settings, rng)
            entry["mean_reward"] = float(np.mean([t.reward for t in found]))
            entry["collect_s"] = round(collected - started, 1)
            entry["update_s"] = round(time.perf_counter() - collected, 1)

            if iteration % settings.snapshot_every == 0:
                magnet.load_state_dict(policy.state_dict())
                slot = (iteration // settings.snapshot_every - 1) % settings.max_snapshots
                runner.broadcast(snapshot_name(slot), "load", _cpu_state(policy))
                if snapshot_name(slot) not in snapshots:
                    snapshots.append(snapshot_name(slot))
            if eval_every and iteration % eval_every == 0:
                runner.broadcast(EVAL, "load", _cpu_state(policy))
                name, field = ("rulebot", RULE) if target is None else ("target", TARGET)
                result = evaluate(runner, EVAL, field, eval_positions)
                entry[f"vs_{name}"], entry[f"vs_{name}_ci95"] = result.mean, result.ci95
            if out is not None:
                out.mkdir(parents=True, exist_ok=True)
                with (out / "log.jsonl").open("a") as log:
                    log.write(json.dumps(entry) + "\n")
                if iteration % settings.snapshot_every == 0 or iteration == iterations:
                    save(policy, str(out / "policy.pt"))
                    save(critic, str(out / "critic.pt"))
                    export(policy, str(out / "policy.npz"))
            print(
                " ".join(
                    f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in entry.items()
                ),
                flush=True,
            )
            history.append(entry)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play training with PPO.")
    parser.add_argument("--init", type=Path, help="start the policy (and critic) from a checkpoint")
    parser.add_argument("--resume", type=Path, help="carry on from a run's policy.pt and critic.pt")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--exploit", type=Path, help="train an exploiter against this policy")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--deals", type=int, default=Settings.deals_per_iteration)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-deals", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/rl"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    if args.resume:
        policy, critic = load(str(args.resume / "policy.pt")), load(str(args.resume / "critic.pt"))
    elif args.init:
        policy, critic = load(str(args.init)), load(str(args.init))
    else:
        policy, critic = Net(NetConfig()), Net(NetConfig())
    settings = Settings(deals_per_iteration=args.deals)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "settings.json").write_text(json.dumps(asdict(settings), indent=2))
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
    )


if __name__ == "__main__":
    main()
