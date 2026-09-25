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
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from danish_wist.bidding import NUM_PLAYERS
from danish_wist.bots import Agent, RuleBot
from danish_wist.game import Deal

from .arena import duplicate, random_positions
from .encoding import ACTIONS, Observation, encode_oracle, observe
from .model import Net, NetAgent, NetConfig, collate, export, load, save


@dataclass(frozen=True)
class Settings:
    deals_per_iteration: int = 256
    opponent_share: float = 0.25  # deals where some seats are RuleBot or past versions
    snapshot_every: int = 10  # iterations between refreshing the magnet and saving a snapshot
    max_snapshots: int = 8
    gae_lambda: float = 0.95
    ppo_epochs: int = 2
    batch_size: int = 512
    clip: float = 0.2
    policy_lr: float = 1e-4
    critic_lr: float = 3e-4
    entropy: float = 0.01
    magnet: float = 0.02  # weight of KL(policy || magnet)


# --- Collecting experience ---------------------------------------------------


@dataclass
class Step:
    """One learner decision, stored compactly."""

    observation: Observation  # NumPy arrays
    oracle: np.ndarray  # tokens including hidden cards, for the critic
    action: int
    log_prob: float


@dataclass
class Trajectory:
    """Everything one learner seat decided in one deal, and what it scored."""

    steps: list[Step]
    reward: float


LEARNER = None  # a seat played by the learner


def _compact(observation: Observation) -> Observation:
    return Observation(
        np.asarray(observation.tokens, dtype=np.int16),
        np.asarray(observation.legal, dtype=np.int16),
    )


@torch.no_grad()
def collect(policy: Net, lineups: list[list[Agent | None]], rng: random.Random) -> list[Trajectory]:
    """Play one deal per lineup (seat -> agent, None for the learner) and record the learner.

    All deals advance together so the learner's decisions are batched.
    """
    policy.eval()
    generator = torch.Generator().manual_seed(rng.getrandbits(63))
    deals = [Deal.new(rng.randrange(NUM_PLAYERS), rng) for _ in lineups]
    steps: dict[tuple[int, int], list[Step]] = defaultdict(list)
    active = list(range(len(deals)))
    while active:
        waiting = []
        for game in active:
            deal = deals[game]
            seat = deal.to_act
            agent = lineups[game][seat]
            if agent is LEARNER:
                waiting.append(game)
            else:
                deal.apply(agent.choose(deal.view(seat)))
        if waiting:
            views = [deals[g].view(deals[g].to_act) for g in waiting]
            observations = [observe(v) for v in views]
            logits, _ = policy(*collate(observations))
            log_probs = torch.log_softmax(logits, dim=-1)
            actions = torch.multinomial(log_probs.exp(), 1, generator=generator).squeeze(-1)
            chosen = log_probs.gather(1, actions[:, None]).squeeze(-1)
            for game, observation, action, log_prob in zip(
                waiting, observations, actions.tolist(), chosen.tolist(), strict=True
            ):
                deal = deals[game]
                seat = deal.to_act
                oracle = np.asarray(encode_oracle(deal, seat), dtype=np.int16)
                steps[game, seat].append(Step(_compact(observation), oracle, action, log_prob))
                deal.apply(ACTIONS[action])
        active = [g for g in active if not deals[g].is_over]
    return [
        Trajectory(seat_steps, float(deals[game].scores[seat]))
        for (game, seat), seat_steps in steps.items()
    ]


def choose_lineups(
    count: int, opponents: list[Agent], share: float, rng: random.Random
) -> list[list[Agent | None]]:
    """Mostly pure self-play; in `share` of deals, 1–3 seats go to random opponents."""
    lineups = []
    for _ in range(count):
        lineup: list[Agent | None] = [LEARNER] * NUM_PLAYERS
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
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    targets: torch.Tensor  # value targets, in symlog units


@torch.no_grad()
def _critic_values(critic: Net, oracles: list[np.ndarray], batch_size: int) -> list[float]:
    critic.eval()
    values = []
    for start in range(0, len(oracles), batch_size):
        chunk = oracles[start : start + batch_size]
        inputs = collate([Observation(o, np.zeros(0, dtype=np.int64)) for o in chunk])
        values += symexp(critic(*inputs)[1]).tolist()
    return values


def prepare(trajectories: list[Trajectory], critic: Net, settings: Settings) -> Batch:
    steps = [step for trajectory in trajectories for step in trajectory.steps]
    values = _critic_values(critic, [s.oracle for s in steps], settings.batch_size)
    all_advantages, start = [], 0
    for trajectory in trajectories:
        end = start + len(trajectory.steps)
        all_advantages += advantages(values[start:end], trajectory.reward, settings.gae_lambda)
        start = end
    adv = torch.tensor(all_advantages, dtype=torch.float32)
    returns = adv + torch.tensor(values, dtype=torch.float32)
    return Batch(
        [s.observation for s in steps],
        [s.oracle for s in steps],
        torch.tensor([s.action for s in steps]),
        torch.tensor([s.log_prob for s in steps]),
        (adv - adv.mean()) / (adv.std() + 1e-8),
        symlog(returns),
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
            tokens, padding, legal = collate([batch.observations[i] for i in index])
            logits, _ = policy(tokens, padding, legal)
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
            loss = policy_loss - settings.entropy * entropy + settings.magnet * kl
            policy_optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            policy_optimiser.step()

            oracle = collate(
                [Observation(batch.oracles[i], np.zeros(0, dtype=np.int64)) for i in index]
            )
            value_loss = F.mse_loss(critic(*oracle)[1], batch.targets[index])
            critic_optimiser.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_optimiser.step()

            for name, value in [
                ("policy_loss", policy_loss),
                ("value_loss", value_loss),
                ("entropy", entropy),
                ("magnet_kl", kl),
                ("clip_fraction", ((ratio - 1).abs() > settings.clip).float().mean()),
            ]:
                totals[name] += value.item()
            count += 1
    return {name: total / count for name, total in totals.items()}


# --- The training loop -------------------------------------------------------


def train(
    policy: Net,
    critic: Net,
    iterations: int,
    settings: Settings,
    rng: random.Random,
    out: Path | None = None,
    eval_every: int = 0,
    eval_deals: int = 100,
) -> list[dict]:
    """Run self-play training; returns one log entry per iteration."""
    magnet = copy.deepcopy(policy)
    snapshots: list[Agent] = []
    optimisers = (
        torch.optim.AdamW(policy.parameters(), lr=settings.policy_lr),
        torch.optim.AdamW(critic.parameters(), lr=settings.critic_lr),
    )
    eval_positions = random_positions(eval_deals, random.Random(12345)) if eval_every else []
    history = []
    for iteration in range(1, iterations + 1):
        started = time.perf_counter()
        opponents = [RuleBot(), *snapshots]
        lineups = choose_lineups(
            settings.deals_per_iteration, opponents, settings.opponent_share, rng
        )
        trajectories = collect(policy, lineups, rng)
        collected = time.perf_counter()
        batch = prepare(trajectories, critic, settings)
        entry = {"iteration": iteration, "decisions": len(batch.actions)}
        entry |= update(policy, critic, magnet, optimisers, batch, settings, rng)
        entry["mean_reward"] = float(np.mean([t.reward for t in trajectories]))
        entry["collect_s"] = round(collected - started, 1)
        entry["update_s"] = round(time.perf_counter() - collected, 1)

        if iteration % settings.snapshot_every == 0:
            magnet.load_state_dict(policy.state_dict())
            snapshots = (snapshots + [NetAgent(copy.deepcopy(policy))])[-settings.max_snapshots :]
        if eval_every and iteration % eval_every == 0:
            result = duplicate(NetAgent(policy), RuleBot(), eval_positions)
            entry["vs_rulebot"], entry["vs_rulebot_ci95"] = result.mean, result.ci95
            policy.train()
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
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--deals", type=int, default=Settings.deals_per_iteration)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-deals", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/rl"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    policy = load(str(args.init)) if args.init else Net(NetConfig())
    critic = load(str(args.init)) if args.init else Net(NetConfig())
    settings = Settings(deals_per_iteration=args.deals)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "settings.json").write_text(json.dumps(asdict(settings), indent=2))
    train(
        policy, critic, args.iterations, settings, rng, args.out, args.eval_every, args.eval_deals
    )


if __name__ == "__main__":
    main()
