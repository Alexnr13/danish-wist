"""Behaviour cloning: train the network to copy a teacher bot (LEARNING.md §5, step 3).

This proves the whole pipeline (encoding, network, training, playing) before
self-play, and gives self-play a sensible starting policy.

    python -m learn.imitate --deals 5000 --epochs 3 --out runs/bc.pt
    python -m learn.imitate --deals 20000 --explore 0.1 --out runs/bc-explore.pt

A copy of RuleBot never tries what RuleBot never does (it never bids Flip or
Halves), and self-play cannot learn from choices it never samples. With
`--explore`, that share of the target for each bid and contract decision is
spread over its alternatives (see `alternatives`), so self-play starts with
a real chance of trying them.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from danish_wist.actions import Action
from danish_wist.bidding import Bid
from danish_wist.bots import Agent, RandomBot, RuleBot
from danish_wist.game import Deal, Phase, PlayerView

from .arena import duplicate, random_positions
from .encoding import NUM_ACTIONS, Observation, action_index, observe
from .model import Net, NetAgent, NetConfig, collate, export, save


@dataclass(frozen=True)
class Sample:
    observation: Observation  # tokens and legal indices stored as small NumPy arrays
    target: int  # the teacher's action index
    alternatives: tuple[int, ...] = ()  # other choices `--explore` keeps open


EXPLORED = {Phase.CALL_ACE, Phase.NAME_TRUMPS, Phase.FLIP, Phase.EXCHANGE, Phase.FUCDIC}


def alternatives(view: PlayerView, action: Action) -> tuple[int, ...]:
    """The choices besides the teacher's that exploration keeps a chance on.

    For a bid, the other kinds of contract at the same level (plain, Flip,
    Clubs or Halves); in the contract's decisions (the ace, trumps, Flip, the
    exchange, the fucdic), every other choice. Passes, discards and card play
    stay the teacher's alone, so no one is taught to bid 13 at random.
    """
    if isinstance(action, Bid):
        others = [a for a in view.legal_actions if isinstance(a, Bid) and a.level == action.level]
    elif view.phase in EXPLORED:
        others = view.legal_actions
    else:
        return ()
    return tuple(action_index(a) for a in others if a != action)


def teacher_samples(teacher: Agent, deals: int, rng: random.Random) -> list[Sample]:
    """Every decision the teacher makes in self-play, skipping forced moves."""
    samples = []
    for _ in range(deals):
        deal = Deal.new(rng.randrange(4), rng)
        while not deal.is_over:
            view = deal.view(deal.to_act)
            action = teacher.choose(view)
            if len(view.legal_actions) > 1:
                observation = observe(view)
                compact = Observation(
                    np.asarray(observation.tokens, dtype=np.int16),
                    np.asarray(observation.legal, dtype=np.int16),
                )
                samples.append(Sample(compact, action_index(action), alternatives(view, action)))
            deal.apply(action)
    return samples


def groups(samples: list[Sample], size: int, rng: random.Random, shuffle: bool) -> list[list[int]]:
    """Index batches of similar length (less padding), in random order."""
    order = list(range(len(samples)))
    if shuffle:
        rng.shuffle(order)
    chunk = size * 32
    result = []
    for start in range(0, len(order), chunk):
        part = sorted(
            order[start : start + chunk], key=lambda i: len(samples[i].observation.tokens)
        )
        result += [part[i : i + size] for i in range(0, len(part), size)]
    if shuffle:
        rng.shuffle(result)
    return result


def target_odds(samples: list[Sample], explore: float) -> torch.Tensor:
    """What the network learns to choose: the teacher's action, with `explore` of it
    spread evenly over the sample's alternatives."""
    odds = torch.zeros(len(samples), NUM_ACTIONS)
    for i, sample in enumerate(samples):
        share = explore if sample.alternatives else 0.0
        odds[i, sample.target] = 1 - share
        if share:
            odds[i, list(sample.alternatives)] = share / len(sample.alternatives)
    return odds


def batches(
    samples: list[Sample], size: int, rng: random.Random, shuffle: bool = True, explore: float = 0
):
    """Inputs, the teacher's actions and the target odds, one batch at a time."""
    for group in groups(samples, size, rng, shuffle):
        chosen = [samples[i] for i in group]
        targets = torch.tensor([s.target for s in chosen])
        yield collate([s.observation for s in chosen]), targets, target_odds(chosen, explore)


def loss_against(logits: torch.Tensor, legal: torch.Tensor, odds: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with the target odds (illegal actions have odds and logits of 0 and -inf)."""
    log_probs = torch.log_softmax(logits, dim=-1).masked_fill(~legal, 0.0)
    return -(odds * log_probs).sum(-1).mean()


def train(
    net: Net,
    samples: list[Sample],
    epochs: int,
    rng: random.Random,
    lr: float = 3e-4,
    batch_size: int = 256,
    explore: float = 0.0,
) -> None:
    device = next(net.parameters()).device
    optimiser = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * len(groups(samples, batch_size, rng, shuffle=False))
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, lr, total_steps=steps)
    net.train()
    for epoch in range(epochs):
        started, total, count = time.perf_counter(), 0.0, 0
        for inputs, targets, odds in batches(samples, batch_size, rng, explore=explore):
            tokens, padding, legal = (t.to(device) for t in inputs)
            logits, _ = net(tokens, padding, legal)
            loss = loss_against(logits, legal, odds.to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            total, count = total + loss.item() * len(targets), count + len(targets)
        print(f"epoch {epoch + 1}: loss {total / count:.3f} ({time.perf_counter() - started:.0f}s)")
    net.eval()


@torch.no_grad()
def accuracy(net: Net, samples: list[Sample]) -> float:
    net.eval()
    device = next(net.parameters()).device
    correct = 0
    for inputs, targets, _ in batches(samples, 512, random.Random(0), shuffle=False):
        logits, _ = net(*(t.to(device) for t in inputs))
        correct += (logits.argmax(-1).cpu() == targets).sum().item()
    return correct / len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the network to copy RuleBot.")
    parser.add_argument("--deals", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--eval-deals", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/bc.pt"))
    parser.add_argument(
        "--explore", type=float, default=0.0, help="share of the target kept for alternatives"
    )
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    samples = teacher_samples(RuleBot(), args.deals, rng)
    held_out = teacher_samples(RuleBot(), max(1, args.deals // 10), rng)
    print(f"{len(samples)} training decisions ({time.perf_counter() - started:.0f}s)")

    net = Net(NetConfig()).to(args.device)
    train(net, samples, args.epochs, rng, explore=args.explore)
    net = net.cpu()
    print(f"held-out agreement with RuleBot: {accuracy(net, held_out):.1%}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save(net, str(args.out))
    export(net, str(args.out.with_suffix(".npz")))
    print(f"saved {args.out} and {args.out.with_suffix('.npz')} (for playing)")

    positions = random_positions(args.eval_deals, rng)
    agent = NetAgent(net)
    for name, field in [("RuleBot", RuleBot()), ("RandomBot", RandomBot(rng))]:
        print(f"network against a field of {name}:")
        print(duplicate(agent, field, positions).summary())


if __name__ == "__main__":
    main()
