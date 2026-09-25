"""Behaviour cloning: train the network to copy a teacher bot (LEARNING.md §5, step 3).

This proves the whole pipeline (encoding, network, training, playing) before
self-play, and gives self-play a sensible starting policy.

    python -m learn.imitate --deals 5000 --epochs 3 --out runs/bc.pt
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from danish_wist.bots import Agent, RandomBot, RuleBot
from danish_wist.game import Deal

from .arena import duplicate, random_positions
from .encoding import Observation, action_index, observe
from .model import Net, NetAgent, NetConfig, collate, export, save


@dataclass(frozen=True)
class Sample:
    observation: Observation  # tokens and legal indices stored as small NumPy arrays
    target: int  # the teacher's action index


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
                samples.append(Sample(compact, action_index(action)))
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


def batches(samples: list[Sample], size: int, rng: random.Random, shuffle: bool = True):
    for group in groups(samples, size, rng, shuffle):
        chosen = [samples[i] for i in group]
        yield collate([s.observation for s in chosen]), torch.tensor([s.target for s in chosen])


def train(
    net: Net,
    samples: list[Sample],
    epochs: int,
    rng: random.Random,
    lr: float = 3e-4,
    batch_size: int = 256,
) -> None:
    optimiser = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * len(groups(samples, batch_size, rng, shuffle=False))
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, lr, total_steps=steps)
    net.train()
    for epoch in range(epochs):
        started, total, count = time.perf_counter(), 0.0, 0
        for inputs, targets in batches(samples, batch_size, rng):
            logits, _ = net(*inputs)
            loss = F.cross_entropy(logits, targets)
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
    correct = 0
    for inputs, targets in batches(samples, 512, random.Random(0), shuffle=False):
        logits, _ = net(*inputs)
        correct += (logits.argmax(-1) == targets).sum().item()
    return correct / len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the network to copy RuleBot.")
    parser.add_argument("--deals", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--eval-deals", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/bc.pt"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    samples = teacher_samples(RuleBot(), args.deals, rng)
    held_out = teacher_samples(RuleBot(), max(1, args.deals // 10), rng)
    print(f"{len(samples)} training decisions ({time.perf_counter() - started:.0f}s)")

    net = Net(NetConfig())
    train(net, samples, args.epochs, rng)
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
