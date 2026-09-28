"""Duplicate evaluation: how much better is an agent than a field of others?

Card luck dominates single deals, so every comparison replays the same cards.
For each deal the field plays all four seats once (the baseline). Then the
candidate replaces the field in each seat in turn. The candidate's
*advantage* in a seat is its score there minus the field's score in that same
seat on the same cards. Averaging over the four seats and many deals cancels
most of the luck.

    python -m learn.arena --candidate rule --field random --deals 1000
    python -m learn.arena --candidate runs/rl/policy.npz --field rule
    python -m learn.arena --candidate search:runs/rl/policy.npz --field rule
    python -m learn.arena --candidate rule --field random --workers 8 --record games.jsonl
    python -m learn.arena --candidate play:runs/rl/policy.npz --field rule  # card play alone
    python -m learn.arena --candidate runs/a.npz runs/b.npz --field rule --seeds 21 23

With `--workers` above 1 the deals are spread over that many processes
(`learn.evaluate`); `--record` writes every deal for `learn.report`. Several
candidates play the same deals (from each of `--seeds`, pooled), and each is
also compared with the first deal by deal: a much tighter interval than
either result's own.

An agent is `random`, `rule`, `search` (RuleBot rollouts), a trained network
(`.npz`, or `.pt` with PyTorch), or `search:` plus a network, which searches
with that network for rollouts and beliefs. `play:<agent>` has RuleBot bid
and set up the contract and the agent play the cards: the fixed-contract
card-play test (`learn.hybrid`, which also has `hybrid:<auction>,<contract>,<play>`).
`critic:<policy.pt>` searches card play one card deep in `--worlds` sampled
deals, judged by the critic saved beside the policy (`learn.critic_search`;
`critic-reply:` judges after the others' replies).
Networks play on `--device`: the GPU when PyTorch has one, else with NumPy
(`--device numpy`).
"""

from __future__ import annotations

import argparse
import math
import os
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from danish_wist.bidding import NUM_PLAYERS
from danish_wist.bots import Agent, RandomBot, RuleBot
from danish_wist.cards import Card, shuffled_deck
from danish_wist.game import CAT_SIZE, HAND_SIZE, Deal

from .search import SearchAgent

AGENTS: dict[str, Callable[[random.Random], Agent]] = {
    "random": RandomBot,
    "rule": lambda rng: RuleBot(),
    "search": lambda rng: SearchAgent(RuleBot(), worlds=8, rng=rng),
}


def make_agent(name: str, rng: random.Random, worlds: int = 8, device: str | None = None) -> Agent:
    """An agent from its command-line name (see the module docstring).

    A network plays with NumPy, or with PyTorch on `device` if one is given:
    on a GPU that is many times faster.
    """
    if name in AGENTS:
        return AGENTS[name](rng)
    if name.startswith("search:"):
        network = make_agent(name.removeprefix("search:"), rng, device=device)
        return SearchAgent(network, worlds=worlds, rng=rng, belief=network)
    if name.startswith(("critic:", "critic-reply:")):
        from .critic_search import CriticSearch  # needs PyTorch

        kind, policy = name.split(":", 1)
        reply = kind == "critic-reply"
        return CriticSearch.load(policy, device or "cpu", worlds=worlds, rng=rng, reply=reply)
    if name.startswith(("play:", "hybrid:")):
        from .hybrid import Hybrid

        names = name.split(":", 1)[1].split(",")
        if name.startswith("play:"):
            names = ["rule", "rule", names[0]] if len(names) == 1 else []
        if len(names) != 3:
            raise ValueError(f"{name!r}: use play:<agent> or hybrid:<auction>,<contract>,<play>")
        made = {n: make_agent(n, rng, worlds, device) for n in dict.fromkeys(names)}
        return Hybrid(*(made[n] for n in names))
    if name.endswith((".npz", ".pt")) and device is not None:
        from .model import NetAgent, load  # needs PyTorch, so only imported when asked for

        return NetAgent(load(name).to(device), rng=rng)
    if name.endswith(".npz"):
        from .inference import NumpyAgent  # needs NumPy, so only imported when asked for

        return NumpyAgent(name)
    raise ValueError(f"unknown agent {name!r}: use {', '.join(AGENTS)}, a .npz, or search:<.npz>")


def default_device() -> str | None:
    """Where networks play unless told: an NVIDIA GPU if PyTorch has one, else NumPy (None)."""
    try:
        import torch
    except ImportError:
        return None
    return "cuda" if torch.cuda.is_available() else None


# Networks on a GPU make playing a matter of the GPU switching between the workers'
# processes more than of cores: on the RTX 5090, 12 workers play as fast as 22 and hold
# 12 GB of its memory instead of 21 (PERFORMANCE.md).
GPU_WORKERS = 12


def default_workers(device: str | None) -> int:
    """How many processes play: all cores but two, and at most `GPU_WORKERS` on a GPU."""
    cores = max(1, (os.cpu_count() or 2) - 2)
    return min(cores, GPU_WORKERS) if device is not None and device.startswith("cuda") else cores


def add_device_argument(parser: argparse.ArgumentParser) -> None:
    """`--device` for the networks, and `--workers` (default: `default_workers` for it)."""
    parser.add_argument(
        "--device",
        type=lambda name: None if name == "numpy" else name,
        default=default_device(),
        help="where networks play: cuda (the default when there is one), cpu, or numpy",
    )
    parser.add_argument("--workers", type=int, help="processes to play on (default: see above)")


def parse_with_device(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()
    args.workers = args.workers or default_workers(args.device)
    return args


ROLES = ("declarer", "partner", "defender", "redeal")


@dataclass(frozen=True)
class Position:
    """The starting cards of a deal, so it can be played again and again."""

    dealer: int
    hands: tuple[tuple[Card, ...], ...]
    cat: tuple[Card, ...]

    def start(self) -> Deal:
        return Deal(self.dealer, [list(hand) for hand in self.hands], list(self.cat))


def random_positions(count: int, rng: random.Random) -> list[Position]:
    positions = []
    for i in range(count):
        deck = shuffled_deck(rng)
        hands = tuple(tuple(deck[s * HAND_SIZE : (s + 1) * HAND_SIZE]) for s in range(NUM_PLAYERS))
        positions.append(Position(i % NUM_PLAYERS, hands, tuple(deck[-CAT_SIZE:])))
    return positions


def play(position: Position, agents: Sequence[Agent]) -> Deal:
    """Play one deal to the end with `agents[seat]` in each seat."""
    deal = position.start()
    while not deal.is_over:
        seat = deal.to_act
        deal.apply(agents[seat].choose(deal.view(seat)))
    return deal


def role(deal: Deal, seat: int) -> str:
    if deal.redeal:
        return "redeal"
    if seat == deal.declarer:
        return "declarer"
    return "partner" if seat == deal.partner else "defender"


@dataclass
class Result:
    """Advantages of a candidate over a field, in points per deal."""

    per_deal: list[float] = field(default_factory=list)  # mean over the four seats
    by_role: dict[str, list[int]] = field(default_factory=lambda: {r: [] for r in ROLES})

    @property
    def mean(self) -> float:
        return sum(self.per_deal) / len(self.per_deal)

    @property
    def ci95(self) -> float:
        """Half-width of a 95% confidence interval for the mean."""
        n = len(self.per_deal)
        if n < 2:
            return math.inf
        variance = sum((x - self.mean) ** 2 for x in self.per_deal) / (n - 1)
        return 1.96 * math.sqrt(variance / n)

    def summary(self) -> str:
        n = len(self.per_deal)
        lines = [f"advantage per deal: {self.mean:+.1f} ± {self.ci95:.1f} (95% CI, {n} deals)"]
        for name, values in self.by_role.items():
            if values:
                share = len(values) / (NUM_PLAYERS * n)
                mean = sum(values) / len(values)
                lines.append(f"  as {name:<8} {share:6.1%} of seats, advantage {mean:+.1f}")
        return "\n".join(lines)


def duplicate(candidate: Agent, field_agent: Agent, positions: Sequence[Position]) -> Result:
    """Compare `candidate` with `field_agent` by duplicate play (see module docstring)."""
    result = Result()
    for position in positions:
        baseline = play(position, [field_agent] * NUM_PLAYERS).scores
        seat_advantages = []
        for seat in range(NUM_PLAYERS):
            agents = [field_agent] * NUM_PLAYERS
            agents[seat] = candidate
            deal = play(position, agents)
            advantage = deal.scores[seat] - baseline[seat]
            seat_advantages.append(advantage)
            result.by_role[role(deal, seat)].append(advantage)
        result.per_deal.append(sum(seat_advantages) / NUM_PLAYERS)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Duplicate evaluation of agents against a field.")
    parser.add_argument(
        "--candidate",
        nargs="+",
        default=["rule"],
        help="one or more agents (see the module docstring); several are paired with the first",
    )
    parser.add_argument("--field", default="random")
    parser.add_argument("--worlds", type=int, default=8, help="worlds per decision for search")
    parser.add_argument("--deals", type=int, default=1000, help="deals from each seed")
    parser.add_argument(
        "--seed", "--seeds", dest="seeds", type=int, nargs="+", default=[0], help="pooled"
    )
    parser.add_argument("--record", type=Path, help="append every deal played to this file")
    add_device_argument(parser)
    args = parse_with_device(parser)

    from .curve import paired
    from .evaluate import screen  # imported here: learn.evaluate builds on this module

    results = screen(
        args.candidate,
        args.field,
        args.deals,
        args.seeds,
        workers=args.workers,
        worlds=args.worlds,
        record=args.record,
        device=args.device,
    )
    for name, result in results.items():
        print(f"{name} against a field of {args.field}, seeds {' '.join(map(str, args.seeds))}")
        print(result.summary())
    (first, baseline), *others = results.items()
    if others:
        print(f"\npaired with {first}, per deal:")
    for name, result in others:
        mean, half = paired(baseline.per_deal, result.per_deal)
        print(f"  {name}: {mean:+.1f} ± {half:.1f}")


if __name__ == "__main__":
    main()
