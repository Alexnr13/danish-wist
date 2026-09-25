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

With `--workers` above 1 the deals are spread over that many processes
(`learn.evaluate`); `--record` writes every deal for `learn.report`.

An agent is `random`, `rule`, `search` (RuleBot rollouts), a trained network
(`.npz`, played with NumPy), or `search:` plus a network, which searches with
that network for rollouts and beliefs.
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


def make_agent(name: str, rng: random.Random, worlds: int = 8) -> Agent:
    """An agent from its command-line name (see the module docstring)."""
    if name in AGENTS:
        return AGENTS[name](rng)
    if name.startswith("search:"):
        network = make_agent(name.removeprefix("search:"), rng)
        return SearchAgent(network, worlds=worlds, rng=rng, belief=network)
    if name.endswith(".npz"):
        from .inference import NumpyAgent  # needs NumPy, so only imported when asked for

        return NumpyAgent(name)
    raise ValueError(f"unknown agent {name!r}: use {', '.join(AGENTS)}, a .npz, or search:<.npz>")


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
    parser = argparse.ArgumentParser(
        description="Duplicate evaluation of one agent against a field."
    )
    parser.add_argument("--candidate", default="rule", help="see the module docstring")
    parser.add_argument("--field", default="random")
    parser.add_argument("--worlds", type=int, default=8, help="worlds per decision for search")
    parser.add_argument("--deals", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--record", type=Path, help="append every deal played to this file")
    args = parser.parse_args()

    from .evaluate import run  # imported here: learn.evaluate builds on this module

    result = run(
        args.candidate,
        args.field,
        args.deals,
        seed=args.seed,
        workers=args.workers,
        worlds=args.worlds,
        record=args.record,
    )
    print(f"{args.candidate} against a field of {args.field}")
    print(result.summary())


if __name__ == "__main__":
    main()
