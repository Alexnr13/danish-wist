"""Duplicate evaluation: how much better is an agent than a field of others?

Card luck dominates single deals, so every comparison replays the same cards.
For each deal the field plays all four seats once (the baseline). Then the
candidate replaces the field in each seat in turn. The candidate's
*advantage* in a seat is its score there minus the field's score in that same
seat on the same cards. Averaging over the four seats and many deals cancels
most of the luck.

    python -m learn.arena --candidate rule --field random --deals 1000
"""

from __future__ import annotations

import argparse
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from danish_wist.bidding import NUM_PLAYERS
from danish_wist.bots import Agent, RandomBot, RuleBot
from danish_wist.cards import Card, shuffled_deck
from danish_wist.game import CAT_SIZE, HAND_SIZE, Deal

AGENTS: dict[str, Callable[[random.Random], Agent]] = {
    "random": RandomBot,
    "rule": lambda rng: RuleBot(),
}

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
    parser.add_argument("--candidate", choices=AGENTS, default="rule")
    parser.add_argument("--field", choices=AGENTS, default="random")
    parser.add_argument("--deals", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    positions = random_positions(args.deals, rng)
    result = duplicate(AGENTS[args.candidate](rng), AGENTS[args.field](rng), positions)
    print(f"{args.candidate} against a field of {args.field}")
    print(result.summary())


if __name__ == "__main__":
    main()
