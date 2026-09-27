"""What a policy's bids were worth against the alternatives (LEARNING.md §4).

    python -m learn.margins runs/rl-003/checkpoints/policy-0110.npz
    python -m learn.margins runs/rl-003/checkpoints/policy-0110.npz --field rule --nth 2

The policy (a name as in `learn.arena`, played greedily) plays `--deals` deals
in all four seats, or with `--field` in one seat at a time among that agent.
Every deal is played four times for each seat, the same except for that
seat's `--nth` decision in the auction: its own choice, a pass, and the same
kind of contract one and two levels higher. Everything else is played as
before, so the differences come from that one bid. The report splits them by
the seat's aces and Jokers:

- **bid - pass**: what bidding gained over passing, where the policy bid.
- **+1 - bid**, **+2 - bid**: what bidding one or two levels higher would
  have gained, where that bid was allowed.

It also gives how often the seat's own contracts were made in each branch.
Bidding that fits the policy's own play shows bid - pass rising with the hand
and +1 and +2 below zero except with the strongest hands. If +1 turns
positive for strong hands while the policy still does not raise, its bidding
lags its play (TRAINING.md). The agents must be deterministic: a network is
played greedily, and RandomBot cannot be probed.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

from danish_wist.actions import Action, Pass, encode
from danish_wist.bidding import MAX_LEVEL, NUM_PLAYERS, Bid
from danish_wist.cards import ACE
from danish_wist.game import Deal, Phase

from .arena import (
    Position,
    add_device_argument,
    make_agent,
    parse_with_device,
    random_positions,
)
from .runner import Runner

BRANCHES = ("own", "pass", "up1", "up2")
POLICY, FIELD = "policy", "field"


def plan(game: int) -> tuple[int, str]:
    """The seat and branch of game number `game`: each seat has one game per branch."""
    return game // len(BRANCHES) % NUM_PLAYERS, BRANCHES[game % len(BRANCHES)]


def top_cards(hand) -> int:
    """Aces and Jokers: a rough measure of a hand's strength (as in `learn.report`)."""
    return sum(card.is_joker or card.rank == ACE for card in hand)


def alternative(own: Action, branch: str, legal: Sequence[Action]) -> Action | None:
    """The move `branch` plays instead of the policy's own bid, or None if there is none."""
    if not isinstance(own, Bid) or branch == "own":
        return None
    if branch == "pass":
        return Pass()
    level = own.level + (1 if branch == "up1" else 2)
    higher = Bid(level, own.attachment) if level <= MAX_LEVEL else None
    return higher if higher in legal else None


class Branching:
    """Plays like `agent`, except at the planned seat's `nth` auction decision."""

    def __init__(self, agent, nth: int) -> None:
        self.agent, self.nth = agent, nth
        self.counts: dict[int, int] = defaultdict(int)
        self.found: dict[int, tuple[Action, int, bool]] = {}  # game: (own, top cards, applies)

    def choose_decisions(self, decisions) -> list[Action]:
        views = [d.view for d in decisions]
        if hasattr(self.agent, "choose_batch"):
            actions = self.agent.choose_batch(views)
        else:
            actions = [self.agent.choose(view) for view in views]
        for i, d in enumerate(decisions):
            seat, branch = plan(d.game)
            if d.seat != seat or d.view.phase is not Phase.AUCTION:
                continue
            self.counts[d.game] += 1
            if self.counts[d.game] == self.nth:
                own = actions[i]
                instead = alternative(own, branch, d.view.legal_actions)
                self.found[d.game] = (own, top_cards(d.view.hand), instead is not None)
                actions[i] = instead or own
        return actions


def _agents(
    worker: int, policy: str, field: str | None, nth: int, seed: int, device: str | None = None
) -> dict:
    rng = random.Random(seed * 1009 + worker)
    agents = {POLICY: Branching(make_agent(policy, rng, device=device), nth)}
    if field is not None:
        agents[FIELD] = make_agent(field, rng, device=device)
    return agents


def _finish(game: int, deal: Deal, agents: dict) -> tuple:
    """Runs in the worker when a game ends: the seat's score and what happened at the bid."""
    seat, _ = plan(game)
    branching = agents[POLICY]
    branching.counts.pop(game, None)
    found = branching.found.pop(game, None)
    bids = [action for s, action in deal.history if s == seat and isinstance(action, Bid | Pass)]
    played = encode(bids[branching.nth - 1]) if len(bids) >= branching.nth else None
    declared = not deal.redeal and deal.declarer == seat
    return game, deal.scores[seat], found, played, declared, declared and deal.scores[seat] > 0


@dataclass
class Row:
    """One seat of one deal, played in every branch."""

    deal: int
    seat: int
    top: int  # the seat's aces and Jokers
    own: Action  # the policy's own choice at the probed decision
    scores: dict[str, int]  # the seat's score, by branch
    applies: dict[str, bool]  # whether the branch played something else
    played: dict[str, str]  # the move made at the probed decision, by branch
    made: dict[str, bool | None]  # whether the seat's own contract was made (None: not declarer)


def probe(
    policy: str,
    positions: Sequence[Position],
    field: str | None = None,
    nth: int = 1,
    workers: int = 1,
    seed: int = 0,
    device: str | None = None,
) -> list[Row]:
    """Play every position in every branch; one row per seat that reached its `nth` bid."""
    games = []
    for position in positions:
        for seat in range(NUM_PLAYERS):
            lineup = [POLICY if field is None else FIELD] * NUM_PLAYERS
            lineup[seat] = POLICY
            games += [(position, lineup)] * len(BRANCHES)
    make = partial(_agents, policy=policy, field=field, nth=nth, seed=seed, device=device)
    in_flight = 64 if device is None else 256  # a GPU gains from bigger batches; one core does not
    with Runner(make, workers=workers, games_in_flight=in_flight) as runner:
        results = {game: rest for game, *rest in runner.play(games, _finish)}
    rows = []
    for first in range(0, len(games), len(BRANCHES)):
        branches = {b: results[first + i] for i, b in enumerate(BRANCHES)}
        if branches["own"][1] is None:
            continue  # thrown in (iron hand) before the seat's nth decision in the auction
        own, top, _ = branches["own"][1]
        rows.append(
            Row(
                deal=first // (len(BRANCHES) * NUM_PLAYERS),
                seat=plan(first)[0],
                top=top,
                own=own,
                scores={b: r[0] for b, r in branches.items()},
                # A branch counts only if it met the same decision and played something else.
                applies={
                    b: r[1] is not None and r[1][0] == own and r[1][2] for b, r in branches.items()
                },
                played={b: r[2] for b, r in branches.items()},
                made={b: r[4] if r[3] else None for b, r in branches.items()},
            )
        )
    return rows


def _mean_ci(values: list[float]) -> str:
    if len(values) < 2:
        return f"{'n/a':>13}"
    mean = sum(values) / len(values)
    spread = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    return f"{mean:+7.0f} ±{1.96 * spread / math.sqrt(len(values)):4.0f}"


BUCKETS = [(str(n), lambda top, n=n: top == n) for n in range(4)] + [
    ("4+", lambda top: top >= 4),
    ("all", lambda top: True),
]


def table(rows: list[Row]) -> str:
    lines = [
        f"  {'aces+Jokers':>11}{'seats':>7}{'bid':>6}{'bid - pass':>15}"
        f"{'+1 - bid':>15}{'+2 - bid':>15}"
    ]
    for label, test in BUCKETS:
        chosen = [r for r in rows if test(r.top)]
        if not chosen:
            continue
        bids = [r for r in chosen if isinstance(r.own, Bid)]
        cells = []
        for better, worse, forced in (
            ("own", "pass", "pass"),
            ("up1", "own", "up1"),
            ("up2", "own", "up2"),
        ):
            diffs = [r.scores[better] - r.scores[worse] for r in bids if r.applies[forced]]
            cells.append(f"{_mean_ci(diffs):>15}")
        lines.append(
            f"  {label:>11}{len(chosen):>7}{len(bids) / len(chosen):>6.0%}{''.join(cells)}"
        )
    made = []
    for branch, name in (("own", "as bid"), ("up1", "one level up"), ("up2", "two levels up")):
        found = [
            r.made[branch]
            for r in rows
            if r.made[branch] is not None and (branch == "own" or r.applies[branch])
        ]
        if found:
            made.append(f"{name} {sum(found) / len(found):.0%}")
    lines.append("  its own contracts made: " + ", ".join(made))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="What a policy's bids were worth.")
    parser.add_argument("policy", help="an agent name, as in learn.arena (greedy)")
    parser.add_argument("--field", help="play in one seat among this agent, not self-play")
    parser.add_argument("--nth", type=int, default=1, help="probe the seat's nth bid (default 1)")
    parser.add_argument("--deals", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    add_device_argument(parser)
    args = parse_with_device(parser)
    positions = random_positions(args.deals, random.Random(args.seed))
    rows = probe(args.policy, positions, args.field, args.nth, args.workers, args.seed, args.device)
    where = f"among {args.field}" if args.field else "in self-play"
    print(f"{args.policy} {where}, decision {args.nth} in the auction, {args.deals} deals:")
    print(table(rows))


if __name__ == "__main__":
    main()
