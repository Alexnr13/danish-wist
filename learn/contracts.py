"""What contracts a policy bids, and how they go (LEARNING.md §4).

    python -m learn.contracts runs/bc.npz runs/rl-001/checkpoints/policy-*.npz
    python -m learn.contracts --field rule rule runs/rl-001/policy.npz
    python -m learn.contracts --phases runs/bc.npz runs/rl-001/policy.npz

Each policy (a name as in `learn.arena`) plays the same `--deals` deals,
greedily (or with `--sample`, drawing its moves as it does in training, which
shows the contracts it only sometimes tries): in all four seats (self-play,
the setting it trains in), or with `--field rule` in one seat at a time among
RuleBots. For the contracts it declares, the report gives how often each
kind (plain, Clubs, Flip, Halves) is bid, its mean level, how often it is
played alone and made, and the declarer's mean score. A last table puts
each policy on one row, which shows how the bidding changes over a run's
checkpoints.

`--phases` adds how each network's choices on RuleBot's own decisions differ,
phase by phase, from RuleBot's and from the first network's, and its entropy.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial

from danish_wist.bidding import NUM_PLAYERS
from danish_wist.bots import RuleBot
from danish_wist.game import Deal, Phase

from .arena import (
    Position,
    add_device_argument,
    make_agent,
    parse_with_device,
    random_positions,
)
from .runner import Runner

KINDS = ("plain", "clubs", "flip", "halves")


@dataclass
class Contracts:
    """The contracts one policy declared in `seats` seat-deals, as (kind, level, alone, score)."""

    seats: int = 0
    declared: list[tuple[str, int, bool, int]] = field(default_factory=list)

    def rows(self, kind: str) -> list[tuple[str, int, bool, int]]:
        return [c for c in self.declared if kind in (c[0], "all")]

    def table(self) -> str:
        lines = [f"  {'contract':<9}{'seats':>7}{'level':>7}{'alone':>7}{'made':>7}{'score':>9}"]
        for kind in (*KINDS, "all"):
            if rows := self.rows(kind):
                lines.append(
                    f"  {kind:<9}{len(rows) / self.seats:>7.1%}"
                    f"{_mean(r[1] for r in rows):>7.2f}{_mean(r[2] for r in rows):>7.0%}"
                    f"{_mean(r[3] > 0 for r in rows):>7.0%}{_mean(r[3] for r in rows):>+9.0f}"
                )
        return "\n".join(lines)

    def summary(self) -> list[float]:
        """Declarer share, mean level, shares of Flip/Halves/Clubs, alone, made, score."""
        rows = self.declared
        share = [_mean(r[0] == kind for r in rows) for kind in ("flip", "halves", "clubs")]
        return [
            len(rows) / self.seats,
            _mean(r[1] for r in rows),
            *share,
            _mean(r[2] for r in rows),
            _mean(r[3] > 0 for r in rows),
            _mean(r[3] for r in rows),
        ]


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def _contract(game: int, deal: Deal, agents: dict) -> tuple:
    """Runs in the worker when a game ends: its contract, or None after a redeal."""
    if deal.redeal:
        return game, None
    return game, (
        deal.declarer,
        contract_kind(deal.bid),
        deal.bid.level,
        deal.alone,
        deal.scores[deal.declarer],
    )


def contract_kind(bid) -> str:
    """plain, clubs, flip or halves."""
    return bid.attachment.value if bid.attachment else "plain"


def _agents(names: list[str], sample: bool, device: str | None, worker: int) -> dict:
    rng = random.Random(worker)
    agents = {name: make_agent(name, rng, device=device) for name in names}
    if sample:
        import numpy as np

        for agent in agents.values():
            if hasattr(agent, "temperature"):  # a network: draw its moves, reproducibly
                agent.temperature, agent.rng = 1.0, np.random.default_rng(worker)
    return agents


def study(
    names: list[str],
    positions: list[Position],
    field: str | None,
    workers: int = 1,
    sample: bool = False,
    device: str | None = None,
) -> dict[str, Contracts]:
    """The contracts each named policy declares on `positions` (see the module docstring)."""
    games, whose = [], []
    for name in names:
        for position in positions:
            if field is None:
                games.append((position, [name] * NUM_PLAYERS))
                whose.append((name, None))
            else:
                for seat in range(NUM_PLAYERS):
                    lineup = [field] * NUM_PLAYERS
                    lineup[seat] = name
                    games.append((position, lineup))
                    whose.append((name, seat))
    found = {name: Contracts() for name in names}
    make = partial(_agents, sorted({*names, field} - {None}), sample, device)
    with Runner(make, workers=workers) as runner:
        for game, contract in runner.play(games, _contract):
            name, seat = whose[game]
            found[name].seats += NUM_PLAYERS if seat is None else 1
            if contract is not None and seat in (None, contract[0]):
                found[name].declared.append(contract[1:])
    return found


GROUPS = {Phase.IRON_HAND: "auction", Phase.AUCTION: "auction", Phase.PLAY: "card play"}


def phases(paths: list[str], deals: int, rng: random.Random) -> str:
    """Each network on RuleBot's own decisions, by phase: how often it chooses as RuleBot
    does and as the first network does, and its entropy; and in the auction, its
    probability of bidding each kind of contract (which sampling in training explores)."""
    import numpy as np

    from danish_wist.bidding import Bid

    from .encoding import ACTIONS, action_index, observe
    from .inference import NumpyNet, collate

    bids = {
        kind: np.array([isinstance(a, Bid) and contract_kind(a) == kind for a in ACTIONS])
        for kind in KINDS
    }

    samples = defaultdict(list)
    bot = RuleBot()
    for position in random_positions(deals, rng):
        deal = position.start()
        while not deal.is_over:
            view = deal.view(deal.to_act)
            action = bot.choose(view)
            if len(view.legal_actions) > 1:
                group = GROUPS.get(view.phase, "contract")
                samples[group].append((observe(view), action_index(action)))
            deal.apply(action)
    groups = ["auction", "contract", "card play"]
    lines = [
        "as RuleBot / as the first / entropy; in the auction, the chance of bidding each kind",
        f"{'policy':<44}"
        + "".join(f"{f'{g} ({len(samples[g])})':>22}" for g in groups)
        + "".join(f"{kind:>8}" for kind in KINDS),
    ]
    first = {}
    for path in paths:
        net, cells = NumpyNet(path), []
        for group in groups:
            observations = [o for o, _ in samples[group]]
            logits = np.concatenate(
                [
                    net(*collate(observations[i : i + 512]))[0]
                    for i in range(0, len(observations), 512)
                ]
            )
            shifted = logits - logits.max(-1, keepdims=True)
            log_probs = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
            safe = np.where(np.isfinite(log_probs), log_probs, 0.0)  # illegal: 0 * -inf
            entropy = -(np.exp(log_probs) * safe).sum(-1)
            choices = logits.argmax(-1)
            first.setdefault(group, choices)
            teacher = np.array([a for _, a in samples[group]])
            agree, same = (choices == teacher).mean(), (choices == first[group]).mean()
            cells.append(f"{f'{agree:.0%} {same:.0%} {entropy.mean():.2f}':>22}")
            if group == "auction":
                chances = [np.exp(log_probs)[:, bids[kind]].sum(-1).mean() for kind in KINDS]
        cells += [f"{chance:>8.1%}" for chance in chances]
        lines.append(f"{path[-44:]:<44}" + "".join(cells))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="The contracts policies bid, and how they go.")
    parser.add_argument("policies", nargs="+", help="agent names, as in learn.arena")
    parser.add_argument("--field", help="play in one seat among these (e.g. rule), not self-play")
    parser.add_argument("--deals", type=int, default=1000)
    parser.add_argument("--phases", action="store_true", help="also compare choices by phase")
    parser.add_argument("--sample", action="store_true", help="networks draw moves as in training")
    parser.add_argument("--seed", type=int, default=0)
    add_device_argument(parser)
    args = parse_with_device(parser)

    positions = random_positions(args.deals, random.Random(args.seed))
    found = study(args.policies, positions, args.field, args.workers, args.sample, args.device)
    setting = f"among {args.field}" if args.field else "in self-play"
    setting += ", sampling" if args.sample else ""
    for name, contracts in found.items():
        print(f"{name} {setting}, {args.deals} deals: contracts it declared")
        print(contracts.table())
    print(
        f"\n{'policy':<44}{'declares':>9}{'level':>7}{'flip':>7}{'halves':>7}"
        f"{'clubs':>7}{'alone':>7}{'made':>7}{'score':>8}"
    )
    for name, contracts in found.items():
        declares, level, *rest, score = contracts.summary()
        cells = "".join(f"{x:>7.0%}" for x in rest)
        print(f"{name[-44:]:<44}{declares:>9.1%}{level:>7.2f}{cells}{score:>+8.0f}")
    if args.phases:
        networks = [name for name in args.policies if name.endswith(".npz")]
        print(f"\nChoices on RuleBot's own decisions ({args.deals} deals), by phase:")
        print(phases(networks, args.deals, random.Random(args.seed)))


if __name__ == "__main__":
    main()
