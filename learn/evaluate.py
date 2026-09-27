"""Duplicate evaluation on all cores, optionally recording every deal.

The same comparison as `learn.arena.duplicate` (each position played once by
the field and then with the candidate in each seat), but run through
`learn.runner.Runner`: games are spread over worker processes, and each
agent decides a batch at a time. With `record`, every deal is written as a
replayable record (`danish_wist.record`, JSON Lines) whose `meta` names the
agent in each seat, for `learn.report` to analyse.

`python -m learn.arena ... --workers 8 --record games.jsonl` uses this, and
with several candidates or seeds (`screen`) plays them all on the same deals.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from functools import partial
from pathlib import Path

from danish_wist.bidding import NUM_PLAYERS
from danish_wist.game import Deal
from danish_wist.record import to_record

from .arena import Position, Result, make_agent, random_positions, role
from .runner import Runner


def _outcome(game: int, deal: Deal, agents: dict, record: bool) -> tuple:
    """Runs in the worker when a game ends: what the evaluation needs, and the record."""
    roles = [role(deal, seat) for seat in range(NUM_PLAYERS)]
    return game, deal.scores, roles, to_record(deal) if record else None


def evaluate_all(
    runner: Runner,
    candidates: Sequence[str],
    field: str,
    positions: Sequence[Position],
    record: Path | None = None,
    seed: int | None = None,
) -> dict[str, Result]:
    """Duplicate play of each named candidate against `field`, all agents of `runner`.

    Every position is played once by the field alone, the baseline all the
    candidates share, and then with each candidate in each seat, all in one
    play. `seed` only goes into the records, to tell their positions apart.
    """
    candidates = list(dict.fromkeys(candidates))
    games = []
    for position in positions:
        games.append((position, [field] * NUM_PLAYERS))
        for candidate in candidates:
            for seat in range(NUM_PLAYERS):
                lineup = [field] * NUM_PLAYERS
                lineup[seat] = candidate
                games.append((position, lineup))
    each = 1 + NUM_PLAYERS * len(candidates)  # games per position
    finish = partial(_outcome, record=record is not None)
    played = {game: rest for game, *rest in runner.play(games, finish)}

    if record is not None:
        with record.open("a") as out:
            for game in range(len(games)):
                entry = played[game][2]
                entry["meta"] = {"seats": list(games[game][1]), "position": game // each}
                if seed is not None:
                    entry["meta"]["seed"] = seed
                out.write(json.dumps(entry) + "\n")

    results = {candidate: Result() for candidate in candidates}
    for i in range(len(positions)):
        baseline = played[each * i][0]
        for k, candidate in enumerate(candidates):
            advantages = []
            for seat in range(NUM_PLAYERS):
                scores, roles, _ = played[each * i + 1 + NUM_PLAYERS * k + seat]
                advantages.append(scores[seat] - baseline[seat])
                results[candidate].by_role[roles[seat]].append(advantages[-1])
            results[candidate].per_deal.append(sum(advantages) / NUM_PLAYERS)
    return results


def evaluate(
    runner: Runner,
    candidate: str,
    field: str,
    positions: Sequence[Position],
    record: Path | None = None,
) -> Result:
    """Duplicate play between two named agents of `runner`; see the module docstring."""
    return evaluate_all(runner, [candidate], field, positions, record)[candidate]


def named_agents(
    worker: int, names: Sequence[str], worlds: int, seed: int, device: str | None = None
) -> dict:
    """The agents for command-line names, made in each worker with its own randomness."""
    rng = random.Random(seed * 1009 + worker)
    return {name: make_agent(name, rng, worlds, device) for name in dict.fromkeys(names)}


def screen(
    candidates: Sequence[str],
    field: str,
    deals: int,
    seeds: Sequence[int] = (0,),
    *,
    workers: int = 1,
    worlds: int = 8,
    record: Path | None = None,
    device: str | None = None,
) -> dict[str, Result]:
    """Command-line agents against `field` on `deals` positions from each seed, pooled.

    Every candidate plays the same deals, so any two are best compared per
    deal (`learn.curve.paired`). Networks play on `device`.
    """
    names = list(dict.fromkeys(candidates))
    make = partial(named_agents, names=[*names, field], worlds=worlds, seed=seeds[0], device=device)
    pooled = {name: Result() for name in names}
    in_flight = 64 if device is None else 256  # a GPU gains from bigger batches; one core does not
    with Runner(make, workers=workers, games_in_flight=in_flight) as runner:
        for seed in seeds:
            positions = random_positions(deals, random.Random(seed))
            for name, found in evaluate_all(runner, names, field, positions, record, seed).items():
                pooled[name].per_deal += found.per_deal
                for role_name, values in found.by_role.items():
                    pooled[name].by_role[role_name] += values
    return pooled


def run(
    candidate: str,
    field: str,
    deals: int,
    *,
    seed: int = 0,
    workers: int = 1,
    worlds: int = 8,
    record: Path | None = None,
    device: str | None = None,
) -> Result:
    """Compare two command-line agents over `deals` random positions (networks on `device`)."""
    return screen(
        [candidate],
        field,
        deals,
        [seed],
        workers=workers,
        worlds=worlds,
        record=record,
        device=device,
    )[candidate]
