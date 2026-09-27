"""Duplicate evaluation on all cores, optionally recording every deal.

The same comparison as `learn.arena.duplicate` (each position played once by
the field and then with the candidate in each seat), but run through
`learn.runner.Runner`: games are spread over worker processes, and each
agent decides a batch at a time. With `record`, every deal is written as a
replayable record (`danish_wist.record`, JSON Lines) whose `meta` names the
agent in each seat, for `learn.report` to analyse.

`python -m learn.arena ... --workers 8 --record games.jsonl` uses this.
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


def evaluate(
    runner: Runner,
    candidate: str,
    field: str,
    positions: Sequence[Position],
    record: Path | None = None,
) -> Result:
    """Duplicate play between two named agents of `runner`; see the module docstring."""
    games = []
    for position in positions:
        games.append((position, [field] * NUM_PLAYERS))
        for seat in range(NUM_PLAYERS):
            lineup = [field] * NUM_PLAYERS
            lineup[seat] = candidate
            games.append((position, lineup))
    finish = partial(_outcome, record=record is not None)
    played = {game: rest for game, *rest in runner.play(games, finish)}

    if record is not None:
        with record.open("a") as out:
            for game in range(len(games)):
                entry = played[game][2]
                entry["meta"] = {"seats": list(games[game][1]), "position": game // 5}
                out.write(json.dumps(entry) + "\n")

    result = Result()
    for i in range(len(positions)):
        baseline = played[5 * i][0]
        advantages = []
        for seat in range(NUM_PLAYERS):
            scores, roles, _ = played[5 * i + 1 + seat]
            advantages.append(scores[seat] - baseline[seat])
            result.by_role[roles[seat]].append(advantages[-1])
        result.per_deal.append(sum(advantages) / NUM_PLAYERS)
    return result


def named_agents(
    worker: int, names: Sequence[str], worlds: int, seed: int, device: str | None = None
) -> dict:
    """The agents for command-line names, made in each worker with its own randomness."""
    rng = random.Random(seed * 1009 + worker)
    return {name: make_agent(name, rng, worlds, device) for name in dict.fromkeys(names)}


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
    positions = random_positions(deals, random.Random(seed))
    make = partial(named_agents, names=[candidate, field], worlds=worlds, seed=seed, device=device)
    in_flight = 64 if device is None else 256  # a GPU gains from bigger batches; one core does not
    with Runner(make, workers=workers, games_in_flight=in_flight) as runner:
        return evaluate(runner, candidate, field, positions, record)
