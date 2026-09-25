"""Play many deals at once: in batches, and on all cores (see PERFORMANCE.md).

Neural agents are fast only in batches, so each process keeps many deals in
flight. Every round it collects the pending decisions, asks each agent for all
of its decisions in one `choose_batch` call, applies them, and replaces
finished deals with new ones from the stream of positions.

    from learn.arena import random_positions
    from learn.runner import play_many

    for deal in play_many(random_positions(10_000, rng), [RuleBot()] * 4):
        print(deal.scores)

A finished deal's `Position` is `Position(deal.dealer, tuple(deal.initial_hands),
tuple(deal.cat))`, for matching results to positions.
"""

from __future__ import annotations

import gc
import multiprocessing
import os
import pickle
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import islice
from typing import Any, Protocol

from danish_wist.actions import Action
from danish_wist.bots import Agent
from danish_wist.game import Deal, PlayerView
from learn.arena import Position

CHUNKS_PER_WORKER = 2  # chunks of positions queued per worker, to keep each one busy


class BatchAgent(Protocol):
    """An agent that decides many views in one call, e.g. one network evaluation."""

    def choose_batch(self, views: list[PlayerView]) -> list[Action]: ...


class Batched:
    """A plain `Agent` as a `BatchAgent`."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent

    def choose_batch(self, views: list[PlayerView]) -> list[Action]:
        return [self.agent.choose(view) for view in views]


AnyAgent = Agent | BatchAgent
AgentsBySeat = Sequence[AnyAgent] | Callable[[int], Sequence[AnyAgent]]


def play_many(
    positions: Iterable[Position],
    agents_by_seat: AgentsBySeat,
    *,
    games_in_flight: int = 256,
    workers: int | None = None,
    finish: Callable[[Deal], Any] | None = None,
) -> Iterator[Any]:
    """Play every position to the end, yielding the finished deals in any order.

    `agents_by_seat` is the four agents, one per seat; the same agent object
    in several seats gets one batch for all of them. Plain `Agent`s are wrapped.
    It may instead be a function that makes the four agents, given a worker
    number (0, 1, ...): use that for agents holding randomness, so each worker
    gets its own, or holding state too big to send to every worker.

    `workers` is the number of processes, by default one per core. Each keeps
    up to `games_in_flight` deals going and gets its own copy of the agents
    (sent by pickling), so worker results cannot update agents here. With
    `workers=1` everything runs in this process, lazily, with these agents.
    `positions` is read lazily in both cases, so it may be endless.

    `finish`, if given, is applied to each finished deal where it was played,
    and its result is yielded instead of the deal. Sending a whole deal back
    from a worker costs about 25 µs to pickle there and 40 µs to unpickle and
    collect here, which limits all cores to some 15,000-20,000 deals/s. So
    send back only what you need; the result must pickle.
    """
    if workers is None:
        workers = os.cpu_count() or 1
    if workers <= 1:
        deals = _play(positions, _agents(agents_by_seat, 0), games_in_flight)
        return map(finish, deals) if finish else deals
    return _play_in_workers(positions, agents_by_seat, games_in_flight, workers, finish)


def _play(
    positions: Iterable[Position], agents: Sequence[AnyAgent], games_in_flight: int
) -> Iterator[Deal]:
    """The batched loop, in this process."""
    groups = _group_seats(agents)
    positions = iter(positions)
    in_flight = [position.start() for position in islice(positions, games_in_flight)]
    while in_flight:
        for agent, seats in groups:
            if len(groups) == 1:
                waiting = in_flight  # one agent plays every seat
            else:
                waiting = [deal for deal in in_flight if deal.to_act in seats]
            if waiting:
                actions = agent.choose_batch([deal.view(deal.to_act) for deal in waiting])
                for deal, action in zip(waiting, actions, strict=True):
                    deal.apply(action)
        playing = []
        for deal in in_flight:
            if deal.is_over:
                yield deal
            else:
                playing.append(deal)
        free = games_in_flight - len(playing)
        in_flight = playing + [position.start() for position in islice(positions, free)]


def _group_seats(agents: Sequence[AnyAgent]) -> list[tuple[BatchAgent, set[int]]]:
    """Each distinct agent, as a `BatchAgent`, with the seats it plays."""
    assert len(agents) == 4, "one agent per seat"
    groups: dict[int, tuple[AnyAgent, set[int]]] = {}
    for seat, agent in enumerate(agents):
        groups.setdefault(id(agent), (agent, set()))[1].add(seat)
    return [
        (agent if hasattr(agent, "choose_batch") else Batched(agent), seats)
        for agent, seats in groups.values()
    ]


def _agents(agents_by_seat: AgentsBySeat, worker: int) -> Sequence[AnyAgent]:
    return agents_by_seat(worker) if callable(agents_by_seat) else agents_by_seat


# --- Worker processes --------------------------------------------------------

_worker_agents: Sequence[AnyAgent] = ()
_worker_finish: Callable[[Deal], Any] | None = None


def _start_worker(agents_by_seat: AgentsBySeat, finish, numbers) -> None:
    global _worker_agents, _worker_finish
    _worker_agents, _worker_finish = _agents(agents_by_seat, numbers.get()), finish


def _play_chunk(positions: list[Position], games_in_flight: int) -> bytes:
    deals = _play(positions, _worker_agents, games_in_flight)
    results = list(map(_worker_finish, deals) if _worker_finish else deals)
    return pickle.dumps(results, pickle.HIGHEST_PROTOCOL)


def _unpickle(data: bytes) -> list[Any]:
    """Unpickle a chunk of results with the garbage collector paused.

    Otherwise collections keep starting part-way through and scanning the new
    objects again and again: for whole deals that is over half the time spent
    here, and this process is what limits all cores.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        return pickle.loads(data)
    finally:
        if was_enabled:
            gc.enable()


def _play_in_workers(
    positions: Iterable[Position],
    agents_by_seat: AgentsBySeat,
    games_in_flight: int,
    workers: int,
    finish: Callable[[Deal], Any] | None,
) -> Iterator[Any]:
    # Chunks big enough to refill the deals in flight a few times, but small
    # enough that a short list of positions is still shared by every worker.
    size = 4 * games_in_flight
    if isinstance(positions, Sequence):
        size = max(1, min(size, -(-len(positions) // workers)))
    positions = iter(positions)
    chunks = iter(lambda: list(islice(positions, size)), [])

    context = multiprocessing.get_context("spawn")  # macOS's default; safe everywhere
    numbers = context.Queue()
    for number in range(workers):
        numbers.put(number)
    pool = ProcessPoolExecutor(
        workers,
        mp_context=context,
        initializer=_start_worker,
        initargs=(agents_by_seat, finish, numbers),
    )
    pending = set()
    try:
        for chunk in chunks:
            pending.add(pool.submit(_play_chunk, chunk, games_in_flight))
            while len(pending) >= CHUNKS_PER_WORKER * workers:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield from _unpickle(future.result())
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield from _unpickle(future.result())
    finally:
        pool.shutdown(cancel_futures=True)
