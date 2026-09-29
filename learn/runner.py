"""Play many deals at once: in batches, and on all cores (see PERFORMANCE.md).

Neural agents are fast only in batches, so each process keeps many deals in
flight. Every round it collects the pending decisions, asks each agent for all
of its decisions in one call, applies them, and replaces finished deals with
new ones from the stream of games.

A `Runner` keeps its worker processes, and the agents in them, between calls
to `play`, so a training loop pays for starting them once:

    def make_agents(worker):          # runs once in each worker
        return {"learner": Learner(worker), "rule": RuleBot()}

    with Runner(make_agents, workers=8) as runner:
        for iteration in range(100):
            runner.broadcast("learner", "load", weights)
            games = [(position, ["learner", "rule", "learner", "learner"]), ...]
            for result in runner.play(games, finish=trajectories):
                ...

Networks can instead run in the runner's own process, for every worker's
agents at once (`networks`, and agents that define `ask` and `act`): then
each network gets one batch per round from all the workers together, instead
of every worker running its own small batches on its own copy of it.

`play_many` is the one-off form, with the same agents in every deal:

    for deal in play_many(random_positions(10_000, rng), [RuleBot()] * 4):
        print(deal.scores)
"""

from __future__ import annotations

import gc
import multiprocessing
import os
import pickle
import queue
import threading
import traceback
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from itertools import islice
from typing import Any, Protocol

from danish_wist.actions import Action
from danish_wist.bots import Agent
from danish_wist.game import Deal, PlayerView
from learn.arena import Position

CHUNKS_PER_WORKER = 2  # chunks of games queued per worker, to keep each one busy
# A worker is one process per core, so its numerical libraries get one thread each. They
# read these when they load, before any agent is made. Otherwise NumPy's BLAS starts a
# thread per core in every worker: on 24 cores that made a NumPy arena several times slower.
# Workers sharing a GPU keep what memory PyTorch caches in segments that can grow, so that
# batches of many shapes do not leave each worker holding gigabytes of fragments.
WORKER_ENVIRONMENT = dict.fromkeys(
    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"), "1"
) | {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
SEATS = ("seat 0", "seat 1", "seat 2", "seat 3")  # agent names used by play_many


class BatchAgent(Protocol):
    """An agent that decides many views in one call, e.g. one network evaluation."""

    def choose_batch(self, views: list[PlayerView]) -> list[Action]: ...


@dataclass(frozen=True)
class Decision:
    """A decision with its context, for agents that record what they do in training."""

    game: int  # the game's index in the stream given to `Runner.play`
    seat: int
    view: PlayerView  # what the player may know: a policy must use only this
    deal: Deal  # everything, hidden cards included: for training targets only


class DecisionAgent(Protocol):
    """An agent given `Decision`s instead of views (it defines `choose_decisions`)."""

    def choose_decisions(self, decisions: list[Decision]) -> list[Action]: ...


class ServedAgent(Protocol):
    """An agent whose network runs in the runner's process (`Runner(networks=...)`).

    Each round, `ask` turns the agent's decisions into a question for its
    network, and `act` chooses from the network's answer. The questions of
    every served agent in every worker go to the runner's `networks` in one
    call, so each network sees one batch per round for all the workers.
    """

    network: str  # the name of its network, for the runner's `networks`

    def ask(self, decisions: list[Decision]) -> Any: ...

    def act(self, decisions: list[Decision], question: Any, answer: Any) -> list[Action]: ...


AnyAgent = Agent | BatchAgent | DecisionAgent | ServedAgent
Networks = Callable[[list[tuple[str, Any]]], list[Any]]  # (network, question) pairs -> answers
Game = tuple[Position, Sequence[str]]  # a position and the name of the agent in each seat
Finish = Callable[[int, Deal, Mapping[str, AnyAgent]], Any]  # (game, deal, agents) -> result


class Runner:
    """Worker processes, each with its own agents, that stay up between plays.

    `make_agents(worker)` returns the agents by name. It runs once in each
    worker (numbered 0, 1, ...), so it must be a module-level function (or a
    `functools.partial` of one), as workers are started with *spawn*. With
    `workers=1` everything runs in this process instead, with the agents made
    here. `workers=None` means one per core.

    Games are dealt to workers in fixed chunks, chunk k to worker k mod
    `workers`, and each worker plays its chunks in order. So an agent that
    seeds its randomness from a broadcast seed and its worker number makes
    the same choices whenever the same games are played with the same number
    of workers.

    `networks`, if given, answers the questions of served agents (see
    `ServedAgent`): here, for all the workers together. The workers then keep
    in step, one round each, so that the networks see the same batches
    whenever the same games are played.
    """

    def __init__(
        self,
        make_agents: Callable[[int], Mapping[str, AnyAgent]],
        *,
        workers: int | None = None,
        games_in_flight: int = 256,
        networks: Networks | None = None,
    ) -> None:
        self.workers = (os.cpu_count() or 1) if workers is None else max(workers, 1)
        self.games_in_flight = games_in_flight
        self.networks = networks
        self._playing = self._closed = False
        if self.workers == 1:
            self._agents = make_agents(0)
            return
        context = multiprocessing.get_context("spawn")  # macOS's default; safe everywhere
        self._results = context.Queue()
        self._tasks = [context.Queue() for _ in range(self.workers)]
        pipes = [context.Pipe(duplex=False) for _ in range(self.workers)]  # the networks' answers
        self._answers = [send for _, send in pipes]
        self._processes = [
            context.Process(
                target=_work,
                args=(make_agents, number, tasks, self._results, receive),
                daemon=True,
                name=f"runner worker {number}",
            )
            for number, (tasks, (receive, _)) in enumerate(zip(self._tasks, pipes, strict=True))
        ]
        with _environment(WORKER_ENVIRONMENT):
            for process in self._processes:
                process.start()
        try:
            self._gather("ready")  # errors in make_agents surface here
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> Runner:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def play(self, games: Iterable[Game], finish: Finish | None = None) -> Iterator[Any]:
        """Play every game to the end, yielding the results in any order.

        Each game is a `Position` and the name of the agent in each seat; the
        same agent object in several seats or games gets one batch for all of
        them. `finish(game, deal, agents)` runs where the game was played, when
        it ends, and its result is sent back instead of the deal; `game` is the
        game's index in `games`. Sending a whole deal back from a worker costs
        about 25 µs there and 40 µs here, so send back only what you need.
        `games` is read lazily, so it may be endless. One play at a time: it
        starts with the first result asked for and ends with the last, or when
        the iterator is closed.
        """
        self._check()
        if self.workers == 1:
            return self._play_here(games, finish)
        return self._play_in_workers(games, finish)

    def broadcast(self, name: str, method: str, *args: Any) -> list[Any]:
        """Call `method(*args)` on the agent called `name` in every worker.

        Returns each worker's result, in worker order. Plays started afterwards
        see the change; it may not be called while a play is going.
        """
        self._check()
        self._playing = True
        try:
            if self.workers == 1:
                return [getattr(self._agents[name], method)(*args)]
            call = _pickle(("call", name, method, args))  # once, for every worker
            for tasks in self._tasks:
                tasks.put(call)
            return self._gather("called")
        finally:
            self._playing = False

    def close(self) -> None:
        """Stop the workers. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        if self.workers == 1:
            return
        for tasks, process in zip(self._tasks, self._processes, strict=True):
            if process.is_alive():
                tasks.put(None)
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()

    def _check(self) -> None:
        if self._closed:
            raise RuntimeError("the runner is closed")
        if self._playing:
            raise RuntimeError("a play is still going: finish or close it first")

    # --- In this process -----------------------------------------------------

    def _play_here(self, games: Iterable[Game], finish: Finish | None) -> Iterator[Any]:
        self._check()
        self._playing = True
        try:
            numbered = ((index, *game) for index, game in enumerate(games))
            playing = _play(numbered, self._agents, self.games_in_flight, self.networks)
            for index, deal in playing:
                yield finish(index, deal, self._agents) if finish else deal
        finally:
            self._playing = False

    # --- In worker processes -------------------------------------------------

    def _play_in_workers(self, games: Iterable[Game], finish: Finish | None) -> Iterator[Any]:
        count = len(games) if isinstance(games, Sequence) else None
        size = chunk_size(count, self.workers, self.games_in_flight)
        numbered = ((index, *game) for index, game in enumerate(games))
        chunks = iter(lambda: list(islice(numbered, size)), [])

        self._check()
        self._playing = True
        out = [0] * self.workers  # chunks sent to each worker and not yet back
        held: list[deque] = [deque() for _ in range(self.workers)]  # messages not yet handled
        sent, chunk = 0, next(chunks, None)  # chunk k goes to worker k mod workers
        error = None
        try:
            while True:
                while chunk is not None and out[sent % self.workers] < CHUNKS_PER_WORKER:
                    task = _pickle(("play", chunk, self.games_in_flight, finish))
                    self._tasks[sent % self.workers].put(task)
                    out[sent % self.workers] += 1
                    sent, chunk = sent + 1, next(chunks, None)
                if not any(out):
                    break
                results, error = self._round(out, held)
                if error:
                    break
                for found in results:
                    yield from found
        finally:
            # If the caller stopped early or a worker failed, play out (and drop) the
            # chunks still out, so that the next play starts clean.
            self._drain(out, held)
            self._playing = False
        if error:
            raise error

    def _round(self, out: list[int], held: list[deque]) -> tuple[list[list], Exception | None]:
        """Handle a round of messages; the chunks' results that came back, and any error.

        Without networks a round is the next message. With them it is one
        message from every worker still playing, and the questions among them
        are answered together: the workers keep in step, so the networks see
        the same batches whenever the same games are played.
        """
        if self.networks is None:
            kind, worker, payload = self._receive()
            messages = [(worker, kind, payload)]
        else:
            busy = [worker for worker in range(self.workers) if out[worker]]
            while not all(held[worker] for worker in busy):
                kind, worker, payload = self._receive()
                held[worker].append((kind, payload))
            messages = [(worker, *held[worker].popleft()) for worker in busy]
        results, error, questions = [], None, []
        for worker, kind, payload in messages:
            if kind == "ask":
                questions.append((worker, _unpickle(payload)))
                continue
            out[worker] -= 1
            if kind == "error":
                error = error or _unpickle(payload)
            else:
                results.append(_unpickle(payload))
        if questions:
            error = error or self._answer(questions)
        return results, error

    def _answer(self, questions: list[tuple[int, list]]) -> Exception | None:
        """Answer workers' questions (worker, [(network, question), ...]) in one call of the
        networks; if that fails, tell the workers to stop, and return the error."""
        try:
            answers = iter(self.networks([entry for _, entries in questions for entry in entries]))
            replies = [[next(answers) for _ in entries] for _, entries in questions]
        except Exception as error:
            error.add_note("in the runner's networks")
            replies, failed = [None] * len(questions), error
        else:
            failed = None
        for (worker, _), reply in zip(questions, replies, strict=True):
            self._answers[worker].send_bytes(_pickle(reply))
        return failed

    def _drain(self, out: list[int], held: list[deque]) -> None:
        """Play out the chunks still out, answering their questions, and drop their results."""
        while any(out) and not self._closed:
            worker = next((worker for worker in range(self.workers) if held[worker]), None)
            if worker is None:
                kind, worker, payload = self._receive()
            else:
                kind, payload = held[worker].popleft()
            if kind == "ask":
                self._answer([(worker, _unpickle(payload))])
            else:
                out[worker] -= 1

    def _gather(self, kind: str) -> list[Any]:
        """One reply of `kind` from every worker, in worker order; raises the first error."""
        replies: list[Any] = [None] * self.workers
        errors = []
        for _ in range(self.workers):
            got, worker, payload = self._receive()
            if got == "error":
                errors.append(_unpickle(payload))
            else:
                assert got == kind, (got, kind)
                replies[worker] = _unpickle(payload)
        if errors:
            raise errors[0]
        return replies

    def _receive(self) -> tuple[str, int, Any]:
        while True:
            try:
                return self._results.get(timeout=1)
            except queue.Empty:
                stopped = [p.name for p in self._processes if not p.is_alive()]
                if stopped:
                    self.close()
                    raise RuntimeError(f"{', '.join(stopped)} stopped") from None


def chunk_size(count: int | None, workers: int, games_in_flight: int) -> int:
    """How many games to send a worker at a time.

    Big enough to refill the deals in flight a few times. For a known number
    of games, also small enough that every worker gets the same number of
    chunks, since chunk k always goes to worker k mod `workers`.
    """
    largest = 4 * games_in_flight
    if count is None:
        return largest
    rounds = max(1, -(-count // (workers * largest)))  # chunks per worker
    return max(1, -(-count // (workers * rounds)))


def play_many(
    positions: Iterable[Position],
    agents_by_seat: Sequence[AnyAgent] | Callable[[int], Sequence[AnyAgent]],
    *,
    games_in_flight: int = 256,
    workers: int | None = None,
    finish: Callable[[Deal], Any] | None = None,
) -> Iterator[Any]:
    """Play every position with the same agents, yielding finished deals in any order.

    `agents_by_seat` is the four agents, or a function that makes them from a
    worker number (use that for agents holding randomness, so each worker gets
    its own). `finish(deal)`, if given, runs where the deal was played and its
    result is yielded instead. Starts a `Runner` for this call alone; see it
    for `workers` and `games_in_flight`.
    """
    if isinstance(positions, Sequence):
        games: Iterable[Game] = [(position, SEATS) for position in positions]
    else:
        games = ((position, SEATS) for position in positions)
    make_agents = partial(_seated, agents_by_seat)
    with Runner(make_agents, workers=workers, games_in_flight=games_in_flight) as runner:
        yield from runner.play(games, partial(_finish_deal, finish) if finish else None)


def _seated(agents_by_seat, worker: int) -> dict[str, AnyAgent]:
    agents = agents_by_seat(worker) if callable(agents_by_seat) else agents_by_seat
    assert len(agents) == len(SEATS), "one agent per seat"
    return dict(zip(SEATS, agents, strict=True))


def _finish_deal(finish: Callable[[Deal], Any], game: int, deal: Deal, agents) -> Any:
    return finish(deal)


# --- The batched loop --------------------------------------------------------


@dataclass(slots=True)
class _Playing:
    index: int  # the game's index in the stream
    deal: Deal
    queues: list[list[_Playing]]  # for each seat, the queue of its agent's decisions


def _play(
    games: Iterable[tuple[int, Position, Sequence[str]]],
    agents: Mapping[str, AnyAgent],
    games_in_flight: int,
    ask: Networks | None = None,
) -> Iterator[tuple[int, Deal]]:
    """Play numbered games, keeping up to `games_in_flight` going; yield (game, deal).

    `ask` answers the served agents' questions (see `ServedAgent`), all of a round's at once.
    """
    # Each round, every deal joins the queue of the agent whose turn it is.
    queues = {id(agent): (agent, []) for agent in agents.values()}

    def start(index: int, position: Position, names: Sequence[str]) -> _Playing:
        assert len(names) == len(SEATS), f"game {index}: one agent name per seat"
        return _Playing(index, position.start(), [queues[id(agents[n])][1] for n in names])

    games = iter(games)
    in_flight = [start(*game) for game in islice(games, games_in_flight)]
    while in_flight:
        for game in in_flight:
            game.queues[game.deal.to_act].append(game)
        served = []
        for agent, waiting in queues.values():
            if waiting and hasattr(agent, "network"):
                served.append((agent, list(waiting)))
            elif waiting:
                for game, action in zip(waiting, _decide(agent, waiting), strict=True):
                    game.deal.apply(action)
            waiting.clear()
        if served:
            _serve(served, ask)
        playing = []
        for game in in_flight:
            if game.deal.is_over:
                yield game.index, game.deal
            else:
                playing.append(game)
        free = games_in_flight - len(playing)
        in_flight = playing + [start(*game) for game in islice(games, free)]


def _decide(agent: AnyAgent, games: list[_Playing]) -> list[Action]:
    if hasattr(agent, "choose_decisions"):
        return agent.choose_decisions([_decision(game) for game in games])
    views = [game.deal.view(game.deal.to_act) for game in games]
    if hasattr(agent, "choose_batch"):
        return agent.choose_batch(views)
    return [agent.choose(view) for view in views]


def _decision(game: _Playing) -> Decision:
    return Decision(game.index, game.deal.to_act, game.deal.view(game.deal.to_act), game.deal)


def _serve(served: list[tuple[ServedAgent, list[_Playing]]], ask: Networks | None) -> None:
    """Ask the networks of every served agent's decisions at once, then apply their choices."""
    if ask is None:
        raise TypeError(f"{type(served[0][0]).__name__} needs a runner with networks")
    decisions = [[_decision(game) for game in games] for _, games in served]
    questions = [agent.ask(found) for (agent, _), found in zip(served, decisions, strict=True)]
    answers = ask([(agent.network, q) for (agent, _), q in zip(served, questions, strict=True)])
    for (agent, games), found, question, answer in zip(
        served, decisions, questions, answers, strict=True
    ):
        for game, action in zip(games, agent.act(found, question, answer), strict=True):
            game.deal.apply(action)


# --- Worker processes --------------------------------------------------------


def _work(make_agents, number: int, tasks, results, answers) -> None:
    """A worker's life: make its agents, then play chunks and run calls until told to stop.

    Everything sent is pickled here first, so that a failure to pickle is
    reported like any other error instead of being lost in the queue's thread.
    """
    parent = multiprocessing.parent_process()
    threading.Thread(target=_exit_with, args=(parent,), daemon=True).start()
    try:
        agents = make_agents(number)
        results.put(("ready", number, _pickle(None)))
    except BaseException as error:
        results.put(("error", number, _pickled_error(error)))
        return
    ask = partial(_ask, number, results, answers)
    while (task := tasks.get()) is not None:
        try:
            kind, *details = pickle.loads(task)
            if kind == "play":
                chunk, games_in_flight, finish = details
                out = [
                    finish(index, deal, agents) if finish else deal
                    for index, deal in _play(chunk, agents, games_in_flight, ask)
                ]
                results.put(("done", number, _pickle(out)))
            else:
                name, method, args = details
                results.put(("called", number, _pickle(getattr(agents[name], method)(*args))))
        except BaseException as error:
            results.put(("error", number, _pickled_error(error)))


class _Stopped(Exception):
    """The runner could not answer: its networks failed."""


def _ask(number: int, results, answers, entries: list[tuple[str, Any]]) -> list[Any]:
    """In a worker: the answers of the runner's networks to its served agents' questions."""
    results.put(("ask", number, _pickle(entries)))
    found = pickle.loads(answers.recv_bytes())
    if found is None:
        raise _Stopped("the runner's networks failed")
    return found


def _exit_with(parent: multiprocessing.process.BaseProcess) -> None:
    """Stop this worker as soon as the process that started it has gone.

    A parent that is killed cannot stop its workers, and they would play on
    for nothing, holding cores (and GPU memory) until they finished.
    """
    parent.join()
    os._exit(1)


@contextmanager
def _environment(variables: Mapping[str, str]) -> Iterator[None]:
    """Set environment variables for processes started meanwhile, then restore them."""
    saved = {name: os.environ.get(name) for name in variables}
    os.environ.update(variables)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                del os.environ[name]
            else:
                os.environ[name] = value


def _pickle(value: Any) -> bytes:
    return pickle.dumps(value, pickle.HIGHEST_PROTOCOL)


def _pickled_error(error: BaseException) -> bytes:
    """The error itself, noting where it happened, or a RuntimeError if it will not pickle."""
    error.add_note(f"in runner worker:\n{''.join(traceback.format_exception(error))}")
    try:
        return _pickle(error)
    except Exception:
        return _pickle(RuntimeError(f"{type(error).__name__}: {error}\n{error.__notes__[-1]}"))


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
