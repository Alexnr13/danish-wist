"""TRAINING.md "Next", Task A, step 1, the fallback: collecting in a process of its own, with
its own CUDA context and its own copy of the networks, while this process prepares and
updates. Timing only; see `timing.py`.

The collecting process owns the runner (22 workers) and the served networks. Each iteration
this process sends it the weights to load (the learner's, and league members new to a slot),
the workers' seed and the games, and it sends back the learner's trajectories.

    python results/overlap-prototype/timing_proc.py --iterations 15
"""

import argparse
import json
import multiprocessing
import sys
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from timing import WORKERS, proximal, restore  # noqa: E402

from learn import selfplay as sp  # noqa: E402
from learn.arena import random_positions  # noqa: E402
from learn.model import NetConfig  # noqa: E402
from learn.runner import Runner  # noqa: E402


def collecting(connection, settings: dict, config: dict, device: str) -> None:
    """The collecting process: play what it is sent until it is sent None."""
    settings = sp.Settings(**settings)
    make = partial(
        sp.make_agents,
        slots=sp.SLOTS,
        explore=settings.explore_bids,
        explore_levels=settings.explore_levels,
    )
    served = sp.networks(NetConfig(**config), sp.SLOTS, device)
    with Runner(
        make, workers=WORKERS, games_in_flight=settings.games_in_flight, networks=served
    ) as runner:
        connection.send("ready")
        while (message := connection.recv()) is not None:
            loads, seed, games = message
            started, cpu = time.perf_counter(), time.process_time()
            for name, state in loads.items():
                runner.networks.load(name, state)
            runner.broadcast(sp.LEARNER, "seed", seed)
            found = []
            for _, trajectories, _scores in runner.play(games, finish=sp.trajectories):
                found += trajectories
            connection.send((found, time.perf_counter() - started, time.process_time() - cpu))


class _Loads:
    def __init__(self) -> None:
        self.loads: dict[str, dict] = {}

    def load(self, name: str, state: dict) -> None:
        self.loads[name] = state


class _Recorder:
    """Stands in for the runner while drawing lineups: it records the slots' loads."""

    def __init__(self) -> None:
        self.networks = _Loads()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=15, help="of sync and of overlap")
    args = parser.parse_args()
    device = "cuda"
    settings, learning, league, rng = restore(device)
    loaded: dict[int, str] = {}
    context = multiprocessing.get_context("spawn")
    here, there = context.Pipe()
    config = asdict(learning.policy.config)
    child = context.Process(target=collecting, args=(there, asdict(settings), config, device))
    child.start()
    assert here.recv() == "ready"

    def message() -> tuple:
        recorder = _Recorder()
        lineups, _ = sp._league_lineups(recorder, league, loaded, settings, rng)
        seed = rng.getrandbits(32)
        games = list(zip(random_positions(len(lineups), rng), lineups, strict=True))
        loads = {sp.LEARNER: sp._cpu_state(learning.policy)} | recorder.networks.loads
        return loads, seed, games

    def learn(found: list) -> int:
        batch = sp.prepare(found, learning.critic, settings)
        batch.old_log_probs = proximal(learning.policy, batch, settings.chunk)
        sp.update(
            learning.policy,
            learning.critic,
            learning.magnet,
            learning.optimisers,
            batch,
            settings,
            rng,
        )
        torch.cuda.current_stream().synchronize()
        return len(batch.actions)

    rows = []
    for i in range(3):  # compiling and capturing
        here.send(message())
        found, collect_s, _ = here.recv()
        started = time.perf_counter()
        learn(found)
        update_s = time.perf_counter() - started
        print(f"warm-up {i}: collect {collect_s:.2f} s, update {update_s:.2f} s", flush=True)
    for _ in range(args.iterations):  # one after the other, through the other process
        started = time.perf_counter()
        here.send(message())
        found, collect_s, cpu = here.recv()
        received = time.perf_counter()
        count = learn(found)
        ended = time.perf_counter()
        rows.append(
            {
                "mode": "sync-proc",
                "collect_there": collect_s,
                "collect": received - started,
                "update": ended - received,
                "both": ended - started,
                "decisions": count,
                "cpu_there": cpu,
            }
        )
    here.send(message())
    pending, *_ = here.recv()
    for _ in range(args.iterations):  # overlapped
        started = time.perf_counter()
        here.send(message())
        sent = time.perf_counter()
        count = learn(pending)
        updated = time.perf_counter()
        pending, collect_s, cpu = here.recv()
        ended = time.perf_counter()
        rows.append(
            {
                "mode": "overlap-proc",
                "send": sent - started,
                "collect_there": collect_s,
                "update": updated - sent,
                "receive": ended - updated,
                "both": ended - started,
                "decisions": count,
                "cpu_there": cpu,
            }
        )
    here.send(None)
    child.join()
    for mode in ("sync-proc", "overlap-proc"):
        found = [row for row in rows if row["mode"] == mode]
        values = {key: [row[key] for row in found] for key in found[0] if key != "mode"}
        summary = " ".join(f"{k}={np.mean(v):.3f}±{np.std(v):.3f}" for k, v in values.items())
        print(f"{mode} ({len(found)}): {summary}", flush=True)
    (HERE / "rows").mkdir(exist_ok=True)
    (HERE / "rows" / f"timing-proc-{args.iterations}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
