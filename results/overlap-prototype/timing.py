"""TRAINING.md "Next", Task A, step 1: a timing prototype. Nothing learns differently and
nothing is saved but the timings.

From rl-006's state at 18,000 with its league (read only), on the workstation with 22 workers,
it times per iteration (1024 deals, lineups drawn from the league as in the run):

- update-alone: `prepare` and `update` on one saved batch, with the proximal pass
- collect-alone: collecting only
- sync: collecting, then preparing and updating, as the run does now
- overlap: collecting batch k+1 in a thread, on a CUDA stream of its own, with the served
  copy of the weights, while this thread recomputes the proximal log-chances of batch k
  (`proximal`), prepares it and updates on it; the design in TRAINING.md "Next"
- with --stand-in: an update stand-in of few launches and little Python (`stand_in`), alone and
  overlapped with collecting: how far the overlap would go if the update left Python's lock free

    python results/overlap-prototype/timing.py --iterations 20
    python results/overlap-prototype/timing.py --iterations 6 --profile   # under nsys: trace.py

Graphs of the served forward pass are captured with `capture_error_mode="thread_local"`, so
that one captured in the collecting thread does not fail the other thread's CUDA calls.
"""

import argparse
import json
import random
import sys
import threading
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from learn import model  # noqa: E402
from learn import selfplay as sp  # noqa: E402
from learn.arena import random_positions  # noqa: E402
from learn.model import device_of, load  # noqa: E402
from learn.runner import Runner  # noqa: E402

HERE = Path(__file__).parent
STATE = ROOT / "runs/rl-006-state-18000"
WORKERS = 22
CAPTURES: list[str] = []  # the thread each graph was captured in


def _capture(self, rows: int, length: int) -> tuple:
    """`model._Graphed._capture`, with a capture that other threads' CUDA calls do not fail."""
    device = self.weights[0].device
    tokens = torch.zeros(rows, length, 5, dtype=torch.int32, device=device)
    padding = torch.ones(rows, length, dtype=torch.bool, device=device)
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        for _ in range(2):
            model._logits(self.net, tokens, padding)
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=self.pool, capture_error_mode="thread_local"):
        logits = model._logits(self.net, tokens, padding)
    self.pool = graph.pool()
    CAPTURES.append(threading.current_thread().name)
    return graph, tokens, padding, logits


model._Graphed._capture = _capture


def _log_probs_of(policy, tokens, padding, legal, actions):
    logits = policy.heads(sp._summarise(policy, tokens, padding), legal)[0]
    return torch.log_softmax(logits, dim=-1).gather(1, actions[:, None]).squeeze(-1)


@torch.no_grad()
def proximal(policy, batch: sp.Batch, chunk: int) -> torch.Tensor:
    """The log-chance of each move under `policy` as it is: the proximal policy's, in the
    update's compiled bfloat16 passes (`_passes`, `_compiled`)."""
    device = device_of(policy)
    count = len(batch.actions)
    rows = sp._Rows()
    passes = [(rows.add(part), counted) for part, counted in sp._passes(list(range(count)), chunk)]
    rows.to(device)
    found = torch.empty(count, device=device)
    log_probs_of = sp._compiled(_log_probs_of, device)
    with sp._attention(device):
        for part, counted in passes:
            index = rows[part]
            chunk_found = log_probs_of(
                policy, *batch.inputs.take(index), batch.legal[index], batch.actions[index]
            )
            found[index[:counted]] = chunk_found[:counted]
    return found


def restore(device: str) -> tuple[sp.Settings, sp.Learning, sp.League, random.Random]:
    """rl-006's networks, optimisers, league and random state at 18,000."""
    started = json.loads((STATE / "settings.json").read_text())
    settings = sp.Settings(**(started | {"league_exploiters": 20, "exploit_every": 200}))
    policy, critic = load(str(STATE / "policy.pt")), load(str(STATE / "critic.pt"))
    learning = sp.Learning.start(policy, critic, settings, device)
    state = torch.load(STATE / "state.pt", map_location="cpu", weights_only=True)
    for net, name in [
        (learning.policy, "policy"),
        (learning.critic, "critic"),
        (learning.magnet, "magnet"),
    ]:
        net.load_state_dict(state[name])
    for optimiser, saved in zip(learning.optimisers, state["optimisers"], strict=True):
        optimiser.load_state_dict(saved)
    league = sp.League(settings.league_size, STATE / "league", settings.league_exploiters)
    league.load_state_dict(state["league"])  # reads the members' files; nothing is written
    rng = random.Random(0)
    rng.setstate(state["rng"])
    return settings, learning, league, rng


def stand_in(square: torch.Tensor) -> None:
    """About the update's time on the GPU (0.85 s) in 144 steps of nine large products, each
    step ending in a wait for the GPU as `_step` does: few launches and little Python."""
    x = square
    for _ in range(144):
        for _ in range(9):
            x = (x @ square).clamp(-1, 1)
        x.sum().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20, help="of sync and of overlap")
    parser.add_argument("--priority", type=int, default=-5, help="the collecting stream's")
    parser.add_argument("--switch", type=float, default=0.005, help="the GIL's interval, s")
    parser.add_argument("--profile", action="store_true", help="for nsys: sync and overlap")
    parser.add_argument("--stand-in", action="store_true", help="also the update stand-in")
    args = parser.parse_args()
    sys.setswitchinterval(args.switch)
    device = "cuda"
    settings, learning, league, rng = restore(device)
    make = partial(
        sp.make_agents,
        slots=sp.SLOTS,
        explore=settings.explore_bids,
        explore_levels=settings.explore_levels,
    )
    served = sp.networks(learning.policy.config, sp.SLOTS, device)
    stream = torch.cuda.Stream(device, priority=args.priority)
    print(f"collecting stream priority {stream.priority}", flush=True)
    loaded: dict[int, str] = {}
    rows: list[dict] = []
    alone = 0 if args.profile else 5

    with Runner(
        make, workers=WORKERS, games_in_flight=settings.games_in_flight, networks=served
    ) as runner:

        def draw() -> tuple[int, list]:
            """Everything a batch's collecting draws, drawn here: lineups, seed, positions."""
            lineups, _ = sp._league_lineups(runner, league, loaded, settings, rng)
            seed = rng.getrandbits(32)
            return seed, list(zip(random_positions(len(lineups), rng), lineups, strict=True))

        def play(seed: int, games: list) -> list:
            runner.broadcast(sp.LEARNER, "seed", seed)
            found = []
            for _, trajectories, _scores in runner.play(games, finish=sp.trajectories):
                found += trajectories
            return found

        def serve() -> None:  # the learner's weights into the served copy
            runner.networks.load(sp.LEARNER, learning.policy.state_dict())

        def learn(found: list, with_proximal: bool) -> tuple[int, float]:
            batch = sp.prepare(found, learning.critic, settings)
            started = time.perf_counter()
            if with_proximal:
                batch.old_log_probs = proximal(learning.policy, batch, settings.chunk)
                torch.cuda.current_stream().synchronize()
            proximal_s = time.perf_counter() - started
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
            return len(batch.actions), proximal_s

        def overlapped(update: Callable) -> tuple[list, object, dict]:
            """Collect the next batch in a thread while `update` runs here: the batch, what
            `update` returned, and the timings."""
            started = time.perf_counter()
            serve()
            seed, games = draw()
            stream.wait_stream(torch.cuda.current_stream())  # the served copy is loaded
            box: dict = {}

            def collecting() -> None:
                with torch.cuda.stream(stream):
                    begun, cpu = time.perf_counter(), time.thread_time()
                    box["found"] = play(seed, games)
                    box["collect"] = time.perf_counter() - begun
                    box["cpu_collect_thread"] = time.thread_time() - cpu

            thread = threading.Thread(target=collecting, name="collecting")
            cpu = time.process_time()
            thread.start()
            begun = time.perf_counter()
            result = update()
            updated = time.perf_counter()
            thread.join()
            timings = {
                "update": updated - begun,
                "both": time.perf_counter() - started,
                "cpu_all": time.process_time() - cpu,
            }
            return box.pop("found"), result, box | timings

        for i in range(3):  # compiling and capturing
            started = time.perf_counter()
            serve()
            found = play(*draw())
            collected = time.perf_counter()
            count, _ = learn(found, True)
            print(
                f"warm-up {i}: collect {collected - started:.2f} s, "
                f"update {time.perf_counter() - collected:.2f} s, {count} decisions",
                flush=True,
            )
        if args.profile:
            torch.cuda.profiler.start()
        serve()
        saved = play(*draw())
        for _ in range(alone):  # the weights move, the batch stays: timing only
            started = time.perf_counter()
            count, proximal_s = learn(saved, True)
            update_s = time.perf_counter() - started
            rows.append({"mode": "update-alone", "update": update_s, "prox": proximal_s})
        for _ in range(alone):
            started = time.perf_counter()
            serve()
            play(*draw())
            rows.append({"mode": "collect-alone", "collect": time.perf_counter() - started})
        for _ in range(args.iterations):
            torch.cuda.nvtx.range_push("sync")
            started, cpu = time.perf_counter(), time.process_time()
            serve()
            found = play(*draw())
            collected, cpu_collected = time.perf_counter(), time.process_time()
            main_cpu_collected = time.thread_time()
            count, _ = learn(found, False)
            torch.cuda.nvtx.range_pop()
            ended = time.perf_counter()
            rows.append(
                {
                    "mode": "sync",
                    "collect": collected - started,
                    "update": ended - collected,
                    "both": ended - started,
                    "decisions": count,
                    "cpu_collect": cpu_collected - cpu,
                    "cpu_update": time.process_time() - cpu_collected,
                    "cpu_update_main": time.thread_time() - main_cpu_collected,
                }
            )
        serve()
        pending = play(*draw())
        for _ in range(args.iterations):
            torch.cuda.nvtx.range_push("overlap")
            pending, (count, proximal_s), timings = overlapped(partial(learn, pending, True))
            torch.cuda.nvtx.range_pop()
            rows.append({"mode": "overlap", "prox": proximal_s, **timings, "decisions": count})
        if args.profile:
            torch.cuda.profiler.stop()
        if args.stand_in:
            square = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16) / 64
            for _ in range(alone):
                started = time.perf_counter()
                stand_in(square)
                rows.append({"mode": "stand-in-alone", "update": time.perf_counter() - started})
            for _ in range(args.iterations):
                _, _, timings = overlapped(partial(stand_in, square))
                rows.append({"mode": "stand-in-overlap", **timings})

    modes = ["update-alone", "collect-alone", "sync", "overlap", "stand-in-alone"]
    for mode in [*modes, "stand-in-overlap"]:
        found = [row for row in rows if row["mode"] == mode]
        if found:
            values = {key: [row[key] for row in found] for key in found[0] if key != "mode"}
            summary = " ".join(
                f"{key}={np.mean(v):.3f}±{np.std(v):.3f}" for key, v in values.items()
            )
            print(f"{mode} ({len(found)}): {summary}", flush=True)
    print(f"graphs captured: {len(CAPTURES)}, in the collecting thread:", end=" ")
    print(CAPTURES.count("collecting"), flush=True)
    name = f"timing-{args.iterations}-p{args.priority}-s{args.switch}"
    name += "-profile" if args.profile else "-stand-in" if args.stand_in else ""
    (HERE / "rows").mkdir(exist_ok=True)
    (HERE / "rows" / f"{name}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
